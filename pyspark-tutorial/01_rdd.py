"""
01 — RDD: the low-level API
===========================

REAL-WORLD SCENARIO
-------------------
You run a web service. Nginx access logs arrive as raw text, one request per
line, in a format that is NOT tabular — the user-agent contains spaces and
quotes, some lines are truncated by log rotation, some are malformed entirely.

You need: requests per status code, top offending IPs, p95 response size, and
a count of unparseable lines for an alert.

WHY RDD AND NOT DATAFRAME
-------------------------
This is one of the few cases where RDDs genuinely fit. The input is arbitrary
text with no schema, parsing needs Python logic that returns EITHER a parsed
record OR a failure marker, and you want to route bad records to a separate
path. Once parsed, you convert to a DataFrame and never touch RDDs again.

For anything already tabular, DataFrames win on every axis — see 02.

Run:
    spark-submit 01_rdd.py        # or paste into a Databricks notebook
"""

import re
from pyspark.sql import SparkSession, Row

spark = (SparkSession.builder
         .appName("rdd-log-parsing")
         .master("local[4]")
         .getOrCreate())
sc = spark.sparkContext
sc.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Section 1 — Sample data (a realistic nginx combined log, including bad lines)
# ---------------------------------------------------------------------------

RAW_LINES = [
    '10.0.0.1 - - [09/Aug/2026:06:15:01 +0000] "GET /api/orders HTTP/1.1" 200 1534 "-" "Mozilla/5.0"',
    '10.0.0.2 - - [09/Aug/2026:06:15:02 +0000] "POST /api/checkout HTTP/1.1" 201 892 "-" "curl/8.1"',
    '10.0.0.1 - - [09/Aug/2026:06:15:03 +0000] "GET /api/orders/99 HTTP/1.1" 404 128 "-" "Mozilla/5.0"',
    '10.0.0.9 - - [09/Aug/2026:06:15:04 +0000] "GET /health HTTP/1.1" 200 15 "-" "kube-probe/1.28"',
    'TRUNCATED LINE FROM LOG ROTATION',                                    # malformed
    '10.0.0.3 - - [09/Aug/2026:06:15:06 +0000] "DELETE /api/cart HTTP/1.1" 500 245 "-" "okhttp/4.9"',
    '',                                                                    # empty
    '10.0.0.1 - - [09/Aug/2026:06:15:08 +0000] "GET /api/orders HTTP/1.1" 200 2201 "-" "Mozilla/5.0"',
    '10.0.0.7 - - [09/Aug/2026:06:15:09 +0000] "GET /admin HTTP/1.1" 403 90 "-" "python-requests/2.31"',
    'garbage ### not a log line at all',                                   # malformed
]

# Simulate a bigger file so partitioning is meaningful
lines = sc.parallelize(RAW_LINES * 500, numSlices=4)

print("partitions:", lines.getNumPartitions())
print("total lines:", lines.count())

# EXPECTED:
#   partitions: 4
#   total lines: 5000
#
# parallelize(data, numSlices) distributes a local collection. numSlices sets
# the partition count directly. In real code you would use sc.textFile(path),
# where partition count comes from file size and block layout instead.

# ---------------------------------------------------------------------------
# Section 2 — Parse, routing failures instead of dropping them
# ---------------------------------------------------------------------------

LOG_RE = re.compile(
    r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+) [^"]*" (\d{3}) (\d+)'
)


def parse(line):
    """Return ('ok', Row) or ('bad', original_line). Never raises."""
    m = LOG_RE.match(line)
    if not m:
        return ("bad", line)
    ip, ts, method, path, status, size = m.groups()
    return ("ok", Row(ip=ip, ts=ts, method=method, path=path,
                      status=int(status), size=int(size)))


parsed = lines.map(parse).cache()

# .cache() matters here: `parsed` is consumed twice below (ok + bad branches).
# Without it the regex runs twice over every line — RDDs have no automatic
# result reuse, exactly like DataFrames.

ok  = parsed.filter(lambda t: t[0] == "ok").map(lambda t: t[1])
bad = parsed.filter(lambda t: t[0] == "bad").map(lambda t: t[1])

n_ok, n_bad = ok.count(), bad.count()
print(f"parsed {n_ok}  failed {n_bad}  ({n_bad / (n_ok + n_bad):.1%} bad)")

# EXPECTED:
#   parsed 4000  failed 1000  (20.0% bad)
#
# THE PATTERN THAT MATTERS: parse() never raises. A Python exception inside a
# map() kills the TASK, Spark retries it, it fails again, and after 4 attempts
# the whole JOB dies — because of one bad line out of 50 million. Returning a
# tagged tuple and splitting afterwards is the standard "dead letter" approach.
#
# In production `bad` would be written to a quarantine path and alerted on,
# not just counted.

print("\nsample bad lines:")
for b in bad.distinct().take(3):
    print("  ", repr(b[:60]))

# ---------------------------------------------------------------------------
# Section 3 — Transformations: the ones worth knowing
# ---------------------------------------------------------------------------

# map — 1 row in, 1 row out
statuses = ok.map(lambda r: r.status)

# filter — keep matching
errors = ok.filter(lambda r: r.status >= 400)

# flatMap — 1 row in, N rows out. Here: split paths into segments.
segments = ok.flatMap(lambda r: [s for s in r.path.split("/") if s])
print("\ntop path segments:", segments.countByValue())

# EXPECTED (approximate):
#   top path segments: defaultdict(<class 'int'>, {'api': 2500, 'orders': 1500, ...})
#
# countByValue() returns a LOCAL dict to the driver. Safe here; dangerous on
# high-cardinality data — it is a collect() in disguise.

# distinct — dedupe (this is a SHUFFLE)
print("distinct IPs:", ok.map(lambda r: r.ip).distinct().count())

# ---------------------------------------------------------------------------
# Section 4 — reduceByKey vs groupByKey (the classic interview question)
# ---------------------------------------------------------------------------

pairs = ok.map(lambda r: (r.status, 1))

# GOOD — combines on the map side before shuffling
by_status = pairs.reduceByKey(lambda a, b: a + b).collectAsMap()
print("\nrequests by status:", dict(sorted(by_status.items())))

# EXPECTED:
#   requests by status: {200: 1500, 201: 500, 403: 500, 404: 500, 500: 500}

# BAD — shuffles every individual record, then counts
# by_status_slow = ok.map(lambda r: (r.status, 1)).groupByKey().mapValues(len)
#
# WHY reduceByKey WINS:
#   reduceByKey applies the reduce function LOCALLY on each partition first
#   (a "combiner"), so only one partial result per key per partition crosses
#   the network. groupByKey ships every raw value to the reducer, then groups.
#   With 100M rows and 5 keys, reduceByKey shuffles ~5 values per partition;
#   groupByKey shuffles 100M. groupByKey is also an OOM risk, because all
#   values for one key must fit in one executor's memory.
#
#   Use groupByKey only when you genuinely need every value (e.g. to sort
#   within a key), and even then prefer aggregateByKey or a DataFrame window.

# aggregateByKey — different input and output types
# (count, total_size) per status, in one pass
size_stats = (ok
    .map(lambda r: (r.status, r.size))
    .aggregateByKey(
        (0, 0),                                        # zero value
        lambda acc, v: (acc[0] + 1, acc[1] + v),       # within a partition
        lambda a, b: (a[0] + b[0], a[1] + b[1]))       # across partitions
    .mapValues(lambda t: round(t[1] / t[0], 1)))

print("avg response size by status:", dict(sorted(size_stats.collect())))

# EXPECTED:
#   avg response size by status: {200: 1250.0, 201: 892.0, 403: 90.0, ...}
#
# The two-function shape is the same combiner/reducer split as reduceByKey,
# but lets the accumulator differ from the value type. Note the merge function
# must be ASSOCIATIVE and COMMUTATIVE — Spark applies it in nondeterministic
# order.

# ---------------------------------------------------------------------------
# Section 5 — Actions, and the ones that will hurt you
# ---------------------------------------------------------------------------

print("\ntop 3 IPs by request count:")
top_ips = (ok.map(lambda r: (r.ip, 1))
             .reduceByKey(lambda a, b: a + b)
             .takeOrdered(3, key=lambda t: -t[1]))
for ip, n in top_ips:
    print(f"   {ip:12} {n}")

# EXPECTED:
#   10.0.0.1     1500
#   10.0.0.2      500
#   ...
#
# takeOrdered(n, key) is the SAFE top-N. It keeps only n per partition, then
# merges. Compare with .sortBy(...).take(n), which sorts the ENTIRE dataset
# (a full shuffle) to return 3 rows.
#
# ACTIONS THAT RETURN EVERYTHING TO THE DRIVER — treat as dangerous:
#   collect()        the whole RDD. Driver OOM on real data.
#   countByValue()   a dict of every distinct value.
#   collectAsMap()   the whole pair RDD as a dict.
# Safe: count(), take(n), takeOrdered(n), first(), reduce(), foreach().

# ---------------------------------------------------------------------------
# Section 6 — Lineage and fault tolerance
# ---------------------------------------------------------------------------

print("\nlineage of `errors`:")
print(errors.toDebugString().decode()[:400])

# toDebugString shows the RDD dependency chain. Indentation marks stage
# boundaries — each shift means a shuffle.
#
# THIS IS WHAT "RESILIENT" MEANS: an RDD does not store data, it stores the
# RECIPE (partitions + a compute function + parent dependencies). If an
# executor dies, Spark walks this graph and recomputes only the lost
# partitions from surviving parents or the original source.

# ---------------------------------------------------------------------------
# Section 7 — Leave RDD-land as soon as you can
# ---------------------------------------------------------------------------

df = ok.toDF()
df.printSchema()
df.createOrReplaceTempView("access_log")

spark.sql("""
    SELECT status,
           count(*)                            AS requests,
           round(avg(size), 1)                 AS avg_bytes,
           percentile_approx(size, 0.95)       AS p95_bytes
    FROM access_log
    GROUP BY status
    ORDER BY requests DESC
""").show()

# EXPECTED:
#   +------+--------+---------+---------+
#   |status|requests|avg_bytes|p95_bytes|
#   +------+--------+---------+---------+
#   |   200|    1500|   1250.0|     2201|
#   |   201|     500|    892.0|      892|
#   ...
#
# THE RULE: use RDDs for the messy parse, convert to DataFrame immediately,
# do everything else there. Once the data has a schema, Catalyst can optimise
# it and Tungsten can store it compactly. An RDD is opaque — Spark sees an
# arbitrary Python closure and can optimise nothing, and in PySpark every
# record crosses the JVM/Python boundary.
#
# Rough cost: this same aggregation as an RDD reduceByKey in PySpark is
# typically several times slower than the SQL above, on identical data.

parsed.unpersist()
spark.stop()

# ---------------------------------------------------------------------------
# TAKEAWAYS
# ---------------------------------------------------------------------------
# 1. RDDs are for unstructured input and arbitrary Python logic. Nothing else.
# 2. Never let a parse function raise — tag and split instead.
# 3. reduceByKey combines map-side; groupByKey does not. Know why.
# 4. takeOrdered, not sortBy+take.
# 5. collect / countByValue / collectAsMap all return everything to the driver.
# 6. Lineage is the recipe, and it is what makes recomputation possible.
# 7. Convert to DataFrame the moment the data has a shape.
