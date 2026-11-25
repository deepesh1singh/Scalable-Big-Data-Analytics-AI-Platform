"""
05 — Joins: types, strategies, skew
===================================

REAL-WORLD SCENARIO
-------------------
Retail analytics. You must combine orders (the large fact table) with customers
and products (small dimensions) and a returns table, then answer: revenue by
segment, which products are never ordered, and which orders reference a
customer that no longer exists.

Joins are where Spark jobs die. Understanding join TYPE (what rows come back)
and join STRATEGY (how Spark executes it) are two separate skills; you need
both.

Run:
    spark-submit 05_joins.py
"""

from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.appName("joins")
         .master("local[4]")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

orders = spark.createDataFrame([
    ("O-1", "C-01", "P-100", 2,  50.0, "2026-08-01"),
    ("O-2", "C-02", "P-200", 1, 300.0, "2026-08-01"),
    ("O-3", "C-01", "P-100", 5,  50.0, "2026-08-02"),
    ("O-4", "C-99", "P-300", 1,  75.0, "2026-08-02"),   # orphan customer
    ("O-5", "C-03", "P-200", 3, 300.0, "2026-08-03"),
    ("O-6", "C-02", "P-999", 1,  10.0, "2026-08-03"),   # orphan product
], "order_id STRING, customer_id STRING, product_id STRING, qty INT, price DOUBLE, order_date STRING")

customers = spark.createDataFrame([
    ("C-01", "Acme Corp",   "enterprise", "US"),
    ("C-02", "Beta LLC",    "smb",        "US"),
    ("C-03", "Gamma Gmbh",  "enterprise", "DE"),
    ("C-04", "Delta Inc",   "smb",        "GB"),   # no orders
], "customer_id STRING, name STRING, segment STRING, country STRING")

products = spark.createDataFrame([
    ("P-100", "Widget",  "hardware"),
    ("P-200", "Gizmo",   "hardware"),
    ("P-300", "Service", "services"),
    ("P-400", "Gadget",  "hardware"),   # never ordered
], "product_id STRING, product_name STRING, category STRING")

# ---------------------------------------------------------------------------
# Section 1 — Join types: what rows come back
# ---------------------------------------------------------------------------

for how in ["inner", "left", "right", "full", "left_semi", "left_anti"]:
    n = orders.join(customers, "customer_id", how).count()
    print(f"{how:12} -> {n} rows")

# EXPECTED:
#   inner        -> 5     (O-4 dropped: customer C-99 does not exist)
#   left         -> 6     (all orders; C-99 gets null customer columns)
#   right        -> 6     (all customers; C-04 gets null order columns)
#   full         -> 7     (everything from both sides)
#   left_semi    -> 5     (orders WITH a matching customer; customer cols NOT returned)
#   left_anti    -> 1     (orders WITHOUT a matching customer — the orphan)
#
# >>> SEMI AND ANTI JOINS ARE UNDERUSED <<<
#   left_semi  = "EXISTS"     — filters the left side, returns ONLY left columns,
#                               and never duplicates rows even if the right side
#                               has many matches.
#   left_anti  = "NOT EXISTS" — the cleanest way to find orphans/missing keys.
#
# Compare: `orders.join(customers,"customer_id","inner").select(orders["*"])`
# is NOT equivalent — if a customer appeared twice, the inner join duplicates
# the order row. left_semi does not. Use semi for filtering, always.

# ---------------------------------------------------------------------------
# Section 2 — Data quality via anti-join
# ---------------------------------------------------------------------------

print("\norphan orders (customer missing):")
orders.join(customers, "customer_id", "left_anti").show()

print("products never ordered:")
products.join(orders, "product_id", "left_anti").show()

# EXPECTED:
#   orphan: O-4 / C-99
#   never ordered: P-400 Gadget
#
# These two queries are referential-integrity checks. Run them as tests in a
# pipeline, not as ad-hoc curiosity — an orphan rate that jumps from 0.1% to
# 8% means an upstream dimension load failed.

# ---------------------------------------------------------------------------
# Section 3 — Join syntax, and the ambiguous-column trap
# ---------------------------------------------------------------------------

# GOOD — string key: produces ONE customer_id column
a = orders.join(customers, "customer_id", "inner")
print("\nstring-key columns:", a.columns)

# GOOD — list of keys
b = orders.join(customers, ["customer_id"], "inner")

# PROBLEMATIC — expression form: produces TWO customer_id columns
c = orders.join(customers, orders.customer_id == customers.customer_id, "inner")
print("expression-form columns:", c.columns)
# c.select("customer_id")  ->  AMBIGUOUS_REFERENCE at runtime

# FIX when you must use the expression form (different column names,
# or a non-equality condition): alias both sides.
o = orders.alias("o")
cu = customers.alias("c")
d = (o.join(cu, F.col("o.customer_id") == F.col("c.customer_id"), "inner")
       .select("o.order_id", "o.customer_id", "c.name", "c.segment"))
d.show()

# RULE: use the string/list form whenever the key names match. Use the
# expression form only when they differ or the condition is non-equality,
# and alias both sides when you do.

# Non-equality join (range/interval):
#   orders.join(price_ranges,
#       (orders.price >= price_ranges.lo) & (orders.price < price_ranges.hi))
# These CANNOT use sort-merge or broadcast-hash and fall back to
# BroadcastNestedLoopJoin or CartesianProduct — O(n*m). Watch for them; a
# non-equi join on two large tables will not finish.

# ---------------------------------------------------------------------------
# Section 4 — Join strategy: broadcast vs sort-merge
# ---------------------------------------------------------------------------

enriched = (orders
    .join(F.broadcast(customers), "customer_id", "left")
    .join(F.broadcast(products),  "product_id",  "left")
    .withColumn("revenue", F.round(F.col("qty") * F.col("price"), 2)))

enriched.explain("formatted")
enriched.select("order_id", "name", "segment", "product_name", "revenue").show()

# LOOK FOR IN THE PLAN:
#   BroadcastHashJoin ... BuildRight
#   BroadcastExchange HashedRelationBroadcastMode
#   and NO Exchange under the orders side
#
# WHAT BROADCAST PHYSICALLY DOES:
#   1. Spark collects the small side TO THE DRIVER.
#   2. The driver builds a hash table (HashedRelation).
#   3. It ships that to every executor via the block manager (torrent-style).
#   4. The large side is NEVER shuffled — each task streams its local
#      partition and probes the in-memory hash table.
#
# CONSEQUENCES PEOPLE MISS:
#   - The relation passes THROUGH THE DRIVER, so a 2 GB broadcast needs 2 GB of
#     DRIVER heap. Broadcasts cause driver OOM, not only executor OOM.
#   - Every executor holds a full copy: cost = size x n_executors.
#   - Spark's size estimate for a file source is often just the FILE size; a
#     3 MB compressed Parquet can expand to 60 MB of JVM objects.
#   - broadcast() is a HINT. Spark ignores it for FULL OUTER joins, and you can
#     only broadcast the RIGHT side of a LEFT OUTER (and left of a RIGHT OUTER).
#     ALWAYS verify in the plan.
#
# FORCING THE ALTERNATIVES — hints work everywhere, including Databricks
# serverless where spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
# raises CONFIG_NOT_AVAILABLE:
#     df.join(dim.hint("merge"),        "k")   # SortMergeJoin
#     df.join(dim.hint("shuffle_hash"), "k")   # ShuffledHashJoin
#     df.join(dim.hint("broadcast"),    "k")   # BroadcastHashJoin
#
# THE THREE STRATEGIES:
#   BroadcastHashJoin  small side replicated; NO shuffle on the big side.
#                      Best when one side genuinely fits in memory.
#   SortMergeJoin      both sides shuffled by key AND sorted, then merged.
#                      Scales to any size; pays 2 shuffles + 2 sorts.
#   ShuffledHashJoin   both sides shuffled; a hash table built per partition.
#                      No sort. Chosen when one side is small per-partition.

# ---------------------------------------------------------------------------
# Section 5 — The many-to-many explosion
# ---------------------------------------------------------------------------

dup_customers = customers.unionByName(
    spark.createDataFrame([("C-01", "Acme Corp (dup)", "enterprise", "US")],
                          "customer_id STRING, name STRING, segment STRING, country STRING"))

print("\norders:", orders.count(),
      " after join with duplicated dim:",
      orders.join(dup_customers, "customer_id").count())

# EXPECTED:
#   orders: 6   after join: 7
#
# C-01 appears twice on the right, so both C-01 orders duplicate. With
# duplicates on BOTH sides the growth is multiplicative: 1000 x 1000 matching
# rows for one key = 1,000,000 output rows from 2,000 input rows.
#
# >>> THIS IS THE #1 CAUSE OF A JOIN THAT "SHOULD BE FAST" RUNNING FOR HOURS <<<
#
# DIAGNOSE IT BEFORE JOINING:
#     for name, d, k in [("orders", orders, "customer_id"),
#                        ("customers", dup_customers, "customer_id")]:
#         dupes = d.groupBy(k).count().filter("count > 1")
#         print(name, "duplicate keys:", dupes.count())
#
# FIX: deduplicate the dimension first, or use left_semi if you only need to
# filter rather than enrich.

for name, d in [("orders", orders), ("customers", dup_customers)]:
    n = d.groupBy("customer_id").count().filter("count > 1").count()
    print(f"  {name}: {n} duplicate keys")

# ---------------------------------------------------------------------------
# Section 6 — Skew, and salting
# ---------------------------------------------------------------------------

# Build a skewed fact table: one hot key dominates
skewed = (spark.range(0, 200_000)
    .withColumn("customer_id",
        F.when(F.col("id") % 10 < 8, F.lit("C-01"))       # 80% one key
         .otherwise(F.concat(F.lit("C-"), F.lpad((F.col("id") % 50).cast("string"), 2, "0"))))
    .withColumn("amount", (F.rand(seed=1) * 100).cast("double")))

dist = (skewed.groupBy("customer_id").count()
        .agg(F.max("count").alias("max"),
             F.expr("percentile_approx(count, 0.5)").alias("p50")))
dist.show()

# EXPECTED: max ~160,000, p50 ~400  ->  ratio ~400x
#
# READ max/p50. Above ~10x means one task does an order of magnitude more work
# than the median, and A STAGE CANNOT FINISH UNTIL ITS SLOWEST TASK DOES.
# Real keys follow a Zipf distribution — users, merchants, device ids, SKUs
# routinely show 1000x+ ratios.

# See the physical imbalance
(skewed.repartition(8, "customer_id")
       .withColumn("pid", F.spark_partition_id())
       .groupBy("pid").count().orderBy("pid").show())

# spark_partition_id() turns an invisible property into a queryable table.
# It is the best diagnostic in Spark, and it works on Databricks serverless
# where df.rdd.getNumPartitions() is blocked.

# --- SALTING ---
SALT = 16

big = (skewed
    .withColumn("salt", (F.rand(seed=42) * SALT).cast("int"))
    .withColumn("skey", F.concat_ws("_", "customer_id", "salt")))

small = (customers
    .withColumn("salt", F.explode(F.array([F.lit(i) for i in range(SALT)])))
    .withColumn("skey", F.concat_ws("_", "customer_id", "salt"))
    .select("skey", "segment"))

salted = (big.join(small, "skey")
    .groupBy("customer_id", "segment")          # back to the REAL key
    .agg(F.round(F.sum("amount"), 2).alias("total")))

salted.orderBy(F.desc("total")).show(5)

# WHY EACH PIECE IS NECESSARY:
#
#   rand(seed=42) * SALT -> int
#     Uniform random in [0,16). The hot key's 160k rows now spread across 16
#     partitions, so the biggest task shrinks ~16x. SEED IT — an unseeded
#     rand() makes the job nondeterministic, so a RETRIED task can assign
#     different salts than its first attempt.
#
#   The small side is REPLICATED, not randomly salted.
#     If you salted the small side randomly too, a big-side row with salt 7
#     would never meet its partner at salt 3 and you would SILENTLY LOSE ROWS.
#     >>> This is the step candidates get wrong in interviews. <<<
#
#   Final groupBy on the real key collapses the shards back.
#
#   COST: the small side is 16x larger. Salting trades VOLUME for EVENNESS.
#   Only worth it when a straggler dominates.
#
# PRODUCTION REFINEMENT: salt only the HOT keys. Compute the top-N keys by
# frequency, broadcast that list, and salt only when the key is in it. Cold
# keys keep salt 0 and the small side is replicated only for the few hot keys.
#
# AQE ALTERNATIVE (Spark 3+): adaptive skew join splits oversized shuffle
# partitions automatically and replicates the matching side. Configure with
# spark.sql.adaptive.skewJoin.{enabled,skewedPartitionFactor,
# skewedPartitionThresholdInBytes} — a partition must exceed BOTH thresholds,
# which is why raising only one appears to do nothing.
# LIMITS: AQE skew handling helps JOINS only — not skewed groupBy, not windows,
# not skew introduced by a UDF. For those you still salt by hand.

# ---------------------------------------------------------------------------
# Section 7 — Bucketing: pay the shuffle once
# ---------------------------------------------------------------------------

# If two tables are joined on the same key repeatedly, pre-bucket them:
#
#   orders.write.bucketBy(64, "customer_id").sortBy("customer_id") \
#         .mode("overwrite").saveAsTable("orders_bucketed")
#   customers.write.bucketBy(64, "customer_id") \
#         .mode("overwrite").saveAsTable("customers_bucketed")
#
# Joining the two bucketed tables on customer_id needs NO shuffle — matching
# keys are already co-located in the same bucket file. The plan shows the
# SortMergeJoin with no Exchange beneath it.
#
# CAVEATS: both sides need the SAME number of buckets and the SAME key;
# bucketing requires saveAsTable (a metastore), not a plain path write; and
# the bucket count is effectively permanent. On Databricks, ZORDER + Delta
# data skipping usually gives more benefit for less rigidity.

# ---------------------------------------------------------------------------
# Section 8 — Final analysis
# ---------------------------------------------------------------------------

result = (enriched
    .filter(F.col("segment").isNotNull())
    .groupBy("segment", "category")
    .agg(F.countDistinct("order_id").alias("orders"),
         F.round(F.sum("revenue"), 2).alias("revenue"),
         F.round(F.avg("revenue"), 2).alias("avg_order"))
    .orderBy(F.desc("revenue")))
result.show()

spark.stop()

# ---------------------------------------------------------------------------
# TAKEAWAYS
# ---------------------------------------------------------------------------
# 1. left_semi filters without duplicating; left_anti finds orphans. Use both.
# 2. String-key join form avoids ambiguous columns. Alias when you can't use it.
# 3. Check for duplicate keys on BOTH sides before joining. Explosion is the
#    #1 cause of a "should be fast" join running for hours.
# 4. Broadcast passes through the DRIVER — it causes driver OOM too.
# 5. broadcast() is a hint. Verify BroadcastHashJoin in the plan.
# 6. Use .hint("merge") to force sort-merge; works where conf changes are blocked.
# 7. max/p50 > 10x means skew. spark_partition_id() shows it directly.
# 8. When salting, REPLICATE the small side. Seed the rand().
# 9. AQE skew join covers joins only — not groupBy, windows or UDF-induced skew.
# 10. Non-equi joins fall back to nested loops. Avoid on large tables.
