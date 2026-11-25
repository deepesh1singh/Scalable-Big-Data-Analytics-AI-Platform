# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Transform: Bronze → Silver → Gold
# MAGIC
# MAGIC **Covers:** the Spark internals — pushdown, shuffles, broadcast vs sort-merge
# MAGIC joins, skew detection and salting, window functions, partitioning.
# MAGIC
# MAGIC **Prereq:** run `01_ingest.py` first.
# MAGIC
# MAGIC **Runtime:** ~10 minutes.
# MAGIC
# MAGIC > **On the dimensional layer.** The public Yelp review dataset ships only
# MAGIC > `label` + `text`. To demonstrate joins and skew we DERIVE business and user
# MAGIC > dimensions deterministically from a hash of the review. These are synthetic
# MAGIC > and labelled as such — but the hash is designed to produce a realistic
# MAGIC > Zipf-like popularity distribution, which makes the skew lesson genuine and,
# MAGIC > unlike real data, reproducible.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG, SCHEMA = "workspace", "default"
VOL    = f"/Volumes/{CATALOG}/{SCHEMA}/lab"
BRONZE = f"{CATALOG}.{SCHEMA}.yelp_bronze"
SILVER = f"{CATALOG}.{SCHEMA}.yelp_silver"

bronze = spark.table(BRONZE)
print("bronze rows:", bronze.count())

# EXPECTED OUTPUT:
#   bronze rows: 700000

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1 — Derive the dimensional keys
# MAGIC
# MAGIC `pmod(hash(x), n)` is the standard trick for deterministic bucketing.
# MAGIC We square the hash for business_id so popularity follows a heavy-tailed
# MAGIC distribution instead of a uniform one — that is what creates the skew
# MAGIC we exploit in Section 6.

# COMMAND ----------

N_BUSINESS = 2000
N_USERS    = 50000

keyed = (bronze
    # SYNTHETIC: heavy-tailed business assignment (low ids get many more reviews)
    # NOTE the .cast("long") on _h itself. hash() returns INT32, and squaring a
    # value near 1e6 gives ~1e12, which overflows int32. Under Spark 3.4+ ANSI
    # mode (the default on Databricks) that raises ARITHMETIC_OVERFLOW rather
    # than silently wrapping. Casting BEFORE the multiply is the fix; casting
    # the product afterwards is too late — the overflow already happened.
    .withColumn("_h", F.pmod(F.hash("review_id"), F.lit(1_000_000)).cast("long"))
    .withColumn("business_id",
        F.concat(F.lit("b_"),
                 F.lpad(F.pmod(F.col("_h") * F.col("_h"),
                               F.lit(N_BUSINESS).cast("long")).cast("string"),
                        5, "0")))
    # SYNTHETIC: near-uniform user assignment
    .withColumn("user_id",
        F.concat(F.lit("u_"),
                 F.lpad(F.pmod(F.hash(F.concat_ws("|", "review_id", F.lit("salt"))),
                               F.lit(N_USERS)).cast("string"), 6, "0")))
    # SYNTHETIC: a plausible review date spread over 6 years
    .withColumn("review_date",
        F.date_add(F.lit("2019-01-01").cast("date"),
                   F.pmod(F.hash("review_id"), F.lit(2190))))
    .drop("_h"))

display(keyed.select("review_id", "business_id", "user_id", "review_date", "stars").limit(5))

# EXPECTED OUTPUT: 5 rows like
#   review_id      | business_id | user_id  | review_date | stars
#   ---------------+-------------+----------+-------------+------
#   3f2a-...       | b_00417     | u_028311 | 2021-07-14  |  5
#
# EXPLANATION of the functions used:
#
#   F.hash(col)      Spark's Murmur3 hash. DETERMINISTIC across runs and
#                    clusters — unlike Python's hash(), which is randomised
#                    per process unless PYTHONHASHSEED is set.
#
#   F.pmod(a, b)     Positive modulo. Plain `%` in Spark can return NEGATIVE
#                    values for negative input, and hash() returns negatives
#                    about half the time. pmod is the correct choice for
#                    bucketing; using % here silently produces negative ids.
#
#   F.lpad(s, 5,"0") Zero-pad so ids sort lexicographically the way you expect.
#
#   F.date_add(d, n) Date arithmetic. n is a COLUMN here, not a literal —
#                    many people assume it must be a constant. It need not be.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2 — Silver: clean, validate, enrich

# COMMAND ----------

silver = (keyed
    .filter(F.col("text").isNotNull())
    .filter(F.length("text") >= 20)                       # drop stubs
    .withColumn("text_len",   F.length("text"))
    .withColumn("word_count", F.size(F.split(F.trim("text"), r"\s+")))
    .withColumn("year",       F.year("review_date"))
    .withColumn("month",      F.month("review_date"))
    .withColumn("dow",        F.date_format("review_date", "E"))
    .withColumn("is_positive", (F.col("stars") >= 4).cast("int"))
    .withColumn("sentiment_bucket",
        F.when(F.col("stars") >= 5, "delighted")
         .when(F.col("stars") >= 4, "happy")
         .when(F.col("stars") >= 3, "neutral")
         .when(F.col("stars") >= 2, "unhappy")
         .otherwise("angry"))
    .withColumn("has_exclaim", F.col("text").rlike(r"!").cast("int"))
    .withColumn("caps_ratio",
        F.round(F.length(F.regexp_replace("text", r"[^A-Z]", "")) /
                F.greatest(F.col("text_len"), F.lit(1)), 4))
    .select("review_id", "business_id", "user_id", "review_date", "year", "month",
            "dow", "stars", "is_positive", "sentiment_bucket",
            "text", "text_len", "word_count", "has_exclaim", "caps_ratio",
            "_source_file", "_ingest_ts"))

kept = silver.count()
print(f"silver rows: {kept:,}  ({kept/700000:.2%} retained)")

# EXPECTED OUTPUT:
#   silver rows: 699,9xx  (99.99% retained)
#
# ALWAYS PRINT THE RETENTION RATIO. A filter that unexpectedly drops 40% of
# rows is a bug you want to find here, not in a dashboard three weeks later.
#
# LINE BY LINE — the non-obvious ones:
#
#   F.size(F.split(F.trim(text), r"\s+"))
#     Word count. trim() first, or a leading space produces an empty first
#     element and every count is off by one.
#
#   F.greatest(col, F.lit(1))
#     Division-by-zero guard. Spark returns NULL for x/0 rather than raising,
#     so without this a zero-length text silently yields a NULL ratio that
#     then poisons any downstream average.
#
#   .when(...).when(...).otherwise(...)
#     Spark's CASE WHEN, CHAINED not nested. Omitting .otherwise() yields NULL
#     for unmatched rows — usually a bug, occasionally intentional.
#
#   .cast("int") on a boolean
#     Booleans do not aggregate. Casting to int lets you sum/avg them, so
#     avg(is_positive) is directly the positive rate.

# COMMAND ----------

silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SILVER)
silver = spark.table(SILVER)
print("written:", SILVER)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 — Read the plan: what pushes down and what does not

# COMMAND ----------

silver.filter(F.col("stars") == 5).select("review_id", "stars").explain("formatted")

# EXPECTED OUTPUT — look for these lines:
#   PushedFilters: [IsNotNull(stars), EqualTo(stars,5)]
#   ReadSchema: struct<review_id:string,stars:int>
#
# TWO THINGS PROVEN AT ONCE:
#
#   PushedFilters  The predicate was handed to the Parquet reader, which uses
#                  per-row-group min/max statistics to skip entire row groups
#                  without decoding them. Note Spark injected IsNotNull for
#                  free — a null can never equal 5.
#
#   ReadSchema     Only 2 of 17 columns are read. Parquet stores columns
#                  contiguously, so this is a real I/O reduction, not
#                  bookkeeping. This is why SELECT * is a performance bug on
#                  columnar storage, not merely untidy.

# COMMAND ----------

silver.filter(F.col("word_count") > F.col("text_len") / 10).explain("formatted")

# EXPECTED OUTPUT:
#   PushedFilters: []          <-- EMPTY
#
# WHY: row-group statistics compare a column to a LITERAL. They cannot compare
# one column to another. Column-to-column predicates ALWAYS evaluate in Spark,
# after the data is read.
#
# The same applies to filters on columns derived in the query itself — the
# column does not exist in the file, so there is nothing to push.
#
# PRACTICAL CONSEQUENCE: prefer `col > 100` over `col_a > col_b / 10` when you
# have the choice, and put pushable filters first in a pipeline so they shrink
# the data before the unpushable ones run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4 — The first shuffle: groupBy dissected

# COMMAND ----------

biz_stats = (silver
    .groupBy("business_id")
    .agg(
        F.count("*").alias("n_reviews"),
        F.round(F.avg("stars"), 3).alias("avg_stars"),
        F.round(F.stddev("stars"), 3).alias("stddev_stars"),
        F.round(F.avg("is_positive"), 3).alias("positive_rate"),
        F.round(F.avg("word_count"), 1).alias("avg_words"),
        F.approx_count_distinct("user_id", 0.02).alias("approx_reviewers"),
        F.min("review_date").alias("first_review"),
        F.max("review_date").alias("last_review"),
    ))

biz_stats.explain("formatted")

# EXPECTED PLAN SHAPE (read BOTTOM-UP):
#   HashAggregate(keys=[business_id], functions=[... FINAL ...])
#   +- AQEShuffleRead coalesced
#      +- Exchange hashpartitioning(business_id, 200)          <-- THE SHUFFLE
#         +- HashAggregate(keys=[business_id], ... PARTIAL ...) <-- map-side combine
#            +- Scan parquet
#
# WHAT PHYSICALLY HAPPENS:
#
#  1. PARTIAL HashAggregate (map side). Each input partition builds a local hash
#     map keyed by business_id holding running (count, sum, sumsq, min, max).
#     700k rows collapse to at most 2,000 partial rows PER PARTITION before
#     anything crosses the network. This is why count/avg/min/max are cheap.
#
#  2. Exchange hashpartitioning. Each partial row goes to partition
#     hash(business_id) % n. Each map task writes n shuffle blocks to local disk
#     plus an index file.
#
#  3. Fetch. Each reduce task pulls its block from every map task. In a real
#     cluster this is all-to-all network traffic.
#
#  4. FINAL HashAggregate merges the partials. avg is decomposed into sum and
#     count precisely BECAUSE that decomposition is associative — which is what
#     makes distributed averaging possible at all.
#
# WHY approx_count_distinct AND NOT countDistinct:
#   Distinct counts are NOT associative — you cannot merge two partial distinct
#   counts without the underlying values, so Spark must shuffle the raw
#   (business_id, user_id) pairs rather than compressed partials. Several
#   countDistinct calls also generate an Expand operator that MULTIPLIES your
#   rows by the number of distinct expressions.
#   approx_count_distinct uses HyperLogLog++: fixed-size, fully mergeable,
#   typically 10-100x faster, with the error bound you pass (0.02 = 2%).

# COMMAND ----------

display(biz_stats.orderBy(F.desc("n_reviews")).limit(10))

# EXPECTED OUTPUT — note the heavy tail we engineered:
#   business_id | n_reviews | avg_stars | positive_rate | approx_reviewers
#   ------------+-----------+-----------+---------------+-----------------
#   b_00000     |   ~2400   |   ~3.00   |     ~0.40     |      ~2350
#   b_00001     |   ~1900   |   ~3.00   |     ~0.40     |      ~1870
#   ...
#
# avg_stars hovers near 3.0 for every business because the source labels are
# balanced and our business assignment is independent of the label. That is
# expected and is itself a lesson: SYNTHETIC KEYS CARRY NO SIGNAL. Do not
# build a model on them (file 03 uses the real text instead).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5 — Joins: broadcast vs sort-merge, measured

# COMMAND ----------

import time

# A small dimension: one row per business
biz_dim = (biz_stats
    .select("business_id", "n_reviews", "avg_stars")
    .withColumn("tier",
        F.when(F.col("n_reviews") >= 1000, "large")
         .when(F.col("n_reviews") >= 300,  "medium")
         .otherwise("small")))

print("dimension rows:", biz_dim.count())

# EXPECTED OUTPUT:
#   dimension rows: 2000        <-- tiny; ideal broadcast candidate

# COMMAND ----------

# --- Force SORT-MERGE ---
#
# The textbook way is:
#     spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
# On DATABRICKS SERVERLESS that raises:
#     [CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION]
#       Configuration spark.sql.autoBroadcastJoinThreshold is not available.
# Serverless manages join planning, AQE and memory itself and gates most
# spark.conf.set calls.
#
# JOIN HINTS still work everywhere, and are better practice anyway because
# they are scoped to ONE join instead of changing global state:
#     .hint("merge")            -> SortMergeJoin
#     .hint("shuffle_hash")     -> ShuffledHashJoin
#     .hint("broadcast")        -> BroadcastHashJoin  (same as F.broadcast())
#     .hint("shuffle_replicate_nl") -> cartesian-style
#
# If you are on a classic (non-serverless) cluster, the conf line works too.

smj = silver.join(biz_dim.hint("merge"), "business_id")
smj.explain("formatted")

t0 = time.time(); smj.count(); t_smj = time.time() - t0
print(f"SortMergeJoin: {t_smj:.1f}s")

# EXPECTED PLAN:
#   SortMergeJoin [business_id], [business_id], Inner
#   :- Sort [business_id ASC]
#   :  +- Exchange hashpartitioning(business_id, 200)     <-- shuffle 1
#   +- Sort [business_id ASC]
#      +- Exchange hashpartitioning(business_id, 200)     <-- shuffle 2
#
# TWO shuffles and TWO sorts. Both sides are repartitioned by the join key and
# sorted within each partition so the merge can walk them in lockstep.

# COMMAND ----------

# --- Force BROADCAST ---
bhj = silver.join(F.broadcast(biz_dim), "business_id")
bhj.explain("formatted")

t0 = time.time(); bhj.count(); t_bhj = time.time() - t0
print(f"BroadcastHashJoin: {t_bhj:.1f}s   speedup {t_smj/t_bhj:.1f}x")

# EXPECTED PLAN:
#   BroadcastHashJoin [business_id], [business_id], Inner, BuildRight
#   :- Scan parquet ...                                  <-- NO Exchange!
#   +- BroadcastExchange HashedRelationBroadcastMode
#
# EXPECTED OUTPUT:
#   BroadcastHashJoin: ~2s   speedup 2-5x
#
# WHAT BROADCAST PHYSICALLY DOES:
#   1. Spark runs a job to collect the small side TO THE DRIVER.
#   2. The driver builds a HashedRelation from it.
#   3. It ships that to every executor via the block manager (torrent-style).
#   4. The large side is NEVER shuffled — each task streams its local
#      partition and probes the in-memory hash table.
#
# CONSEQUENCES PEOPLE MISS:
#   - The relation passes THROUGH THE DRIVER, so a 2 GB broadcast needs 2 GB of
#     DRIVER heap. Broadcasts cause driver OOM, not only executor OOM.
#   - Every executor holds a full copy: memory cost = size x n_executors.
#   - broadcast() is a HINT, not a guarantee. Spark ignores it for FULL OUTER
#     joins, and you can only broadcast the RIGHT side of a LEFT OUTER join.
#     ALWAYS verify in the plan rather than trusting the hint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6 — Skew: detect it, then fix it two ways

# COMMAND ----------

skew = (silver.groupBy("business_id").count()
    .agg(
        F.count("*").alias("n_keys"),
        F.min("count").alias("min"),
        F.expr("percentile_approx(count, 0.5)").alias("p50"),
        F.expr("percentile_approx(count, 0.99)").alias("p99"),
        F.max("count").alias("max"),
    ))
display(skew)

# EXPECTED OUTPUT (approximate — heavy tail by construction):
#   n_keys | min | p50 | p99  | max
#   -------+-----+-----+------+------
#    2000  |  ~5 | ~180| ~1800| ~2400
#
# READ THE RATIO max/p50. Anything above ~10x means one task will do an order
# of magnitude more work than the median, and the STAGE CANNOT FINISH UNTIL
# ITS SLOWEST TASK DOES. That is what a stage boundary means.
#
# Real-world skew is far worse than this: human-generated keys follow a Zipf
# distribution, so users / merchants / device ids / SKUs routinely show
# 1000x-30000x ratios. Anyone who has run Spark in production has been paged
# because of it.

# COMMAND ----------

# See the physical distribution directly
(silver
    .repartition(16, "business_id")
    .withColumn("pid", F.spark_partition_id())
    .groupBy("pid").count()
    .orderBy("pid")
    .display())

# EXPECTED OUTPUT: 16 rows with visibly UNEVEN counts.
#
# spark_partition_id() exposes the physical partition index as a COLUMN,
# turning an invisible property into a queryable table. It is the single best
# diagnostic function in Spark.
#
# NOTE FOR DATABRICKS SERVERLESS: df.rdd.getNumPartitions() and
# spark.sparkContext are BLOCKED. spark_partition_id() is a SQL function,
# not an RDD API, so it works — and it is the better tool anyway.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fix 1 — AQE skew join (let the engine do it)

# COMMAND ----------

# On a CLASSIC cluster you would tune AQE like this:
#
#   spark.conf.set("spark.sql.adaptive.enabled", "true")
#   spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
#   spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "3")
#   spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "16MB")
#
# On DATABRICKS SERVERLESS these raise CONFIG_NOT_AVAILABLE — AQE is always on
# and self-tuned. You cannot turn it off to observe the difference, which is
# precisely why the "before" experiment in this section is worth running
# locally (pip install pyspark) if you want to SEE skew hurt.
#
# Read the mechanism regardless — it is a common interview question, and you
# can still verify AQE fired by looking for `AQEShuffleRead` in the plan.
#
# HOW AQE SKEW HANDLING WORKS:
#  1. After the shuffle stage completes, Spark has REAL byte sizes per partition.
#  2. A partition is "skewed" if it exceeds BOTH
#        skewedPartitionFactor x median   AND   skewedPartitionThresholdInBytes
#     Both conditions — which is why raising only one config often appears to
#     do nothing at all.
#  3. The skewed partition on the big side is SPLIT into sub-partitions.
#  4. The matching partition on the other side is REPLICATED to each split,
#     so every pair still meets and the result stays correct.
#  5. Proof in the plan: `AQEShuffleRead ... skewed=true`.
#
# LIMITS — important, and a good interview answer:
#   AQE skew handling only helps JOINS, and only sort-merge / shuffle-hash
#   joins. It does NOT fix a skewed groupBy, a skewed window, or skew that a
#   UDF introduced. For those you still need salting.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fix 2 — Salting (the technique you must be able to write from memory)

# COMMAND ----------

SALT = 16

# BIG side: append a RANDOM salt to the key
big_salted = (silver
    .withColumn("salt", (F.rand(seed=42) * SALT).cast("int"))
    .withColumn("skey", F.concat_ws("_", "business_id", "salt")))

# SMALL side: REPLICATE each row once per salt value
small_salted = (biz_dim
    .withColumn("salt", F.explode(F.array([F.lit(i) for i in range(SALT)])))
    .withColumn("skey", F.concat_ws("_", "business_id", "salt"))
    .select("skey", "tier"))

salted = (big_salted
    .join(small_salted, "skey")
    .groupBy("business_id", "tier")          # aggregate back on the REAL key
    .agg(F.count("*").alias("n"),
         F.round(F.avg("stars"), 3).alias("avg_stars")))

display(salted.orderBy(F.desc("n")).limit(5))

# EXPECTED OUTPUT: identical numbers to the unsalted join. Only the physical
# distribution changed.
#
# WHY EACH PIECE IS NECESSARY:
#
#   F.rand(seed=42) * SALT -> int
#     Uniform random integer in [0, 16). The hottest key's ~2400 rows now
#     spread across 16 partitions instead of 1, so the biggest task shrinks ~16x.
#     SEED IT — an unseeded rand() makes the job non-deterministic, so a retried
#     task can produce different assignments than its first attempt.
#
#   The small side must be REPLICATED, not randomly salted.
#     If you salted the small side randomly too, a big-side row with salt 7
#     would never meet its partner sitting at salt 3, and you would SILENTLY
#     LOSE ROWS. Replication guarantees every big-side salt finds a match.
#     >>> This is the step candidates get wrong in interviews. <<<
#
#   explode(array([lit(i) for i in range(SALT)]))
#     The idiomatic way to cross a row against 0..N-1 without a real cross join.
#
#   Final groupBy on business_id
#     Collapses the 16 shards back to the true key.
#
#   COST: the small side is now 16x larger. Salting trades VOLUME for
#   EVENNESS. Only worth it when a straggler dominates.
#
# PRODUCTION REFINEMENT: salt only the HOT keys. Compute the top-N keys by
# frequency, broadcast that small list, and apply salt only when the key is in
# it. Cold keys keep salt 0 and the small side is replicated only for the
# handful of hot keys. Same evenness, a fraction of the cost. Being able to
# describe this refinement is a strong senior signal.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7 — Window functions

# COMMAND ----------

w_rank    = Window.partitionBy("tier").orderBy(F.desc("avg_stars"))
w_tierall = Window.partitionBy("tier")

leaders = (biz_stats
    .join(F.broadcast(biz_dim.select("business_id", "tier")), "business_id")
    .filter(F.col("n_reviews") >= 100)
    .withColumn("rank",      F.row_number().over(w_rank))
    .withColumn("tier_avg",  F.round(F.avg("avg_stars").over(w_tierall), 3))
    .withColumn("vs_tier",   F.round(F.col("avg_stars") - F.col("tier_avg"), 3))
    .filter(F.col("rank") <= 5)
    .select("tier", "rank", "business_id", "n_reviews", "avg_stars", "tier_avg", "vs_tier")
    .orderBy("tier", "rank"))

display(leaders)

# EXPECTED OUTPUT: 15 rows (5 per tier), e.g.
#   tier   | rank | business_id | n_reviews | avg_stars | tier_avg | vs_tier
#   -------+------+-------------+-----------+-----------+----------+--------
#   large  |  1   | b_00312     |   1204    |   3.112   |  3.001   |  0.111
#
# LINE BY LINE:
#
#   Window.partitionBy("tier")
#     This is a SHUFFLE. A window spec is a wide operation dressed up as a
#     column expression. The plan shows Exchange hashpartitioning(tier),
#     then Sort, then Window.
#
#   Both specs partition by "tier"
#     So Spark reuses ONE Exchange and adds a second Window operator.
#     Check the plan: one hashpartitioning node, not two. Partitioning-key
#     reuse is a real and underused optimisation.
#
#   row_number vs rank vs dense_rank
#     row_number  -> 1,2,3,4  strictly, no ties
#     rank        -> 1,1,3    ties share, gaps appear
#     dense_rank  -> 1,1,2    ties share, no gaps
#     Choosing wrong is a SILENT correctness bug: row_number() <= 5 gives
#     exactly 5 rows; rank() <= 5 can give 50 if there are ties.
#
#   filter(rank <= 5) AFTER the window
#     The filter CANNOT be pushed below the window — the window must be
#     computed first. This is a case where Catalyst genuinely cannot help
#     and you must order the code yourself.
#
#   vs_tier
#     Comparing a row to its GROUP's aggregate with NO self-join. That is the
#     main reason window functions exist, and the standard answer to
#     "how do you compare a row to its group average?"

# COMMAND ----------

# MAGIC %md
# MAGIC ### The unbounded-window trap

# COMMAND ----------

# DO NOT RUN THIS ON LARGE DATA — shown to be recognised, not executed.
#
#   w_bad = Window.orderBy("review_date")          # NO partitionBy
#   silver.withColumn("running", F.sum("stars").over(w_bad))
#
# No partitionBy means ALL 700k rows must land in a SINGLE partition.
# The plan shows `Exchange SinglePartition`: one task, one core, whole dataset.
# Spark even logs a warning:
#   "No Partition Defined for Window operation! Moving all data to a single
#    partition, this can cause serious performance degradation."
# A rare case of Spark telling you outright. Always partition a window.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Frames: rowsBetween vs rangeBetween

# COMMAND ----------

w_rows = (Window.partitionBy("business_id").orderBy("review_date").rowsBetween(-9, 0))
w_days = (Window.partitionBy("business_id").orderBy(F.unix_timestamp("review_date"))
                .rangeBetween(-30 * 86400, 0))

trend = (silver
    .withColumn("ma_10_reviews", F.round(F.avg("stars").over(w_rows), 3))
    .withColumn("ma_30_days",    F.round(F.avg("stars").over(w_days), 3))
    .withColumn("n_last_30d",    F.count("*").over(w_days))
    .select("business_id", "review_date", "stars",
            "ma_10_reviews", "ma_30_days", "n_last_30d"))

display(trend.orderBy("business_id", "review_date").limit(20))

# THE DISTINCTION THAT TRIPS EVERYONE:
#
#   rowsBetween(-9, 0)      counts ROWS. Position-based. Always exactly 10 rows
#                           regardless of their dates.
#
#   rangeBetween(-2592000, 0)
#                           counts VALUES OF THE ORDERING COLUMN. Because we
#                           ordered by unix_timestamp (SECONDS), 30 days is
#                           30*86400 = 2,592,000. If the ordering column were
#                           milliseconds the number changes by 1000x.
#                           Range frames are value-semantic and unit-sensitive.
#
#   DEFAULT FRAME when you give orderBy but no frame:
#       rangeBetween(unboundedPreceding, currentRow)  -> a running total.
#   DEFAULT with NO orderBy: the entire partition.
#   Many "why is my window result wrong" bugs are just the implicit default.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 8 — Gold tables and the write-partitioning rule

# COMMAND ----------

GOLD_BIZ   = f"{CATALOG}.{SCHEMA}.yelp_gold_business"
GOLD_DAILY = f"{CATALOG}.{SCHEMA}.yelp_gold_daily"

# Bayesian-adjusted rating: shrink low-count averages toward the global mean.
global_avg = biz_stats.agg(F.avg("avg_stars")).first()[0]
M = 200

gold_biz = (biz_stats
    .withColumn("weighted_stars", F.round(
        (F.col("n_reviews") / (F.col("n_reviews") + M)) * F.col("avg_stars")
        + (M / (F.col("n_reviews") + M)) * F.lit(global_avg), 3))
    .withColumn("rank", F.row_number().over(Window.orderBy(F.desc("weighted_stars")))))

gold_biz.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_BIZ)
display(spark.table(GOLD_BIZ).orderBy("rank").limit(10))

# WHY THE BAYESIAN ADJUSTMENT:
#   A straight avg(stars) puts a business with a single 5-star review above one
#   with 2,000 reviews averaging 4.8. The formula pulls low-count averages
#   toward the global mean, with M controlling how hard. This is the "IMDB
#   weighted rating". Raising it unprompted in an interview signals that you
#   think about correctness, not just SQL.

# COMMAND ----------

gold_daily = (silver
    .groupBy("year", "month", "sentiment_bucket")
    .agg(F.count("*").alias("n_reviews"),
         F.round(F.avg("stars"), 3).alias("avg_stars"),
         F.round(F.avg("word_count"), 1).alias("avg_words"),
         F.approx_count_distinct("business_id", 0.02).alias("approx_businesses")))

(gold_daily
    .repartition("year")            # <-- pair with partitionBy; see note below
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("year")
    .saveAsTable(GOLD_DAILY))

display(spark.table(GOLD_DAILY).orderBy("year", "month").limit(20))

# >>> THE MOST IMPORTANT LINE IN THIS FILE <<<
#
#   .repartition("year")  BEFORE  .partitionBy("year")
#
# WITHOUT it: every task can hold rows for every year, so each task writes one
# small file PER YEAR. With 200 tasks and 6 years that is 1,200 tiny files.
# WITH it: each task holds one year, so each year gets a small number of
# properly-sized files.
#
# WHY MANY SMALL FILES HURT — four separate costs:
#   1. LISTING    the driver must enumerate every file; on S3 this is paginated
#                 API calls and can take MINUTES with the cluster idle.
#   2. TASKS      one task per file — thousands of tasks each doing microseconds
#                 of real work.
#   3. METADATA   every Parquet file has a footer that must be read; with tiny
#                 files the footer can exceed the data.
#   4. COMPRESSION Parquet encodings (dictionary, RLE) need volume to work.
#
# Target 128 MB - 1 GB per file. This is exactly what Iceberg / S3 Tables
# automatic compaction does for you, and why that feature matters.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 9 — Delta maintenance

# COMMAND ----------

spark.sql(f"OPTIMIZE {SILVER} ZORDER BY (business_id, review_date)")
display(spark.sql(f"DESCRIBE DETAIL {SILVER}"))

# WHAT EACH DOES:
#   OPTIMIZE  compacts many small files into fewer large ones. Fixes the
#             small-file problem after the fact.
#   ZORDER    co-locates rows with similar values of the listed columns into
#             the same files, so min/max statistics become far more selective
#             and predicate pushdown skips more. Choose the columns you most
#             often FILTER on, not the ones you group by. Diminishing returns
#             past 3-4 columns.
#   VACUUM    removes data files no longer referenced by any retained version:
#                 VACUUM workspace.default.yelp_silver RETAIN 168 HOURS
#             Note this DESTROYS time travel beyond the retention window.
#
# EXPECTED: numFiles drops noticeably after OPTIMIZE.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What you built
# MAGIC
# MAGIC | Table | Rows | Purpose |
# MAGIC |---|---|---|
# MAGIC | `yelp_silver` | ~700k | Cleaned, enriched, feature-bearing |
# MAGIC | `yelp_gold_business` | 2,000 | Per-business metrics + Bayesian rank |
# MAGIC | `yelp_gold_daily` | ~360 | Monthly sentiment rollup, partitioned by year |
# MAGIC
# MAGIC ### Concepts demonstrated
# MAGIC - Pushdown: literal predicates push, column-to-column do not
# MAGIC - Partial + final HashAggregate around an Exchange (map-side combine)
# MAGIC - `approx_count_distinct` and why mergeability decides the cost
# MAGIC - Broadcast vs sort-merge join, forced both ways and timed
# MAGIC - Skew detection via percentiles and `spark_partition_id()`
# MAGIC - AQE skew join, and salting including the replicate-the-small-side rule
# MAGIC - Window ranking, partition-key reuse, frame semantics
# MAGIC - `repartition` before `partitionBy`
# MAGIC - OPTIMIZE / ZORDER / VACUUM
# MAGIC
# MAGIC **Next:** `03_ml.py` — Feature Store, MLflow Experiments, model training.
