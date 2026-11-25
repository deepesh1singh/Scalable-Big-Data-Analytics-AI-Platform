"""
06 — Aggregations: groupBy, agg, pivot, rollup, cube
====================================================

REAL-WORLD SCENARIO
-------------------
Monthly sales reporting. Finance wants revenue by region and product line, a
cross-tab with months as columns, subtotals at every level, and percentile
latency on order fulfilment. Marketing wants distinct customer counts fast
enough to run interactively.

Aggregation is where shuffles happen, so it is where the cost is.

Run:
    spark-submit 06_aggregations.py
"""

from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.appName("aggregations")
         .master("local[4]")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

sales = spark.createDataFrame([
    ("S-01", "US", "north", "hardware", "C-01", "2026-01-15", 1200.0, 3, 2),
    ("S-02", "US", "north", "software", "C-02", "2026-01-20",  800.0, 1, 1),
    ("S-03", "US", "south", "hardware", "C-03", "2026-02-05", 2400.0, 6, 5),
    ("S-04", "DE", "west",  "hardware", "C-04", "2026-02-11",  950.0, 2, 8),
    ("S-05", "DE", "west",  "services", "C-01", "2026-02-18", 5000.0, 1, 12),
    ("S-06", "US", "south", "software", "C-02", "2026-03-02",  600.0, 2, 1),
    ("S-07", "GB", "east",  "hardware", "C-05", "2026-03-14", 1800.0, 4, 3),
    ("S-08", "GB", "east",  "services", "C-05", "2026-03-22", 3200.0, 1, 21),
    ("S-09", "US", "north", "hardware", "C-01", "2026-03-28",  700.0, 2, 2),
    ("S-10", "DE", "west",  "software", "C-06", "2026-01-09",  450.0, 1, 1),
], """sale_id STRING, country STRING, region STRING, product_line STRING,
      customer_id STRING, sale_date STRING, amount DOUBLE, units INT,
      fulfil_days INT""")

sales = (sales
    .withColumn("sale_date", F.to_date("sale_date"))
    .withColumn("month", F.date_format("sale_date", "yyyy-MM")))

# ---------------------------------------------------------------------------
# Section 1 — Basic groupBy + agg
# ---------------------------------------------------------------------------

by_country = (sales.groupBy("country")
    .agg(
        F.count("*").alias("n_sales"),
        F.round(F.sum("amount"), 2).alias("revenue"),
        F.round(F.avg("amount"), 2).alias("avg_sale"),
        F.round(F.stddev("amount"), 2).alias("stddev"),
        F.min("amount").alias("min_sale"),
        F.max("amount").alias("max_sale"),
        F.sum("units").alias("units"),
        F.countDistinct("customer_id").alias("customers"),
    )
    .orderBy(F.desc("revenue")))
by_country.show()

# EXPECTED:
#   DE revenue 6400.00, US 5700.00, GB 5000.00
#
# ALWAYS .alias() — without it you get column names like
# `round(sum(amount), 2)` which are painful to reference downstream.
#
# count("*")  counts ROWS including nulls
# count("col") counts NON-NULL values of col
# These differ, and the difference is a common silent bug.

by_country.explain("formatted")

# THE PLAN — read bottom-up:
#   HashAggregate(keys=[country], functions=[..., FINAL])
#   +- Exchange hashpartitioning(country, 8)      <-- THE SHUFFLE
#      +- HashAggregate(keys=[country], ..., PARTIAL)   <-- map-side combine
#         +- Scan
#
# WHAT HAPPENS PHYSICALLY:
#  1. PARTIAL aggregate: each input partition builds a local hash map keyed by
#     country holding running (count, sum, sumsq, min, max). Millions of rows
#     collapse to a handful of partial rows BEFORE anything crosses the network.
#  2. Exchange: partial rows go to partition hash(country) % 8.
#  3. FINAL aggregate merges partials per key.
#
# avg is decomposed into sum and count precisely BECAUSE that decomposition is
# ASSOCIATIVE — which is what makes distributed averaging possible at all.

# ---------------------------------------------------------------------------
# Section 2 — countDistinct is the expensive one
# ---------------------------------------------------------------------------

exact  = sales.agg(F.countDistinct("customer_id").alias("exact"))
approx = sales.agg(F.approx_count_distinct("customer_id", 0.02).alias("approx"))
exact.join(approx).show()

# WHY countDistinct IS SLOW:
#   Distinct counts are NOT associative — you cannot merge two partial
#   distinct-counts without the underlying values. So Spark must shuffle the
#   RAW (key, value) pairs rather than compressed partials.
#
#   Worse: multiple countDistinct on DIFFERENT columns generate an `Expand`
#   operator that MULTIPLIES your row count by the number of distinct
#   expressions. A groupBy that was fast becomes 10x slower when someone adds
#   one column. Look for `Expand` in the plan.
#
# approx_count_distinct uses HyperLogLog++: fixed-size sketch state, fully
# MERGEABLE, typically 10-100x faster, with the relative error you pass
# (0.02 = 2%). For dashboards and cardinality monitoring, 2% error is free.
# For billing, use exact.

# ---------------------------------------------------------------------------
# Section 3 — Percentiles
# ---------------------------------------------------------------------------

(sales.groupBy("product_line")
    .agg(
        F.count("*").alias("n"),
        F.round(F.avg("fulfil_days"), 1).alias("mean_days"),
        F.expr("percentile_approx(fulfil_days, 0.5)").alias("p50"),
        F.expr("percentile_approx(fulfil_days, 0.95)").alias("p95"),
        F.expr("percentile_approx(fulfil_days, array(0.25,0.5,0.75))").alias("quartiles"),
        F.max("fulfil_days").alias("worst"))
    .orderBy("product_line").show(truncate=False))

# EXPECTED:
#   services: mean 16.5, p50 12, worst 21   <-- mean hides the tail
#
# >>> USE PERCENTILES, NOT AVERAGES, FOR ANYTHING LATENCY-LIKE <<<
# A handful of 21-day fulfilments drags the mean far above the typical
# experience. p50 tells you the typical case, p95 tells you the bad case;
# the mean tells you neither.
#
# percentile_approx uses a mergeable sketch (same argument as HLL).
# EXACT percentile() requires a global sort per group — fine for small groups,
# very expensive for large ones.
# Pass an ARRAY of probabilities to get several in ONE pass.

# ---------------------------------------------------------------------------
# Section 4 — Conditional aggregation
# ---------------------------------------------------------------------------

(sales.groupBy("country")
    .agg(
        F.round(F.sum("amount"), 2).alias("total"),
        F.round(F.sum(F.when(F.col("product_line") == "hardware", F.col("amount"))
                       .otherwise(0)), 2).alias("hardware_rev"),
        F.sum(F.when(F.col("amount") > 1000, 1).otherwise(0)).alias("big_deals"),
        F.round(F.avg(F.when(F.col("units") > 1, F.col("amount"))), 2).alias("avg_multiunit"),
        F.round(100 * F.avg((F.col("fulfil_days") <= 3).cast("int")), 1).alias("sla_pct"))
    .orderBy("country").show())

# THE IDIOM: sum(when(cond, value).otherwise(0)) is Spark's FILTER/pivot
# equivalent. It computes several conditional metrics in ONE pass over the
# data instead of running N separate filtered queries.
#
# NOTE the avg(when(cond, col)) with NO otherwise: unmatched rows become NULL,
# and avg SKIPS nulls — so this averages only the matching rows. That is
# usually what you want, but it is implicit. Adding .otherwise(0) would
# average over ALL rows counting non-matches as zero. Completely different
# number. Be deliberate.
#
# avg(boolean.cast("int")) * 100 is the clean way to express a rate/percentage.

# ---------------------------------------------------------------------------
# Section 5 — Pivot
# ---------------------------------------------------------------------------

pivot = (sales.groupBy("country")
    .pivot("month", ["2026-01", "2026-02", "2026-03"])   # <-- values listed!
    .agg(F.round(F.sum("amount"), 2))
    .fillna(0)
    .orderBy("country"))
pivot.show()

# EXPECTED:
#   country | 2026-01 | 2026-02 | 2026-03
#   DE      |  450.0  | 5950.0  |    0.0
#   GB      |    0.0  |    0.0  | 5000.0
#   US      | 2000.0  | 2400.0  | 1300.0
#
# >>> ALWAYS PASS THE VALUE LIST TO pivot() <<<
# Without it, Spark runs a SEPARATE JOB first just to find the distinct values
# — an extra full scan before your real query starts. With high-cardinality
# columns it may also produce thousands of columns and blow up the plan.
#
# Multiple aggregations in one pivot produce compound column names:
#   .pivot("month", [...]).agg(F.sum("amount").alias("rev"),
#                              F.count("*").alias("n"))
#   -> columns "2026-01_rev", "2026-01_n", ...
#
# UNPIVOT (wide -> long) via stack:
#   pivot.selectExpr("country",
#       "stack(3, '2026-01', `2026-01`, '2026-02', `2026-02`, "
#       "'2026-03', `2026-03`) as (month, revenue)")

# ---------------------------------------------------------------------------
# Section 6 — rollup and cube: subtotals for free
# ---------------------------------------------------------------------------

print("ROLLUP — hierarchical subtotals:")
(sales.rollup("country", "region")
    .agg(F.round(F.sum("amount"), 2).alias("revenue"),
         F.count("*").alias("n"))
    .orderBy("country", "region").show())

# EXPECTED — note the NULLs marking subtotal levels:
#   country | region | revenue
#   null    | null   | 17100.0   <-- GRAND TOTAL
#   DE      | null   |  6400.0   <-- country subtotal
#   DE      | west   |  6400.0   <-- leaf
#   GB      | null   |  5000.0
#   ...
#
# ROLLUP produces the HIERARCHY: (a,b), (a), ()  — 3 levels for 2 columns.
# CUBE produces EVERY COMBINATION: (a,b), (a), (b), () — 2^n levels.
# Use rollup for a natural hierarchy (country > region > city). Use cube when
# you need every cross-section. Cube on 5 columns is 32 grouping sets — check
# the arithmetic before running it on large data.

print("CUBE — every combination:")
(sales.cube("country", "product_line")
    .agg(F.round(F.sum("amount"), 2).alias("revenue"))
    .orderBy("country", "product_line").show(25))

# TELLING SUBTOTAL NULLS FROM DATA NULLS:
# If the source column can itself be null, you cannot distinguish a subtotal
# row from a real null. Use grouping():
#     .agg(F.grouping("country").alias("is_country_subtotal"), ...)
# grouping() returns 1 when the column was aggregated away, 0 otherwise.
# grouping_id() encodes all of them as a single bitmask.

# ---------------------------------------------------------------------------
# Section 7 — Collecting values into arrays and maps
# ---------------------------------------------------------------------------

(sales.groupBy("customer_id")
    .agg(
        F.collect_list("product_line").alias("all_lines"),
        F.collect_set("product_line").alias("distinct_lines"),
        F.sort_array(F.collect_set("month")).alias("active_months"),
        F.round(F.sum("amount"), 2).alias("ltv"),
        F.max_by("product_line", "amount").alias("biggest_line"),
        F.first("country").alias("country"))
    .orderBy(F.desc("ltv")).show(truncate=False))

# EXPECTED:
#   C-01 | [hardware, services, hardware] | [hardware, services] | ltv 6900.0
#
# collect_list  keeps duplicates and ORDER IS NOT GUARANTEED
# collect_set   dedupes, order not guaranteed
# sort_array    makes the output deterministic — do this if you compare runs
#
# max_by(x, y) returns the x from the row with the maximum y. Extremely handy:
# "which product line generated the largest single sale?" in one expression,
# no self-join and no window.
#
# >>> collect_list IS AN UNBOUNDED-MEMORY OPERATION <<<
# All values for one key must fit in ONE executor's memory. With a skewed key
# holding 50 million values, that executor OOMs. It also forces
# ObjectHashAggregate instead of the faster HashAggregate. Use it on
# low-cardinality groups only, and never as a substitute for a proper join.
#
# first()/last() are NONDETERMINISTIC without an ordering — they return
# whatever arrived first in that partition. Use them only when the value is
# constant within the group (like country here), or use a window.

# ---------------------------------------------------------------------------
# Section 8 — Aggregation on the whole DataFrame
# ---------------------------------------------------------------------------

sales.agg(
    F.count("*").alias("rows"),
    F.round(F.sum("amount"), 2).alias("total_revenue"),
    F.countDistinct("customer_id").alias("customers"),
    F.min("sale_date").alias("first_sale"),
    F.max("sale_date").alias("last_sale"),
).show()

# agg() without groupBy aggregates everything to ONE row. Note this forces
# Exchange SinglePartition — fine for a scalar summary, but be aware it is a
# full collapse to one task.

# ---------------------------------------------------------------------------
# Section 9 — Share-of-total without a self-join
# ---------------------------------------------------------------------------

from pyspark.sql.window import Window

(by_country
    .withColumn("grand_total", F.sum("revenue").over(Window.partitionBy()))
    .withColumn("pct_of_total", F.round(100 * F.col("revenue") / F.col("grand_total"), 1))
    .select("country", "revenue", "pct_of_total")
    .orderBy(F.desc("revenue")).show())

# Window.partitionBy() with NO arguments = the entire DataFrame as one window.
# Convenient for share-of-total, but it means Exchange SinglePartition.
# Acceptable on an already-aggregated 3-row result; NEVER on raw data.
#
# The alternative — cross-joining a 1-row total — is more code and no faster.

spark.stop()

# ---------------------------------------------------------------------------
# TAKEAWAYS
# ---------------------------------------------------------------------------
# 1. Always .alias() aggregate outputs.
# 2. count("*") counts rows; count("col") skips nulls.
# 3. Partial + final HashAggregate around an Exchange = map-side combine.
# 4. countDistinct is not mergeable; approx_count_distinct is. Watch for Expand.
# 5. Percentiles beat averages for anything latency-shaped.
# 6. sum(when(...)) computes many conditional metrics in one pass.
# 7. Always pass the value list to pivot(), or you pay an extra full scan.
# 8. rollup = hierarchy (n+1 levels); cube = every combination (2^n).
# 9. collect_list is unbounded memory and nondeterministic in order.
# 10. max_by(x, y) replaces a self-join for "the x with the biggest y".
