"""
02 — DataFrame basics
=====================

REAL-WORLD SCENARIO
-------------------
An e-commerce order feed lands daily. Before anything downstream can use it you
must: enforce a schema, quarantine bad rows, standardise inconsistent free-text
fields (country codes, casing, whitespace), derive revenue columns, and hand a
clean table to analytics.

This is the single most common job in data engineering. Roughly 80% of
production PySpark is this file.

Run:
    spark-submit 02_dataframe_basics.py
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, StringType, IntegerType,
                               DoubleType, TimestampType)

spark = (SparkSession.builder
         .appName("dataframe-basics")
         .master("local[4]")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Section 1 — Explicit schema, always
# ---------------------------------------------------------------------------

schema = StructType([
    StructField("order_id",    StringType(),    nullable=False),
    StructField("customer_id", StringType(),    nullable=False),
    StructField("product",     StringType(),    nullable=True),
    StructField("category",    StringType(),    nullable=True),
    StructField("quantity",    IntegerType(),   nullable=True),
    StructField("unit_price",  DoubleType(),    nullable=True),
    StructField("discount",    DoubleType(),    nullable=True),
    StructField("country",     StringType(),    nullable=True),
    StructField("ordered_at",  TimestampType(), nullable=True),
])

rows = [
    ("O-1001", "C-01", "Laptop Pro 14",   "electronics", 1, 1899.00, 0.10, "US",  "2026-08-01 10:15:00"),
    ("O-1002", "C-02", " wireless mouse", "Electronics", 3,   24.99, 0.00, "us",  "2026-08-01 11:02:00"),
    ("O-1003", "C-01", "USB-C Hub",       "ELECTRONICS", 2,   49.50, 0.05, "USA", "2026-08-02 09:30:00"),
    ("O-1004", "C-03", "Desk Lamp",       "home",        1,   35.00, None, "GB",  "2026-08-02 14:45:00"),
    ("O-1005", "C-02", None,              "home",        1,   89.99, 0.00, "DE",  "2026-08-03 08:00:00"),
    ("O-1006", "C-04", "Monitor 27",      "electronics", 0,  329.00, 0.15, "US",  "2026-08-03 16:20:00"),
    ("O-1007", "C-03", "Chair Ergo",      "furniture",  -1,  450.00, 0.00, "gb",  "2026-08-04 12:00:00"),
    ("O-1008", "C-05", "Notebook A5",     "stationery", 10,    4.50, 0.00, None,  "2026-08-04 13:10:00"),
]

df = (spark.createDataFrame(rows, schema="""
        order_id STRING, customer_id STRING, product STRING, category STRING,
        quantity INT, unit_price DOUBLE, discount DOUBLE, country STRING,
        ordered_at STRING""")
      .withColumn("ordered_at", F.to_timestamp("ordered_at")))

df.printSchema()
df.show(truncate=False)

# EXPECTED: 8 rows, ordered_at as timestamp not string.
#
# WHY THE DDL STRING FORM: `schema="a INT, b STRING"` is far more readable than
# StructType for wide tables and identical in effect. Use StructType only when
# you need nullable control or nested types.
#
# NOTE ordered_at arrives as STRING and is converted explicitly. Declaring it
# TimestampType in createDataFrame from string input silently yields nulls —
# the classic "my timestamps are all null" bug. Read as string, convert with
# to_timestamp(), and pass a format when it is not ISO-8601:
#     F.to_timestamp("col", "dd/MM/yyyy HH:mm:ss")

# ---------------------------------------------------------------------------
# Section 2 — Inspect before you transform
# ---------------------------------------------------------------------------

print("rows:", df.count(), " columns:", len(df.columns))
print("columns:", df.columns)
print("dtypes:", df.dtypes)

df.select("quantity", "unit_price", "discount").describe().show()

# EXPECTED: count/mean/stddev/min/max per numeric column.
#
# describe() gives 5 stats. summary() gives more, including quartiles:
#     df.summary("count", "min", "25%", "50%", "75%", "max").show()
#
# LOOK AT THE MIN. quantity min is -1, which is impossible. Data quality bugs
# announce themselves in describe() output long before they reach a dashboard.

# Null audit — the check people skip
null_counts = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
])
null_counts.show()

# EXPECTED:
#   order_id=0, customer_id=0, product=1, category=0, quantity=0,
#   unit_price=0, discount=1, country=1, ordered_at=0
#
# THE IDIOM: count(col) SKIPS nulls, so you cannot count nulls with count().
# Convert the condition to 1/0 and sum it. The list comprehension inside
# select() builds one expression per column in a single pass — far better than
# looping and running one query per column.

# ---------------------------------------------------------------------------
# Section 3 — select, alias, and column references
# ---------------------------------------------------------------------------

df.select("order_id", "product", F.col("unit_price").alias("price")).show(3)

# Four equivalent ways to reference a column:
#     "unit_price"          string   — simplest, use by default
#     F.col("unit_price")   Column   — needed for expressions
#     df.unit_price         attr     — breaks on names with spaces/dots
#     df["unit_price"]      index    — always works, needed to disambiguate joins
#
# Use the string form until you need an expression, then F.col().

df.selectExpr("order_id", "quantity * unit_price AS gross").show(3)

# selectExpr takes SQL strings. Handy for quick expressions; less
# refactor-safe than the typed API because typos become runtime errors.

# ---------------------------------------------------------------------------
# Section 4 — Cleaning: the standardisation pass
# ---------------------------------------------------------------------------

clean = (df
    # trim + title-case product names
    .withColumn("product", F.initcap(F.trim(F.col("product"))))
    # normalise category casing
    .withColumn("category", F.lower(F.trim(F.col("category"))))
    # canonical 2-letter country codes
    .withColumn("country",
        F.when(F.upper(F.trim("country")).isin("US", "USA"), "US")
         .when(F.upper(F.trim("country")).isin("GB", "UK"), "GB")
         .when(F.col("country").isNull(), "UNKNOWN")
         .otherwise(F.upper(F.trim("country"))))
    # nulls -> sensible defaults
    .fillna({"discount": 0.0, "product": "UNKNOWN"})
)

clean.select("product", "category", "country", "discount").show(truncate=False)

# EXPECTED:
#   " wireless mouse" -> "Wireless Mouse"
#   "Electronics"/"ELECTRONICS" -> "electronics"
#   "usa"/"us"/"US" -> "US"
#   null discount -> 0.0
#
# WHY THIS MATTERS: without normalisation, `GROUP BY category` returns three
# separate rows for Electronics/electronics/ELECTRONICS, and `GROUP BY country`
# splits US revenue three ways. Silent, and it corrupts every downstream number.
#
# fillna(dict) fills per column. fillna(0) fills ALL numeric columns, which is
# usually too broad — filling a null price with 0 invents a free product.
# Always pass the dict form.

# ---------------------------------------------------------------------------
# Section 5 — Validation: quarantine, do not silently drop
# ---------------------------------------------------------------------------

RULES = (F.col("quantity") > 0) & (F.col("unit_price") > 0) & \
        (F.col("discount").between(0, 1))

valid   = clean.filter(RULES)
invalid = clean.filter(~RULES | RULES.isNull())

print(f"valid {valid.count()}  quarantined {invalid.count()}")
invalid.select("order_id", "quantity", "unit_price", "discount").show()

# EXPECTED:
#   valid 6  quarantined 2
#   O-1006 quantity 0   |  O-1007 quantity -1
#
# >>> NOTE `~RULES | RULES.isNull()` <<<
# Spark uses THREE-VALUED LOGIC. If any operand is NULL the comparison is NULL,
# not False — and NULL fails both `RULES` and `~RULES`. So a naive
# `filter(~RULES)` SILENTLY LOSES null-bearing rows from BOTH branches, and
# valid + invalid < total.
#
# This is the number-one silent row-loss bug in Spark. Always verify:
#     assert valid.count() + invalid.count() == clean.count()

assert valid.count() + invalid.count() == clean.count(), "rows lost!"
print("row conservation check passed")

# ---------------------------------------------------------------------------
# Section 6 — Derived columns
# ---------------------------------------------------------------------------

enriched = (valid
    .withColumn("gross",   F.round(F.col("quantity") * F.col("unit_price"), 2))
    .withColumn("savings", F.round(F.col("gross") * F.col("discount"), 2))
    .withColumn("net",     F.round(F.col("gross") - F.col("savings"), 2))
    .withColumn("order_date", F.to_date("ordered_at"))
    .withColumn("order_hour", F.hour("ordered_at"))
    .withColumn("dow",        F.date_format("ordered_at", "EEEE"))
    .withColumn("size_band",
        F.when(F.col("net") >= 500, "large")
         .when(F.col("net") >= 100, "medium")
         .otherwise("small"))
    .withColumn("is_discounted", (F.col("discount") > 0).cast("int")))

enriched.select("order_id", "gross", "savings", "net",
                "size_band", "dow").show(truncate=False)

# EXPECTED:
#   O-1001 gross 1899.00 savings 189.90 net 1709.10 large Saturday
#   O-1002 gross   74.97 savings   0.00 net   74.97 small Saturday
#   ...
#
# .cast("int") on a boolean: booleans do not aggregate. Casting to int makes
# avg(is_discounted) directly the discount rate.

# ---------------------------------------------------------------------------
# Section 7 — The withColumn-in-a-loop anti-pattern
# ---------------------------------------------------------------------------

string_cols = [c for c, t in enriched.dtypes if t == "string"]

# DON'T — one plan node per iteration; with 500 columns this hangs the DRIVER
# for minutes on ZERO rows of data, because Catalyst rule application is
# roughly quadratic in plan depth.
#
#   for c in string_cols:
#       enriched = enriched.withColumn(c, F.trim(F.col(c)))

# DO — one plan node
trimmed = enriched.select(*[
    F.trim(F.col(c)).alias(c) if c in string_cols else F.col(c)
    for c in enriched.columns
])
print("trimmed columns:", len(trimmed.columns))

# This is a genuine production incident pattern, and an excellent answer to
# "have you seen Spark be slow for a non-data reason?"

# ---------------------------------------------------------------------------
# Section 8 — Sorting, dedup, renaming, dropping
# ---------------------------------------------------------------------------

enriched.orderBy(F.desc("net")).select("order_id", "net").show(3)
enriched.orderBy(F.col("country").asc(), F.col("net").desc_nulls_last()).show(3)

# asc_nulls_first / desc_nulls_last control null placement explicitly.
# Default: nulls first ascending, nulls last descending. Be explicit when it
# matters — the default surprises people.

deduped = enriched.dropDuplicates(["customer_id", "product", "order_date"])
print("after dedup:", deduped.count())

# WARNING: dropDuplicates is (a) a SHUFFLE — a wide operation, visible as an
# Exchange in the plan — and (b) NON-DETERMINISTIC about which duplicate
# survives. For real last-write-wins semantics use a window:
#     w = Window.partitionBy("customer_id","product").orderBy(F.desc("ordered_at"))
#     df.withColumn("rn", F.row_number().over(w)).filter("rn = 1")

final = (enriched
    .withColumnRenamed("net", "net_revenue")
    .drop("savings", "is_discounted"))
print("final columns:", final.columns)

# ---------------------------------------------------------------------------
# Section 9 — See the plan
# ---------------------------------------------------------------------------

final.filter(F.col("country") == "US").select("order_id", "net_revenue").explain("formatted")

# READ THE PLAN FOR:
#   1. Operators prefixed with `*` — whole-stage codegen fused them into one
#      generated Java method. Missing stars mean something broke the pipeline
#      (usually a Python UDF).
#   2. `PushedFilters` — which predicates reached the data source.
#   3. `Exchange` nodes — each one is a shuffle and a stage boundary. Count
#      them; that is your cost estimate before you run anything.
#   4. Here there is NO Exchange: everything is narrow, one stage.

spark.stop()

# ---------------------------------------------------------------------------
# TAKEAWAYS
# ---------------------------------------------------------------------------
# 1. Explicit schemas. Never inferSchema in production.
# 2. Read timestamps as string, convert with to_timestamp().
# 3. Audit nulls with sum(col.isNull().cast("int")) — count() skips nulls.
# 4. Normalise casing and whitespace before any GROUP BY.
# 5. Three-valued logic: ~RULES does NOT capture null rows. Assert conservation.
# 6. fillna(dict), never bare fillna(0).
# 7. One select() with a comprehension, never withColumn in a loop.
# 8. dropDuplicates is a shuffle and is nondeterministic — use a window.
# 9. Read explain() after every change. Count the Exchanges.
