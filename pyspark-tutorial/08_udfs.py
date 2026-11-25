"""
08 — UDFs: when to write one, and which kind
============================================

REAL-WORLD SCENARIO
-------------------
An insurance claims pipeline. You must score each claim for fraud risk using a
scoring function the actuarial team supplied as Python, validate policy numbers
against a checksum algorithm, and geocode postcodes via a lookup.

Then you discover the job takes 6 hours. This file shows why, and how to get
it to 20 minutes without changing the business logic.

THE HIERARCHY, in the order you should try things:
    1. Built-in functions        (400+; stay in JVM codegen)
    2. Higher-order functions    (transform/filter/aggregate on arrays)
    3. Pandas UDF (vectorised)   (Arrow batches, ~10x faster than #4)
    4. Python UDF (row-at-a-time) (last resort)

Run:
    spark-submit 08_udfs.py
"""

import time
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import (DoubleType, StringType, IntegerType,
                               BooleanType, StructType, StructField, ArrayType)

spark = (SparkSession.builder.appName("udfs")
         .master("local[4]")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

claims = (spark.range(0, 400_000)
    .withColumn("claim_id", F.concat(F.lit("CLM-"), F.col("id").cast("string")))
    .withColumn("amount",   F.round(F.rand(seed=1) * 50_000, 2))
    .withColumn("age",      (F.rand(seed=2) * 60 + 18).cast("int"))
    .withColumn("prior_claims", (F.rand(seed=3) * 6).cast("int"))
    .withColumn("policy_no", F.concat(F.lit("POL"),
                 F.lpad((F.col("id") % 99991).cast("string"), 8, "0")))
    .withColumn("postcode", F.concat(
                 F.lit("SW"), (F.col("id") % 20 + 1).cast("string"), F.lit(" 1AA")))
    .drop("id")
    .cache())
print("claims:", claims.count())

# ---------------------------------------------------------------------------
# Section 1 — The benchmark: same logic, three ways
# ---------------------------------------------------------------------------

def score_py(amount, age, prior):
    """Actuarial risk score. Pure Python, as supplied."""
    if amount is None or age is None or prior is None:
        return None
    base = amount / 10_000.0
    age_factor = 1.5 if age < 25 else (1.2 if age > 70 else 1.0)
    return round(min(base * age_factor + prior * 0.4, 10.0), 3)


# --- 1. Built-in expression (stays entirely in the JVM) ---
built_in = (F.round(
    F.least(
        (F.col("amount") / 10_000.0) *
        F.when(F.col("age") < 25, 1.5)
         .when(F.col("age") > 70, 1.2)
         .otherwise(1.0)
        + F.col("prior_claims") * 0.4,
        F.lit(10.0)), 3))

t0 = time.time()
claims.withColumn("risk", built_in).agg(F.avg("risk")).collect()
t_builtin = time.time() - t0

# --- 2. Pandas UDF (vectorised over Arrow batches) ---
@pandas_udf(DoubleType())
def score_pandas(amount: pd.Series, age: pd.Series, prior: pd.Series) -> pd.Series:
    base = amount / 10_000.0
    age_factor = pd.Series(1.0, index=age.index)
    age_factor[age < 25] = 1.5
    age_factor[age > 70] = 1.2
    return (base * age_factor + prior * 0.4).clip(upper=10.0).round(3)

t0 = time.time()
claims.withColumn("risk", score_pandas("amount", "age", "prior_claims")) \
      .agg(F.avg("risk")).collect()
t_pandas = time.time() - t0

# --- 3. Plain Python UDF (row at a time) ---
score_udf = udf(score_py, DoubleType())

t0 = time.time()
claims.withColumn("risk", score_udf("amount", "age", "prior_claims")) \
      .agg(F.avg("risk")).collect()
t_python = time.time() - t0

print(f"\nbuilt-in    {t_builtin:6.2f}s   1.0x")
print(f"pandas udf  {t_pandas:6.2f}s   {t_pandas/t_builtin:5.1f}x slower")
print(f"python udf  {t_python:6.2f}s   {t_python/t_builtin:5.1f}x slower")

# EXPECTED (ratios matter, absolute numbers vary by machine):
#   built-in      0.8s    1.0x
#   pandas udf    2.4s    3.0x slower
#   python udf   14.0s   17.5x slower
#
# >>> WHY, MECHANICALLY — this is the part to be able to explain <<<
#
# BUILT-IN: the expression is compiled into whole-stage-generated Java. Rows
#   stay in Tungsten binary format. Zero serialization, zero process boundary.
#   In explain() the operators carry a `*` prefix (codegen).
#
# PYTHON UDF: for EVERY ROW — the JVM pickles the value, writes it over a
#   socket to a Python worker process, Python unpickles, runs your function,
#   pickles the result, sends it back, the JVM unpickles. Per-row IPC.
#   In the plan you see `BatchEvalPython`, and the surrounding operators
#   LOSE their `*` codegen markers — the pipeline is severed.
#
# PANDAS UDF: Spark batches rows into Arrow columnar record batches (default
#   10,000 rows) and hands whole batches to Python. Your function operates on
#   a pd.Series with vectorised NumPy. The crossing is amortised over thousands
#   of rows. Plan shows `ArrowEvalPython`. Still crosses the boundary — just
#   far less often.

# ---------------------------------------------------------------------------
# Section 2 — See the plan difference
# ---------------------------------------------------------------------------

print("\n--- built-in: note the * codegen markers ---")
claims.withColumn("risk", built_in).explain("formatted")

print("\n--- python UDF: note BatchEvalPython and missing * ---")
claims.withColumn("risk", score_udf("amount", "age", "prior_claims")).explain("formatted")

# Scan a plan for BatchEvalPython / ArrowEvalPython. Their presence is a
# performance smell you can spot in two seconds.

# ---------------------------------------------------------------------------
# Section 3 — Pandas UDF types
# ---------------------------------------------------------------------------

# TYPE 1: Series -> Series (scalar). One value out per value in. Most common.
@pandas_udf(StringType())
def risk_band(score: pd.Series) -> pd.Series:
    return pd.cut(score, bins=[-1, 2, 5, 8, 100],
                  labels=["low", "medium", "high", "severe"]).astype(str)

# TYPE 2: Series -> scalar (aggregate). Used with groupBy().agg().
@pandas_udf(DoubleType())
def weighted_mean(value: pd.Series, weight: pd.Series) -> float:
    if weight.sum() == 0:
        return float("nan")
    return float((value * weight).sum() / weight.sum())

# TYPE 3: Iterator[Series] -> Iterator[Series].
# Use when there is EXPENSIVE PER-WORKER SETUP — loading an ML model, opening
# a DB connection, compiling a regex. The setup runs ONCE PER PARTITION
# instead of once per batch.
from typing import Iterator

@pandas_udf(DoubleType())
def score_with_setup(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # expensive one-time setup per worker
    lookup = {i: 1.0 + i * 0.01 for i in range(100)}     # pretend: a loaded model
    for s in it:
        yield s.map(lambda v: v * lookup.get(int(v) % 100, 1.0))

scored = (claims
    .withColumn("risk", score_pandas("amount", "age", "prior_claims"))
    .withColumn("band", risk_band("risk")))

scored.groupBy("band").agg(
    F.count("*").alias("n"),
    F.round(F.avg("risk"), 3).alias("avg_risk"),
    weighted_mean("risk", "amount").alias("amount_weighted_risk"),
).orderBy("band").show()

# TYPE 4: applyInPandas — the whole GROUP as a pandas DataFrame.
# The most powerful and the most dangerous: EVERY ROW OF ONE GROUP MUST FIT
# IN ONE EXECUTOR'S MEMORY. With a skewed key holding 50M rows, that OOMs.

out_schema = StructType([
    StructField("postcode", StringType()),
    StructField("n", IntegerType()),
    StructField("median_amount", DoubleType()),
    StructField("iqr", DoubleType()),
])

def group_stats(pdf: pd.DataFrame) -> pd.DataFrame:
    q1, q3 = pdf["amount"].quantile([0.25, 0.75])
    return pd.DataFrame([{
        "postcode": pdf["postcode"].iloc[0],
        "n": len(pdf),
        "median_amount": float(pdf["amount"].median()),
        "iqr": float(q3 - q1),
    }])

claims.groupBy("postcode").applyInPandas(group_stats, schema=out_schema) \
      .orderBy("postcode").show(5)

# applyInPandas is genuinely useful for per-group model fitting, interpolation,
# or any algorithm that needs the whole group at once. Check group sizes first:
#     claims.groupBy("postcode").count().describe().show()

# ---------------------------------------------------------------------------
# Section 4 — When a UDF is genuinely unavoidable
# ---------------------------------------------------------------------------

def luhn_valid(policy: str) -> bool:
    """Checksum with control flow that has no SQL equivalent."""
    if not policy:
        return False
    digits = [int(c) for c in policy if c.isdigit()]
    if not digits:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

@pandas_udf(BooleanType())
def luhn_pandas(s: pd.Series) -> pd.Series:
    return s.map(luhn_valid)

# >>> ORDER MATTERS: FILTER BEFORE THE UDF <<<
#
# BAD — UDF runs on all 400,000 rows:
#   claims.withColumn("valid", luhn_pandas("policy_no")).filter("valid")
#
# GOOD — cheap built-in filter first, UDF only on survivors:
narrowed = claims.filter(F.length("policy_no") == 11)
t0 = time.time()
n_valid = narrowed.withColumn("valid", luhn_pandas("policy_no")) \
                  .filter(F.col("valid")).count()
print(f"\nvalid policies: {n_valid} in {time.time()-t0:.1f}s")

# Catalyst will NOT push a filter through a UDF, because it cannot know the
# UDF is deterministic or side-effect-free. You must order it yourself.
# This is the single biggest UDF optimisation and costs nothing.

# ---------------------------------------------------------------------------
# Section 5 — Null handling, the #1 UDF crash
# ---------------------------------------------------------------------------

# A Python UDF receives None for SQL NULL. Arithmetic on None raises TypeError,
# which kills the TASK; Spark retries it 4 times, then the whole JOB dies —
# because of one null in 50 million rows.

@udf(DoubleType())
def unsafe(x):
    return x * 2                      # TypeError when x is None

@udf(DoubleType())
def safe(x):
    return None if x is None else x * 2

nulls = spark.createDataFrame([(1.0,), (None,), (3.0,)], "v DOUBLE")
nulls.withColumn("doubled", safe("v")).show()
# nulls.withColumn("doubled", unsafe("v")).show()    # <- would fail the job

# THREE DEFENCES:
#   1. Guard for None inside the function (above).
#   2. Filter nulls before the UDF:  .filter(col("v").isNotNull())
#   3. Wrap the body in try/except and return None on failure — a "dead letter"
#      value you can then count and alert on.
#
# Also: ALWAYS declare the correct return type. Declaring DoubleType and
# returning a Python int yields NULL silently, not an error.

# ---------------------------------------------------------------------------
# Section 6 — Determinism and closures
# ---------------------------------------------------------------------------

import random

@udf(DoubleType())
def nondeterministic(x):
    return x * random.random()        # DANGEROUS

# A retried task recomputes with DIFFERENT values. Spark also assumes UDFs are
# deterministic when reordering, so results can be inconsistent between the
# filter and the projection. Mark it, or better, use F.rand(seed=...):
#   nondeterministic = nondeterministic.asNondeterministic()

# CLOSURE CAPTURE: a UDF referencing an outer variable SERIALISES that variable
# to every task.
big_lookup = {i: f"region_{i}" for i in range(20)}

# BAD at scale — the dict ships with every task:
#   @udf(StringType())
#   def lookup_bad(k): return big_lookup.get(k)
#
# GOOD — broadcast once per executor:
bc = spark.sparkContext.broadcast(big_lookup)

@udf(StringType())
def lookup_good(k):
    return bc.value.get(k)

claims.withColumn("pc_num", (F.length("postcode")).cast("int")) \
      .withColumn("region", lookup_good("pc_num")).select("postcode", "region").show(3)

# A large captured object shows up as high "Task Deserialization Time" in the
# Spark UI — a symptom that looks like "Spark is slow" but is entirely your
# closure. Broadcast anything above ~1 MB.
#
# Better still: if the lookup is a table, JOIN instead of a UDF. A broadcast
# join is faster than a broadcast dict inside a UDF, and Catalyst can optimise it.

# ---------------------------------------------------------------------------
# Section 7 — Returning complex types
# ---------------------------------------------------------------------------

parse_schema = StructType([
    StructField("area",   StringType()),
    StructField("sector", IntegerType()),
    StructField("unit",   StringType()),
])

@udf(parse_schema)
def parse_postcode(pc):
    if not pc:
        return None
    try:
        outward, inward = pc.split(" ")
        area = "".join(c for c in outward if c.isalpha())
        sector = int("".join(c for c in outward if c.isdigit()) or 0)
        return (area, sector, inward)
    except Exception:
        return None

(claims.limit(5)
 .withColumn("pc", parse_postcode("postcode"))
 .select("postcode", "pc.area", "pc.sector", "pc.unit").show())

# Returning a struct lets ONE UDF call produce several columns. Calling three
# separate UDFs on the same input crosses the Python boundary three times.
#
# Note: this particular parse is expressible with regexp_extract and would be
# ~30x faster as built-ins. Shown for the technique, not as a recommendation.

# ---------------------------------------------------------------------------
# Section 8 — Registering for SQL
# ---------------------------------------------------------------------------

spark.udf.register("risk_score", score_py, DoubleType())
claims.createOrReplaceTempView("claims")
spark.sql("""
    SELECT claim_id, amount, age,
           risk_score(amount, age, prior_claims) AS risk
    FROM claims
    WHERE amount > 40000
    LIMIT 5
""").show()

# spark.udf.register makes it callable from SQL. Same cost profile — the SQL
# syntax does not make it faster.

claims.unpersist()
spark.stop()

# ---------------------------------------------------------------------------
# TAKEAWAYS
# ---------------------------------------------------------------------------
# 1. Search the 400+ built-ins before writing any UDF. It is usually there.
# 2. Python UDF ~10-50x slower than built-in; Pandas UDF ~2-4x. Know why:
#    per-row pickling+socket IPC vs Arrow batches vs staying in the JVM.
# 3. BatchEvalPython / ArrowEvalPython in the plan = codegen broken there.
# 4. Filter BEFORE the UDF. Catalyst will not push a filter through one.
# 5. Guard for None. An exception in a UDF kills the job, not the row.
# 6. Declare the right return type — a mismatch yields silent NULLs.
# 7. Broadcast large lookups; better, replace the lookup with a join.
# 8. Return a struct to produce many columns from one boundary crossing.
# 9. applyInPandas needs the whole group in one executor's memory. Check sizes.
# 10. Pandas UDF memory lives OUTSIDE the JVM heap — tune
#     spark.executor.memoryOverhead, not spark.executor.memory.
