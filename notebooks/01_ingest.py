# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingestion: Raw → Bronze
# MAGIC
# MAGIC **Dataset:** `Yelp/yelp_review_full` — 650,000 real Yelp reviews, Parquet, 299 MB.
# MAGIC Direct download from Hugging Face, no login, no licence click-through.
# MAGIC
# MAGIC **Covers sidebar:** Data Ingestion (Auto Loader), and sets up everything downstream.
# MAGIC
# MAGIC **Runtime:** ~5 minutes, mostly download.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1 — Setup
# MAGIC
# MAGIC A Unity Catalog **Volume** mounts as a real POSIX path, so plain Python file
# MAGIC operations (`open`, `os.path`, `urlretrieve`) work directly on it.
# MAGIC That is the whole reason we land files here instead of `/tmp`:
# MAGIC `/tmp` is driver-local and serverless executors cannot read it.

# COMMAND ----------

import os, time, urllib.request
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

CATALOG = "workspace"
SCHEMA  = "default"
VOL     = f"/Volumes/{CATALOG}/{SCHEMA}/lab"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.lab")

RAW = f"{VOL}/yelp/raw"          # Auto Loader watches this directory
os.makedirs(RAW, exist_ok=True)

print(VOL)

# EXPECTED OUTPUT:
#   /Volumes/workspace/default/lab

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2 — Download the Yelp Parquet files
# MAGIC
# MAGIC Two splits: train (650k rows) and test (50k rows). We land them as separate
# MAGIC files on purpose — Auto Loader in Section 4 will pick them up incrementally,
# MAGIC one at a time, which is how you observe incremental ingestion.

# COMMAND ----------

BASE = "https://huggingface.co/datasets/Yelp/yelp_review_full/resolve/main/yelp_review_full"

FILES = {
    "train-00000-of-00001.parquet": f"{RAW}/reviews_train.parquet",
    "test-00000-of-00001.parquet":  f"{RAW}/reviews_test.parquet",
}

for src, dst in FILES.items():
    if not os.path.exists(dst):
        t0 = time.time()
        urllib.request.urlretrieve(f"{BASE}/{src}", dst)
        print(f"downloaded {os.path.basename(dst)} in {time.time()-t0:.0f}s")
    print(f"{os.path.getsize(dst)/1e6:8.1f} MB  {dst}")

# EXPECTED OUTPUT (first run):
#   downloaded reviews_train.parquet in 45s
#      299.4 MB  /Volumes/workspace/default/lab/yelp/raw/reviews_train.parquet
#   downloaded reviews_test.parquet in 4s
#       23.5 MB  /Volumes/workspace/default/lab/yelp/raw/reviews_test.parquet
#
# The `if not os.path.exists` guard makes this cell IDEMPOTENT. You will re-run
# cells constantly; a cell that re-downloads 300 MB every time is a cell you
# start skipping, and then your notebook breaks on restart.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 — First read: what Parquet gives you for free

# COMMAND ----------

raw = spark.read.parquet(f"{RAW}/reviews_train.parquet")

raw.printSchema()
print("rows:", raw.count())

# EXPECTED OUTPUT:
#   root
#    |-- label: long (nullable = true)
#    |-- text: string (nullable = true)
#   rows: 650000
#
# EXPLANATION — why both lines are instant:
#   printSchema() reads the Parquet FOOTER metadata (a few KB), not the data.
#   count()       is summed from row-group metadata, not by scanning rows.
# Contrast with CSV, where `inferSchema=true` must read the ENTIRE file and
# launches a real Spark job before you have called any action. Parquet is
# self-describing; CSV is not. This is the single biggest practical difference.

# COMMAND ----------

display(raw.limit(5))

# EXPECTED OUTPUT: 5 rows like
#   label | text
#   -----------------------------------------------------------------------
#     4   | "dr. goldberg offers everything i look for in a general practitioner..."
#     1   | "Unfortunately, the frustration of being Dr. Goldberg's patient..."
#
# NOTE: `label` is 0-4, meaning 1-5 stars. It is ZERO-INDEXED.
# Forgetting this produces a model that is systematically one star low —
# a silent correctness bug, not an error. We fix it in Section 5.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4 — Auto Loader: incremental ingestion
# MAGIC
# MAGIC This is the **Data Ingestion** item in the sidebar. Auto Loader tracks which
# MAGIC files it has already processed in a checkpoint, so re-running processes only
# MAGIC new arrivals. It is the file-source equivalent of Kafka offsets.

# COMMAND ----------

BRONZE = f"{CATALOG}.{SCHEMA}.yelp_bronze"

(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "parquet")
    .option("cloudFiles.schemaLocation", f"{VOL}/yelp/_schema")
    .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
    .load(RAW)
    .withColumn("_ingest_ts",   F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .writeStream
    .option("checkpointLocation", f"{VOL}/yelp/_ckpt_bronze")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)      # process what exists, then STOP
    .toTable(BRONZE)
    .awaitTermination())

print("bronze rows:", spark.table(BRONZE).count())

# EXPECTED OUTPUT:
#   bronze rows: 700000        (650,000 train + 50,000 test)
#
# LINE BY LINE:
#
#   format("cloudFiles")
#     Auto Loader. Not a normal file source — it maintains its own file-discovery
#     state so it never reprocesses a file it has already seen.
#
#   cloudFiles.schemaLocation
#     Where the INFERRED schema is persisted. Required. On restart Auto Loader
#     reads the stored schema instead of re-inferring, which keeps schema stable
#     across runs. Delete this directory and schema inference starts over.
#
#   schemaEvolutionMode="addNewColumns"
#     If a new file has an extra column, add it to the table rather than failing.
#     Alternatives: "rescue" (put unknown columns in a _rescued_data blob),
#     "failOnNewColumns", "none". This is exactly the Airport_fee / airport_fee
#     schema-drift problem, handled declaratively.
#
#   _metadata.file_path
#     A hidden column every file source exposes. Recording the source file is
#     essential lineage — when a row looks wrong, you need to know which file
#     it came from. Costs nothing.
#
#   trigger(availableNow=True)
#     Process all currently-available data, then terminate. This is the
#     "incremental batch" pattern: streaming semantics (checkpoints, exactly-once
#     file handling) with batch execution. Use this instead of a scheduled batch
#     job whenever the source is an append-only directory.
#
#   awaitTermination()
#     Block until the stream finishes. Without it the cell returns immediately
#     and the count on the next line races the writer.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prove it is incremental
# MAGIC
# MAGIC Re-run the cell above right now. It processes **zero** new rows.

# COMMAND ----------

# Re-running the same stream: nothing new to do
(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "parquet")
    .option("cloudFiles.schemaLocation", f"{VOL}/yelp/_schema")
    .load(RAW)
    .withColumn("_ingest_ts",   F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .writeStream
    .option("checkpointLocation", f"{VOL}/yelp/_ckpt_bronze")
    .trigger(availableNow=True)
    .toTable(BRONZE)
    .awaitTermination())

print("bronze rows after re-run:", spark.table(BRONZE).count())

# EXPECTED OUTPUT:
#   bronze rows after re-run: 700000        <-- UNCHANGED
#
# The checkpoint knows both files are done. This is the property that makes
# Auto Loader safe to schedule every 5 minutes against a growing S3 prefix.
#
# TO SEE IT INGEST SOMETHING NEW: copy any parquet file into RAW and re-run.
# Only the new file is processed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5 — Bronze inspection and the label fix

# COMMAND ----------

bronze = spark.table(BRONZE)

display(
    bronze.groupBy("label").count().orderBy("label")
)

# EXPECTED OUTPUT — near-perfectly balanced, by construction of the dataset:
#   label | count
#   ------+--------
#     0   | 140000
#     1   | 140000
#     2   | 140000
#     3   | 140000
#     4   | 140000
#   (train split is exactly 130k per class; test adds 10k per class)
#
# INTERPRETATION: a balanced dataset is UNUSUAL and worth noticing. Real review
# data is heavily skewed toward 5 stars. This one was deliberately balanced for
# ML benchmarking. That matters in file 03: accuracy is a meaningful metric here,
# whereas on real skewed data it would be misleading.

# COMMAND ----------

display(
    bronze.select(
        F.count("*").alias("rows"),
        F.countDistinct("_source_file").alias("source_files"),
        F.min(F.length("text")).alias("min_chars"),
        F.round(F.avg(F.length("text")), 0).alias("avg_chars"),
        F.max(F.length("text")).alias("max_chars"),
        F.sum(F.when(F.col("text").isNull(), 1).otherwise(0)).alias("null_text"),
    )
)

# EXPECTED OUTPUT (approximate):
#   rows   | source_files | min_chars | avg_chars | max_chars | null_text
#   700000 |      2       |     3     |    732    |   5000    |     0
#
# EXPLANATION of F.sum(F.when(...)):
#   This is the idiomatic conditional count in Spark. `count(col)` skips nulls,
#   so counting nulls with count() is impossible — you must convert the
#   condition to 1/0 and sum it. Remember this pattern; you will use it
#   constantly for data quality checks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6 — Write the corrected bronze table
# MAGIC
# MAGIC Bronze should hold raw data plus lineage, with **no business logic**.
# MAGIC The one thing we fix here is the zero-indexed label, because leaving a
# MAGIC trap like that in the raw layer guarantees somebody downstream steps in it.

# COMMAND ----------

(bronze
    .withColumn("stars", F.col("label") + 1)          # 0-4  ->  1-5
    .withColumn("review_id", F.expr("uuid()"))
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE))

display(spark.table(BRONZE).select("review_id", "stars", "label", "text").limit(3))

# EXPECTED OUTPUT:
#   review_id                            | stars | label | text
#   -------------------------------------+-------+-------+------------------
#   3f2a...-....                         |   5   |   4   | "dr. goldberg..."
#
# WHY overwriteSchema=true:
#   We are adding columns to an existing Delta table with mode("overwrite").
#   Delta ENFORCES schema by default and would reject this write. That
#   enforcement is a feature — it is what stops the silent column-drift
#   garbage that plain Parquet allows. You override it deliberately, once,
#   rather than never noticing.
#
# WHY uuid() and not monotonically_increasing_id():
#   monotonically_increasing_id() is NOT stable across runs and encodes the
#   partition index in the high bits, so values change if partitioning changes.
#   It is fine as a within-run row marker, wrong as a durable key.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7 — Delta gives you history for free

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {BRONZE}"))

# EXPECTED OUTPUT: one row per write, e.g.
#   version | operation                    | operationMetrics
#   --------+------------------------------+----------------------------------
#      2    | WRITE                        | {numOutputRows: 700000, ...}
#      1    | STREAMING UPDATE             | {numOutputRows: 0, ...}
#      0    | STREAMING UPDATE             | {numOutputRows: 700000, ...}
#
# Note version 1 has numOutputRows 0 — that is the idempotent re-run from
# Section 4, recorded as a real transaction that wrote nothing.
#
# TIME TRAVEL:
#   spark.read.option("versionAsOf", 0).table(BRONZE)
#
# This snapshot model is the same idea as Iceberg snapshots. Both are
# "Parquet data files + a metadata layer that makes commits atomic".

# COMMAND ----------

display(spark.sql(f"DESCRIBE DETAIL {BRONZE}"))

# EXPECTED OUTPUT (key columns):
#   format | numFiles | sizeInBytes | location
#   -------+----------+-------------+---------------------------------
#   delta  |    4     |  ~230 MB    | s3://.../workspace/default/...
#
# numFiles is worth watching over time. Many small files = slow reads
# (listing overhead + task overhead + poor compression). Fix with:
#   OPTIMIZE workspace.default.yelp_bronze

# COMMAND ----------

# MAGIC %md
# MAGIC ## What you built
# MAGIC
# MAGIC | Object | What it is |
# MAGIC |---|---|
# MAGIC | `/Volumes/workspace/default/lab/yelp/raw/` | Landing zone, 2 Parquet files |
# MAGIC | `workspace.default.yelp_bronze` | 700k rows, Delta, with lineage columns |
# MAGIC | `.../_ckpt_bronze` | Auto Loader checkpoint — makes re-runs safe |
# MAGIC | `.../_schema` | Persisted inferred schema |
# MAGIC
# MAGIC ### Concepts demonstrated
# MAGIC - Parquet footer schema + metadata counts (no data read)
# MAGIC - Auto Loader incremental file discovery via checkpoint
# MAGIC - `trigger(availableNow=True)` — streaming semantics, batch execution
# MAGIC - Schema evolution mode as the answer to column drift
# MAGIC - `_metadata.file_path` for lineage
# MAGIC - `sum(when(...))` as the conditional-count idiom
# MAGIC - Delta schema enforcement and `DESCRIBE HISTORY`
# MAGIC
# MAGIC **Next:** `02_transform.py` — silver/gold, joins, skew, windows, and reading plans.
