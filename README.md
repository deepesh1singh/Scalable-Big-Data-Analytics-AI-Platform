# databricks-pyspark-lab

Hands-on PySpark and Databricks, built from real datasets and real failures.

Two halves:

- **`notebooks/`** — an end-to-end Databricks project on 700k real Yelp reviews.
  Ingestion → transformation → ML → serving → agents, covering every item in the
  Databricks sidebar.
- **`pyspark-tutorial/`** — ten standalone PySpark topics, each with its own
  real-world scenario, runnable code, expected output in comments, and an
  explanation of *why* each command behaves the way it does.

Every file runs. Nothing is a snippet.

---

## `pyspark-tutorial/` — ten topics, ten scenarios

| File | Scenario | Core concepts |
|---|---|---|
| `01_rdd.py` | Parsing nginx access logs with malformed lines | When RDDs are still right, dead-letter routing, `reduceByKey` vs `groupByKey`, lineage |
| `02_dataframe_basics.py` | Cleaning a messy e-commerce order feed | Explicit schemas, null auditing, three-valued logic, the `withColumn`-in-a-loop trap |
| `03_data_sources.py` | IoT telemetry across CSV, JSON and Parquet | Corrupt-record capture, nested JSON, partition pruning vs pushdown, small files, JDBC parallelism |
| `04_sql_functions.py` | Standardising a CRM export | Strings, regex, multi-format dates, null semantics, higher-order functions |
| `05_joins.py` | Retail orders × customers × products | Join types, semi/anti joins, broadcast vs sort-merge, many-to-many explosion, salting |
| `06_aggregations.py` | Monthly sales reporting | Map-side combine, `countDistinct` cost, percentiles, pivot, rollup vs cube |
| `07_window_functions.py` | Banking transaction ledger | Ranking, running totals, `lag`/`lead`, frames, the `last()` trap, sessionisation |
| `08_udfs.py` | Insurance claim risk scoring | Built-in vs Pandas vs Python UDF benchmark, null crashes, closures, broadcast |
| `09_streaming.py` | Payment fraud velocity detection | Event time, watermarks, output modes, checkpoints, Kafka, `foreachBatch` |
| `10_mllib.py` | Telecom churn prediction | Leak-free splits, pipelines, feature importance, threshold tuning, calibration |

Run any of them standalone:

```bash
pip install pyspark==3.5.1
python pyspark-tutorial/05_joins.py
```

They generate their own data — no downloads, no external dependencies.

### Why the scenarios matter

Each file solves a problem someone actually has, so the concepts land in the
order you would meet them. `05_joins.py` doesn't list join types abstractly; it
finds orphaned orders with an anti-join, then blows up a join with duplicate
keys, then fixes a 400× skew by salting. The failure comes first, the technique
second.

---

## `notebooks/` — the Databricks project

700,000 real Yelp reviews
([`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full),
Parquet, direct download, no licence click-through).

| File | Sidebar items | Runtime |
|---|---|---|
| `01_ingest.py` | Data Ingestion | ~5 min |
| `02_transform.py` | (Spark internals) | ~10 min |
| `03_ml.py` | Features, Experiments | ~15 min |
| `04_serving.py` | Models, Serving, Runs | ~10 min |
| `05_agent.py` | Playground, AI Gateway, Agents | ~20 min |

```
01  HF Parquet ──Auto Loader──▶ yelp_bronze          700k rows
02  yelp_bronze ──transform───▶ yelp_silver ──▶ yelp_gold_business
03  yelp_silver ──features────▶ yelp_business_features  (PK: business_id)
                ──MLflow──────▶ 5+ tracked runs, best ~0.94 AUC
04  best run ────register─────▶ UC registry @champion ──▶ serving endpoint
05  gold tables ──UC funcs────▶ agent tools ──▶ tool-calling LLM agent
```

These are Databricks notebook source files — `# COMMAND ----------` markers
become cell boundaries. Import via **Workspace → right-click → Import → File**.

See [`notebooks/README.md`](notebooks/README.md) for full details, including the
synthetic-dimension caveat and the negative control that proves it.

---

## Things in here that are easy to get wrong

A partial list of the traps each file demonstrates rather than merely mentions:

- **Three-valued logic.** `filter(~RULES)` does *not* capture rows where the
  predicate is NULL. `valid + invalid < total` and nothing errors. (`02`)
- **`union` vs `unionByName`.** Position matching silently swaps your columns
  when schemas differ in order. (`03`)
- **`repartition(col)` before `partitionBy(col)`.** Without it you write
  `n_tasks × n_partitions` tiny files. (`03`)
- **Column-to-column predicates never push down.** Row-group stats compare a
  column to a *literal*. (`03`)
- **`concat` vs `concat_ws`.** One propagates NULL, one skips it. (`04`)
- **`element_at` is 1-indexed; `getItem` and `vector_to_array` are 0-indexed.**
  Same API, two conventions. (`04`, `10`)
- **Many-to-many join explosion.** Duplicate keys on both sides multiply rows.
  The #1 cause of a "should be fast" join running for hours. (`05`)
- **When salting, replicate the small side.** Salting both sides randomly
  silently loses rows. This is the step people get wrong in interviews. (`05`)
- **`rank()` vs `row_number()`.** `filter(rank <= 3)` can return 50 rows. (`07`)
- **`last()` needs an explicit unbounded frame** or it returns the current row.
  `first()` works with the default, which makes the asymmetry worse. (`07`)
- **No `partitionBy` on a window** = `Exchange SinglePartition` = one task for
  the entire dataset. Spark literally logs a warning. (`07`)
- **A Python exception in a UDF kills the job**, not the row — four task
  retries, then the stage fails. (`08`)
- **Catalyst will not push a filter through a UDF.** Order it yourself. (`08`)
- **Sliding windows multiply streaming state** by `windowSize / slideSize`. (`09`)
- **`startingOffsets` applies only on the first run.** After that the
  checkpoint wins. (`09`)
- **Split before fitting any transformer.** Imputer, StringIndexer and
  StandardScaler all leak test data otherwise. (`10`)
- **0.5 is an arbitrary threshold**, not an optimum. Derive it from cost
  asymmetry. (`10`)

---

## Databricks serverless notes

Free Edition runs serverless, which blocks a few things the docs assume:

| Blocked | Use instead |
|---|---|
| `df.rdd.getNumPartitions()`, `spark.sparkContext` | `F.spark_partition_id()` — a SQL function, and a better diagnostic |
| `spark.conf.set("spark.sql.autoBroadcastJoinThreshold", …)` | Join hints: `df.join(dim.hint("merge"), "k")` |
| AQE config changes | AQE is always on and self-tuned; look for `AQEShuffleRead` in the plan |
| `spark.conf.set("fs.s3a.*")` | Unity Catalog storage credential, or boto3 into a Volume |
| Classic Spark UI | **View → Query profile** on a cell — arguably better for reading plans |

Also: ANSI mode is on by default, so integer overflow **raises** instead of
silently wrapping. That is an improvement. Setting
`spark.sql.ansi.enabled=false` to "fix" an overflow restores the silent
corruption — fix the cast instead.

---

## The habit worth taking away

After every change, run `explain("formatted")` and diff the plan against the
previous one. Not the timing — the *plan*.

Timing tells you something changed. The plan tells you what and why. Count the
`Exchange` nodes; each one is a shuffle, a stage boundary, and a place your job
can die.

---

## Setup

```bash
pip install pyspark==3.5.1 pandas pyarrow
python pyspark-tutorial/01_rdd.py
```
