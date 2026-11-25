# Yelp Lab — Databricks end-to-end

One dataset, one connected pipeline, every Databricks sidebar item.

## Dataset

**`Yelp/yelp_review_full`** — 650,000 real Yelp reviews (train) + 50,000 (test),
Parquet, ~300 MB, direct HTTPS download, no login and no licence click-through.

```
https://huggingface.co/datasets/Yelp/yelp_review_full/resolve/main/yelp_review_full/train-00000-of-00001.parquet
https://huggingface.co/datasets/Yelp/yelp_review_full/resolve/main/yelp_review_full/test-00000-of-00001.parquet
```

Schema is `label` (0–4, meaning 1–5 stars) and `text`.

> **Why not the official Yelp Open Dataset?** It sits behind a click-through
> licence agreement and a signed URL, so it cannot be scripted. This is the same
> review data, redistributed for research.

> **On the dimensional layer.** The public dataset has no business or user
> metadata, so `02_transform.py` derives `business_id` / `user_id` / `review_date`
> deterministically from a hash. Those columns are **synthetic and labelled as
> such in the code**. The hash is shaped to produce a heavy-tailed popularity
> distribution, which makes the skew lesson genuine and — unlike real data —
> exactly reproducible. `03_ml.py` trains on the **real text**, not on these keys,
> and includes a negative control proving the keys carry zero signal.

## Files

| File | Sidebar items covered | Runtime |
|---|---|---|
| `01_ingest.py` | Data Ingestion | ~5 min |
| `02_transform.py` | (Spark internals) | ~10 min |
| `03_ml.py` | Features, Experiments | ~15 min |
| `04_serving.py` | Models, Serving, Runs | ~10 min + endpoint wait |
| `05_agent.py` | Playground, AI Gateway, Agents | ~20 min |

Run in order. Each file's output is the next file's input.

## Import into Databricks

These are Databricks notebook source files — the `# COMMAND ----------` markers
become cell boundaries and `# MAGIC %md` / `# MAGIC %sql` become markdown and
SQL cells.

Workspace → right-click a folder → **Import** → **File** → drag all five in.

Pasting the raw text into one cell also works but loses the cell splits and the
SQL cells will not run.

## Pipeline

```
01  HF Parquet ──Auto Loader──▶ yelp_bronze          700k rows
02  yelp_bronze ──transform───▶ yelp_silver          ~700k, 17 cols
                              ▶ yelp_gold_business   2,000
                              ▶ yelp_gold_daily      ~360, partitioned by year
03  yelp_silver ──features────▶ yelp_business_features  (PK: business_id)
                ──MLflow──────▶ 5+ tracked runs, best ~0.94 AUC
04  best run ────register─────▶ workspace.default.yelp_sentiment @champion
                              ▶ serving endpoint (tier permitting)
                              ▶ yelp_scored, yelp_prediction_monitor
05  gold tables ──UC funcs────▶ agent tools ──▶ tool-calling LLM agent
```

## Concepts, by file

**01 — Ingestion**
Parquet footer schema and metadata counts (no data read) · Auto Loader
incremental discovery via checkpoint · `trigger(availableNow=True)` — streaming
semantics with batch execution · schema evolution modes · `_metadata.file_path`
lineage · `sum(when(...))` as the conditional-count idiom · Delta schema
enforcement · `DESCRIBE HISTORY`

**02 — Transform**
Predicate pushdown: literals push, column-to-column does not · column pruning
via `ReadSchema` · partial + final `HashAggregate` around an `Exchange` ·
`approx_count_distinct` and why mergeability decides cost · broadcast vs
sort-merge join, forced both ways and timed · skew detection by percentile and
`spark_partition_id()` · AQE skew join · salting, including the
replicate-the-small-side rule people get wrong · window ranking, partition-key
reuse, `rowsBetween` vs `rangeBetween` · `repartition` before `partitionBy` ·
`OPTIMIZE` / `ZORDER` / `VACUUM`

**03 — ML**
Seeded and hash-based splits · Spark ML Pipeline as a lazy transformer graph ·
`HashingTF` vs `CountVectorizer`; `IDF` as an estimator · why linear models beat
trees on sparse high-dimensional text · four metrics and when each misleads ·
`CrossValidator` cost arithmetic · **negative control** to detect target
leakage · feature table with a primary key · training/serving skew ·
error-slice analysis over headline metrics

**04 — Serving**
Unity Catalog registry vs legacy workspace registry · immutable versions,
aliases as movable pointers, champion/challenger · rollback as an alias
repoint · `scale_to_zero` and the idle-endpoint cost trap · serving payload
formats · batch vs real-time as a decision, not a default · `element_at` is
1-indexed and `vector_to_array` is 0-indexed · label-free drift signals ·
job dependency graphs, `max_concurrent_runs`, Quartz six-field cron

**05 — Agents**
Low temperature for analytical tasks · LLM vs trained classifier — ~1,000×
latency gap and what each is actually for · LLMs replace the *labelling effort*,
not the classifier · UC function `COMMENT`s **are** the tool spec · multi-step
agent loops where step 2 depends on step 1 · `max_iterations` as a cost
guardrail · inference tables as debugging, audit and training data · SQL
injection risk in hand-rolled tools

## Free Edition notes

Blocked on serverless — the code works around all of these:

- `df.rdd.getNumPartitions()` and `spark.sparkContext` → use
  `F.spark_partition_id()` instead (better diagnostic anyway)
- `spark.conf.set("fs.s3a.*")` → S3 needs a Unity Catalog storage credential;
  use boto3 into a Volume as a fallback
- Most `spark.conf.set()` performance sweeps are ignored — serverless manages
  shuffle partitions, AQE and memory itself
- No classic Spark UI → **View → Query profile** on a cell gives the physical
  plan as a DAG with per-operator row counts and timings

May be unavailable by tier — each section states the expected error and has a
fallback:

- **Model Serving** (`04` §4) → batch scoring in §5 needs no endpoint
- **AI Gateway** (`05` §5) → concepts still apply
- **Agent frameworks** (`05` §6) → a 20-line hand-rolled ReAct loop follows it

## Expected results

Every cell has its expected output in a comment beneath it. Headline numbers:

| Checkpoint | Expect |
|---|---|
| Bronze rows | 700,000 |
| Bronze re-run | 700,000 — unchanged, proving incrementality |
| Silver retention | ~99.99% |
| Business keys | 2,000, max/p50 ratio ~13× |
| Broadcast vs sort-merge | 2–5× speedup |
| Best model AUC | ~0.94 |
| Negative control AUC | ~0.50 — confirms no leakage |
| 3-star slice accuracy | ~0.62 — where all the error lives |

## Two serverless errors you would otherwise hit, already fixed

**`[ARITHMETIC_OVERFLOW] ... SQLSTATE: 22003`** — `02` §1

`F.hash()` returns **int32**, and squaring a value near 1e6 reaches ~1e12.
Under Spark 3.4+ ANSI mode (default on Databricks) that raises rather than
silently wrapping. Fix: cast to `long` **before** the multiply, not after —
casting the product is too late, the overflow already happened.

The general lesson: ANSI mode turns silent integer wraparound into a loud
error. Setting `spark.sql.ansi.enabled=false` to "fix" it restores the silent
corruption.

**`[CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION]`** — `02` §5 and §6

Serverless gates most `spark.conf.set` calls, including
`spark.sql.autoBroadcastJoinThreshold` and the AQE settings. It manages join
planning, AQE and memory itself.

Fix: **join hints**, which work everywhere and are better practice anyway
because they scope to one join instead of mutating global state.

```python
df.join(dim.hint("merge"), "key")          # force SortMergeJoin
df.join(F.broadcast(dim), "key")           # force BroadcastHashJoin
df.join(dim.hint("shuffle_hash"), "key")   # force ShuffledHashJoin
```

AQE cannot be disabled on serverless, so the "watch skew hurt, then heal"
experiment needs a local `pip install pyspark` cluster. The mechanism is
documented in the file either way — it is a common interview question, and you
can still confirm AQE fired by finding `AQEShuffleRead` in the plan.

## Known gotcha, already fixed

`02_transform.py` §1 squares a hash to build a heavy-tailed distribution.
`F.hash()` returns **int32**, and the square reaches ~1e12. Under Spark 3.4+
ANSI mode (default on Databricks) that raises:

```
[ARITHMETIC_OVERFLOW] overflow. SQLSTATE: 22003
```

The fix is casting to `long` **before** the multiply, not after — the file does
this. Casting the product afterwards is too late; the overflow already happened.

The general lesson: ANSI mode turns silent integer wraparound into a loud error.
That is an improvement. Setting `spark.sql.ansi.enabled=false` to "fix" it
restores the silent corruption.
