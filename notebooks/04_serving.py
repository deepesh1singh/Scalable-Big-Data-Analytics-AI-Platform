# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Models + Serving: register, version, deploy
# MAGIC
# MAGIC **Covers sidebar:** Models (Unity Catalog registry), Serving (REST endpoint),
# MAGIC Runs (Workflows / job orchestration).
# MAGIC
# MAGIC **Prereq:** `03_ml.py` — you need at least one logged MLflow run.
# MAGIC
# MAGIC **Runtime:** ~10 minutes, plus up to 15 minutes waiting for the endpoint.
# MAGIC
# MAGIC > **Free Edition caveat.** Model Serving may be capped or unavailable.
# MAGIC > Section 4 will tell you definitively. Sections 1-3 and 6-7 work regardless,
# MAGIC > and Section 5 gives you a batch-scoring path that needs no endpoint at all.

# COMMAND ----------

import mlflow, time, json
from mlflow import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")     # register into Unity Catalog, not the workspace registry
client = MlflowClient()

CATALOG, SCHEMA = "workspace", "default"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.yelp_sentiment"

# WHY set_registry_uri("databricks-uc"):
#   Without it, models land in the legacy WORKSPACE model registry, which uses
#   stages (None/Staging/Production) and has no cross-workspace governance.
#   Unity Catalog models use three-level names (catalog.schema.model) and
#   ALIASES instead of stages. UC is the current model; the workspace registry
#   is legacy. Setting this one line decides which system you are using.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1 — Find the best run and register it

# COMMAND ----------

runs = mlflow.search_runs(
    order_by=["metrics.auc DESC"],
    filter_string="attributes.status = 'FINISHED'",
    max_results=5)

display(runs[["run_id", "tags.mlflow.runName", "metrics.auc", "metrics.f1"]])

best_run_id = runs.iloc[0]["run_id"]
best_auc    = runs.iloc[0]["metrics.auc"]
print(f"registering run {best_run_id} (AUC {best_auc:.4f})")

# EXPECTED OUTPUT: your runs from 03_ml.py ranked by AUC, e.g.
#   run_id     | runName          | auc    | f1
#   -----------+------------------+--------+-------
#   a1b2c3...  | logreg-cv        | 0.9391 | 0.8672
#   d4e5f6...  | logreg-lowreg    | 0.9388 | 0.8669
#   ...
#   9z8y7x...  | synthetic-keys.. | 0.5012 | 0.4890   <-- the negative control
#
# search_runs returns a PANDAS DataFrame, not a Spark one. It is metadata about
# runs, which is small by definition — no need for distribution.

# COMMAND ----------

result = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=MODEL_NAME)

print(f"registered {MODEL_NAME} version {result.version}")

# EXPECTED OUTPUT:
#   registered workspace.default.yelp_sentiment version 1
#
# Re-running this cell creates version 2, 3, ... Versions are IMMUTABLE and
# never reused. That immutability is the whole point: "version 3" means one
# exact set of weights forever, so a production incident can always be traced
# to a specific artefact.
#
# Open the **Models** tab in the sidebar — the model appears with its version
# list, source run, and lineage back to the notebook and the tables it read.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2 — Aliases, descriptions, tags
# MAGIC
# MAGIC Unity Catalog replaced stages with **aliases**. An alias is a movable
# MAGIC pointer to a version. This matters more than it sounds: your serving
# MAGIC endpoint targets `@champion`, so promoting a new model is one API call
# MAGIC that repoints the alias — no redeploy, and instant rollback.

# COMMAND ----------

v = result.version

client.set_registered_model_alias(MODEL_NAME, "champion", v)

client.update_model_version(
    name=MODEL_NAME, version=v,
    description=(f"LogisticRegression + TF-IDF on 700k Yelp reviews. "
                 f"Binary: is_positive (4-5 stars). Test AUC {best_auc:.4f}. "
                 f"Source run {best_run_id}."))

client.set_model_version_tag(MODEL_NAME, v, "dataset", "yelp_review_full")
client.set_model_version_tag(MODEL_NAME, v, "task", "binary-sentiment")
client.set_model_version_tag(MODEL_NAME, v, "validated", "true")

mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
print(f"@champion -> version {mv.version}")

# EXPECTED OUTPUT:
#   @champion -> version 1
#
# THE PROMOTION PATTERN:
#   new model  -> alias @challenger  -> shadow traffic / offline eval
#   if better  -> move @champion to that version
#   if worse   -> delete the @challenger alias, nothing else changes
#
# ROLLBACK is repointing @champion at the previous version. Seconds, not a
# redeploy. This is the single most useful operational property of the
# registry and worth knowing for interviews.
#
# DESCRIPTIONS AND TAGS ARE NOT DECORATION. Six months from now the question
# "which data trained version 3, and was it validated?" has to be answerable
# without archaeology.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 — Load by alias and verify

# COMMAND ----------

model = mlflow.spark.load_model(f"models:/{MODEL_NAME}@champion")

samples = spark.createDataFrame([
    ("r1", "Absolutely fantastic. Best meal I have had all year, staff were lovely.",  70, 13, 1, 0.02),
    ("r2", "Terrible. Cold food, rude server, waited 45 minutes. Never coming back.",   71, 13, 0, 0.02),
    ("r3", "It was fine. Nothing special but nothing wrong either.",                    54,  9, 0, 0.02),
    ("r4", "AMAZING!!! GO HERE NOW!!!",                                                 25,  5, 1, 0.60),
], "review_id string, text string, text_len int, word_count int, has_exclaim int, caps_ratio double")

preds = model.transform(samples)
display(preds.select("review_id", "text", "prediction", "probability"))

# EXPECTED OUTPUT:
#   review_id | prediction | probability
#   ----------+------------+---------------------
#   r1        |    1.0     | [0.05, 0.95]     positive, confident
#   r2        |    0.0     | [0.96, 0.04]     negative, confident
#   r3        |    0.0     | [0.62, 0.38]     ambiguous — near the boundary
#   r4        |    1.0     | [0.15, 0.85]     positive
#
# NOTE r3. "It was fine" is genuinely 3-star territory, and the model's low
# confidence is CORRECT behaviour, not a failure. A model that is confident on
# ambiguous input is worse than one that hedges.
#
# THE INPUT SCHEMA MUST MATCH TRAINING EXACTLY. The PipelineModel expects
# text, text_len, word_count, has_exclaim, caps_ratio because VectorAssembler
# was configured with those columns. Miss one and you get a schema error at
# transform time. This is precisely the training/serving skew that a feature
# store exists to prevent — here we are doing it by hand.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4 — Model Serving endpoint
# MAGIC
# MAGIC This is the **Serving** item in the sidebar. It stands up an
# MAGIC autoscaling REST endpoint in front of a registered model version.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput)

w = WorkspaceClient()
ENDPOINT = "yelp-sentiment"

try:
    w.serving_endpoints.create(
        name=ENDPOINT,
        config=EndpointCoreConfigInput(
            served_entities=[ServedEntityInput(
                entity_name=MODEL_NAME,
                entity_version=v,
                workload_size="Small",
                scale_to_zero_enabled=True,      # <-- cost control
            )]))
    print(f"creating endpoint '{ENDPOINT}' — takes 5-15 minutes")
except Exception as e:
    print("could not create endpoint:")
    print(" ", str(e)[:400])

# EXPECTED OUTCOMES — read carefully, this is where Free Edition may stop you:
#
#   SUCCESS ->
#       creating endpoint 'yelp-sentiment' — takes 5-15 minutes
#
#   ALREADY EXISTS -> harmless, endpoint is there from a previous run
#
#   PERMISSION_DENIED / FEATURE_DISABLED / QUOTA_EXCEEDED ->
#       Model Serving is not available on your tier. THIS IS EXPECTED ON
#       FREE EDITION. Skip to Section 5 — batch scoring gives you the same
#       predictions without an endpoint.
#
# scale_to_zero_enabled=True is the setting that matters for cost. The
# endpoint shuts down when idle and cold-starts on the next request (adding a
# few seconds of latency). Without it you pay continuously for an endpoint
# nobody is calling — a classic surprise bill.
#
# workload_size "Small" is the smallest option. Sizes map to concurrency,
# not model size.

# COMMAND ----------

# Poll until ready (safe to interrupt)
for _ in range(30):
    try:
        st = w.serving_endpoints.get(ENDPOINT)
        state = st.state.ready if st.state else "UNKNOWN"
        print(f"{time.strftime('%H:%M:%S')}  state={state}")
        if str(state).endswith("READY"):
            break
    except Exception as e:
        print("not available:", str(e)[:120]); break
    time.sleep(30)

# COMMAND ----------

# Query the endpoint
try:
    payload = {"dataframe_records": [
        {"review_id": "x1", "text": "Best pizza in the entire city, went back twice.",
         "text_len": 47, "word_count": 9, "has_exclaim": 0, "caps_ratio": 0.02},
        {"review_id": "x2", "text": "Overpriced and bland. Very disappointing.",
         "text_len": 41, "word_count": 6, "has_exclaim": 0, "caps_ratio": 0.02},
    ]}
    resp = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=payload["dataframe_records"])
    print(json.dumps(resp.as_dict(), indent=2)[:800])
except Exception as e:
    print("query failed:", str(e)[:300])

# EXPECTED OUTPUT on success:
#   {"predictions": [1.0, 0.0]}
#
# The equivalent raw HTTP call, which is what an application would issue:
#
#   curl -X POST \
#     -H "Authorization: Bearer $DATABRICKS_TOKEN" \
#     -H "Content-Type: application/json" \
#     -d '{"dataframe_records":[{"text":"great food","text_len":10,
#          "word_count":2,"has_exclaim":0,"caps_ratio":0.0}]}' \
#     https://<workspace>.cloud.databricks.com/serving-endpoints/yelp-sentiment/invocations
#
# ACCEPTED PAYLOAD FORMATS — getting this wrong is the most common serving error:
#   dataframe_records   list of row dicts        <- most readable, use this
#   dataframe_split     {columns:[], data:[[]]}  <- compact for many rows
#   instances           TensorFlow-serving style
#   inputs              named tensors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5 — Batch scoring (works on every tier, and is often correct anyway)

# COMMAND ----------

SCORED = f"{CATALOG}.{SCHEMA}.yelp_scored"

to_score = (spark.table(f"{CATALOG}.{SCHEMA}.yelp_silver")
    .select("review_id", "business_id", "text", "text_len",
            "word_count", "has_exclaim", "caps_ratio", "stars", "is_positive")
    .limit(50000))

scored = (model.transform(to_score)
    .withColumn("p_positive", F.round(
        F.element_at(F.col("probability").cast("array<double>"), 2), 4))
    .select("review_id", "business_id", "stars", "is_positive",
            "prediction", "p_positive"))

scored.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SCORED)

display(spark.table(SCORED).limit(10))

# NOTE on extracting the probability:
#   probability is a Spark ML VectorUDT, not an array. Casting to
#   array<double> then element_at(..., 2) pulls P(class=1). element_at is
#   1-INDEXED in Spark SQL, unlike Python. Off-by-one here silently gives you
#   the NEGATIVE class probability, and everything downstream inverts.
#   Cleaner alternative:
#       from pyspark.ml.functions import vector_to_array
#       .withColumn("p_positive", vector_to_array("probability")[1])
#   (that indexing IS 0-based — read carefully.)
#
# BATCH VS REAL-TIME — the decision, not the default:
#   Batch wins when predictions are consumed on a schedule (dashboards, nightly
#   segmentation, alerting). It is cheaper, simpler, has no cold start, and
#   scales with Spark. Real-time is only required when a user is waiting for
#   the answer. Many teams stand up an endpoint they did not need.

# COMMAND ----------

display(
    spark.table(SCORED)
        .withColumn("correct", (F.col("prediction") == F.col("is_positive")).cast("int"))
        .groupBy("stars")
        .agg(F.count("*").alias("n"),
             F.round(F.avg("correct"), 3).alias("accuracy"),
             F.round(F.avg("p_positive"), 3).alias("avg_p_positive"))
        .orderBy("stars"))

# EXPECTED OUTPUT:
#   stars | n     | accuracy | avg_p_positive
#   ------+-------+----------+---------------
#     1   | 10000 |  0.95    |     0.06
#     2   | 10000 |  0.88    |     0.16
#     3   | 10000 |  0.62    |     0.39     <-- worst, as expected
#     4   | 10000 |  0.87    |     0.79
#     5   | 10000 |  0.95    |     0.92
#
# Accuracy per slice, not just overall. 3-star reviews are the genuinely
# ambiguous class and carry nearly all the error. If you needed to improve
# this model, that row tells you exactly where to spend effort — and it
# suggests the real problem may be the binary framing rather than the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6 — Monitoring: drift detection you can actually run

# COMMAND ----------

MONITOR = f"{CATALOG}.{SCHEMA}.yelp_prediction_monitor"

(spark.table(SCORED)
    .agg(
        F.current_timestamp().alias("scored_at"),
        F.lit(MODEL_NAME).alias("model"),
        F.lit(v).alias("model_version"),
        F.count("*").alias("n_scored"),
        F.round(F.avg("prediction"), 4).alias("positive_rate"),
        F.round(F.avg("p_positive"), 4).alias("avg_confidence"),
        F.round(F.stddev("p_positive"), 4).alias("confidence_spread"),
        F.round(F.avg((F.col("prediction") == F.col("is_positive")).cast("int")), 4)
         .alias("accuracy"),
    )
    .write.mode("append").option("mergeSchema", "true").saveAsTable(MONITOR))

display(spark.table(MONITOR))

# WHAT TO ALERT ON, in order of how early it warns you:
#
#   positive_rate       shifts before accuracy does, and needs NO LABELS.
#                       If it moves from 0.40 to 0.65 with no product change,
#                       either the input distribution shifted or something
#                       upstream broke. This is your earliest signal.
#
#   avg_confidence      falling means the model is seeing unfamiliar input.
#
#   confidence_spread   collapsing toward 0 means predictions bunching at the
#                       threshold — often a broken feature pipeline sending
#                       constants.
#
#   accuracy            the metric you care about, but it REQUIRES GROUND
#                       TRUTH, which in production arrives days or weeks late
#                       or never. That delay is exactly why the label-free
#                       signals above matter.
#
# Appending one row per scoring run makes this a time series you can chart.
# Databricks Lakehouse Monitoring automates this on paid tiers; the manual
# version costs six lines and teaches you what the automation is doing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7 — Runs: turn the notebooks into a scheduled Workflow
# MAGIC
# MAGIC This is the **Runs** item in the sidebar.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Via the UI
# MAGIC 1. Sidebar → **Runs** (Workflows) → **Create job**
# MAGIC 2. Task 1 — `ingest`: notebook `01_ingest`
# MAGIC 3. Task 2 — `transform`: notebook `02_transform`, **depends on** `ingest`
# MAGIC 4. Task 3 — `train`: notebook `03_ml`, **depends on** `transform`
# MAGIC 5. Task 4 — `serve`: notebook `04_serving`, **depends on** `train`
# MAGIC 6. Schedule: daily, or leave manual
# MAGIC 7. Add an email notification **on failure**
# MAGIC
# MAGIC The dependency graph is the point. Task 3 cannot start before task 2
# MAGIC succeeds, and a failure stops the chain rather than training on stale data.

# COMMAND ----------

# Same thing as code — this is what the UI generates
job_spec = {
    "name": "yelp-lab-pipeline",
    "tasks": [
        {"task_key": "ingest",
         "notebook_task": {"notebook_path": "/Workspace/Users/<you>/yelp-lab/01_ingest"}},
        {"task_key": "transform", "depends_on": [{"task_key": "ingest"}],
         "notebook_task": {"notebook_path": "/Workspace/Users/<you>/yelp-lab/02_transform"}},
        {"task_key": "train", "depends_on": [{"task_key": "transform"}],
         "notebook_task": {"notebook_path": "/Workspace/Users/<you>/yelp-lab/03_ml"}},
        {"task_key": "serve", "depends_on": [{"task_key": "train"}],
         "notebook_task": {"notebook_path": "/Workspace/Users/<you>/yelp-lab/04_serving"}},
    ],
    "schedule": {"quartz_cron_expression": "0 0 6 * * ?",
                 "timezone_id": "America/New_York"},
    "email_notifications": {"on_failure": ["you@example.com"]},
    "max_concurrent_runs": 1,
}
print(json.dumps(job_spec, indent=2))

# max_concurrent_runs=1 prevents a slow run overlapping the next schedule —
# two concurrent runs writing the same tables is a real corruption risk, and
# it is the default failure mode people hit when a job gradually slows down.
#
# The cron has SIX fields (Quartz), not five. "0 0 6 * * ?" is 06:00 daily.
# Using standard five-field cron here silently fails to parse.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (run when done — endpoints cost money)

# COMMAND ----------

# w.serving_endpoints.delete(ENDPOINT)
# client.delete_registered_model_alias(MODEL_NAME, "champion")
# spark.sql(f"DROP TABLE IF EXISTS {SCORED}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What you built
# MAGIC
# MAGIC | Artefact | Sidebar location |
# MAGIC |---|---|
# MAGIC | `workspace.default.yelp_sentiment` v1 `@champion` | **Models** |
# MAGIC | `yelp-sentiment` REST endpoint (tier permitting) | **Serving** |
# MAGIC | `yelp_scored` — 50k batch predictions | Catalog |
# MAGIC | `yelp_prediction_monitor` — drift time series | Catalog |
# MAGIC | 4-task job spec with dependencies | **Runs** |
# MAGIC
# MAGIC ### Concepts demonstrated
# MAGIC - UC registry vs legacy workspace registry
# MAGIC - Immutable versions; aliases as movable pointers; champion/challenger
# MAGIC - Rollback as an alias repoint, not a redeploy
# MAGIC - `scale_to_zero` and the idle-endpoint cost trap
# MAGIC - Serving payload formats
# MAGIC - Batch vs real-time as a decision
# MAGIC - `element_at` is 1-indexed; `vector_to_array` is 0-indexed
# MAGIC - Label-free drift signals and why they matter more than accuracy
# MAGIC - Job dependency graphs, `max_concurrent_runs`, Quartz six-field cron
# MAGIC
# MAGIC **Next:** `05_agent.py` — Playground, AI Gateway, and an agent over these tables.
