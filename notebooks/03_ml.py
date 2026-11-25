# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Features + Experiments: predict star rating from review text
# MAGIC
# MAGIC **Covers sidebar:** Features (Feature Store), Experiments (MLflow).
# MAGIC
# MAGIC **Prereq:** `01_ingest.py`, `02_transform.py`.
# MAGIC
# MAGIC **Task:** given review text, predict whether the review is positive
# MAGIC (4-5 stars) — a binary classification problem on 700k real reviews.
# MAGIC
# MAGIC **Runtime:** ~15 minutes.
# MAGIC
# MAGIC > We predict from **text**, not from the synthetic business/user keys.
# MAGIC > Those keys were generated independently of the label, so they carry
# MAGIC > exactly zero signal — a model built on them would score ~0.50 AUC.
# MAGIC > Confirming that is itself a useful exercise (Section 6).

# COMMAND ----------

import mlflow, time
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (Tokenizer, StopWordsRemover, HashingTF, IDF,
                                CountVectorizer, VectorAssembler, StandardScaler)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

CATALOG, SCHEMA = "workspace", "default"
SILVER = f"{CATALOG}.{SCHEMA}.yelp_silver"

silver = spark.table(SILVER)
print("rows:", silver.count())

# EXPECTED OUTPUT:
#   rows: 699xxx

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1 — Train/test split, done correctly

# COMMAND ----------

data = silver.select("review_id", "business_id", "text", "text_len",
                     "word_count", "has_exclaim", "caps_ratio",
                     "stars", "is_positive")

train, test = data.randomSplit([0.8, 0.2], seed=42)
train.cache(); test.cache()
print(f"train {train.count():,}   test {test.count():,}")

# EXPECTED OUTPUT:
#   train 559,xxx   test 139,xxx
#
# WHY seed=42 IS NOT OPTIONAL:
#   randomSplit without a seed produces a DIFFERENT split every run. Your
#   "improvement" between two experiments then reflects a different test set,
#   not a better model. Always seed splits.
#
# A SUBTLER TRAP: randomSplit is only deterministic if the UPSTREAM DataFrame
# is deterministic. If `data` involves an unseeded rand(), a non-deterministic
# dropDuplicates, or an unordered file listing, the split shifts anyway. The
# robust fix is a hash-based split:
#     .withColumn("bucket", F.pmod(F.hash("review_id"), F.lit(10)))
#     train = df.filter("bucket < 8"); test = df.filter("bucket >= 8")
# That is stable forever, regardless of upstream changes.
#
# WHY .cache():
#   Both DataFrames are read many times below — once per model, once per
#   evaluation. Without caching, the entire silver->select->split lineage
#   replays every time. This is the single highest-value cache in the file.

# COMMAND ----------

display(train.groupBy("is_positive").count())

# EXPECTED OUTPUT:
#   is_positive | count
#   ------------+---------
#        0      | ~335,000     (1-3 stars)
#        1      | ~224,000     (4-5 stars)
#
# ~60/40. Mildly imbalanced, not enough to need resampling. Worth CHECKING
# rather than assuming — if it were 99/1, accuracy would be a useless metric
# and you would need AUPRC plus class weights.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2 — The feature pipeline
# MAGIC
# MAGIC Spark ML pipelines are **lazy transformer graphs**, exactly like DataFrames.
# MAGIC `Pipeline.fit()` returns a `PipelineModel` that carries every fitted stage,
# MAGIC so the identical transformation runs at training and at serving time.
# MAGIC That property is what prevents training/serving skew.

# COMMAND ----------

tokenizer = Tokenizer(inputCol="text", outputCol="words_raw")

remover = StopWordsRemover(inputCol="words_raw", outputCol="words")

hashing = HashingTF(inputCol="words", outputCol="tf", numFeatures=1 << 16)

idf = IDF(inputCol="tf", outputCol="tfidf", minDocFreq=5)

assembler = VectorAssembler(
    inputCols=["tfidf", "text_len", "word_count", "has_exclaim", "caps_ratio"],
    outputCol="features")

# STAGE BY STAGE:
#
#   Tokenizer          lowercases and splits on whitespace. RegexTokenizer gives
#                      you control over the pattern; plain Tokenizer keeps
#                      punctuation attached to words ("great!" != "great"),
#                      which costs you a little accuracy. Try swapping it.
#
#   StopWordsRemover   drops "the", "a", "is"... Default list is English.
#                      CAUTION for sentiment: the default list includes "not",
#                      "no", "very" — removing negation can genuinely hurt a
#                      sentiment model. Inspect with remover.getStopWords()
#                      and consider remover.setStopWords([w for w in ... ]).
#
#   HashingTF          maps each token to one of 2^16 buckets via hashing.
#                      FAST and requires no vocabulary pass — but COLLISIONS are
#                      possible and features are not interpretable. The
#                      alternative, CountVectorizer, builds a real vocabulary:
#                      slower (an extra pass) but you can name each feature.
#                      2^16 = 65,536 buckets is a sensible default for this size.
#
#   IDF                down-weights tokens common across all documents.
#                      minDocFreq=5 discards terms in fewer than 5 documents,
#                      which removes typos and cuts the effective dimension.
#                      NOTE IDF is an ESTIMATOR — it must .fit() to learn the
#                      document frequencies, unlike HashingTF which is a pure
#                      transformer.
#
#   VectorAssembler    concatenates the sparse tfidf vector with the dense
#                      numeric features into one column. Spark ML requires a
#                      SINGLE vector column named by featuresCol; you cannot
#                      pass a list of columns to an estimator.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 — MLflow experiment: baseline
# MAGIC
# MAGIC This is the **Experiments** item in the sidebar. Every run is captured with
# MAGIC its parameters, metrics, and the model artifact, so runs are comparable
# MAGIC in the UI rather than in your memory.

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/yelp-sentiment")

def run_experiment(name, classifier, extra_params=None):
    """Train, evaluate and log one model. Returns (run_id, auc, f1)."""
    with mlflow.start_run(run_name=name) as run:
        pipe = Pipeline(stages=[tokenizer, remover, hashing, idf, assembler, classifier])

        t0 = time.time()
        model = pipe.fit(train)
        train_secs = time.time() - t0

        preds = model.transform(test)

        auc = BinaryClassificationEvaluator(
            labelCol="is_positive", metricName="areaUnderROC").evaluate(preds)
        aupr = BinaryClassificationEvaluator(
            labelCol="is_positive", metricName="areaUnderPR").evaluate(preds)
        f1 = MulticlassClassificationEvaluator(
            labelCol="is_positive", metricName="f1").evaluate(preds)
        acc = MulticlassClassificationEvaluator(
            labelCol="is_positive", metricName="accuracy").evaluate(preds)

        mlflow.log_params({
            "classifier": classifier.__class__.__name__,
            "num_features": hashing.getNumFeatures(),
            "min_doc_freq": idf.getMinDocFreq(),
            "train_rows": train.count(),
            **(extra_params or {}),
        })
        mlflow.log_metrics({
            "auc": auc, "aupr": aupr, "f1": f1,
            "accuracy": acc, "train_seconds": train_secs,
        })
        mlflow.spark.log_model(model, "model")

        print(f"{name:22} AUC={auc:.4f}  AUPR={aupr:.4f}  F1={f1:.4f}  "
              f"acc={acc:.4f}  ({train_secs:.0f}s)")
        return run.info.run_id, auc, f1


rid_lr, auc_lr, f1_lr = run_experiment(
    "logreg-baseline",
    LogisticRegression(featuresCol="features", labelCol="is_positive",
                       maxIter=20, regParam=0.01))

# EXPECTED OUTPUT (approximate — text sentiment is a genuinely learnable task):
#   logreg-baseline        AUC=0.9350  AUPR=0.9150  F1=0.8650  acc=0.8660  (95s)
#
# WHY FOUR METRICS AND NOT ONE:
#   accuracy  intuitive, but misleading under class imbalance
#   AUC       threshold-independent ranking quality; the default headline metric
#   AUPR      far more informative than AUC when the positive class is rare
#   F1        balances precision and recall at the chosen threshold
#
# mlflow.log_model stores the ENTIRE PipelineModel — tokenizer, IDF weights,
# coefficients, everything. That is what makes serving reproducible: the same
# transformations that trained the model run at inference.
#
# NOTE: mlflow.autolog() would capture much of this automatically. Logging
# explicitly is used here so you can see exactly what is recorded and why.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4 — More runs, so the Experiments UI has something to compare

# COMMAND ----------

rid_lr2, auc_lr2, _ = run_experiment(
    "logreg-lowreg",
    LogisticRegression(featuresCol="features", labelCol="is_positive",
                       maxIter=50, regParam=0.001, elasticNetParam=0.0),
    {"note": "less regularisation, more iterations"})

rid_rf, auc_rf, _ = run_experiment(
    "randomforest",
    RandomForestClassifier(featuresCol="features", labelCol="is_positive",
                           numTrees=50, maxDepth=10, seed=42),
    {"note": "trees on sparse high-dim text"})

print(f"\nbest: {'logreg' if max(auc_lr, auc_lr2) > auc_rf else 'rf'}")

# EXPECTED OUTPUT (approximate):
#   logreg-lowreg          AUC=0.9390  ...  (140s)
#   randomforest           AUC=0.8600  ...  (210s)
#   best: logreg
#
# WHY LOGISTIC REGRESSION BEATS RANDOM FOREST HERE — worth understanding:
#   Text TF-IDF is very high-dimensional (65k) and very sparse. Linear models
#   handle that natively: one coefficient per feature, and sparsity makes the
#   optimisation cheap. Tree models must pick split points among 65k mostly-zero
#   features, which is both slow and statistically weak. This is why linear
#   models remain the default for bag-of-words text, and why "just use gradient
#   boosting" is not universal advice.
#
# NOW OPEN THE **Experiments** TAB IN THE SIDEBAR.
# Select all three runs -> Compare. You get a parallel-coordinates plot and a
# parameter/metric table. That comparison view is the entire point of MLflow:
# it replaces a spreadsheet of remembered numbers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5 — Hyperparameter search with CrossValidator

# COMMAND ----------

with mlflow.start_run(run_name="logreg-cv"):
    lr = LogisticRegression(featuresCol="features", labelCol="is_positive", maxIter=30)
    pipe = Pipeline(stages=[tokenizer, remover, hashing, idf, assembler, lr])

    grid = (ParamGridBuilder()
        .addGrid(lr.regParam, [0.001, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build())

    cv = CrossValidator(
        estimator=pipe,
        estimatorParamMaps=grid,
        evaluator=BinaryClassificationEvaluator(labelCol="is_positive",
                                                metricName="areaUnderROC"),
        numFolds=3,
        parallelism=2,
        seed=42)

    cv_model = cv.fit(train)
    best = cv_model.bestModel.stages[-1]

    auc = BinaryClassificationEvaluator(
        labelCol="is_positive", metricName="areaUnderROC").evaluate(
            cv_model.transform(test))

    mlflow.log_params({"best_regParam": best.getRegParam(),
                       "best_elasticNet": best.getElasticNetParam()})
    mlflow.log_metric("auc", auc)
    mlflow.spark.log_model(cv_model.bestModel, "model")

    print(f"CV best: regParam={best.getRegParam()} "
          f"elasticNet={best.getElasticNetParam()} AUC={auc:.4f}")

# EXPECTED OUTPUT:
#   CV best: regParam=0.001 elasticNet=0.0 AUC=0.939x
#
# COST WARNING — do the arithmetic before you run a grid:
#   6 param combinations x 3 folds = 18 full pipeline fits.
#   Each fit re-runs tokenisation, TF, and IDF from scratch. This is the
#   slowest cell in the file (5-10 minutes).
#
#   parallelism=2 fits two models concurrently. Raising it uses more cluster
#   at once, not less total work.
#
#   A REAL OPTIMISATION: move the feature stages OUT of the cross-validated
#   pipeline. Fit tokenizer/HashingTF/IDF once, cache the transformed features,
#   and cross-validate only the classifier. That turns 18 full pipelines into
#   1 feature pass + 18 cheap fits. Structuring the pipeline to avoid redundant
#   work is a good thing to be able to explain.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6 — The negative control (do this; it teaches more than it costs)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder

with mlflow.start_run(run_name="synthetic-keys-only"):
    idx = StringIndexer(inputCol="business_id", outputCol="biz_idx", handleInvalid="keep")
    ohe = OneHotEncoder(inputCol="biz_idx", outputCol="biz_vec")
    asm = VectorAssembler(inputCols=["biz_vec"], outputCol="features")
    lr  = LogisticRegression(featuresCol="features", labelCol="is_positive", maxIter=10)

    m = Pipeline(stages=[idx, ohe, asm, lr]).fit(train)
    auc = BinaryClassificationEvaluator(
        labelCol="is_positive", metricName="areaUnderROC").evaluate(m.transform(test))

    mlflow.log_metric("auc", auc)
    print(f"business_id only: AUC={auc:.4f}")

# EXPECTED OUTPUT:
#   business_id only: AUC=0.50xx        <-- pure coin flip
#
# THIS IS THE POINT. business_id was generated from a hash INDEPENDENT of the
# label, so it cannot possibly predict it. AUC ~= 0.50 confirms that.
#
# WHY RUN A NEGATIVE CONTROL AT ALL:
#   If this had scored 0.85, you would know something was LEAKING — the
#   synthetic key would have to encode the label somehow. Deliberately
#   training on features that must be useless is the cheapest way to detect
#   target leakage, and almost nobody does it.
#
# The mirror-image failure is more common and more dangerous: a feature that
# scores suspiciously WELL because it was computed after the fact. If a model
# jumps to 0.99 AUC, suspect leakage before celebrating.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7 — Feature Store
# MAGIC
# MAGIC This is the **Features** item in the sidebar. A feature table is a Delta
# MAGIC table with a declared primary key, registered so that features are
# MAGIC discoverable, lineage-tracked, and — crucially — computed by the SAME code
# MAGIC at training time and at serving time.

# COMMAND ----------

FEATURES = f"{CATALOG}.{SCHEMA}.yelp_business_features"

biz_features = (silver
    .groupBy("business_id")
    .agg(
        F.count("*").alias("f_n_reviews"),
        F.round(F.avg("stars"), 4).alias("f_avg_stars"),
        F.round(F.stddev("stars"), 4).alias("f_stddev_stars"),
        F.round(F.avg("is_positive"), 4).alias("f_positive_rate"),
        F.round(F.avg("word_count"), 2).alias("f_avg_words"),
        F.round(F.avg("caps_ratio"), 4).alias("f_avg_caps"),
        F.round(F.avg("has_exclaim"), 4).alias("f_exclaim_rate"),
        F.max("review_date").alias("f_last_review"),
    )
    .withColumn("f_days_since_last",
                F.datediff(F.current_date(), F.col("f_last_review")))
    .drop("f_last_review"))

(biz_features.write.mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(FEATURES))

spark.sql(f"ALTER TABLE {FEATURES} ALTER COLUMN business_id SET NOT NULL")
spark.sql(f"ALTER TABLE {FEATURES} ADD CONSTRAINT pk_biz PRIMARY KEY (business_id)")

display(spark.table(FEATURES).limit(10))

# EXPECTED OUTPUT: 2,000 rows, one per business, 8 feature columns.
#
# THE PRIMARY KEY IS WHAT MAKES IT A FEATURE TABLE.
# In Unity Catalog, declaring a PK is how the platform knows this table can be
# joined by key for feature lookup, and it is what the Feature Engineering
# client uses to build training sets and to serve features online.
# NOT NULL is required before a PK constraint can be added.
#
# WHY A FEATURE STORE EXISTS AT ALL — the problem it solves:
#   Training code computes avg_stars one way in a notebook. Serving code
#   recomputes it a slightly different way in a microservice. The two drift,
#   and the model silently degrades in production. That is TRAINING/SERVING
#   SKEW, and it is one of the most common causes of "the model worked in
#   testing". A feature store makes the definition singular.
#
# Open the **Features** tab in the sidebar — the table appears with its schema,
# PK, and lineage back to yelp_silver.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build a training set by joining features

# COMMAND ----------

train_with_features = (train
    .join(spark.table(FEATURES), "business_id", "left")
    .fillna(0, subset=[c for c in spark.table(FEATURES).columns if c.startswith("f_")]))

display(train_with_features.select(
    "review_id", "business_id", "stars",
    "f_n_reviews", "f_avg_stars", "f_positive_rate").limit(5))

# fillna(0) on the feature columns handles businesses with no feature row —
# a LEFT join produces NULLs, and Spark ML estimators REJECT nulls in the
# feature vector with an unhelpful error. Deciding the imputation value
# explicitly is better than discovering the failure at fit() time.
#
# IN PRODUCTION you would use databricks.feature_engineering:
#     from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
#     fe = FeatureEngineeringClient()
#     training_set = fe.create_training_set(
#         df=train, label="is_positive",
#         feature_lookups=[FeatureLookup(table_name=FEATURES,
#                                        lookup_key="business_id")])
# That records the lookup IN THE MODEL, so at serving time you send only
# business_id and the platform fetches the features for you. The manual join
# above is the same idea without the packaging.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 8 — Inspect what the model learned

# COMMAND ----------

best_run = mlflow.search_runs(
    order_by=["metrics.auc DESC"], max_results=1).iloc[0]
print("best run:", best_run["run_id"], "AUC:", round(best_run["metrics.auc"], 4))

model = mlflow.spark.load_model(f"runs:/{best_run['run_id']}/model")
scored = model.transform(test)

display(scored.select("text", "stars", "is_positive", "probability", "prediction").limit(10))

# EXPECTED OUTPUT: `probability` is a 2-element vector [P(class0), P(class1)];
# `prediction` is the argmax at the 0.5 threshold.
#
# EXTRACTING THE POSITIVE PROBABILITY as a plain double requires a UDF,
# because Spark has no built-in vector-element accessor:
#     from pyspark.sql.types import DoubleType
#     get_p = F.udf(lambda v: float(v[1]), DoubleType())
#     scored.withColumn("p_positive", get_p("probability"))
# A rare case where a UDF is genuinely the simplest correct answer — though
# vector_to_array() from pyspark.ml.functions avoids it and is faster.

# COMMAND ----------

display(
    scored.groupBy("stars")
          .agg(F.count("*").alias("n"),
               F.round(F.avg("prediction"), 3).alias("predicted_positive_rate"))
          .orderBy("stars"))

# EXPECTED OUTPUT — a clean monotonic ramp:
#   stars | n      | predicted_positive_rate
#   ------+--------+------------------------
#     1   | 28000  | 0.03
#     2   | 28000  | 0.10
#     3   | 28000  | 0.35     <-- the hard middle
#     4   | 28000  | 0.85
#     5   | 28000  | 0.95
#
# READ THE 3-STAR ROW. It is the genuinely ambiguous class, and it is where
# nearly all the model's errors live. Monotonicity across the other classes
# is strong evidence the model learned real sentiment rather than an artefact.
#
# This kind of ERROR-SLICE ANALYSIS matters more than the headline metric.
# "AUC 0.94" tells you nothing about WHERE the model fails; this table does.

# COMMAND ----------

train.unpersist(); test.unpersist()

# ALWAYS unpersist. In a long-running application, cached blocks occupy
# storage memory forever, evicting each other and starving execution memory —
# a common cause of a job that gets slower over time.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What you built
# MAGIC
# MAGIC | Artefact | Where |
# MAGIC |---|---|
# MAGIC | 5+ MLflow runs with params/metrics/models | **Experiments** tab |
# MAGIC | `yelp_business_features` (PK: business_id) | **Features** tab |
# MAGIC | Best model logged and loadable by run id | MLflow artifacts |
# MAGIC
# MAGIC ### Concepts demonstrated
# MAGIC - Seeded and hash-based splits; why unseeded splits invalidate comparisons
# MAGIC - Spark ML Pipeline as a lazy transformer graph; PipelineModel carries fitted state
# MAGIC - HashingTF vs CountVectorizer; IDF as an estimator
# MAGIC - Why linear models beat trees on sparse high-dimensional text
# MAGIC - Four metrics and when each misleads
# MAGIC - CrossValidator cost arithmetic and how to restructure it
# MAGIC - **Negative control** to detect target leakage
# MAGIC - Feature table with a primary key; training/serving skew
# MAGIC - Error-slice analysis over headline metrics
# MAGIC
# MAGIC **Next:** `04_serving.py` — register to Unity Catalog, deploy an endpoint.
