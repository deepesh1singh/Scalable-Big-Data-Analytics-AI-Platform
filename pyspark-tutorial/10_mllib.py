"""
10 — MLlib: a complete supervised pipeline
==========================================

REAL-WORLD SCENARIO
-------------------
Telecom churn prediction. Given customer tenure, contract type, monthly charges,
support-ticket counts and payment method, predict who will cancel next month so
retention can intervene.

This is the canonical tabular ML problem, and it exercises everything: mixed
categorical/numeric features, class imbalance, a proper train/test split,
hyperparameter search, honest evaluation, and model persistence.

Run:
    spark-submit 10_mllib.py
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (StringIndexer, OneHotEncoder, VectorAssembler,
                                StandardScaler, Imputer, Bucketizer)
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier,
                                       GBTClassifier)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit

spark = (SparkSession.builder.appName("mllib-churn")
         .master("local[4]")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Section 1 — Synthesise data with REAL signal
# ---------------------------------------------------------------------------

raw = (spark.range(0, 60_000)
    .withColumn("tenure_months",  (F.rand(seed=1) * 72).cast("int"))
    .withColumn("monthly_charge", F.round(20 + F.rand(seed=2) * 100, 2))
    .withColumn("support_tickets", (F.rand(seed=3) * 8).cast("int"))
    .withColumn("contract",
        F.when(F.rand(seed=4) < 0.5, "month-to-month")
         .when(F.rand(seed=4) < 0.8, "one-year").otherwise("two-year"))
    .withColumn("payment",
        F.when(F.rand(seed=5) < 0.35, "electronic-check")
         .when(F.rand(seed=5) < 0.7, "card").otherwise("bank-transfer"))
    .withColumn("has_fiber", (F.rand(seed=6) < 0.45).cast("int"))
    # ~8% of monthly_charge values missing, so Imputer has work to do
    .withColumn("monthly_charge",
        F.when(F.rand(seed=7) < 0.08, None).otherwise(F.col("monthly_charge")))
)

# Churn probability driven by real relationships + noise
churn_logit = (
    -1.2
    - 0.045 * F.col("tenure_months")
    + 0.012 * F.coalesce(F.col("monthly_charge"), F.lit(60.0))
    + 0.28 * F.col("support_tickets")
    + F.when(F.col("contract") == "month-to-month", 1.4)
       .when(F.col("contract") == "one-year", 0.3).otherwise(0.0)
    + F.when(F.col("payment") == "electronic-check", 0.5).otherwise(0.0)
)

data = (raw
    .withColumn("p_churn", 1 / (1 + F.exp(-churn_logit)))
    .withColumn("churn", (F.rand(seed=99) < F.col("p_churn")).cast("int"))
    .drop("id", "p_churn")
    .cache())

print("rows:", data.count())
data.groupBy("churn").count().orderBy("churn").show()
data.show(5)

# EXPECTED: roughly 25-30% churn — a realistic mild imbalance.
#
# WHY SYNTHESISE RATHER THAN DOWNLOAD: the relationships are KNOWN, so you can
# verify the model recovers them. If feature importance does not rank
# contract / tenure / support_tickets highly, your pipeline has a bug — and you
# would never know that on a real dataset.

# ---------------------------------------------------------------------------
# Section 2 — Split FIRST, before anything is fitted
# ---------------------------------------------------------------------------

train, test = data.randomSplit([0.8, 0.2], seed=42)
train.cache(); test.cache()
print(f"train {train.count():,}  test {test.count():,}")

# >>> SPLIT BEFORE FITTING ANY TRANSFORMER <<<
# Imputer, StringIndexer and StandardScaler all LEARN from data. Fitting them
# on the full dataset leaks test-set information into training — the model
# looks better than it is, and the gap only appears in production.
# The Pipeline below is fitted on `train` only. This ordering is the whole
# point of Pipeline.
#
# seed=42 makes the split reproducible. Without it, every "improvement" you
# measure might just be a different test set.
#
# MORE ROBUST than randomSplit — a hash-based split is stable even if upstream
# data changes:
#     .withColumn("bucket", F.pmod(F.hash("customer_id"), F.lit(10)))
#     train = df.filter("bucket < 8"); test = df.filter("bucket >= 8")
#
# For TIME SERIES, never split randomly — split by DATE, or you train on the
# future and predict the past.

# ---------------------------------------------------------------------------
# Section 3 — The feature pipeline
# ---------------------------------------------------------------------------

CATEGORICAL = ["contract", "payment"]
NUMERIC = ["tenure_months", "monthly_charge", "support_tickets", "has_fiber"]

imputer = Imputer(
    inputCols=["monthly_charge"], outputCols=["monthly_charge_i"],
    strategy="median")

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                          handleInvalid="keep") for c in CATEGORICAL]

encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec",
                          dropLast=True) for c in CATEGORICAL]

bucketizer = Bucketizer(
    splits=[-float("inf"), 6, 24, 48, float("inf")],
    inputCol="tenure_months", outputCol="tenure_band")

assembler = VectorAssembler(
    inputCols=["tenure_months", "monthly_charge_i", "support_tickets",
               "has_fiber", "tenure_band"] + [f"{c}_vec" for c in CATEGORICAL],
    outputCol="features_raw",
    handleInvalid="skip")

scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                        withMean=True, withStd=True)

FEATURE_STAGES = [imputer] + indexers + encoders + [bucketizer, assembler, scaler]

# STAGE BY STAGE:
#
#   Imputer          fills nulls with mean/median/mode. LEARNS the fill value
#                    from training data only. Spark ML estimators REJECT nulls
#                    in the feature vector with an unhelpful error, so this is
#                    not optional when nulls exist.
#
#   StringIndexer    string -> numeric index, ordered by frequency.
#                    handleInvalid: "error" (default, fails on unseen values at
#                    predict time), "skip" (drops the row), "keep" (assigns a
#                    special index). >>> USE "keep" IN PRODUCTION <<< — a new
#                    contract type appearing next month should not crash
#                    inference.
#
#   OneHotEncoder    index -> sparse one-hot vector. dropLast=True omits the
#                    last category to avoid the DUMMY VARIABLE TRAP (perfect
#                    multicollinearity), which matters for linear models and is
#                    irrelevant for trees.
#                    NEVER feed a raw StringIndexer output to a linear model —
#                    it implies a false ordering (card=1 < check=2).
#
#   Bucketizer       continuous -> discrete bands. Lets a LINEAR model capture
#                    a non-linear relationship. Note the infinite bounds; a
#                    value outside the splits errors otherwise.
#
#   VectorAssembler  concatenates everything into ONE vector column. Spark ML
#                    requires a single vector named by featuresCol — you cannot
#                    pass a list of columns to an estimator.
#
#   StandardScaler   zero mean, unit variance. ESSENTIAL for LogisticRegression
#                    and SVMs (gradient descent converges badly on unscaled
#                    features). USELESS for trees, which are scale-invariant.

# ---------------------------------------------------------------------------
# Section 4 — Train three models
# ---------------------------------------------------------------------------

def evaluate(model, df, name):
    p = model.transform(df)
    auc = BinaryClassificationEvaluator(labelCol="churn",
            metricName="areaUnderROC").evaluate(p)
    aupr = BinaryClassificationEvaluator(labelCol="churn",
            metricName="areaUnderPR").evaluate(p)
    f1 = MulticlassClassificationEvaluator(labelCol="churn",
            metricName="f1").evaluate(p)
    acc = MulticlassClassificationEvaluator(labelCol="churn",
            metricName="accuracy").evaluate(p)
    print(f"{name:22} AUC={auc:.4f}  AUPR={aupr:.4f}  F1={f1:.4f}  acc={acc:.4f}")
    return auc, p


models = {
    "logistic-regression": LogisticRegression(featuresCol="features", labelCol="churn",
                                              maxIter=50, regParam=0.01),
    "random-forest":       RandomForestClassifier(featuresCol="features", labelCol="churn",
                                                  numTrees=80, maxDepth=8, seed=42),
    "gradient-boosted":    GBTClassifier(featuresCol="features", labelCol="churn",
                                         maxIter=40, maxDepth=5, seed=42),
}

fitted, best_auc, best_name, best_model = {}, 0.0, None, None
for name, clf in models.items():
    m = Pipeline(stages=FEATURE_STAGES + [clf]).fit(train)
    auc, _ = evaluate(m, test, name)
    fitted[name] = m
    if auc > best_auc:
        best_auc, best_name, best_model = auc, name, m

print(f"\nbest: {best_name} (AUC {best_auc:.4f})")

# EXPECTED (approximate):
#   logistic-regression    AUC=0.80  AUPR=0.62  F1=0.75  acc=0.76
#   random-forest          AUC=0.80  AUPR=0.62  F1=0.75  acc=0.76
#   gradient-boosted       AUC=0.81  AUPR=0.63  F1=0.75  acc=0.76
#
# All three land close together because the true relationship is LOGISTIC —
# exactly what LogisticRegression models. GBT edges ahead by capturing the
# bucketised non-linearity.
#
# >>> COMPARE WITH FILE 03 OF THE YELP LAB <<<
# There, on sparse high-dimensional TEXT, LogisticRegression beat RandomForest
# decisively (0.94 vs 0.86). Here on dense tabular data they tie and GBT wins
# slightly. The lesson: model choice depends on DATA SHAPE, not on which
# algorithm is fashionable. "Just use gradient boosting" is not universal advice.
#
# FOUR METRICS, because each hides something:
#   accuracy  intuitive, misleading under imbalance (predict all-zero here
#             already gives ~72%)
#   AUC       threshold-independent ranking quality; the default headline
#   AUPR      far more informative when the positive class is rare
#   F1        balances precision and recall AT THE CHOSEN THRESHOLD

# ---------------------------------------------------------------------------
# Section 5 — Did it learn the real relationships?
# ---------------------------------------------------------------------------

rf = fitted["random-forest"].stages[-1]
names = ["tenure_months", "monthly_charge_i", "support_tickets", "has_fiber",
         "tenure_band"] + [f"{c}_vec[{i}]" for c in CATEGORICAL for i in range(2)]

imp = sorted(zip(names, rf.featureImportances.toArray()),
             key=lambda t: -t[1])
print("\nfeature importances:")
for n, v in imp:
    print(f"   {n:22} {v:.4f}  {'#' * int(v * 60)}")

# EXPECTED: tenure_months, contract_vec and support_tickets dominate — which
# are exactly the drivers we built into the data generator.
#
# >>> THIS IS A CORRECTNESS CHECK, NOT A CURIOSITY <<<
# We KNOW the true relationships. If has_fiber (pure noise) ranked first, the
# pipeline has a bug. On real data you cannot make this check, which is
# precisely why validating a pipeline on synthetic data with known structure
# is worth the effort.
#
# CAUTION: tree feature importances are BIASED toward high-cardinality and
# continuous features. For decisions that matter, use permutation importance
# or SHAP rather than the built-in numbers.

lr = fitted["logistic-regression"].stages[-1]
print(f"\nlogreg intercept {lr.intercept:.3f}")
print("coefficients:", [round(c, 3) for c in lr.coefficients.toArray()])

# Because StandardScaler was applied, coefficient MAGNITUDES are directly
# comparable across features. Without scaling they are not, and ranking them
# would be meaningless.

# ---------------------------------------------------------------------------
# Section 6 — Confusion matrix and threshold tuning
# ---------------------------------------------------------------------------

pred = best_model.transform(test)

print("\nconfusion matrix:")
pred.groupBy("churn").pivot("prediction", [0.0, 1.0]).count().fillna(0).show()

from pyspark.ml.functions import vector_to_array

with_p = pred.withColumn("p_churn", vector_to_array("probability")[1])

print("threshold sweep:")
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    d = with_p.withColumn("pred_t", (F.col("p_churn") >= t).cast("double"))
    tp = d.filter("churn = 1 AND pred_t = 1").count()
    fp = d.filter("churn = 0 AND pred_t = 1").count()
    fn = d.filter("churn = 1 AND pred_t = 0").count()
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"   t={t:.1f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")

# >>> 0.5 IS AN ARBITRARY DEFAULT, NOT AN OPTIMUM <<<
# The right threshold follows from BUSINESS COST asymmetry:
#   - a retention call costs $5; a lost customer costs $500
#   - so a false negative is 100x more expensive than a false positive
#   - therefore push the threshold DOWN, accept more false positives,
#     and catch more real churners
# Choosing the threshold from a cost matrix rather than accepting 0.5 is one of
# the highest-value and least-practised things in applied ML.
#
# vector_to_array("probability")[1] extracts P(class=1). Note this indexing is
# 0-BASED, whereas element_at() in Spark SQL is 1-based. Two conventions in one
# API; off-by-one here silently inverts your probabilities.

# ---------------------------------------------------------------------------
# Section 7 — Hyperparameter search
# ---------------------------------------------------------------------------

grid = (ParamGridBuilder()
    .addGrid(models["random-forest"].numTrees, [40, 80])
    .addGrid(models["random-forest"].maxDepth, [5, 10])
    .build())

tvs = TrainValidationSplit(
    estimator=Pipeline(stages=FEATURE_STAGES + [models["random-forest"]]),
    estimatorParamMaps=grid,
    evaluator=BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC"),
    trainRatio=0.8,
    parallelism=2,
    seed=42)

tvs_model = tvs.fit(train)
best_rf = tvs_model.bestModel.stages[-1]
print(f"\nbest params: numTrees={best_rf.getNumTrees} maxDepth={best_rf.getMaxDepth()}")
evaluate(tvs_model.bestModel, test, "tuned-rf")

# >>> DO THE COST ARITHMETIC BEFORE RUNNING A GRID <<<
#   TrainValidationSplit: 4 combos x 1 fit  =  4 pipeline fits
#   CrossValidator(numFolds=3): 4 x 3       = 12 pipeline fits
# and EACH fit re-runs the entire feature pipeline from scratch.
#
# A REAL OPTIMISATION: move the feature stages OUT of the searched pipeline.
# Fit imputer/indexers/encoders/scaler ONCE, cache the transformed features,
# and search only the classifier. That turns 12 full pipelines into 1 feature
# pass + 12 cheap fits. Structuring a pipeline to avoid redundant work is a
# good thing to be able to explain.
#
# parallelism=2 fits two models concurrently — more cluster at once, not less
# total work.
#
# CrossValidator when data is small and variance matters; TrainValidationSplit
# when data is large and you want one pass.

# ---------------------------------------------------------------------------
# Section 8 — Persist and reload
# ---------------------------------------------------------------------------

PATH = "/tmp/pyspark_lab_10/churn_model"
best_model.write().overwrite().save(PATH)
reloaded = PipelineModel.load(PATH)

check = reloaded.transform(test.limit(100))
print("\nreloaded model predictions:")
check.select("tenure_months", "contract", "support_tickets",
             "churn", "prediction").show(5)

# >>> THE WHOLE PipelineModel IS SAVED — every fitted stage <<<
# Imputer medians, StringIndexer vocabularies, scaler means/stds, tree
# structures. So inference applies the IDENTICAL transformations that training
# used. That property is what prevents TRAINING/SERVING SKEW, which is one of
# the most common causes of "the model worked in testing".
#
# Saving only the classifier and re-implementing feature engineering in a
# serving service is how teams create that skew. Save the whole pipeline.
#
# In production, register with MLflow instead of a raw path — you get versions,
# aliases, lineage and rollback. See 04_serving.py in the notebooks/ folder.

# ---------------------------------------------------------------------------
# Section 9 — Batch scoring
# ---------------------------------------------------------------------------

scored = (reloaded.transform(data)
    .withColumn("p_churn", F.round(vector_to_array("probability")[1], 4))
    .withColumn("risk_tier",
        F.when(F.col("p_churn") >= 0.7, "critical")
         .when(F.col("p_churn") >= 0.5, "high")
         .when(F.col("p_churn") >= 0.3, "medium").otherwise("low"))
    .select("tenure_months", "contract", "monthly_charge",
            "support_tickets", "churn", "p_churn", "risk_tier"))

scored.groupBy("risk_tier").agg(
    F.count("*").alias("customers"),
    F.round(F.avg("p_churn"), 3).alias("avg_p"),
    F.round(F.avg("churn"), 3).alias("actual_churn_rate"),
    F.round(F.avg("monthly_charge"), 2).alias("avg_monthly"),
).orderBy(F.desc("avg_p")).show()

# >>> READ avg_p AGAINST actual_churn_rate — THIS IS CALIBRATION <<<
# A well-calibrated model predicting 0.70 should see ~70% of those customers
# actually churn. If predicted and actual diverge badly, the RANKING may still
# be fine (AUC unaffected) but the PROBABILITIES are not trustworthy — and any
# expected-value calculation built on them is wrong.
#
# Tree ensembles are frequently poorly calibrated. Fix with Platt scaling or
# isotonic regression. AUC will not tell you about this; only a calibration
# table like this one will.

train.unpersist(); test.unpersist(); data.unpersist()
spark.stop()

# ---------------------------------------------------------------------------
# TAKEAWAYS
# ---------------------------------------------------------------------------
# 1. Split BEFORE fitting any transformer. Imputer/Indexer/Scaler leak otherwise.
# 2. Pipeline keeps training and serving transformations identical.
# 3. handleInvalid="keep" on StringIndexer, or unseen values crash inference.
# 4. OneHot before linear models; raw indices imply a false ordering.
# 5. Scale for LR/SVM; pointless for trees.
# 6. Model choice depends on data SHAPE — linear wins on sparse text, GBT on
#    dense tabular. There is no universally best algorithm.
# 7. Four metrics; accuracy alone is misleading under imbalance.
# 8. 0.5 is arbitrary. Choose the threshold from business cost asymmetry.
# 9. Grid search cost = combos x folds x full pipeline. Cache features first.
# 10. Save the WHOLE PipelineModel, never just the classifier.
# 11. Check calibration (predicted vs actual rate), not only AUC.
# 12. Validate on synthetic data with known structure — it catches real bugs.
