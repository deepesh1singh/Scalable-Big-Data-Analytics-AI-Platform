"""
09 — Structured Streaming: files and Kafka
==========================================

REAL-WORLD SCENARIO
-------------------
Payment fraud detection. Card authorisations arrive on Kafka. You must maintain
a 5-minute rolling count and sum per card, flag velocity anomalies (more than
N transactions in the window), tolerate events arriving late, and survive
restarts without double-counting or losing data.

Structured Streaming is batch code over an unbounded input table. Almost every
DataFrame operation you already know works unchanged. The genuinely new
concepts are event time, watermarks, output modes and checkpoints.

Run:
    spark-submit 09_streaming.py            # file-source demo runs standalone
"""

import os, shutil, json, time
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, LongType, IntegerType)

spark = (SparkSession.builder.appName("structured-streaming")
         .master("local[4]")
         .config("spark.sql.shuffle.partitions", "4")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE = "/tmp/pyspark_lab_09"
IN, OUT, CKPT = f"{BASE}/in", f"{BASE}/out", f"{BASE}/ckpt"
shutil.rmtree(BASE, ignore_errors=True)
for d in (IN, OUT, CKPT):
    os.makedirs(d, exist_ok=True)

SCHEMA = StructType([
    StructField("txn_id",   StringType()),
    StructField("card_id",  StringType()),
    StructField("amount",   DoubleType()),
    StructField("merchant", StringType()),
    StructField("event_ts", LongType()),      # epoch millis
])

# ---------------------------------------------------------------------------
# Section 1 — Streaming read: schema is MANDATORY
# ---------------------------------------------------------------------------

stream = (spark.readStream
    .schema(SCHEMA)                       # required — cannot infer future data
    .option("maxFilesPerTrigger", 1)      # rate limit: one file per micro-batch
    .json(IN))

print("isStreaming:", stream.isStreaming)

# EXPECTED: isStreaming: True
#
# .schema() IS MANDATORY for a streaming file source. Inference would require
# reading data that has not arrived yet. Omitting it throws immediately.
#
# maxFilesPerTrigger is the RATE LIMITER and the most important operational
# setting here. Without it, a job restarting after a two-day outage tries to
# consume the entire backlog in ONE micro-batch and OOMs. The Kafka equivalent
# is maxOffsetsPerTrigger. This is also the standard fix for "my streaming job
# died on restart after a backlog built up".

# ---------------------------------------------------------------------------
# Section 2 — Event time, watermark, windows
# ---------------------------------------------------------------------------

WINDOW = "5 minutes"
SLIDE = "1 minute"
LATENESS = "10 minutes"

agg = (stream
    .withColumn("event_time", (F.col("event_ts") / 1000).cast("timestamp"))
    .withColumn("ingest_time", F.current_timestamp())
    .withWatermark("event_time", LATENESS)
    .groupBy(F.window("event_time", WINDOW, SLIDE), "card_id")
    .agg(
        F.count("*").alias("n_txns"),
        F.round(F.sum("amount"), 2).alias("total"),
        F.round(F.max("amount"), 2).alias("largest"),
        F.approx_count_distinct("merchant").alias("merchants"))
    .withColumn("window_start", F.col("window.start"))
    .withColumn("window_end",   F.col("window.end"))
    .withColumn("is_velocity_alert",
        (F.col("n_txns") >= 4) | (F.col("total") > 5000))
    .drop("window"))

# >>> THE THREE TIME CONCEPTS <<<
#   EVENT TIME       when it happened, carried IN the record. What you want.
#   PROCESSING TIME  when Spark saw it. Depends on lag, retries, backlogs —
#                    so results are not reproducible.
#   INGESTION TIME   when it entered the system (e.g. Kafka's record timestamp).
# Always window on EVENT TIME. Carrying event_ts inside the payload is what
# makes that possible; Kafka's own timestamp is often ingestion time.
#
# >>> WATERMARK: the single most important streaming concept <<<
#   withWatermark("event_time", "10 minutes") declares: "I will wait up to 10
#   minutes past the max event time I have seen before finalising a window."
#
#   TWO effects:
#     1. STATE CLEANUP — windows older than the watermark are dropped from the
#        state store. WITHOUT a watermark, state grows FOREVER and the job dies
#        days or weeks later. That delayed failure is what makes it nasty.
#     2. In append mode, it decides WHEN a window is emitted.
#
#   THE TRADE-OFF:
#     too long  -> unbounded state growth -> eventual OOM
#     too short -> late data silently DROPPED
#   There is no safe default. Measure your actual lateness distribution.
#
# WINDOW TYPES:
#   window(col, "5 minutes")                tumbling — non-overlapping
#   window(col, "5 minutes", "1 minute")    sliding  — OVERLAPPING
#   session_window(col, "30 minutes")       session  — gap-based (Spark 3.2+)
#
#   >>> SLIDING WINDOWS MULTIPLY STATE <<<
#   Each record belongs to windowSize/slideSize windows simultaneously. Here
#   5min/1min = every record is held in FIVE windows. A 1-hour window sliding
#   every minute means 60x the state. Check this arithmetic before shipping.

# ---------------------------------------------------------------------------
# Section 3 — Output modes
# ---------------------------------------------------------------------------
#
#   append    emit each row ONCE, when the watermark passes the window end.
#             Requires a watermark for aggregations. Bounded state.
#             >>> The only combination that is both correct and bounded. <<<
#   update    emit only rows that CHANGED this batch. Lower latency, but the
#             sink must handle upserts.
#   complete  re-emit the ENTIRE result table every batch. Requires unbounded
#             state. Only viable for small aggregations; never for production
#             with a growing key space.
#
# Not every sink supports every mode. File sinks support append only.

query = (agg.writeStream
    .outputMode("append")
    .format("parquet")
    .option("path", OUT)
    .option("checkpointLocation", CKPT)
    .trigger(processingTime="5 seconds")
    .queryName("fraud-velocity")
    .start())

# >>> checkpointLocation IS NOT OPTIONAL <<<
# It stores source offsets, the query plan version, and the state store.
#   - Delete it and the query reprocesses everything from scratch.
#   - Point two queries at the same checkpoint and you corrupt both.
#   - Many code changes INVALIDATE it, because the plan version is recorded.
#     That is the awkward part of Spark streaming deployments: there is no
#     Flink-style savepoint that survives an operator change.
# It is the single most operationally important line in a streaming job.
#
# TRIGGERS:
#   processingTime="5 seconds"  micro-batch every 5s
#   availableNow=True           process all available data, then STOP.
#                               >>> Use this instead of a scheduled batch job
#                               whenever the source is an append-only
#                               directory: you get checkpointed exactly-once
#                               file handling with batch execution. <<<
#   once=True                   deprecated in favour of availableNow
#   continuous="1 second"       experimental, very few operators supported

# ---------------------------------------------------------------------------
# Section 4 — Feed it and watch
# ---------------------------------------------------------------------------

now_ms = int(time.time() * 1000)
CARDS = ["C-1111", "C-2222", "C-3333"]

for batch in range(6):
    events = []
    for i in range(8):
        card = CARDS[(batch + i) % 3]
        # card C-1111 gets a burst — should trigger the velocity alert
        n = 5 if (card == "C-1111" and batch == 2) else 1
        for k in range(n):
            events.append({
                "txn_id": f"T-{batch}-{i}-{k}",
                "card_id": card,
                "amount": round(20 + (i * 37 + k * 13) % 900, 2),
                "merchant": f"M-{(i + k) % 5}",
                "event_ts": now_ms + batch * 60_000 + i * 1000 + k * 100,
            })
    with open(f"{IN}/batch_{batch:03d}.json", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    time.sleep(6)
    p = query.lastProgress
    if p:
        state = p["stateOperators"][0]["numRowsTotal"] if p.get("stateOperators") else 0
        print(f"batch {p['batchId']:>2}  in={p['numInputRows']:>4}  "
              f"rate={p.get('processedRowsPerSecond', 0):>7.0f}/s  "
              f"stateRows={state:>5}")

# >>> WATCH stateRows <<<
# That is your state store size. If it grows MONOTONICALLY and never drops,
# your watermark is not cleaning state and the job WILL eventually die.
# Plotting state size over time is the number-one thing to monitor in a
# stateful streaming job. query.lastProgress / recentProgress are the hooks;
# in production push them to your metrics system.

query.processAllAvailable()
query.stop()

print("\n--- results ---")
res = spark.read.parquet(OUT)
res.orderBy("card_id", "window_start").show(20, truncate=False)

print("velocity alerts:")
res.filter("is_velocity_alert").select(
    "card_id", "window_start", "n_txns", "total").show(truncate=False)

# EXPECTED: C-1111 shows elevated n_txns in the burst window, flagged.
#
# NOTE some windows may be missing: in APPEND mode a window is only emitted
# once the watermark passes its end. The final windows stay in state until
# more data (or a shutdown) advances the watermark. That is correct behaviour
# and surprises people constantly — "where are my last results?".

# ---------------------------------------------------------------------------
# Section 5 — Restart is safe
# ---------------------------------------------------------------------------

q2 = (agg.writeStream
      .outputMode("append").format("parquet")
      .option("path", OUT).option("checkpointLocation", CKPT)
      .trigger(availableNow=True).start())
q2.awaitTermination()
print("rows after restart:", spark.read.parquet(OUT).count())

# The checkpoint knows which files were already processed, so the restart
# consumes nothing new. This is the property that makes streaming jobs safe
# to schedule and safe to restart after a crash.

# ---------------------------------------------------------------------------
# Section 6 — Kafka source and sink (reference)
# ---------------------------------------------------------------------------
#
# spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 ...
#
# raw = (spark.readStream.format("kafka")
#     .option("kafka.bootstrap.servers", "localhost:9092")
#     .option("subscribe", "authorisations")
#     .option("startingOffsets", "earliest")
#     .option("maxOffsetsPerTrigger", 100000)
#     .option("failOnDataLoss", "true")
#     .load())
#
# The Kafka source ALWAYS returns this fixed schema:
#     key BINARY, value BINARY, topic STRING, partition INT,
#     offset LONG, timestamp TIMESTAMP, timestampType INT
# Your payload is opaque bytes — you deserialise it yourself:
#
# events = (raw.select(
#     F.from_json(F.col("value").cast("string"), SCHEMA).alias("j"),
#     "partition", "offset", F.col("timestamp").alias("kafka_ts"))
#   .select("j.*", "partition", "offset", "kafka_ts"))
#
# OPTIONS THAT MATTER:
#   startingOffsets   earliest | latest | explicit JSON per partition.
#                     >>> Applies ONLY on the FIRST run. <<< After that the
#                     checkpoint wins. Changing it later does nothing, which
#                     confuses people expecting it to force a reprocess. To
#                     truly reprocess, delete the checkpoint.
#   maxOffsetsPerTrigger  the rate limiter. See Section 1.
#   failOnDataLoss=true   abort if requested offsets were deleted by retention.
#                     Setting false SILENTLY SKIPS missing data.
#
# from_json returns NULL for the WHOLE struct on a malformed message — not an
# error. Always count and alert on those; Spark has no built-in dead-letter
# mechanism, you build it with a filter and a second sink.
#
# NUMBER OF SPARK TASKS = NUMBER OF KAFKA PARTITIONS. Four Kafka partitions
# means four reader tasks regardless of cluster size. Going wider requires an
# explicit repartition(), which costs a shuffle. Kafka partition count is a
# capacity-planning decision, not a detail.
#
# ALSO: Spark does NOT use Kafka's consumer-group protocol for assignment or
# offsets. It assigns partitions itself and stores offsets in ITS OWN
# checkpoint. So consumer-lag monitoring via Kafka's group API can show stale
# or empty results, and deleting the checkpoint resets your position even
# though Kafka shows committed offsets for your group id.
#
# WRITING BACK TO KAFKA:
# (out.select(F.col("card_id").alias("key"),
#             F.to_json(F.struct("*")).alias("value"))
#     .writeStream.format("kafka")
#     .option("kafka.bootstrap.servers", "localhost:9092")
#     .option("topic", "fraud-alerts")
#     .option("checkpointLocation", "/ckpt/alerts").start())

# ---------------------------------------------------------------------------
# Section 7 — foreachBatch: the escape hatch
# ---------------------------------------------------------------------------
#
# def upsert(batch_df, batch_id):
#     # batch_df is a NORMAL DataFrame — every batch API works here
#     batch_df.persist()
#     batch_df.write.mode("append").saveAsTable("alerts_audit")
#     (DeltaTable.forName(spark, "alerts")
#         .alias("t").merge(batch_df.alias("s"), "t.card_id = s.card_id")
#         .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())
#     batch_df.unpersist()
#
# agg.writeStream.foreachBatch(upsert) \
#    .option("checkpointLocation", CKPT).start()
#
# WHY IT MATTERS: it unlocks operations streaming does not support natively —
# MERGE/upsert, writing to multiple sinks, JDBC writes, arbitrary Python.
#
# >>> batch_id IS THE IDEMPOTENCY KEY <<<
# On failure Spark may RE-RUN a batch. Exactly-once end-to-end requires the
# sink to be idempotent or transactional, and batch_id is how you achieve it:
# a MERGE keyed on it, or a "skip if batch_id already written" check. Writing
# to a REST API in foreachBatch is at-least-once no matter what Spark does.
#
# Also persist() inside foreachBatch if you use batch_df more than once —
# otherwise the whole streaming lineage recomputes per use.

# ---------------------------------------------------------------------------
# Section 8 — Stream-stream joins
# ---------------------------------------------------------------------------
#
# STREAM-STATIC: easy, no watermark needed on the static side.
#   stream.join(F.broadcast(dim_df), "card_id", "left")
#
# STREAM-STREAM: requires watermarks on BOTH sides AND a time constraint,
# or state grows without bound:
#   auth.withWatermark("auth_time", "10 minutes").alias("a").join(
#       settle.withWatermark("settle_time", "20 minutes").alias("s"),
#       F.expr("""a.txn_id = s.txn_id AND
#                 s.settle_time >= a.auth_time AND
#                 s.settle_time <= a.auth_time + interval 1 hour"""),
#       "leftOuter")
#
# The time constraint is what lets Spark drop state once the window has passed.
# Without it the join must remember every row forever.
#
# NOT SUPPORTED in Spark Structured Streaming: TEMPORAL / versioned joins
# ("enrich with the dimension value AS IT WAS at event time"). That is an
# ordinary business requirement with no clean Spark answer — Flink has
# FOR SYSTEM_TIME AS OF. If you need it, that alone can decide the engine.

spark.stop()

# ---------------------------------------------------------------------------
# TAKEAWAYS
# ---------------------------------------------------------------------------
# 1. Schema is mandatory for streaming sources.
# 2. Rate-limit (maxFilesPerTrigger / maxOffsetsPerTrigger) or a backlog OOMs you.
# 3. Window on EVENT time, carried inside the payload.
# 4. Watermark bounds state. Too long = OOM in weeks; too short = dropped data.
# 5. Sliding windows multiply state by windowSize/slideSize.
# 6. append + watermark is the only correct AND bounded combination.
# 7. checkpointLocation is not optional; many code changes invalidate it.
# 8. trigger(availableNow=True) replaces scheduled batch jobs on file sources.
# 9. Monitor stateRows. Monotonic growth = a job that dies later.
# 10. Exactly-once needs a replayable source AND an idempotent sink. Use batch_id.
# 11. Kafka partitions cap parallelism; Spark ignores Kafka consumer groups.
