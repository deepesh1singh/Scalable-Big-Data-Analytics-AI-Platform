"""
07 — Window functions
=====================

REAL-WORLD SCENARIO
-------------------
Retail banking. From a raw transaction ledger you must produce: a running
balance per account, each customer's top 3 largest transactions, month-over-
month change, a 7-day rolling average for anomaly detection, gaps between
transactions to detect dormancy, and each transaction's percentile within its
account.

Every one of those is impossible with groupBy alone, and each is one line with
a window.

Run:
    spark-submit 07_window_functions.py
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = (SparkSession.builder.appName("window-functions")
         .master("local[4]")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

txn = spark.createDataFrame([
    ("T-01", "A-100", "2026-01-05",  2500.00, "deposit"),
    ("T-02", "A-100", "2026-01-12",  -320.50, "purchase"),
    ("T-03", "A-100", "2026-01-28", -1200.00, "transfer"),
    ("T-04", "A-100", "2026-02-14",  3000.00, "deposit"),
    ("T-05", "A-100", "2026-02-15",   -89.99, "purchase"),
    ("T-06", "A-200", "2026-01-08",  5000.00, "deposit"),
    ("T-07", "A-200", "2026-01-09", -2500.00, "transfer"),
    ("T-08", "A-200", "2026-03-01",  -150.00, "purchase"),
    ("T-09", "A-200", "2026-03-02",  -150.00, "purchase"),
    ("T-10", "A-300", "2026-02-20",   750.00, "deposit"),
    ("T-11", "A-300", "2026-02-21",  -750.00, "transfer"),
], "txn_id STRING, account_id STRING, txn_date STRING, amount DOUBLE, txn_type STRING")

txn = txn.withColumn("txn_date", F.to_date("txn_date"))

# ---------------------------------------------------------------------------
# Section 1 — The anatomy of a window
# ---------------------------------------------------------------------------
#
#   Window.partitionBy(cols)   WHICH rows share a window   -> this is a SHUFFLE
#         .orderBy(cols)       ORDER within the window     -> this is a SORT
#         .rowsBetween(a, b)   FRAME by row position
#      or .rangeBetween(a, b)  FRAME by ordering-column VALUE
#
# A window spec is a WIDE operation dressed up as a column expression. In the
# plan you will see:
#     Window [...]
#     +- Sort [account_id, txn_date]
#        +- Exchange hashpartitioning(account_id, 8)
#
# Nothing about the syntax hints at a shuffle. That is the main thing to know.

w_acct = Window.partitionBy("account_id").orderBy("txn_date")

# ---------------------------------------------------------------------------
# Section 2 — Ranking: row_number vs rank vs dense_rank
# ---------------------------------------------------------------------------

w_amt = Window.partitionBy("account_id").orderBy(F.desc("amount"))

ranked = (txn
    .withColumn("row_number",  F.row_number().over(w_amt))
    .withColumn("rank",        F.rank().over(w_amt))
    .withColumn("dense_rank",  F.dense_rank().over(w_amt))
    .withColumn("pct_rank",    F.round(F.percent_rank().over(w_amt), 3))
    .withColumn("ntile_4",     F.ntile(4).over(w_amt)))

ranked.select("account_id", "txn_id", "amount", "row_number",
              "rank", "dense_rank", "pct_rank").orderBy("account_id", "row_number").show()

# EXPECTED for A-200, where T-08 and T-09 both equal -150.00:
#   amount   row_number  rank  dense_rank
#   5000.00      1        1        1
#   -150.00      2        2        2
#   -150.00      3        2        2      <-- TIE
#   -2500.00     4        4        3
#
# >>> THE DIFFERENCE IS A CORRECTNESS ISSUE, NOT A STYLE CHOICE <<<
#   row_number  1,2,3,4   strictly sequential, ties broken ARBITRARILY
#   rank        1,2,2,4   ties share, gaps appear
#   dense_rank  1,2,2,3   ties share, no gaps
#
#   filter(row_number <= 3)  -> EXACTLY 3 rows
#   filter(rank <= 3)        -> 3 OR MORE rows if there are ties
#   filter(dense_rank <= 3)  -> the top 3 DISTINCT VALUES, any number of rows
#
# "Top 3 per group" almost always means row_number. Using rank silently
# returns more rows than expected, and nothing errors.
#
# row_number with ties is NONDETERMINISTIC unless the orderBy fully
# disambiguates. Add a tiebreaker: .orderBy(F.desc("amount"), "txn_id").
#
# ntile(n) splits into n roughly-equal buckets — quartiles, deciles, cohorts.

print("top 2 transactions per account:")
ranked.filter(F.col("row_number") <= 2) \
      .select("account_id", "txn_id", "amount").show()

# ---------------------------------------------------------------------------
# Section 3 — Running totals: the default frame
# ---------------------------------------------------------------------------

running = (txn
    .withColumn("balance",     F.round(F.sum("amount").over(w_acct), 2))
    .withColumn("txn_seq",     F.row_number().over(w_acct))
    .withColumn("cum_count",   F.count("*").over(w_acct))
    .withColumn("running_max", F.max("amount").over(w_acct)))

running.select("account_id", "txn_date", "amount", "balance", "txn_seq") \
       .orderBy("account_id", "txn_date").show()

# EXPECTED for A-100:
#   2026-01-05  2500.00  ->  2500.00
#   2026-01-12  -320.50  ->  2179.50
#   2026-01-28 -1200.00  ->   979.50
#   2026-02-14  3000.00  ->  3979.50
#   2026-02-15   -89.99  ->  3889.51
#
# >>> THE IMPLICIT DEFAULT FRAME <<<
# With orderBy and NO explicit frame, the default is
#     rangeBetween(unboundedPreceding, currentRow)
# which is why sum() gives a RUNNING total rather than the group total.
#
# With NO orderBy, the default frame is the ENTIRE PARTITION, so sum() gives
# the group total instead. Same function, completely different answer.
#
# Most "why is my window result wrong" bugs are this default.
#
# For the group total alongside each row, omit orderBy deliberately:
w_total = Window.partitionBy("account_id")
(txn.withColumn("account_total", F.round(F.sum("amount").over(w_total), 2))
    .withColumn("pct_of_account",
        F.round(100 * F.col("amount") / F.sum(F.abs("amount")).over(w_total), 1))
    .select("account_id", "txn_id", "amount", "account_total", "pct_of_account")
    .orderBy("account_id").show())

# Comparing a row to its GROUP's aggregate with NO self-join is the single
# biggest reason window functions exist.

# ---------------------------------------------------------------------------
# Section 4 — lag / lead: period-over-period and gap detection
# ---------------------------------------------------------------------------

deltas = (txn
    .withColumn("prev_amount", F.lag("amount", 1).over(w_acct))
    .withColumn("next_amount", F.lead("amount", 1).over(w_acct))
    .withColumn("prev_date",   F.lag("txn_date", 1).over(w_acct))
    .withColumn("days_since_prev", F.datediff("txn_date", F.lag("txn_date", 1).over(w_acct)))
    .withColumn("delta", F.round(F.col("amount") - F.lag("amount", 1, 0.0).over(w_acct), 2))
    .withColumn("pct_change",
        F.round(100 * (F.col("amount") - F.lag("amount").over(w_acct))
                / F.abs(F.lag("amount").over(w_acct)), 1))
    .withColumn("is_dormant_gap", F.col("days_since_prev") > 30))

deltas.select("account_id", "txn_date", "amount", "prev_amount",
              "days_since_prev", "is_dormant_gap") \
      .orderBy("account_id", "txn_date").show()

# EXPECTED: first row of each account has NULL prev_amount and NULL gap.
#   A-200 jumps 2026-01-09 -> 2026-03-01 = 51 days -> is_dormant_gap true
#
# lag(col, offset, default) — the THIRD argument fills the boundary instead of
# NULL. lag("amount", 1, 0.0) makes the first delta equal the amount rather
# than NULL. Choose deliberately: 0 is right for a delta, wrong for a
# percentage change (it would produce a division by zero -> NULL anyway).
#
# THE DORMANCY PATTERN — gaps between events — is the standard way to build
# sessions, detect churn, or find missing days in a feed. It replaces a
# self-join to the "previous row", which on 100M rows is a very different
# cost.

# ---------------------------------------------------------------------------
# Section 5 — Frames: rowsBetween vs rangeBetween
# ---------------------------------------------------------------------------

w_rows3 = Window.partitionBy("account_id").orderBy("txn_date").rowsBetween(-2, 0)
w_days7 = (Window.partitionBy("account_id")
                 .orderBy(F.unix_timestamp("txn_date"))
                 .rangeBetween(-7 * 86400, 0))
w_center = Window.partitionBy("account_id").orderBy("txn_date").rowsBetween(-1, 1)

frames = (txn
    .withColumn("ma_3_txns",  F.round(F.avg("amount").over(w_rows3), 2))
    .withColumn("sum_7_days", F.round(F.sum("amount").over(w_days7), 2))
    .withColumn("n_7_days",   F.count("*").over(w_days7))
    .withColumn("centered_avg", F.round(F.avg("amount").over(w_center), 2)))

frames.select("account_id", "txn_date", "amount",
              "ma_3_txns", "sum_7_days", "n_7_days") \
      .orderBy("account_id", "txn_date").show()

# >>> THE DISTINCTION THAT TRIPS EVERYONE <<<
#
#   rowsBetween(-2, 0)     counts ROWS. Position-based. Always this row plus
#                          the 2 physically preceding rows, whatever their
#                          dates. Gives exactly 3 rows (fewer at the start).
#
#   rangeBetween(-604800, 0)
#                          counts VALUES OF THE ORDERING COLUMN. Because we
#                          ordered by unix_timestamp (SECONDS), 7 days is
#                          7*86400 = 604,800. Rows outside that VALUE range are
#                          excluded even if physically adjacent.
#                          If the ordering column were milliseconds the number
#                          changes by 1000x. Range frames are unit-sensitive.
#
# For A-200, T-08 (2026-03-01) and T-09 (2026-03-02) are within 7 days of each
# other but ~51 days from T-07 — so sum_7_days excludes T-07 while a
# rowsBetween(-2,0) frame would include it. Different questions, different
# answers.
#
# FRAME BOUNDARY CONSTANTS:
#   Window.unboundedPreceding / unboundedFollowing / currentRow
#   rowsBetween(Window.unboundedPreceding, Window.currentRow)   = running total
#   rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing) = whole partition
#
# rangeBetween requires a SINGLE numeric/date ordering column. It errors on
# multi-column orderBy.

# ---------------------------------------------------------------------------
# Section 6 — first_value / last_value, and the frame gotcha
# ---------------------------------------------------------------------------

w_full = (Window.partitionBy("account_id").orderBy("txn_date")
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))

(txn
 .withColumn("first_txn_amt", F.first("amount").over(w_full))
 .withColumn("last_txn_amt",  F.last("amount").over(w_full))
 .withColumn("first_date",    F.first("txn_date").over(w_full))
 .withColumn("last_date",     F.last("txn_date").over(w_full))
 .select("account_id", "txn_date", "amount", "first_txn_amt", "last_txn_amt")
 .orderBy("account_id", "txn_date").show())

# >>> THE last() TRAP <<<
# With the DEFAULT frame (unboundedPreceding -> currentRow), last() returns the
# CURRENT ROW, not the last row of the partition — because the frame ends at
# the current row. People expect the partition's final value and get a copy of
# the current one.
#
# You MUST specify rowsBetween(unboundedPreceding, unboundedFollowing) for
# last() to mean what you think. first() happens to work with the default
# frame, which makes the asymmetry even more confusing.
#
# Also note: first()/last() with ignorenulls=True skip nulls —
#     F.last("amount", ignorenulls=True)
# That is the standard "forward fill / last known value" pattern for filling
# gaps in a time series.

# ---------------------------------------------------------------------------
# Section 7 — Multiple windows and Exchange reuse
# ---------------------------------------------------------------------------

w_a = Window.partitionBy("account_id").orderBy("txn_date")
w_b = Window.partitionBy("account_id").orderBy(F.desc("amount"))
w_c = Window.partitionBy("account_id")

multi = (txn
    .withColumn("seq",     F.row_number().over(w_a))
    .withColumn("balance", F.round(F.sum("amount").over(w_a), 2))
    .withColumn("amt_rank", F.rank().over(w_b))
    .withColumn("acct_avg", F.round(F.avg("amount").over(w_c), 2)))

multi.explain("formatted")

# LOOK AT THE PLAN: all three specs partition by account_id, so Spark reuses
# ONE Exchange hashpartitioning(account_id) and stacks Window operators above
# it. Two different SORT orders still cost two sorts, but only one shuffle.
#
# If the specs partitioned by DIFFERENT columns you would get one Exchange
# each. >>> Reusing the partitioning key across windows is a real and
# underused optimisation. <<<
#
# Practical rule: define window specs so they share partitionBy wherever the
# business logic allows.

# ---------------------------------------------------------------------------
# Section 8 — The unbounded-window trap
# ---------------------------------------------------------------------------

# DO NOT RUN ON LARGE DATA:
#   w_bad = Window.orderBy("txn_date")            # NO partitionBy
#   txn.withColumn("global_running", F.sum("amount").over(w_bad))
#
# No partitionBy means ALL rows must land in a SINGLE partition. The plan shows
# `Exchange SinglePartition`: one task, one core, the whole dataset, and almost
# certainly an OOM or a multi-hour stall.
#
# Spark even logs:
#   "No Partition Defined for Window operation! Moving all data to a single
#    partition, this can cause serious performance degradation."
# A rare case of Spark warning you outright. ALWAYS partition a window.
#
# If you genuinely need a global running total, either (a) pre-aggregate to a
# small enough result first, or (b) compute per-partition totals and combine
# them with a second, tiny window.

# ---------------------------------------------------------------------------
# Section 9 — Filters cannot be pushed through a window
# ---------------------------------------------------------------------------

top_n = (txn
    .withColumn("rn", F.row_number().over(Window.partitionBy("account_id")
                                                .orderBy(F.desc("amount"))))
    .filter(F.col("rn") <= 2))
top_n.explain("formatted")

# The Filter sits ABOVE the Window in the plan and CANNOT move below it — the
# window must be computed before rn exists. This is a case where Catalyst
# genuinely cannot help you and you must order the code yourself:
# filter to the relevant accounts/date range BEFORE the window, not after.

# ---------------------------------------------------------------------------
# Section 10 — Sessionisation: the capstone pattern
# ---------------------------------------------------------------------------

SESSION_GAP_DAYS = 30

sessions = (txn
    .withColumn("prev_date", F.lag("txn_date").over(w_acct))
    .withColumn("gap_days", F.datediff("txn_date", "prev_date"))
    .withColumn("is_new_session",
        F.when(F.col("prev_date").isNull(), 1)
         .when(F.col("gap_days") > SESSION_GAP_DAYS, 1)
         .otherwise(0))
    .withColumn("session_num", F.sum("is_new_session").over(w_acct))
    .withColumn("session_id",
        F.concat_ws("-", "account_id", F.col("session_num").cast("string"))))

sessions.select("account_id", "txn_date", "gap_days",
                "is_new_session", "session_id") \
        .orderBy("account_id", "txn_date").show()

(sessions.groupBy("session_id")
    .agg(F.min("txn_date").alias("start"),
         F.max("txn_date").alias("end"),
         F.count("*").alias("n_txns"),
         F.round(F.sum("amount"), 2).alias("net"))
    .orderBy("session_id").show())

# EXPECTED: A-200 splits into two sessions because of the 51-day gap.
#
# >>> THE lag -> flag -> CUMULATIVE SUM PATTERN <<<
# This is THE canonical sessionisation technique and it appears constantly:
#   1. lag() to see the previous event
#   2. flag 1 when the gap exceeds a threshold (or any "new group" condition)
#   3. running sum() of the flag becomes a monotonically increasing group id
#
# It generalises far beyond sessions: detect state changes, number contiguous
# runs, split a stream into batches, find gaps-and-islands. Learn it once.
#
# Note that step 3 uses the DEFAULT running frame from Section 3 — which is
# exactly why understanding the default frame matters.

spark.stop()

# ---------------------------------------------------------------------------
# TAKEAWAYS
# ---------------------------------------------------------------------------
# 1. A window spec is a shuffle plus a sort. The syntax hides that.
# 2. row_number for "top N per group". rank returns more rows on ties.
# 3. Default frame WITH orderBy = running total; WITHOUT orderBy = whole partition.
# 4. last() needs an explicit unboundedFollowing frame or it returns the current row.
# 5. rowsBetween counts rows; rangeBetween counts ordering-column VALUES.
# 6. lag(col, 1, default) fills the boundary instead of NULL.
# 7. Reuse partitionBy across specs to reuse one Exchange.
# 8. No partitionBy = Exchange SinglePartition = one task for everything.
# 9. Filters cannot push below a window. Filter before, not after.
# 10. lag -> flag -> cumulative sum is the sessionisation pattern. Memorise it.
