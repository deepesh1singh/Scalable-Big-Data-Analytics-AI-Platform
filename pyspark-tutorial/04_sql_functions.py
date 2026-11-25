"""
04 — Built-in SQL functions: strings, dates, nulls, conditionals
===============================================================

REAL-WORLD SCENARIO
-------------------
A CRM export arrives from a system nobody maintains. Names have inconsistent
casing and stray whitespace, emails are dirty, phone numbers use four different
formats, dates arrive as three different string formats, and "missing" is
encoded as null, empty string, "N/A", and "NULL" depending on which decade the
row was written in.

You must produce a clean, deduplicated customer table.

Spark ships 400+ built-in functions. They stay inside JVM whole-stage codegen,
so they are 10-50x faster than the Python UDF most people reach for. Learning
what exists is the highest-leverage PySpark skill there is.

Run:
    spark-submit 04_sql_functions.py
"""

from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.appName("sql-functions")
         .master("local[4]").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

rows = [
    ("C-001", "  john SMITH ", "JOHN.SMITH@Example.COM ", "(555) 123-4567", "2021-03-15",  "gold",  "1250.50"),
    ("C-002", "jane doe",      "jane@test.org",           "555.987.6543",   "15/04/2022",  "SILVER", "890.00"),
    ("C-003", "BOB o'brien",   "bob@@bad-email",          "+1-555-222-3333","2023-01-09",  "gold",  "N/A"),
    ("C-004", "alice  wong",   "",                        "5551119999",     "2020-11-30",  None,     "3400.75"),
    ("C-005", "N/A",           "carlos@example.com",      "",               "invalid-date","bronze", "0"),
    ("C-006", "Émile Zolá",    "emile@example.fr",        "555 444 8888",   "2024-06-01",  "Gold",   "77.25"),
]

df = spark.createDataFrame(rows, """
    customer_id STRING, full_name STRING, email STRING,
    phone STRING, signup_date STRING, tier STRING, lifetime_value STRING""")

df.show(truncate=False)

# ---------------------------------------------------------------------------
# Section 1 — Normalise "missing": four encodings, one representation
# ---------------------------------------------------------------------------

MISSING = ["", " ", "N/A", "n/a", "NULL", "null", "-", "unknown"]

def blank_to_null(c):
    """Map every 'missing' sentinel to a real NULL."""
    return F.when(F.trim(F.col(c)).isin(MISSING) | F.col(c).isNull(),
                  None).otherwise(F.trim(F.col(c)))

std = df.select(*[blank_to_null(c).alias(c) for c in df.columns])
std.show(truncate=False)

# EXPECTED: "N/A" and "" become real nulls.
#
# WHY THIS MUST BE STEP ONE: "" and "N/A" are TRUTHY strings. They pass
# isNotNull(), they form their own GROUP BY bucket, and they break every
# coalesce(). Until missing values are actually NULL, no null handling works.
#
# The list comprehension applies one expression per column in a SINGLE
# select — never loop with withColumn (see 02 Section 7).

# ---------------------------------------------------------------------------
# Section 2 — Strings
# ---------------------------------------------------------------------------

names = (std
    .withColumn("name_clean", F.initcap(F.lower(F.trim("full_name"))))
    .withColumn("first_name", F.split("name_clean", " ").getItem(0))
    .withColumn("last_name",  F.element_at(F.split("name_clean", " "), -1))
    .withColumn("initials",
        F.concat_ws(".",
            F.substring(F.col("first_name"), 1, 1),
            F.substring(F.col("last_name"), 1, 1)))
    .withColumn("name_len", F.length("name_clean"))
    .withColumn("name_ascii", F.regexp_replace("name_clean", r"[^\x00-\x7F]", "?"))
    .withColumn("padded_id", F.lpad("customer_id", 10, "0")))

names.select("full_name", "name_clean", "first_name", "last_name",
             "initials", "name_ascii").show(truncate=False)

# EXPECTED:
#   "  john SMITH " -> "John Smith", first "John", last "Smith", initials "J.S"
#   "Émile Zolá"    -> name_ascii "?mile Zol?"
#
# NOTE element_at(array, -1) — NEGATIVE index counts from the end, and
# element_at is 1-INDEXED. getItem(0) is 0-indexed. Two accessors, two
# conventions, in the same API. Mixing them up is a classic off-by-one.
#
# STRING FUNCTIONS WORTH MEMORISING:
#   trim/ltrim/rtrim, lower/upper/initcap
#   concat, concat_ws (ws SKIPS nulls; concat returns NULL if ANY arg is null)
#   substring, left, right, lpad, rpad, repeat, reverse
#   split, element_at, array_join
#   regexp_replace, regexp_extract, regexp_extract_all, rlike
#   translate (char-by-char), overlay, locate, instr
#   levenshtein, soundex  <- fuzzy matching without a UDF
#
# >>> concat vs concat_ws IS A REAL BUG SOURCE <<<
#   concat("a", null, "b")            -> NULL       (whole thing)
#   concat_ws("-", "a", null, "b")    -> "a-b"      (null skipped)
# Use concat_ws whenever any input might be null.

# ---------------------------------------------------------------------------
# Section 3 — Regex: validate and extract
# ---------------------------------------------------------------------------

valid = (names
    .withColumn("email_clean", F.lower(F.trim("email")))
    .withColumn("email_valid",
        F.col("email_clean").rlike(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"))
    .withColumn("email_domain", F.regexp_extract("email_clean", r"@(.+)$", 1))
    # phone: strip everything that is not a digit, then take the last 10
    .withColumn("phone_digits", F.regexp_replace("phone", r"\D", ""))
    .withColumn("phone_e164",
        F.when(F.length("phone_digits") >= 10,
               F.concat(F.lit("+1"), F.substring(F.col("phone_digits"), -10, 10)))
         .otherwise(None)))

valid.select("email_clean", "email_valid", "email_domain",
             "phone", "phone_e164").show(truncate=False)

# EXPECTED:
#   "bob@@bad-email"     -> email_valid false
#   "(555) 123-4567"     -> "+15551234567"
#   "555.987.6543"       -> "+15559876543"
#   "5551119999"         -> "+15551119999"
#
# ONE regexp_replace normalises four phone formats. The Python-UDF version of
# this is ~30x slower and breaks whole-stage codegen.
#
# regexp_extract(col, pattern, group) returns "" (empty string) when there is
# no match — NOT null. Wrap in blank_to_null() if you need a real null.

# ---------------------------------------------------------------------------
# Section 4 — Dates: multiple input formats
# ---------------------------------------------------------------------------

dated = (valid
    .withColumn("signup",
        F.coalesce(
            F.to_date("signup_date", "yyyy-MM-dd"),
            F.to_date("signup_date", "dd/MM/yyyy"),
            F.to_date("signup_date", "MM/dd/yyyy")))
    .withColumn("signup_year",    F.year("signup"))
    .withColumn("signup_month",   F.date_format("signup", "MMMM"))
    .withColumn("signup_dow",     F.date_format("signup", "EEEE"))
    .withColumn("days_active",    F.datediff(F.current_date(), F.col("signup")))
    .withColumn("months_active",  F.round(F.months_between(F.current_date(), F.col("signup")), 1))
    .withColumn("anniversary",    F.add_months(F.col("signup"), 12))
    .withColumn("quarter_start",  F.trunc("signup", "quarter"))
    .withColumn("is_weekend",     F.dayofweek("signup").isin(1, 7)))

dated.select("signup_date", "signup", "signup_dow",
             "days_active", "quarter_start").show(truncate=False)

# EXPECTED:
#   "2021-03-15"   -> 2021-03-15, Monday
#   "15/04/2022"   -> 2022-04-15   (second format matched)
#   "invalid-date" -> null          (no format matched)
#
# >>> THE coalesce-OF-FORMATS PATTERN <<<
# to_date returns NULL when the format does not match, so coalesce tries each
# in turn and takes the first success. This is the standard way to handle a
# feed with inconsistent date formats. ORDER MATTERS: put the most specific
# and most common format first, and beware that "03/04/2022" parses under BOTH
# dd/MM and MM/dd with different meanings.
#
# SPARK 3 STRICT PARSING: legacy Spark silently coerced bad dates. Spark 3+
# raises unless you set spark.sql.legacy.timeParserPolicy=LEGACY. Do not set
# it — fix the formats instead.
#
# DATE FUNCTIONS: current_date, current_timestamp, to_date, to_timestamp,
# date_add/date_sub, datediff, months_between, add_months, next_day, last_day,
# trunc (date), date_trunc (timestamp), year/month/day/hour/minute/second,
# weekofyear, dayofweek/dayofyear, date_format, unix_timestamp, from_unixtime,
# from_utc_timestamp / to_utc_timestamp.

# ---------------------------------------------------------------------------
# Section 5 — Nulls and conditionals
# ---------------------------------------------------------------------------

final = (dated
    .withColumn("ltv",
        F.coalesce(F.col("lifetime_value").cast("double"), F.lit(0.0)))
    .withColumn("tier_clean",
        F.coalesce(F.lower(F.col("tier")), F.lit("unassigned")))
    .withColumn("segment",
        F.when(F.col("ltv") >= 3000, "vip")
         .when(F.col("ltv") >= 1000, "high")
         .when(F.col("ltv") >= 100,  "medium")
         .when(F.col("ltv") > 0,     "low")
         .otherwise("dormant"))
    .withColumn("contactable",
        F.col("email_valid") & F.col("phone_e164").isNotNull())
    .withColumn("data_quality_score",
        (F.col("name_clean").isNotNull().cast("int") +
         F.col("email_valid").cast("int") +
         F.col("phone_e164").isNotNull().cast("int") +
         F.col("signup").isNotNull().cast("int")) / 4.0))

final.select("customer_id", "ltv", "tier_clean", "segment",
             "contactable", "data_quality_score").show()

# EXPECTED:
#   C-003 ltv 0.0 (was "N/A"), segment dormant
#   C-004 tier_clean "unassigned"
#   data_quality_score between 0.25 and 1.00
#
# NULL SEMANTICS — the rules that catch everyone:
#   null == null          -> NULL, not true.  Use isNull() / eqNullSafe().
#   null + 1              -> NULL
#   count(col)            SKIPS nulls; count(*) does not
#   sum/avg               SKIP nulls (so avg is over non-nulls only)
#   NOT (x = 1)           does NOT match rows where x is NULL
#   'a' IN (null)         -> NULL
#
#   df1.join(df2, df1.k == df2.k)              drops null keys
#   df1.join(df2, df1.k.eqNullSafe(df2.k))     treats null == null as a match
#
# NULL FUNCTIONS: coalesce, nvl, nanvl, ifnull, nullif, isnull, isnotnull,
# na.fill / na.drop / na.replace, eqNullSafe.
#
# THE cast("double") ON "N/A": Spark returns NULL for an unparseable cast
# rather than raising, which is why coalesce catches it. Under ANSI mode
# (default on Databricks) some casts DO raise — behaviour differs, so test.

# ---------------------------------------------------------------------------
# Section 6 — Higher-order functions (no UDF needed)
# ---------------------------------------------------------------------------

arr = spark.createDataFrame(
    [("A", [3, 1, 4, 1, 5, 9]), ("B", [2, 7, 1]), ("C", [])],
    "id STRING, vals ARRAY<INT>")

(arr
 .withColumn("doubled",   F.transform("vals", lambda x: x * 2))
 .withColumn("evens",     F.filter("vals", lambda x: x % 2 == 0))
 .withColumn("total",     F.aggregate("vals", F.lit(0), lambda acc, x: acc + x))
 .withColumn("has_big",   F.exists("vals", lambda x: x > 5))
 .withColumn("all_pos",   F.forall("vals", lambda x: x > 0))
 .withColumn("sorted",    F.sort_array("vals"))
 .withColumn("uniq",      F.array_distinct("vals"))
 .withColumn("n",         F.size("vals"))
 .show(truncate=False))

# EXPECTED:
#   A: doubled [6,2,8,2,10,18], evens [4], total 23, has_big true, uniq [3,1,4,5,9]
#   C: empty array -> total 0, has_big false, all_pos TRUE (vacuous), n 0
#
# NOTE forall on an EMPTY array returns TRUE — vacuous truth. Mathematically
# correct, frequently surprising, and a real source of bugs when the rule is
# "all items must pass" and the array is empty.
#
# transform / filter / aggregate / exists / forall / zip_with are HIGHER-ORDER
# FUNCTIONS: they take lambdas but execute entirely in the JVM. They replace
# most array-processing UDFs at a fraction of the cost. Most people do not know
# they exist and write a UDF instead.

# ---------------------------------------------------------------------------
# Section 7 — SQL is the same engine
# ---------------------------------------------------------------------------

final.createOrReplaceTempView("customers")

spark.sql("""
    SELECT segment,
           count(*)                              AS n,
           round(avg(ltv), 2)                    AS avg_ltv,
           round(avg(data_quality_score), 2)     AS avg_dq,
           sum(CASE WHEN contactable THEN 1 ELSE 0 END) AS contactable
    FROM customers
    GROUP BY segment
    ORDER BY avg_ltv DESC
""").show()

# DataFrame API and SQL compile to the SAME logical plan and are optimised
# identically. Choose per readability: SQL for set-oriented aggregation,
# DataFrame API for programmatic column generation and reuse.

spark.stop()

# ---------------------------------------------------------------------------
# TAKEAWAYS
# ---------------------------------------------------------------------------
# 1. Normalise "", "N/A", "NULL" to real NULLs FIRST. Nothing else works until.
# 2. concat_ws skips nulls; concat propagates them.
# 3. element_at is 1-indexed and accepts negatives; getItem is 0-indexed.
# 4. regexp_extract returns "" on no match, not null.
# 5. coalesce(to_date(fmt1), to_date(fmt2), ...) handles mixed date formats.
# 6. null == null is NULL. Use eqNullSafe for null-matching joins.
# 7. Higher-order functions replace most array UDFs and stay in the JVM.
# 8. forall on an empty array is TRUE.
# 9. Search the function list before writing a UDF. It is almost always there.
