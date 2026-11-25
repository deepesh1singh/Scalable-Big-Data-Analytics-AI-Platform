# ClickHouse DDL / DML Reference

Verified against ClickHouse **26.8.1.1028**.

ClickHouse is SQL-shaped but not a row store. The places it differs from
Postgres/MySQL are exactly where people get hurt, and they are flagged
throughout with **⚠**.

---

## 1. Databases

```sql
CREATE DATABASE IF NOT EXISTS analytics;
CREATE DATABASE analytics ENGINE = Atomic;          -- default
CREATE DATABASE mysql_db ENGINE = MySQL('host:3306', 'db', 'user', 'pass');
CREATE DATABASE pg_db    ENGINE = PostgreSQL('host:5432', 'db', 'user', 'pass');

-- Data lake catalogs (the ones that matter for lakehouse work)
CREATE DATABASE lake ENGINE = DataLakeCatalog
SETTINGS catalog_type = 'glue', region = 'us-east-1',
         aws_access_key_id = '...', aws_secret_access_key = '...';

CREATE DATABASE ice ENGINE = DataLakeCatalog
SETTINGS catalog_type = 'rest', catalog_url = 'http://host:8181/api/catalog';

SHOW DATABASES;
USE analytics;
DROP DATABASE IF EXISTS analytics;
DROP DATABASE analytics SYNC;        -- immediate, skips the drop delay
```

---

## 2. Tables

### Basic create

```sql
CREATE TABLE events
(
    event_id     UUID       DEFAULT generateUUIDv4(),
    company_id   UInt32,
    device_id    String,
    event_time   DateTime64(3),
    event_type   LowCardinality(String),
    speed        Float32,
    lat          Float64,
    lon          Float64,
    payload      String      CODEC(ZSTD(3)),
    ingested_at  DateTime    DEFAULT now(),
    event_date   Date        MATERIALIZED toDate(event_time),
    speed_kph    Float32     ALIAS speed * 1.60934
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (company_id, device_id, event_time)
PRIMARY KEY (company_id, device_id)
TTL event_time + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;
```

**⚠ `ORDER BY` is the single most important line.** It is the sparse primary
index *and* the physical sort order on disk. Lead with the column you filter on
in nearly every query — for multi-tenant serving that is the tenant id.
Get it wrong and no amount of hardware recovers it.

`PRIMARY KEY` may be a **prefix** of `ORDER BY`. Shorter primary key = smaller
index in memory, same sort order. Omit it and it equals `ORDER BY`.

### Column modifiers

| Modifier | Stored? | Meaning |
|---|---|---|
| `DEFAULT expr` | yes | Used when the column is omitted on insert |
| `MATERIALIZED expr` | yes | Always computed; **not returned by `SELECT *`** |
| `ALIAS expr` | no | Computed at query time, zero storage |
| `EPHEMERAL` | no | Insert-only helper, never stored |
| `CODEC(...)` | — | Per-column compression |

Useful codecs: `ZSTD(1..22)`, `LZ4` (default), `Delta` then `ZSTD` for
timestamps/counters, `DoubleDelta` for slowly-changing numerics, `Gorilla` for
floats, `T64` for small-range integers.

```sql
event_time DateTime CODEC(DoubleDelta, ZSTD(1)),
counter    UInt64   CODEC(Delta, ZSTD(1))
```

### Types worth knowing

```sql
LowCardinality(String)        -- dictionary-encoded; use under ~10k distinct values
Nullable(T)                   -- ⚠ costs a separate null mask; avoid unless needed
Enum8('a'=1,'b'=2)
Array(String), Map(String,String), Tuple(a UInt8, b String)
Nested(k String, v UInt32)    -- parallel arrays
Decimal(18,4)                 -- exact money
DateTime64(3, 'UTC')          -- millisecond precision
AggregateFunction(sum, UInt64)
SimpleAggregateFunction(max, UInt32)
IPv4, IPv6, UUID, JSON, Point, Polygon
```

**⚠ `Nullable` blocks some optimisations and cannot be in `ORDER BY` unless
`allow_nullable_key=1`.** Prefer a sentinel value where semantics allow.

### Create variants

```sql
CREATE TABLE t2 AS t1;                              -- same schema + engine
CREATE TABLE t2 AS t1 ENGINE = Memory;              -- same schema, new engine
CREATE TABLE t2 ENGINE = MergeTree ORDER BY id AS SELECT * FROM t1;   -- CTAS
CREATE TABLE t2 AS remote('host', 'db', 't1');      -- schema from remote
CREATE TABLE t2 AS s3('https://.../f.parquet');     -- schema inferred from file
CREATE OR REPLACE TABLE t (...) ENGINE = MergeTree ORDER BY id;
CREATE TEMPORARY TABLE tmp (id UInt32) ENGINE = Memory;   -- session-scoped
```

---

## 3. Table engines

### MergeTree family — the ones you actually use

| Engine | Behaviour |
|---|---|
| `MergeTree` | Plain. Append-only, no dedup. |
| `ReplacingMergeTree([ver])` | Keeps the last row per `ORDER BY` key. **Upsert.** |
| `SummingMergeTree([cols])` | Sums numeric columns per key on merge. |
| `AggregatingMergeTree` | Merges `AggregateFunction` states. Backs materialized views. |
| `CollapsingMergeTree(sign)` | `sign = 1/-1` rows cancel. |
| `VersionedCollapsingMergeTree(sign, ver)` | Same, order-independent. |
| `Replicated*MergeTree` | Any of the above, replicated via Keeper. |

```sql
CREATE TABLE devices_current
(
    device_id String,
    status    String,
    updated   DateTime,
    is_deleted UInt8 DEFAULT 0
)
ENGINE = ReplacingMergeTree(updated)
ORDER BY device_id;
```

**⚠ Deduplication is eventual, not immediate.** It happens during background
merges, at an unpredictable time. To read deduplicated results *now*:

```sql
SELECT * FROM devices_current FINAL;                -- correct, slower
SELECT * FROM devices_current ORDER BY device_id;   -- may show duplicates
SET final = 1;                                       -- apply FINAL session-wide
OPTIMIZE TABLE devices_current FINAL;               -- force a merge (expensive)
```

`FINAL` is the single biggest ClickHouse gotcha coming from an OLTP background.

### Integration engines

```sql
-- Kafka
CREATE TABLE kafka_events (raw String) ENGINE = Kafka
SETTINGS kafka_broker_list = 'localhost:9092',
         kafka_topic_list  = 'events',
         kafka_group_name  = 'ch_consumer',
         kafka_format      = 'JSONEachRow',
         kafka_num_consumers = 4;

-- Iceberg / Delta / S3
CREATE TABLE ice ENGINE = Iceberg('s3://bucket/path/', 'key', 'secret');
CREATE TABLE dl  ENGINE = DeltaLake('s3://bucket/path/', 'key', 'secret');
CREATE TABLE raw ENGINE = S3('https://bucket.s3.amazonaws.com/*.parquet',
                             'key', 'secret', 'Parquet');

-- Other
ENGINE = MySQL('host:3306','db','table','user','pass')
ENGINE = PostgreSQL('host:5432','db','table','user','pass')
ENGINE = Distributed(cluster, db, table[, sharding_key])
ENGINE = Dictionary(dict_name)
ENGINE = Buffer(db, table, 16, 10, 100, 10000, 1000000, 10000000, 100000000)
ENGINE = Memory | Log | TinyLog | Null | File(Parquet) | URL(url, format)
```

`Null` is genuinely useful: writes go nowhere, but materialized views attached
to it still fire. The standard pattern for "transform on ingest, keep nothing raw".

---

## 4. ALTER — DDL

```sql
-- Columns
ALTER TABLE t ADD COLUMN c UInt32 DEFAULT 0 AFTER b;
ALTER TABLE t ADD COLUMN IF NOT EXISTS c String FIRST;
ALTER TABLE t DROP COLUMN c;
ALTER TABLE t RENAME COLUMN old TO new;
ALTER TABLE t MODIFY COLUMN c String;                  -- ⚠ rewrites data
ALTER TABLE t MODIFY COLUMN c UInt32 DEFAULT 42;       -- metadata only
ALTER TABLE t MODIFY COLUMN c CODEC(ZSTD(3));
ALTER TABLE t COMMENT COLUMN c 'meaning';
ALTER TABLE t CLEAR COLUMN c IN PARTITION '202608';

-- ⚠ ORDER BY can only be EXTENDED, never changed or reordered.
ALTER TABLE t MODIFY ORDER BY (company_id, device_id, event_time);

-- TTL
ALTER TABLE t MODIFY TTL event_time + INTERVAL 30 DAY;
ALTER TABLE t REMOVE TTL;

-- Table
RENAME TABLE a TO b;
EXCHANGE TABLES a AND b;             -- atomic swap; the blue/green deploy trick
ALTER TABLE t MODIFY SETTING merge_with_ttl_timeout = 3600;
TRUNCATE TABLE t;
DROP TABLE IF EXISTS t SYNC;
DETACH TABLE t;  ATTACH TABLE t;
```

### Indexes and projections

```sql
-- Data-skipping indexes (secondary; they skip granules, they do not point to rows)
ALTER TABLE t ADD INDEX idx_type event_type TYPE set(100) GRANULARITY 4;
ALTER TABLE t ADD INDEX idx_speed speed TYPE minmax GRANULARITY 1;
ALTER TABLE t ADD INDEX idx_bloom device_id TYPE bloom_filter(0.01) GRANULARITY 1;
ALTER TABLE t ADD INDEX idx_tok payload TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 1;
ALTER TABLE t MATERIALIZE INDEX idx_type;     -- build it on existing data
ALTER TABLE t DROP INDEX idx_type;

-- Projections: an alternate sort order, maintained automatically
ALTER TABLE t ADD PROJECTION by_device
    (SELECT * ORDER BY device_id, event_time);
ALTER TABLE t MATERIALIZE PROJECTION by_device;
```

**Projections are the answer to "I need two different `ORDER BY`s".** The
optimiser picks the cheaper one per query.

### Partitions

```sql
ALTER TABLE t DROP PARTITION '202601';                -- instant, no rewrite
ALTER TABLE t DETACH PARTITION '202601';
ALTER TABLE t ATTACH PARTITION '202601';
ALTER TABLE t FREEZE PARTITION '202601';              -- hardlink snapshot
ALTER TABLE t MOVE PARTITION '202601' TO DISK 's3';
ALTER TABLE dst REPLACE PARTITION '202601' FROM src;  -- atomic swap-in
ALTER TABLE dst ATTACH PARTITION '202601' FROM src;

SELECT partition, rows, formatReadableSize(bytes_on_disk)
FROM system.parts WHERE table = 't' AND active;
```

**⚠ Do not over-partition.** Monthly or daily is normal. Partitioning by hour or
by a high-cardinality column creates thousands of parts and destroys performance.
Partitions are a data-management unit (drop old data instantly), **not** an index.

---

## 5. DML — inserting

```sql
INSERT INTO t VALUES (1, 'a'), (2, 'b');
INSERT INTO t (id, name) VALUES (1, 'a');
INSERT INTO t SELECT * FROM other WHERE dt >= '2026-01-01';
INSERT INTO t FORMAT JSONEachRow {"id":1,"name":"a"}
INSERT INTO FUNCTION s3('https://.../out.parquet','key','secret','Parquet')
    SELECT * FROM t;
```

From the shell:
```bash
clickhouse client -q "INSERT INTO t FORMAT Parquet"  < data.parquet
clickhouse client -q "INSERT INTO t FORMAT CSVWithNames" < data.csv
cat *.jsonl | clickhouse client -q "INSERT INTO t FORMAT JSONEachRow"
```

From files and URLs directly:
```sql
INSERT INTO t SELECT * FROM file('/path/data.parquet', Parquet);
INSERT INTO t SELECT * FROM url('https://host/data.csv', CSVWithNames);
INSERT INTO t SELECT * FROM s3('https://bucket/*.parquet','key','secret','Parquet');
```

**⚠ Insert in large batches.** Every `INSERT` creates a new *part*. Thousands of
tiny inserts create thousands of parts, background merges cannot keep up, and you
hit `TOO_MANY_PARTS`. Target 10k–1M rows per insert, or enable async inserts:

```sql
SET async_insert = 1, wait_for_async_insert = 1;
```

Idempotency: identical consecutive inserts into a Replicated table are
deduplicated by block checksum (`insert_deduplicate = 1`).

---

## 6. DML — updating and deleting

**⚠ This is where ClickHouse is least like a normal database.**

### Lightweight DELETE — the one to use

```sql
DELETE FROM t WHERE company_id = 42;
```
Marks rows deleted immediately; physical removal happens at merge time.
Cheap, and synchronous by default (`lightweight_deletes_sync = 2`).

### Mutations — heavyweight, asynchronous

```sql
ALTER TABLE t UPDATE status = 'archived' WHERE event_date < '2026-01-01';
ALTER TABLE t DELETE WHERE company_id = 42;
ALTER TABLE t UPDATE a = 1, b = 2 WHERE id IN (SELECT id FROM other);
```

**⚠ Mutations rewrite entire parts.** They are asynchronous, not transactional,
and there is no rollback. Updating one row can rewrite gigabytes.

```sql
-- watch progress
SELECT database, table, mutation_id, command, is_done, latest_fail_reason
FROM system.mutations WHERE NOT is_done;

KILL MUTATION WHERE mutation_id = '...';
SET mutations_sync = 1;    -- block until complete (2 = wait for all replicas)
```

### The idiomatic alternative

Do not update. **Insert a new version and let the engine collapse it:**

```sql
CREATE TABLE state (id String, status String, ver DateTime, deleted UInt8 DEFAULT 0)
ENGINE = ReplacingMergeTree(ver) ORDER BY id;

INSERT INTO state VALUES ('d1','active', now(), 0);
INSERT INTO state VALUES ('d1','retired', now(), 0);   -- "update"
INSERT INTO state VALUES ('d1','', now(), 1);          -- "delete"

SELECT * FROM state FINAL WHERE deleted = 0;
```

This is the pattern to use for CDC and for anything mutable.

### Experimental lightweight UPDATE (enabled in 26.8)

```sql
UPDATE t SET status = 'x' WHERE id = 1;
```
Newer, patch-part based. Test before relying on it in production.

---

## 7. Views

```sql
-- Normal view: stored query, no data
CREATE VIEW v AS SELECT * FROM t WHERE speed > 100;

-- ⚠ MATERIALIZED VIEW is an INSERT TRIGGER, not a cached result.
-- It fires on rows inserted AFTER creation. It does NOT see existing data.
CREATE MATERIALIZED VIEW mv_hourly
ENGINE = SummingMergeTree
ORDER BY (company_id, hour)
POPULATE                              -- backfill existing rows (see warning)
AS SELECT company_id,
          toStartOfHour(event_time) AS hour,
          count()      AS events,
          sum(speed)   AS speed_sum
   FROM events
   GROUP BY company_id, hour;

-- Explicit target table — the production pattern
CREATE TABLE hourly_agg (company_id UInt32, hour DateTime, events UInt64)
ENGINE = SummingMergeTree ORDER BY (company_id, hour);

CREATE MATERIALIZED VIEW mv TO hourly_agg AS
SELECT company_id, toStartOfHour(event_time) AS hour, count() AS events
FROM events GROUP BY company_id, hour;
```

**⚠ `POPULATE` misses rows inserted while it runs.** For production, create the
MV without it, then backfill manually with a bounded `INSERT ... SELECT`.

Prefer `TO existing_table` — it separates the trigger from the storage, so you
can drop and recreate the MV without losing data.

### AggregatingMergeTree pattern

```sql
CREATE TABLE agg (company_id UInt32, day Date,
                  uniq_devices AggregateFunction(uniq, String),
                  avg_speed    AggregateFunction(avg, Float32))
ENGINE = AggregatingMergeTree ORDER BY (company_id, day);

CREATE MATERIALIZED VIEW mv_agg TO agg AS
SELECT company_id, toDate(event_time) AS day,
       uniqState(device_id) AS uniq_devices,
       avgState(speed)      AS avg_speed
FROM events GROUP BY company_id, day;

-- read with -Merge
SELECT company_id, day, uniqMerge(uniq_devices), avgMerge(avg_speed)
FROM agg GROUP BY company_id, day;
```

`-State` writes, `-Merge` reads. Mixing them up returns binary garbage.

### Other view types

```sql
CREATE LIVE VIEW lv AS SELECT count() FROM t;         -- experimental
CREATE WINDOW VIEW wv ...;                            -- experimental
```

---

## 8. Dictionaries

```sql
CREATE DICTIONARY dim_company
(
    company_id UInt32,
    name       String,
    tier       String
)
PRIMARY KEY company_id
SOURCE(CLICKHOUSE(TABLE 'companies' DB 'analytics'))
LAYOUT(HASHED())
LIFETIME(MIN 300 MAX 600);

SELECT dictGet('dim_company', 'name', toUInt32(company_id)) AS name FROM events;
SYSTEM RELOAD DICTIONARY dim_company;
```

Layouts: `FLAT`, `HASHED`, `SPARSE_HASHED`, `COMPLEX_KEY_HASHED`, `CACHE`,
`RANGE_HASHED` (for time-versioned lookups), `DIRECT`, `IP_TRIE`.

**Dictionaries replace dimension joins.** `dictGet` on a hashed dictionary is a
memory lookup, dramatically faster than a join. This is the ClickHouse answer to
star-schema enrichment.

`RANGE_HASHED` gives you point-in-time lookups — the nearest thing ClickHouse has
to a temporal join.

---

## 9. SELECT — the ClickHouse-specific parts

```sql
SELECT ... FROM t FINAL                    -- force dedup/collapse at read time
SELECT ... PREWHERE cheap_filter           -- filter before reading other columns
SELECT ... SAMPLE 0.1                      -- requires SAMPLE BY in the DDL
SELECT ... LIMIT 3 BY company_id           -- top-N per group, no window needed
SELECT ... WITH TOTALS                     -- adds a totals row
SELECT ... WITH FILL FROM a TO b STEP c    -- gap filling in time series
SELECT ... ARRAY JOIN arr                  -- explode an array column
SELECT ... LEFT ARRAY JOIN arr             -- keeps empty arrays
SELECT ... GROUP BY a, b WITH ROLLUP / WITH CUBE / GROUPING SETS
SELECT ... SETTINGS max_threads = 8
SELECT ... FORMAT JSONEachRow
```

`LIMIT n BY expr` is a genuine differentiator — top-N per group without a window
function, and far cheaper.

```sql
-- Useful functions
arrayJoin, groupArray, groupUniqArray, arrayMap, arrayFilter, arrayReduce
uniq, uniqExact, uniqHLL12, quantile, quantiles, quantileTDigest
argMax(x, y), argMin(x, y)          -- the x with the largest/smallest y
any(x), anyLast(x)
runningDifference, neighbor, lagInFrame, leadInFrame
toStartOfInterval(t, INTERVAL 5 MINUTE)
sumIf(x, cond), countIf(cond), avgIf(x, cond)      -- -If combinator
sumMap, minMap, maxMap
JSONExtractString, JSONExtractInt, JSONExtract
```

The `-If` combinator is idiomatic and fast: `countIf(status = 'error')` in one
pass, no `CASE WHEN`.

---

## 10. Maintenance and introspection

```sql
OPTIMIZE TABLE t;                            -- merge parts
OPTIMIZE TABLE t FINAL;                      -- ⚠ expensive; forces full merge
OPTIMIZE TABLE t PARTITION '202608' FINAL;
OPTIMIZE TABLE t DEDUPLICATE BY id;

SYSTEM STOP MERGES t;  SYSTEM START MERGES t;
SYSTEM FLUSH LOGS;
SYSTEM RELOAD CONFIG;
SYSTEM DROP MARK CACHE;  SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM SYNC REPLICA t;
SYSTEM RESTART REPLICA t;

SHOW TABLES;  SHOW CREATE TABLE t;  DESCRIBE TABLE t;
EXPLAIN SELECT ...;                       -- logical plan
EXPLAIN PIPELINE SELECT ...;              -- execution pipeline
EXPLAIN ESTIMATE SELECT ...;              -- parts/rows/marks to be read
EXPLAIN indexes = 1 SELECT ...;           -- ⭐ shows index granule skipping
```

`EXPLAIN indexes = 1` is the ClickHouse equivalent of reading a Spark plan for
`PushedFilters`. It tells you whether your `ORDER BY` is actually being used.

### System tables worth memorising

```sql
system.parts            -- per-part size, rows, compression
system.columns          -- per-column compressed/uncompressed bytes
system.query_log        -- every query, duration, rows read, memory
system.mutations        -- running/failed mutations
system.merges           -- in-flight merges
system.replicas         -- replication lag and queue
system.settings, system.tables, system.databases
system.asynchronous_metrics, system.metrics, system.events

-- storage by table
SELECT table,
       formatReadableSize(sum(data_compressed_bytes))   AS compressed,
       formatReadableSize(sum(data_uncompressed_bytes)) AS uncompressed,
       round(sum(data_uncompressed_bytes)/sum(data_compressed_bytes),1) AS ratio,
       sum(rows) AS rows, count() AS parts
FROM system.parts WHERE active GROUP BY table ORDER BY sum(data_compressed_bytes) DESC;

-- slowest queries
SELECT query_duration_ms, read_rows, formatReadableSize(read_bytes),
       formatReadableSize(memory_usage), substring(query, 1, 120) AS q
FROM system.query_log
WHERE type = 'QueryFinish' AND event_time > now() - INTERVAL 1 HOUR
ORDER BY query_duration_ms DESC LIMIT 10;

-- which columns cost the most storage
SELECT name, formatReadableSize(sum(data_compressed_bytes)) AS sz
FROM system.columns WHERE table = 'events' GROUP BY name ORDER BY sum(data_compressed_bytes) DESC;
```

---

## 11. Users, roles, quotas

```sql
CREATE USER analyst IDENTIFIED WITH sha256_password BY 'secret';
CREATE USER app IDENTIFIED WITH sha256_password BY 'x' HOST IP '10.0.0.0/8';
ALTER USER default IDENTIFIED WITH sha256_password BY 'newpass';
DROP USER analyst;

CREATE ROLE reader;
GRANT SELECT ON analytics.* TO reader;
GRANT reader TO analyst;
SET DEFAULT ROLE reader TO analyst;

GRANT SELECT, INSERT ON analytics.events TO app;
GRANT SELECT(company_id, event_time) ON analytics.events TO analyst;   -- column level
REVOKE INSERT ON analytics.* FROM app;

-- Row-level security — the multi-tenant control
CREATE ROW POLICY tenant_filter ON analytics.events
    FOR SELECT USING company_id = toUInt32(getSetting('SQL_custom_tenant'))
    TO analyst;

CREATE QUOTA q FOR INTERVAL 1 HOUR MAX queries = 1000, read_rows = 1000000000 TO analyst;
CREATE SETTINGS PROFILE p SETTINGS max_memory_usage = 10000000000 TO analyst;

SHOW USERS;  SHOW GRANTS FOR analyst;  SHOW CREATE USER analyst;
SELECT * FROM system.users;
```

---

## 12. Patterns for a Kafka → ClickHouse pipeline

```sql
-- 1. Kafka engine table: a consumer, not storage. Reading it CONSUMES messages.
CREATE TABLE kafka_raw (raw String) ENGINE = Kafka
SETTINGS kafka_broker_list = 'localhost:9092',
         kafka_topic_list = 'events', kafka_group_name = 'ch',
         kafka_format = 'JSONAsString', kafka_num_consumers = 4;

-- 2. Real storage
CREATE TABLE events (company_id UInt32, device_id String,
                     event_time DateTime64(3), speed Float32)
ENGINE = MergeTree PARTITION BY toYYYYMM(event_time)
ORDER BY (company_id, device_id, event_time);

-- 3. MV moves and parses
CREATE MATERIALIZED VIEW mv_kafka TO events AS
SELECT JSONExtractUInt(raw, 'company_id')      AS company_id,
       JSONExtractString(raw, 'device_id')     AS device_id,
       toDateTime64(JSONExtractUInt(raw,'ts')/1000, 3) AS event_time,
       JSONExtractFloat(raw, 'speed')          AS speed
FROM kafka_raw;

-- 4. Rollups chain off the same table
CREATE TABLE hourly (company_id UInt32, hour DateTime, events UInt64, speed_sum Float64)
ENGINE = SummingMergeTree ORDER BY (company_id, hour);

CREATE MATERIALIZED VIEW mv_hourly TO hourly AS
SELECT company_id, toStartOfHour(event_time) AS hour,
       count() AS events, sum(speed) AS speed_sum
FROM events GROUP BY company_id, hour;
```

**⚠ Never `SELECT` from a Kafka engine table directly** — it consumes messages
and they are gone. Always read the MergeTree table behind the MV.

---

## 13. The gotchas, collected

| # | Trap |
|---|---|
| 1 | `ORDER BY` is the primary index. Lead with your most-filtered column. |
| 2 | `ReplacingMergeTree` dedup is eventual. Use `FINAL` to read correct results now. |
| 3 | Mutations (`ALTER UPDATE/DELETE`) rewrite whole parts, are async, and have no rollback. |
| 4 | Insert in large batches. Small inserts → part explosion → `TOO_MANY_PARTS`. |
| 5 | A materialized view is an **insert trigger**, not a cached result. It never sees existing rows. |
| 6 | `POPULATE` misses rows inserted while it runs. |
| 7 | Over-partitioning is far worse than no partitioning. Monthly/daily only. |
| 8 | `Nullable` costs a mask column and blocks optimisations. |
| 9 | `SELECT` on a Kafka engine table consumes the messages. |
| 10 | `OPTIMIZE FINAL` on a large table is very expensive. Not routine maintenance. |
| 11 | No transactions across statements. No foreign keys. No real constraints. |
| 12 | Joins are weak compared to a warehouse. Prefer dictionaries and denormalisation. |
| 13 | `-State` writes, `-Merge` reads. Mixing them returns binary garbage. |
| 14 | `EXPLAIN indexes = 1` is how you verify the index is actually used. |

---

## 14. Migration notes (S3 Tables / Iceberg → ClickHouse)

Two paths, and the choice is a benchmark, not an opinion.

**Option A — query the lake in place**
```sql
CREATE DATABASE lake ENGINE = DataLakeCatalog
SETTINGS catalog_type = 'glue', region = 'us-east-1';

SELECT company_id, count() FROM lake.refined_events
WHERE event_date >= today() - 7 GROUP BY company_id;
```
No copy, S3 stays the source of truth. But: cold S3 reads per query, no
ClickHouse sparse index, only Iceberg partition pruning, and concurrency
multiplies S3 requests.

**Option B — materialise into MergeTree**
```sql
CREATE TABLE serving_events (...)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_date)
ORDER BY (company_id, event_date, device_id);

INSERT INTO serving_events
SELECT * FROM lake.refined_events WHERE event_date >= today() - 90;
```
Duplicate storage and a sync job, but you get the index, the sort order, and
the compression that produce millisecond dashboard queries.

**Measured on a laptop, 2.96M rows, single core:**

| Operation | Time |
|---|---|
| Load 2.96M rows from Parquet | 0.72 s |
| `count()` | 0.001 s |
| Point lookup on `ORDER BY` prefix | 0.002 s |
| Full aggregation by hour | 0.018 s |
| Filter + group + p95 quantile | 0.021 s |
| Compression ratio | 4.5× |

Benchmark Option A at **real concurrency**, not single-query. A hybrid is often
correct: recent 90 days in MergeTree for serving, older data queried in place.
