-- =============================================================================
-- 01_load_and_clean.sql
-- Purpose: Load raw Kaggle CSVs into SQLite, verify data integrity, apply
--          scope filters (6 stores, 2015-2016), and create cleaned tables.
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  -- 64 MB cache

-- ---------------------------------------------------------------------------
-- Step 1: Create raw staging tables (mirror CSV structure exactly)
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS stg_train;
CREATE TABLE stg_train (
    id           INTEGER PRIMARY KEY,
    date         TEXT NOT NULL,
    store_nbr    INTEGER NOT NULL,
    item_nbr     INTEGER NOT NULL,
    unit_sales   REAL NOT NULL,
    onpromotion  TEXT
);

DROP TABLE IF EXISTS stg_stores;
CREATE TABLE stg_stores (
    store_nbr    INTEGER PRIMARY KEY,
    city         TEXT,
    state        TEXT,
    type         TEXT,
    cluster      INTEGER
);

DROP TABLE IF EXISTS stg_items;
CREATE TABLE stg_items (
    item_nbr     INTEGER PRIMARY KEY,
    family       TEXT,
    class        INTEGER,
    perishable   INTEGER
);

DROP TABLE IF EXISTS stg_holidays;
CREATE TABLE stg_holidays (
    date         TEXT NOT NULL,
    type         TEXT,
    locale       TEXT,
    locale_name  TEXT,
    description  TEXT,
    transferred  TEXT
);

DROP TABLE IF EXISTS stg_oil;
CREATE TABLE stg_oil (
    date         TEXT NOT NULL,
    dcoilwtico   REAL
);

DROP TABLE IF EXISTS stg_transactions;
CREATE TABLE stg_transactions (
    date         TEXT NOT NULL,
    store_nbr    INTEGER NOT NULL,
    transactions INTEGER
);

-- ---------------------------------------------------------------------------
-- Step 2: Load CSVs (run via Python script - see scripts/01_load_data.py)
-- These are placeholder comments; actual .import happens in Python/pandas
-- ---------------------------------------------------------------------------
-- The Python loader will:
--   1. Read each CSV with pandas
--   2. Validate columns match expected schema
--   3. Insert into staging tables above

-- ---------------------------------------------------------------------------
-- Step 3: Data quality verification queries (run after loading)
-- ---------------------------------------------------------------------------

-- Check: confirm expected row counts (adjust if your scope differs)
SELECT 'stg_train' AS table_name, COUNT(*) AS row_count FROM stg_train
UNION ALL
SELECT 'stg_stores', COUNT(*) FROM stg_stores
UNION ALL
SELECT 'stg_items', COUNT(*) FROM stg_items
UNION ALL
SELECT 'stg_holidays', COUNT(*) FROM stg_holidays;

-- Check: date range
SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM stg_train;

-- Check: negative sales (returns) - should be removed
SELECT COUNT(*) AS negative_sales_count
FROM stg_train
WHERE unit_sales < 0;

-- Check: zero sales records
SELECT COUNT(*) AS zero_sales_count
FROM stg_train
WHERE unit_sales = 0;

-- Check: null store or item
SELECT COUNT(*) AS null_store_item
FROM stg_train
WHERE store_nbr IS NULL OR item_nbr IS NULL;

-- ---------------------------------------------------------------------------
-- Step 4: Select scoping stores (Quito cluster for consistency)
-- ---------------------------------------------------------------------------
-- Rationale: Select stores from the same city to minimize cross-store
-- geographic variance. Quito stores provide largest transaction volume.
SELECT store_nbr, city, state, type, cluster
FROM stg_stores
WHERE city = 'Quito'
ORDER BY store_nbr;

-- ---------------------------------------------------------------------------
-- Step 5: Create clean daily_demand table (scoped, filtered)
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS daily_demand;
CREATE TABLE daily_demand AS
SELECT
    t.id,
    t.date,
    t.store_nbr,
    t.item_nbr,
    i.family,
    i.class,
    i.perishable,
    t.unit_sales,
    CASE WHEN t.onpromotion = 'True' THEN 1 ELSE 0 END AS on_promotion
FROM stg_train t
INNER JOIN stg_items i ON t.item_nbr = i.item_nbr
INNER JOIN stg_stores s ON t.store_nbr = s.store_nbr
WHERE
    -- Scope: 2015-2016 data only
    t.date >= '2015-01-01'
    AND t.date <= '2016-12-31'
    -- Remove returns (negative sales)
    AND t.unit_sales > 0
    -- Scope: selected Quito stores
    AND t.store_nbr IN (SELECT store_nbr FROM stg_stores WHERE city = 'Quito')
;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_daily_demand_store_item ON daily_demand(store_nbr, item_nbr);
CREATE INDEX IF NOT EXISTS idx_daily_demand_date ON daily_demand(date);
CREATE INDEX IF NOT EXISTS idx_daily_demand_family ON daily_demand(family);

-- Verification: confirm cleaned row count and date range
SELECT
    COUNT(*) AS total_rows,
    MIN(date) AS start_date,
    MAX(date) AS end_date,
    COUNT(DISTINCT store_nbr) AS num_stores,
    COUNT(DISTINCT item_nbr) AS num_items,
    COUNT(DISTINCT family) AS num_families
FROM daily_demand;
