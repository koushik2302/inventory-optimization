-- =============================================================================
-- 02_daily_demand.sql
-- Purpose: Create sku_demand_summary table with per-SKU statistics needed
--          for safety stock calculation and ABC-XYZ classification.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Demand Summary Statistics per SKU per Store
-- Minimum 30 demand observations filter applied here
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS sku_demand_summary;
CREATE TABLE sku_demand_summary AS
SELECT
    store_nbr,
    item_nbr,
    family,
    class,
    perishable,

    -- Volume statistics
    COUNT(*)                              AS num_observations,
    SUM(unit_sales)                       AS total_units_sold,
    AVG(unit_sales)                       AS mean_daily_demand,
    MIN(unit_sales)                       AS min_daily_demand,
    MAX(unit_sales)                       AS max_daily_demand,

    -- Variability (SQLite doesn't have STDDEV natively - computed in Python)
    -- Storing sum and sum of squares for STDDEV computation
    SUM(unit_sales * unit_sales)          AS sum_sq_demand,

    -- Promotion frequency
    SUM(on_promotion)                     AS promo_days,
    AVG(CAST(on_promotion AS REAL))       AS promo_rate,

    -- Intermittence metrics
    SUM(CASE WHEN unit_sales > 0 THEN 1 ELSE 0 END) AS non_zero_demand_days,

    -- Date range
    MIN(date)                             AS first_sale_date,
    MAX(date)                             AS last_sale_date

FROM daily_demand
GROUP BY store_nbr, item_nbr, family, class, perishable
HAVING COUNT(*) >= 30  -- Minimum 30 observations for statistical validity
;

CREATE INDEX IF NOT EXISTS idx_sku_demand_store ON sku_demand_summary(store_nbr);
CREATE INDEX IF NOT EXISTS idx_sku_demand_item ON sku_demand_summary(item_nbr);

-- Verify
SELECT
    COUNT(*) AS total_skus,
    COUNT(DISTINCT store_nbr) AS num_stores,
    SUM(total_units_sold) AS grand_total_units,
    AVG(num_observations) AS avg_observations_per_sku
FROM sku_demand_summary;
