-- =============================================================================
-- 03_demand_stats.sql
-- Purpose: Add unit cost proxy (using mean unit sales as proxy in absence
--          of real price data), and compute CV after Python has added std_dev.
--          Also computes total revenue per SKU per store.
-- Note: Real CV and std_dev are computed in Python (02_classify_skus.py)
--       and written back to sku_demand_stats table.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Demand stats with computed fields (after Python enrichment)
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS sku_demand_stats;
CREATE TABLE sku_demand_stats AS
SELECT
    ds.*,

    -- Coefficient of variation (std_dev / mean) - will be updated by Python
    0.0 AS std_daily_demand,
    0.0 AS cv,

    -- Unit cost proxy: we don't have true cost, use mean daily demand as
    -- a relative proxy for ABC (revenue = units * 1.0 as price proxy)
    -- In real deployment, replace with actual unit price
    1.0 AS unit_cost_proxy,
    total_units_sold AS revenue_proxy

FROM sku_demand_summary ds;

CREATE INDEX IF NOT EXISTS idx_stats_store_item ON sku_demand_stats(store_nbr, item_nbr);
CREATE INDEX IF NOT EXISTS idx_stats_revenue ON sku_demand_stats(revenue_proxy);

-- Verify distribution of observation counts
SELECT
    CASE
        WHEN num_observations < 50  THEN '30-49'
        WHEN num_observations < 100 THEN '50-99'
        WHEN num_observations < 200 THEN '100-199'
        WHEN num_observations < 365 THEN '200-364'
        ELSE '365+'
    END AS obs_bucket,
    COUNT(*) AS sku_count
FROM sku_demand_summary
GROUP BY obs_bucket
ORDER BY obs_bucket;
