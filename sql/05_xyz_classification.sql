-- =============================================================================
-- 05_xyz_classification.sql
-- Purpose: XYZ classification based on Coefficient of Variation (CV) of
--          daily demand.
--          X: CV < 0.5  (stable, predictable demand)
--          Y: 0.5 <= CV <= 1.0  (moderate variability)
--          Z: CV > 1.0  (highly erratic demand)
-- Note: std_daily_demand and cv columns must be populated by Python first
--       (scripts/02_classify_skus.py updates sku_demand_stats).
-- =============================================================================

DROP TABLE IF EXISTS xyz_classification;
CREATE TABLE xyz_classification AS
SELECT
    store_nbr,
    item_nbr,
    family,
    mean_daily_demand,
    std_daily_demand,
    ROUND(cv, 4) AS cv,
    CASE
        WHEN cv < 0.5  THEN 'X'
        WHEN cv <= 1.0 THEN 'Y'
        ELSE                'Z'
    END AS xyz_class
FROM sku_demand_stats
WHERE cv > 0  -- exclude items with zero variance (shouldn't exist after filtering)
;

CREATE INDEX IF NOT EXISTS idx_xyz_store_item ON xyz_classification(store_nbr, item_nbr);
CREATE INDEX IF NOT EXISTS idx_xyz_class ON xyz_classification(xyz_class);

-- Verification: Distribution of XYZ classes
SELECT
    store_nbr,
    xyz_class,
    COUNT(*) AS num_skus,
    ROUND(AVG(cv), 4) AS avg_cv,
    ROUND(MIN(cv), 4) AS min_cv,
    ROUND(MAX(cv), 4) AS max_cv
FROM xyz_classification
GROUP BY store_nbr, xyz_class
ORDER BY store_nbr, xyz_class;
