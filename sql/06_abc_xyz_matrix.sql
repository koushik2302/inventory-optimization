-- =============================================================================
-- 06_abc_xyz_matrix.sql
-- Purpose: Join ABC and XYZ classifications to form the 9-cell ABC-XYZ matrix.
--          Also assigns service level and z-score per cell for safety stock
--          computation in the subsequent Python analysis.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Step 1: Create the combined ABC-XYZ classification table
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS abc_xyz_matrix;
CREATE TABLE abc_xyz_matrix AS
SELECT
    a.store_nbr,
    a.item_nbr,
    a.family,
    a.abc_class,
    x.xyz_class,
    a.abc_class || x.xyz_class AS cell,  -- e.g., 'AX', 'BZ'
    a.revenue_proxy,
    a.cumulative_pct       AS abc_cumulative_pct,
    x.cv,
    x.mean_daily_demand,
    x.std_daily_demand,

    -- Assign differentiated service levels (9-cell policy)
    CASE a.abc_class || x.xyz_class
        WHEN 'AX' THEN 0.99
        WHEN 'AY' THEN 0.97
        WHEN 'AZ' THEN 0.95
        WHEN 'BX' THEN 0.95
        WHEN 'BY' THEN 0.93
        WHEN 'BZ' THEN 0.90
        WHEN 'CX' THEN 0.90
        WHEN 'CY' THEN 0.88
        WHEN 'CZ' THEN 0.85
    END AS service_level_9cell,

    -- Corresponding z-scores (NORM.S.INV approximations)
    CASE a.abc_class || x.xyz_class
        WHEN 'AX' THEN 2.33
        WHEN 'AY' THEN 1.88
        WHEN 'AZ' THEN 1.65
        WHEN 'BX' THEN 1.65
        WHEN 'BY' THEN 1.48
        WHEN 'BZ' THEN 1.28
        WHEN 'CX' THEN 1.28
        WHEN 'CY' THEN 1.17
        WHEN 'CZ' THEN 1.04
    END AS z_score_9cell,

    -- Uniform policy (95% for all)
    0.95 AS service_level_uniform,
    1.65 AS z_score_uniform,

    -- 3-tier ABC-only policy
    CASE a.abc_class
        WHEN 'A' THEN 0.99
        WHEN 'B' THEN 0.95
        WHEN 'C' THEN 0.90
    END AS service_level_3tier,
    CASE a.abc_class
        WHEN 'A' THEN 2.33
        WHEN 'B' THEN 1.65
        WHEN 'C' THEN 1.28
    END AS z_score_3tier

FROM abc_classification a
INNER JOIN xyz_classification x
    ON a.store_nbr = x.store_nbr AND a.item_nbr = x.item_nbr
;

CREATE INDEX IF NOT EXISTS idx_matrix_store ON abc_xyz_matrix(store_nbr);
CREATE INDEX IF NOT EXISTS idx_matrix_cell ON abc_xyz_matrix(cell);
CREATE INDEX IF NOT EXISTS idx_matrix_store_item ON abc_xyz_matrix(store_nbr, item_nbr);

-- ---------------------------------------------------------------------------
-- Step 2: ABC-XYZ Matrix Summary (the 3x3 heatmap data)
-- ---------------------------------------------------------------------------
SELECT
    abc_class,
    xyz_class,
    cell,
    COUNT(*)                         AS num_skus,
    ROUND(SUM(revenue_proxy), 2)     AS total_revenue,
    ROUND(AVG(cv), 4)                AS avg_cv,
    service_level_9cell              AS target_service_level
FROM abc_xyz_matrix
GROUP BY abc_class, xyz_class, cell, service_level_9cell
ORDER BY abc_class, xyz_class;

-- ---------------------------------------------------------------------------
-- Step 3: Store-level matrix distribution
-- ---------------------------------------------------------------------------
SELECT
    store_nbr,
    cell,
    COUNT(*) AS num_skus,
    ROUND(SUM(revenue_proxy) * 100.0 / SUM(SUM(revenue_proxy)) OVER (PARTITION BY store_nbr), 2) AS revenue_share_pct
FROM abc_xyz_matrix
GROUP BY store_nbr, cell
ORDER BY store_nbr, abc_class, xyz_class;
