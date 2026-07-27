-- =============================================================================
-- 04_abc_classification.sql
-- Purpose: ABC classification using cumulative revenue contribution.
--          A = top 80% of revenue, B = next 15%, C = bottom 5%
--          Classification is per store (each store classified independently).
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Step 1: Rank SKUs by revenue within each store
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS abc_classification;
CREATE TABLE abc_classification AS
WITH ranked_skus AS (
    SELECT
        store_nbr,
        item_nbr,
        family,
        revenue_proxy,
        SUM(revenue_proxy) OVER (PARTITION BY store_nbr) AS store_total_revenue,
        ROW_NUMBER() OVER (PARTITION BY store_nbr ORDER BY revenue_proxy DESC) AS revenue_rank
    FROM sku_demand_stats
),
cumulative AS (
    SELECT
        r.*,
        SUM(revenue_proxy) OVER (
            PARTITION BY store_nbr
            ORDER BY revenue_rank
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue,
        SUM(revenue_proxy) OVER (
            PARTITION BY store_nbr
            ORDER BY revenue_rank
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) * 100.0 / store_total_revenue AS cumulative_pct
    FROM ranked_skus
)
SELECT
    store_nbr,
    item_nbr,
    family,
    revenue_proxy,
    revenue_rank,
    store_total_revenue,
    cumulative_revenue,
    ROUND(cumulative_pct, 4) AS cumulative_pct,
    CASE
        WHEN cumulative_pct <= 80 THEN 'A'
        WHEN cumulative_pct <= 95 THEN 'B'
        ELSE 'C'
    END AS abc_class
FROM cumulative
ORDER BY store_nbr, revenue_rank;

CREATE INDEX IF NOT EXISTS idx_abc_store_item ON abc_classification(store_nbr, item_nbr);
CREATE INDEX IF NOT EXISTS idx_abc_class ON abc_classification(abc_class);

-- ---------------------------------------------------------------------------
-- Verification: Distribution of ABC classes
-- ---------------------------------------------------------------------------
SELECT
    store_nbr,
    abc_class,
    COUNT(*) AS num_skus,
    ROUND(SUM(revenue_proxy), 2) AS total_revenue,
    ROUND(SUM(revenue_proxy) * 100.0 / SUM(SUM(revenue_proxy)) OVER (PARTITION BY store_nbr), 2) AS revenue_pct,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY store_nbr), 2) AS sku_pct
FROM abc_classification
GROUP BY store_nbr, abc_class
ORDER BY store_nbr, abc_class;
