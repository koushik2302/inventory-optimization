"""
04_promotion_impact.py
----------------------
Week 4 deliverable: Analyze the impact of promotions and holidays on
demand, quantify demand lift per product family, and compute adjusted
safety stock buffers for promotional periods.

Run AFTER: scripts/03_safety_stock.py

Usage:
    python scripts/04_promotion_impact.py
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys
import logging

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "inventory.db"
DUCKDB_PATH  = PROJECT_ROOT / "data" / "inventory.duckdb"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_125m" / "extracted"
CLEANED_DIR  = PROJECT_ROOT / "data" / "cleaned"


def _duckdb_conn() -> duckdb.DuckDBPyConnection:
    dcon = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    dcon.execute("PRAGMA memory_limit='4GB'")
    dcon.execute("PRAGMA threads=2")
    return dcon

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[_stream_handler])
log = logging.getLogger(__name__)

HOLDING_COST_RATE = 0.22
BASE_LEAD_TIME = 7


def analyze_promotion_lift(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Compute demand lift during promotions by product family.
    Returns: DataFrame with family, baseline_demand, promo_demand, lift_pct

    Reads from data/inventory.duckdb, not SQLite `conn` — this GROUP BY
    over the full 125M-row daily_demand table took ~8 min via SQLite;
    DuckDB's own native storage is dramatically faster for this shape of
    aggregation (benchmarked ~10-65x elsewhere in the pipeline, see
    02_classify_skus.py / CHANGELOG.md / PIPELINE_LOG.md).
    """
    log.info("▶  Analyzing promotion demand lift by family (via DuckDB)...")

    dcon = _duckdb_conn()
    df = dcon.execute("""
        SELECT
            family,
            on_promotion,
            AVG(unit_sales) AS avg_demand,
            COUNT(*) AS num_obs
        FROM daily_demand
        GROUP BY family, on_promotion
        ORDER BY family, on_promotion
    """).fetchdf()
    dcon.close()

    baseline = df[df["on_promotion"]==0].set_index("family")["avg_demand"]
    promo    = df[df["on_promotion"]==1].set_index("family")["avg_demand"]

    families = baseline.index.intersection(promo.index)
    lift_df = pd.DataFrame({
        "family": families,
        "baseline_avg_demand": baseline[families].values,
        "promo_avg_demand": promo[families].values,
        "lift_pct": ((promo[families] / baseline[families] - 1) * 100).values,
        "baseline_obs": df[df["on_promotion"]==0].set_index("family").loc[families, "num_obs"].values,
        "promo_obs": df[df["on_promotion"]==1].set_index("family").loc[families, "num_obs"].values,
    }).sort_values("lift_pct", ascending=False).reset_index(drop=True)

    log.info(f"\n   Promotion Lift by Family:\n{lift_df[['family','baseline_avg_demand','promo_avg_demand','lift_pct']].round(2).to_string(index=False)}")
    return lift_df


def compute_promo_safety_stock(conn: sqlite3.Connection, lift_df: pd.DataFrame, lead_time: int = BASE_LEAD_TIME) -> pd.DataFrame:
    """
    Compute adjusted safety stock for promotional periods.
    Compare baseline SS vs. promo-adjusted SS per family.
    """
    log.info("▶  Computing promotional safety stock adjustments...")

    # Load per-SKU data
    df = pd.read_sql("""
        SELECT m.store_nbr, m.item_nbr, m.family, m.cell,
               m.mean_daily_demand, m.std_daily_demand, m.cv,
               m.z_score_9cell, m.service_level_9cell,
               COALESCE(s.ss_9cell, m.z_score_9cell * m.std_daily_demand * ? ) AS ss_baseline,
               COALESCE(s.hc_9cell, 0) AS hc_baseline
        FROM abc_xyz_matrix m
        LEFT JOIN safety_stock_results s ON m.store_nbr = s.store_nbr AND m.item_nbr = s.item_nbr
    """, conn, params=(np.sqrt(lead_time),))

    # Merge lift data
    df = df.merge(lift_df[["family", "lift_pct"]], on="family", how="left")
    df["lift_pct"] = df["lift_pct"].fillna(0)

    # Compute promo-adjusted demand and SS
    df["promo_mean_demand"] = df["mean_daily_demand"] * (1 + df["lift_pct"] / 100)

    # Increased variability during promos: assume std scales with mean
    df["promo_std_demand"]  = df["std_daily_demand"] * (1 + df["lift_pct"] / 200)  # less than proportional

    df["ss_promo"] = df["z_score_9cell"] * df["promo_std_demand"] * np.sqrt(lead_time)
    df["hc_promo"] = df["ss_promo"] * HOLDING_COST_RATE
    df["ss_increase"] = df["ss_promo"] - df["ss_baseline"]
    df["hc_increase"] = df["hc_promo"] - df["hc_baseline"]

    # Summary by family
    family_summary = df.groupby("family").agg(
        num_skus=("item_nbr", "count"),
        lift_pct=("lift_pct", "first"),
        ss_baseline_total=("ss_baseline", "sum"),
        ss_promo_total=("ss_promo", "sum"),
        ss_increase_total=("ss_increase", "sum"),
        hc_increase_total=("hc_increase", "sum"),
    ).reset_index()

    log.info(f"\n   Promo SS Adjustment Summary:\n{family_summary.round(2).to_string(index=False)}")
    return df, family_summary


def analyze_holiday_impact(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Analyze demand impact of national/regional holidays.

    Reads daily_demand from data/inventory.duckdb, not SQLite `conn` — this
    LEFT JOIN + GROUP BY over the full 125M-row table took ~12 min via
    SQLite (after an earlier fix moved it off a naive "pull all 125M
    joined rows into pandas" approach that OOM'd — see CHANGELOG.md /
    PIPELINE_LOG.md). DuckDB's native storage does the same aggregation in
    a fraction of the time. holidays_events.csv is read directly (small
    file, no reason to round-trip through SQLite's stg_holidays).
    Only the small aggregated result (~66 rows: 33 families x 2) ever
    reaches pandas either way — that principle from the earlier fix still
    holds, DuckDB just makes getting there faster.
    """
    log.info("▶  Analyzing holiday impact on demand (via DuckDB)...")

    try:
        dcon = _duckdb_conn()
        holiday_impact = dcon.execute(f"""
            SELECT
                d.family,
                CASE WHEN h.date IS NOT NULL THEN 1 ELSE 0 END AS is_holiday,
                AVG(d.unit_sales) AS avg_demand,
                COUNT(*) AS num_obs
            FROM daily_demand d
            LEFT JOIN read_csv('{(RAW_DATA_DIR / "holidays_events.csv").as_posix()}') h
                ON d.date = h.date
            GROUP BY d.family, is_holiday
        """).fetchdf()
        dcon.close()
    except Exception as e:
        log.warning(f"   Could not join holidays: {e}")
        return pd.DataFrame()

    baseline = holiday_impact[holiday_impact["is_holiday"]==0].set_index("family")["avg_demand"]
    holiday  = holiday_impact[holiday_impact["is_holiday"]==1].set_index("family")["avg_demand"]

    common = baseline.index.intersection(holiday.index)
    impact = pd.DataFrame({
        "family": common,
        "baseline_demand": baseline[common].values,
        "holiday_demand": holiday[common].values,
        "holiday_lift_pct": ((holiday[common] / baseline[common] - 1) * 100).values,
    }).sort_values("holiday_lift_pct", ascending=False)

    log.info(f"\n   Holiday Impact:\n{impact.round(2).to_string(index=False)}")
    return impact


def main():
    log.info("=" * 60)
    log.info("  Inventory Optimization — Promotion Impact (Week 4)")
    log.info("=" * 60)

    if not DB_PATH.exists():
        log.error(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    CLEANED_DIR.mkdir(exist_ok=True)

    try:
        # Promotion lift analysis
        lift_df = analyze_promotion_lift(conn)
        lift_df.to_csv(CLEANED_DIR / "promotion_lift_by_family.csv", index=False)

        # Promo safety stock adjustment
        promo_sku_df, promo_summary = compute_promo_safety_stock(conn, lift_df)
        promo_sku_df.to_csv(CLEANED_DIR / "promo_safety_stock_skus.csv", index=False)
        promo_summary.to_csv(CLEANED_DIR / "promo_safety_stock_summary.csv", index=False)

        # Holiday impact
        holiday_df = analyze_holiday_impact(conn)
        if not holiday_df.empty:
            holiday_df.to_csv(CLEANED_DIR / "holiday_impact.csv", index=False)

        log.info("\n✅  Promotion impact analysis complete!")
        log.info("   Output: data/cleaned/promotion_lift_by_family.csv")
        log.info("   Output: data/cleaned/promo_safety_stock_summary.csv")

    except Exception as e:
        log.error(f"❌  Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
