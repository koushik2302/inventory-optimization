"""
02_classify_skus.py
-------------------
Week 2 deliverable: Compute demand statistics (mean, std dev, CV),
run ABC classification (by revenue contribution) and XYZ classification
(by demand variability), and build the 9-cell ABC-XYZ matrix.

Run AFTER: scripts/01_load_data.py
Run BEFORE: scripts/03_safety_stock.py

Usage:
    python scripts/02_classify_skus.py            # skips reclassification if
                                                    # abc_xyz_matrix already reflects
                                                    # the current daily_demand
    python scripts/02_classify_skus.py --force     # reclassify unconditionally
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import time

import duckdb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "inventory.db"
DUCKDB_PATH  = PROJECT_ROOT / "data" / "inventory.duckdb"
CLEANED_DIR  = PROJECT_ROOT / "data" / "cleaned"

# ABC thresholds (cumulative revenue %)
ABC_A_THRESHOLD = 80.0   # Top 80% of revenue → A
ABC_B_THRESHOLD = 95.0   # Next 15% (80-95%) → B; remainder → C

# XYZ thresholds (Coefficient of Variation)
XYZ_X_MAX = 0.5    # CV < 0.5 → X (stable)
XYZ_Y_MAX = 1.0    # 0.5 ≤ CV ≤ 1.0 → Y (moderate); else → Z

# Minimum observations filter
MIN_OBSERVATIONS = 30

# Service level and z-score lookup per ABC-XYZ cell
CELL_POLICY = {
    "AX": {"service_level_9cell": 0.99, "z_score_9cell": 2.326},
    "AY": {"service_level_9cell": 0.97, "z_score_9cell": 1.881},
    "AZ": {"service_level_9cell": 0.95, "z_score_9cell": 1.645},
    "BX": {"service_level_9cell": 0.95, "z_score_9cell": 1.645},
    "BY": {"service_level_9cell": 0.93, "z_score_9cell": 1.476},
    "BZ": {"service_level_9cell": 0.90, "z_score_9cell": 1.282},
    "CX": {"service_level_9cell": 0.90, "z_score_9cell": 1.282},
    "CY": {"service_level_9cell": 0.88, "z_score_9cell": 1.175},
    "CZ": {"service_level_9cell": 0.85, "z_score_9cell": 1.036},
}

# Logging – use UTF-8 to avoid Windows cp1252 emoji crashes
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        _stream_handler,
        logging.FileHandler(PROJECT_ROOT / "data" / "classify_log.txt", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline-state guard (skip reclassification if nothing changed)
# ---------------------------------------------------------------------------
# _pipeline_state records which daily_demand row count each script's output
# was generated from, so a later run can tell "is my output still valid" without
# re-running the expensive query just to check. Shared table name/schema —
# 03_safety_stock.py and 04_promotion_impact.py could use the same pattern.
def _ensure_pipeline_state_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _pipeline_state (
            script TEXT PRIMARY KEY,
            source_rows INTEGER,
            completed_at TEXT
        )
    """)


def already_classified(conn: sqlite3.Connection) -> bool:
    """
    True if abc_xyz_matrix already exists and was built from the current
    daily_demand row count (i.e. a prior run already did this work and
    daily_demand hasn't changed since). Used to skip the ~18-minute
    classification query — pass --force to reclassify anyway.
    """
    _ensure_pipeline_state_table(conn)
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='abc_xyz_matrix'"
    ).fetchone()
    if row[0] == 0:
        return False
    state = conn.execute(
        "SELECT source_rows FROM _pipeline_state WHERE script = '02_classify_skus'"
    ).fetchone()
    if state is None:
        return False
    current_rows = conn.execute("SELECT COUNT(*) FROM daily_demand").fetchone()[0]
    return state[0] == current_rows


def record_pipeline_state(conn: sqlite3.Connection):
    _ensure_pipeline_state_table(conn)
    current_rows = conn.execute("SELECT COUNT(*) FROM daily_demand").fetchone()[0]
    conn.execute(
        "INSERT OR REPLACE INTO _pipeline_state (script, source_rows, completed_at) "
        "VALUES ('02_classify_skus', ?, datetime('now'))",
        (current_rows,),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Step 1: Compute demand statistics
# ---------------------------------------------------------------------------
def compute_demand_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Compute per-SKU per-store demand statistics from daily_demand.
    Key outputs: mean, std, CV, total units.

    Reads from data/inventory.duckdb (built by 01_load_data.py directly
    from the raw CSVs), not the SQLite `conn` — benchmarked ~65x faster
    for this exact aggregation (16.6s vs. SQLite's 1075s across two
    queries), because DuckDB's columnar engine only gets its speed
    advantage over its own native storage, not bridging through SQLite.
    DuckDB also has STDDEV_SAMP built in, collapsing what SQLite needed
    as two separate queries (SQLite has neither STDDEV nor SQRT) into one.
    `conn` (SQLite) is unused here but kept in the signature — the
    fallback in case DuckDB is ever unavailable is the git history of
    this function, not a runtime branch, to keep this function simple.

    NOTE ON FILTERING: daily_demand itself is unfiltered (it retains every
    raw row, including zero/negative "unit_sales" from returns and no-sale
    days, so the LLM agent can query the complete history via SQLite). Here,
    and only here, a `unit_sales > 0` filter is applied — the mean/std/CV
    formulas that drive ABC-XYZ classification and safety stock are only
    valid over genuine positive-demand observations; a return recorded as
    e.g. -5 units would otherwise pull the mean down and inflate CV.
    """
    log.info("▶  Computing demand statistics per SKU per store (via DuckDB)...")
    t0 = time.time()

    dcon = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    dcon.execute("PRAGMA memory_limit='4GB'")
    dcon.execute("PRAGMA threads=2")
    df = dcon.execute("""
        SELECT
            store_nbr,
            item_nbr,
            family,
            class,
            perishable,
            COUNT(*) AS num_observations,
            SUM(unit_sales) AS total_units_sold,
            AVG(unit_sales) AS mean_daily_demand,
            MIN(unit_sales) AS min_demand,
            MAX(unit_sales) AS max_demand,
            SUM(on_promotion) AS promo_days,
            AVG(CAST(on_promotion AS DOUBLE)) AS promo_rate,
            MIN(date) AS first_date,
            MAX(date) AS last_date,
            COALESCE(STDDEV_SAMP(unit_sales), 0.0) AS std_daily_demand
        FROM daily_demand
        WHERE unit_sales > 0
        GROUP BY store_nbr, item_nbr, family, class, perishable
        HAVING COUNT(*) >= ?
        ORDER BY store_nbr, item_nbr
    """, [MIN_OBSERVATIONS]).fetchdf()
    dcon.close()

    log.info(f"   Demand stats (incl. std dev): {len(df):,} SKU-store pairs ({time.time()-t0:.1f}s)")

    # Coefficient of Variation
    df["cv"] = df["std_daily_demand"] / df["mean_daily_demand"]
    df["cv"] = df["cv"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Unit cost proxy (no real prices — use mean_daily_demand as relative proxy)
    # Revenue proxy = total_units_sold (all prices treated as 1.0)
    df["unit_cost_proxy"] = 1.0
    df["revenue_proxy"]   = df["total_units_sold"]

    log.info(f"   Demand stats complete: {len(df):,} SKU-store pairs")
    log.info(f"   CV distribution:\n{df['cv'].describe().round(4).to_string()}")

    return df


# ---------------------------------------------------------------------------
# Step 2: ABC classification
# ---------------------------------------------------------------------------
def abc_classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    ABC classification per store based on cumulative revenue contribution.
    Returns DataFrame with added columns: revenue_rank, cumulative_pct, abc_class.
    """
    log.info("▶  Running ABC classification...")

    results = []
    for store_nbr, grp in df.groupby("store_nbr"):
        grp = grp.copy()

        # Rank by descending revenue
        grp = grp.sort_values("revenue_proxy", ascending=False).reset_index(drop=True)
        grp["revenue_rank"] = range(1, len(grp) + 1)

        # Cumulative revenue %
        total_rev = grp["revenue_proxy"].sum()
        grp["cumulative_revenue"] = grp["revenue_proxy"].cumsum()
        grp["cumulative_pct"]     = grp["cumulative_revenue"] / total_rev * 100

        # ABC assignment
        grp["abc_class"] = "C"
        grp.loc[grp["cumulative_pct"] <= ABC_A_THRESHOLD, "abc_class"] = "A"
        grp.loc[
            (grp["cumulative_pct"] > ABC_A_THRESHOLD) &
            (grp["cumulative_pct"] <= ABC_B_THRESHOLD),
            "abc_class"
        ] = "B"

        results.append(grp)

    out = pd.concat(results, ignore_index=True)

    # Log distribution
    abc_dist = out.groupby(["store_nbr", "abc_class"]).agg(
        num_skus=("item_nbr", "count"),
        revenue_share=("revenue_proxy", "sum"),
    ).reset_index()

    for store_nbr, g in abc_dist.groupby("store_nbr"):
        total = g["revenue_share"].sum()
        g = g.copy()
        g["rev_pct"] = g["revenue_share"] / total * 100
        log.info(f"\n   Store {store_nbr} ABC distribution:")
        for _, row in g.iterrows():
            log.info(f"      {row['abc_class']}: {row['num_skus']} SKUs | {row['rev_pct']:.1f}% revenue")

    return out


# ---------------------------------------------------------------------------
# Step 3: XYZ classification
# ---------------------------------------------------------------------------
def xyz_classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    XYZ classification based on Coefficient of Variation.
    X: CV < 0.5 | Y: 0.5 <= CV <= 1.0 | Z: CV > 1.0
    """
    log.info("▶  Running XYZ classification...")

    df = df.copy()
    df["xyz_class"] = "Z"
    df.loc[df["cv"] < XYZ_X_MAX, "xyz_class"] = "X"
    df.loc[(df["cv"] >= XYZ_X_MAX) & (df["cv"] <= XYZ_Y_MAX), "xyz_class"] = "Y"

    xyz_dist = df.groupby("xyz_class")["item_nbr"].count()
    log.info(f"   XYZ distribution (all stores):\n{xyz_dist.to_string()}")

    return df


# ---------------------------------------------------------------------------
# Step 4: Build ABC-XYZ matrix
# ---------------------------------------------------------------------------
def build_abc_xyz_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine ABC and XYZ classifications into the 9-cell matrix.
    Assigns differentiated service levels and z-scores.
    """
    log.info("▶  Building ABC-XYZ 9-cell matrix...")

    df = df.copy()
    df["cell"] = df["abc_class"] + df["xyz_class"]

    # Map policy parameters
    df["service_level_9cell"] = df["cell"].map(lambda c: CELL_POLICY[c]["service_level_9cell"])
    df["z_score_9cell"]       = df["cell"].map(lambda c: CELL_POLICY[c]["z_score_9cell"])

    # Uniform policy (95% for all)
    df["service_level_uniform"] = 0.95
    df["z_score_uniform"]       = 1.645

    # 3-tier ABC-only policy
    tier3_sl = {"A": 0.99, "B": 0.95, "C": 0.90}
    tier3_z  = {"A": 2.326, "B": 1.645, "C": 1.282}
    df["service_level_3tier"] = df["abc_class"].map(tier3_sl)
    df["z_score_3tier"]       = df["abc_class"].map(tier3_z)

    # Log 9-cell distribution
    matrix_summary = df.groupby(["abc_class", "xyz_class", "cell"]).agg(
        num_skus=("item_nbr", "count"),
        total_revenue=("revenue_proxy", "sum"),
        avg_cv=("cv", "mean"),
    ).reset_index()
    matrix_summary["service_level"] = matrix_summary["cell"].map(
        lambda c: CELL_POLICY[c]["service_level_9cell"]
    )

    log.info("\n   === ABC-XYZ Matrix (all stores combined) ===")
    log.info(f"\n{matrix_summary.to_string(index=False)}")

    return df


# ---------------------------------------------------------------------------
# Step 5: Write to SQLite
# ---------------------------------------------------------------------------
def write_to_db(df: pd.DataFrame, conn: sqlite3.Connection):
    """Write classification results back to SQLite."""
    log.info("▶  Writing classification results to SQLite...")

    df.to_sql("abc_xyz_matrix", conn, if_exists="replace", index=False, chunksize=5000)

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_matrix_cell ON abc_xyz_matrix(cell)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_matrix_store ON abc_xyz_matrix(store_nbr)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_matrix_store_item ON abc_xyz_matrix(store_nbr, item_nbr)")
    conn.commit()

    log.info(f"   Written {len(df):,} rows to abc_xyz_matrix table")


# ---------------------------------------------------------------------------
# Step 6: Export CSVs
# ---------------------------------------------------------------------------
def export_classifications(df: pd.DataFrame):
    """Export classification results to CSV for reporting."""
    log.info("▶  Exporting classification CSVs...")
    CLEANED_DIR.mkdir(exist_ok=True)

    df.to_csv(CLEANED_DIR / "abc_xyz_matrix.csv", index=False)
    log.info(f"   Saved abc_xyz_matrix.csv ({len(df):,} rows)")

    # Summary pivot (for Excel / paper)
    summary = df.groupby(["abc_class", "xyz_class", "cell"]).agg(
        num_skus=("item_nbr", "count"),
        total_revenue=("revenue_proxy", "sum"),
        avg_cv=("cv", "mean"),
        avg_mean_demand=("mean_daily_demand", "mean"),
        avg_std_demand=("std_daily_demand", "mean"),
    ).reset_index()
    summary.to_csv(CLEANED_DIR / "abc_xyz_summary.csv", index=False)
    log.info(f"   Saved abc_xyz_summary.csv ({len(summary)} rows)")

    # Per-store distribution
    store_dist = df.groupby(["store_nbr", "abc_class", "xyz_class", "cell"]).agg(
        num_skus=("item_nbr", "count"),
        total_revenue=("revenue_proxy", "sum"),
    ).reset_index()
    store_dist.to_csv(CLEANED_DIR / "abc_xyz_by_store.csv", index=False)
    log.info(f"   Saved abc_xyz_by_store.csv ({len(store_dist)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Reclassify even if abc_xyz_matrix already reflects current daily_demand.")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  Inventory Optimization — SKU Classification (Week 2)")
    log.info("=" * 60)
    log.info(f"  ABC thresholds: A≤{ABC_A_THRESHOLD}%, B≤{ABC_B_THRESHOLD}%")
    log.info(f"  XYZ thresholds: X<{XYZ_X_MAX}, Y≤{XYZ_Y_MAX}")
    log.info(f"  Min observations: {MIN_OBSERVATIONS}")
    log.info("")

    if not DB_PATH.exists():
        log.error(f"Database not found: {DB_PATH}")
        log.error("Please run: python scripts/01_load_data.py")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA cache_size = -65536")

    try:
        if not args.force and already_classified(conn):
            log.info("[SKIP]  abc_xyz_matrix already reflects the current daily_demand "
                      "row count. Nothing to do — pass --force to reclassify anyway.")
            return

        # Step 1: Demand statistics
        df = compute_demand_stats(conn)

        # Step 2: ABC
        df = abc_classify(df)

        # Step 3: XYZ
        df = xyz_classify(df)

        # Step 4: Full matrix
        df = build_abc_xyz_matrix(df)

        # Step 5: Write to DB
        write_to_db(df, conn)

        # Step 6: Export CSVs
        export_classifications(df)

        # Record state so a future run can skip if daily_demand hasn't changed
        record_pipeline_state(conn)

        log.info("\n✅  Classification complete!")
        log.info(f"   Total SKU-store pairs classified: {len(df):,}")
        log.info(f"   Cells covered: {sorted(df['cell'].unique())}")
        log.info("\n   Next step: python scripts/03_safety_stock.py")

    except Exception as e:
        log.error(f"❌  Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
