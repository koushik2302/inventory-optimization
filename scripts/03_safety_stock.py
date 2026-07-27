"""
03_safety_stock.py
------------------
Week 3 deliverable: Safety stock and reorder point computation under
three policy scenarios:
  1. Uniform    — 95% service level for all SKUs
  2. 3-tier     — ABC-only differentiation (A=99%, B=95%, C=90%)
  3. 9-cell     — Full ABC-XYZ differentiated policy

Also runs sensitivity analysis across lead times (3, 5, 7, 10 days)
and service level ranges, producing comparison tables and cost curves.

Run AFTER: scripts/02_classify_skus.py

Usage:
    python scripts/03_safety_stock.py
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys
import logging
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "inventory.db"
CLEANED_DIR  = PROJECT_ROOT / "data" / "cleaned"

# Lead time assumption (days) — primary scenario
BASE_LEAD_TIME = 7

# Sensitivity analysis lead times
SENSITIVITY_LEAD_TIMES = [3, 5, 7, 10]

# Holding cost rate (22% of unit cost annually)
HOLDING_COST_RATE = 0.22

# Logging – use UTF-8 to avoid Windows cp1252 emoji crashes
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        _stream_handler,
        logging.FileHandler(PROJECT_ROOT / "data" / "safety_stock_log.txt", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_safety_stock(
    mean_demand: float,
    std_demand: float,
    lead_time: int,
    z_score: float,
    unit_cost: float,
    holding_rate: float = HOLDING_COST_RATE,
) -> dict:
    """
    Compute safety stock, ROP, and annual holding cost for a single SKU.

    Formula:
        SS  = z × σ_d × √(LT)
        ROP = (d̄ × LT) + SS
        HC  = SS × C_unit × h_rate

    Args:
        mean_demand: Mean daily demand (units/day)
        std_demand:  Std dev of daily demand
        lead_time:   Lead time in days
        z_score:     z-score for target service level
        unit_cost:   Unit cost (proxy)
        holding_rate: Annual holding cost as fraction of unit cost

    Returns:
        dict with SS, ROP, annual_holding_cost, demand_during_LT
    """
    demand_during_lt = mean_demand * lead_time
    std_during_lt    = std_demand * np.sqrt(lead_time)  # demand variability over LT
    safety_stock     = z_score * std_during_lt
    reorder_point    = demand_during_lt + safety_stock
    annual_holding   = safety_stock * unit_cost * holding_rate

    return {
        "demand_during_lt": round(demand_during_lt, 4),
        "std_during_lt":    round(std_during_lt, 4),
        "safety_stock":     round(safety_stock, 4),
        "reorder_point":    round(reorder_point, 4),
        "annual_holding_cost": round(annual_holding, 4),
    }


def compute_all_scenarios(df: pd.DataFrame, lead_time: int) -> pd.DataFrame:
    """
    Compute safety stock for all SKUs under all 3 policy scenarios.

    Args:
        df:         abc_xyz_matrix DataFrame
        lead_time:  Lead time in days

    Returns:
        DataFrame with all scenario columns added
    """
    results = []

    for _, row in df.iterrows():
        base = {
            "store_nbr":  row["store_nbr"],
            "item_nbr":   row["item_nbr"],
            "family":     row["family"],
            "abc_class":  row["abc_class"],
            "xyz_class":  row["xyz_class"],
            "cell":       row["cell"],
            "mean_demand": row["mean_daily_demand"],
            "std_demand":  row["std_daily_demand"],
            "cv":          row["cv"],
            "revenue":     row["revenue_proxy"],
            "lead_time":   lead_time,
            "unit_cost":   row.get("unit_cost_proxy", 1.0),
        }

        # Scenario 1: Uniform (95% for all)
        s1 = compute_safety_stock(
            row["mean_daily_demand"], row["std_daily_demand"],
            lead_time, row["z_score_uniform"], row.get("unit_cost_proxy", 1.0)
        )
        base.update({
            "z_uniform":    row["z_score_uniform"],
            "sl_uniform":   row["service_level_uniform"],
            "ss_uniform":   s1["safety_stock"],
            "rop_uniform":  s1["reorder_point"],
            "hc_uniform":   s1["annual_holding_cost"],
        })

        # Scenario 2: 3-tier ABC only
        s2 = compute_safety_stock(
            row["mean_daily_demand"], row["std_daily_demand"],
            lead_time, row["z_score_3tier"], row.get("unit_cost_proxy", 1.0)
        )
        base.update({
            "z_3tier":    row["z_score_3tier"],
            "sl_3tier":   row["service_level_3tier"],
            "ss_3tier":   s2["safety_stock"],
            "rop_3tier":  s2["reorder_point"],
            "hc_3tier":   s2["annual_holding_cost"],
        })

        # Scenario 3: 9-cell ABC-XYZ differentiated
        s3 = compute_safety_stock(
            row["mean_daily_demand"], row["std_daily_demand"],
            lead_time, row["z_score_9cell"], row.get("unit_cost_proxy", 1.0)
        )
        base.update({
            "z_9cell":    row["z_score_9cell"],
            "sl_9cell":   row["service_level_9cell"],
            "ss_9cell":   s3["safety_stock"],
            "rop_9cell":  s3["reorder_point"],
            "hc_9cell":   s3["annual_holding_cost"],
        })

        results.append(base)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Policy comparison
# ---------------------------------------------------------------------------
def compare_policies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare total holding costs and safety stock levels across 3 policies.
    Returns summary table.
    """
    log.info("▶  Computing policy comparison...")

    total_uniform = df["hc_uniform"].sum()
    total_3tier   = df["hc_3tier"].sum()
    total_9cell   = df["hc_9cell"].sum()

    ss_uniform = df["ss_uniform"].sum()
    ss_3tier   = df["ss_3tier"].sum()
    ss_9cell   = df["ss_9cell"].sum()

    summary = pd.DataFrame([
        {
            "policy": "Uniform (95% all)",
            "total_annual_holding_cost": round(total_uniform, 2),
            "total_safety_stock_units": round(ss_uniform, 2),
            "avg_service_level": 0.95,
            "cost_vs_uniform_pct": 0.0,
            "ss_vs_uniform_pct": 0.0,
        },
        {
            "policy": "3-Tier (ABC only)",
            "total_annual_holding_cost": round(total_3tier, 2),
            "total_safety_stock_units": round(ss_3tier, 2),
            "avg_service_level": df["sl_3tier"].mean(),
            "cost_vs_uniform_pct": round((total_3tier - total_uniform) / total_uniform * 100, 2),
            "ss_vs_uniform_pct": round((ss_3tier - ss_uniform) / ss_uniform * 100, 2),
        },
        {
            "policy": "9-Cell (ABC-XYZ)",
            "total_annual_holding_cost": round(total_9cell, 2),
            "total_safety_stock_units": round(ss_9cell, 2),
            "avg_service_level": df["sl_9cell"].mean(),
            "cost_vs_uniform_pct": round((total_9cell - total_uniform) / total_uniform * 100, 2),
            "ss_vs_uniform_pct": round((ss_9cell - ss_uniform) / ss_uniform * 100, 2),
        },
    ])

    log.info(f"\n{'='*70}")
    log.info("  POLICY COMPARISON RESULTS (Lead Time = 7 days)")
    log.info(f"{'='*70}")
    log.info(f"\n{summary.to_string(index=False)}")
    log.info(f"\n  Key Insight:")
    savings = total_uniform - total_9cell
    savings_pct = savings / total_uniform * 100
    log.info(f"  9-cell vs Uniform: {savings_pct:+.1f}% change in holding cost")
    log.info(f"  9-cell vs 3-tier:  {(total_9cell-total_3tier)/total_3tier*100:+.1f}% change in holding cost")

    return summary


def compare_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Compare holding costs per ABC-XYZ cell."""
    cell_comparison = df.groupby(["abc_class", "xyz_class", "cell"]).agg(
        num_skus=("item_nbr", "count"),
        hc_uniform=("hc_uniform", "sum"),
        hc_3tier=("hc_3tier", "sum"),
        hc_9cell=("hc_9cell", "sum"),
        ss_uniform=("ss_uniform", "sum"),
        ss_9cell=("ss_9cell", "sum"),
        avg_cv=("cv", "mean"),
        service_level_9cell=("sl_9cell", "first"),
    ).reset_index()

    cell_comparison["hc_diff_9cell_vs_uniform_pct"] = (
        (cell_comparison["hc_9cell"] - cell_comparison["hc_uniform"])
        / cell_comparison["hc_uniform"] * 100
    ).round(2)

    log.info(f"\n{'='*70}")
    log.info("  HOLDING COST COMPARISON BY CELL")
    log.info(f"{'='*70}")
    log.info(f"\n{cell_comparison[['cell','num_skus','hc_uniform','hc_9cell','hc_diff_9cell_vs_uniform_pct']].to_string(index=False)}")

    return cell_comparison


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
def sensitivity_by_lead_time(df_base: pd.DataFrame) -> pd.DataFrame:
    """Run sensitivity analysis across different lead times."""
    log.info(f"▶  Running lead time sensitivity ({SENSITIVITY_LEAD_TIMES} days)...")

    rows = []
    for lt in SENSITIVITY_LEAD_TIMES:
        df_lt = compute_all_scenarios(df_base, lt)
        rows.append({
            "lead_time_days": lt,
            "hc_uniform":  round(df_lt["hc_uniform"].sum(), 2),
            "hc_3tier":    round(df_lt["hc_3tier"].sum(), 2),
            "hc_9cell":    round(df_lt["hc_9cell"].sum(), 2),
            "ss_uniform":  round(df_lt["ss_uniform"].sum(), 2),
            "ss_9cell":    round(df_lt["ss_9cell"].sum(), 2),
        })

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df["savings_9cell_vs_uniform_pct"] = (
        (sensitivity_df["hc_uniform"] - sensitivity_df["hc_9cell"])
        / sensitivity_df["hc_uniform"] * 100
    ).round(2)

    log.info(f"\n{'='*70}")
    log.info("  LEAD TIME SENSITIVITY ANALYSIS")
    log.info(f"{'='*70}")
    log.info(f"\n{sensitivity_df.to_string(index=False)}")

    return sensitivity_df


def sensitivity_by_service_level(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Cost tradeoff curve: total holding cost vs. uniform service level
    (the 'money chart' for the paper and Excel model).
    """
    log.info("▶  Computing service level vs. holding cost curve...")

    service_levels = np.arange(0.80, 1.00, 0.005)
    rows = []

    for sl in service_levels:
        z = stats.norm.ppf(sl)  # Inverse normal
        ss_total = (df_base["std_daily_demand"] * np.sqrt(BASE_LEAD_TIME) * z).sum()
        hc_total = (ss_total * HOLDING_COST_RATE)  # Using unit_cost=1

        rows.append({
            "service_level": round(sl, 3),
            "z_score": round(z, 4),
            "total_safety_stock_units": round(ss_total, 2),
            "total_annual_holding_cost": round(hc_total, 2),
        })

    curve_df = pd.DataFrame(rows)
    log.info(f"   Curve computed: {len(curve_df)} service level points")
    return curve_df


# ---------------------------------------------------------------------------
# Write results to DB and CSV
# ---------------------------------------------------------------------------
def write_results(
    df_results: pd.DataFrame,
    policy_summary: pd.DataFrame,
    cell_comparison: pd.DataFrame,
    sensitivity_lt: pd.DataFrame,
    sensitivity_sl: pd.DataFrame,
    conn: sqlite3.Connection,
):
    """Write all results to SQLite and CSV."""
    log.info("▶  Writing results...")
    CLEANED_DIR.mkdir(exist_ok=True)

    # Main results table
    df_results.to_sql("safety_stock_results", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ss_store_item ON safety_stock_results(store_nbr, item_nbr)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ss_cell ON safety_stock_results(cell)")
    conn.commit()

    df_results.to_csv(CLEANED_DIR / "safety_stock_results.csv", index=False)
    policy_summary.to_csv(CLEANED_DIR / "policy_comparison.csv", index=False)
    cell_comparison.to_csv(CLEANED_DIR / "cell_comparison.csv", index=False)
    sensitivity_lt.to_csv(CLEANED_DIR / "sensitivity_lead_time.csv", index=False)
    sensitivity_sl.to_csv(CLEANED_DIR / "sensitivity_service_level.csv", index=False)

    log.info(f"   ✅  Saved 5 output files to data/cleaned/")
    log.info(f"   ✅  Saved safety_stock_results to SQLite ({len(df_results):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("  Inventory Optimization — Safety Stock Model (Week 3)")
    log.info("=" * 60)
    log.info(f"  Base lead time:   {BASE_LEAD_TIME} days")
    log.info(f"  Holding cost rate: {HOLDING_COST_RATE:.0%} annually")
    log.info("")

    if not DB_PATH.exists():
        log.error(f"Database not found: {DB_PATH}")
        log.error("Please run: python scripts/02_classify_skus.py")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    try:
        # Load classification matrix
        log.info("▶  Loading ABC-XYZ matrix from database...")
        df_matrix = pd.read_sql("SELECT * FROM abc_xyz_matrix", conn)
        log.info(f"   Loaded {len(df_matrix):,} SKU-store pairs")

        # Validate required columns exist
        required_cols = [
            "mean_daily_demand", "std_daily_demand", "cv",
            "z_score_uniform", "z_score_3tier", "z_score_9cell",
            "service_level_uniform", "service_level_3tier", "service_level_9cell",
        ]
        missing = [c for c in required_cols if c not in df_matrix.columns]
        if missing:
            log.error(f"Missing columns in abc_xyz_matrix: {missing}")
            log.error("Please re-run: python scripts/02_classify_skus.py")
            sys.exit(1)

        # Drop rows with NaN std_dev (shouldn't happen after filtering)
        before = len(df_matrix)
        df_matrix = df_matrix.dropna(subset=["std_daily_demand", "mean_daily_demand"])
        df_matrix = df_matrix[df_matrix["std_daily_demand"] > 0]
        log.info(f"   After NaN filter: {len(df_matrix):,} rows (removed {before - len(df_matrix)})")

        # Add unit_cost_proxy if missing
        if "unit_cost_proxy" not in df_matrix.columns:
            df_matrix["unit_cost_proxy"] = 1.0

        # Step 1: Compute safety stock for all scenarios (base lead time)
        log.info(f"\n▶  Computing safety stock (LT={BASE_LEAD_TIME} days)...")
        t0 = time.time()
        df_results = compute_all_scenarios(df_matrix, BASE_LEAD_TIME)
        log.info(f"   Computed {len(df_results):,} SKU-scenarios ({time.time()-t0:.1f}s)")

        # Step 2: Policy comparison
        policy_summary = compare_policies(df_results)

        # Step 3: Cell-level comparison
        cell_comparison = compare_by_cell(df_results)

        # Step 4: Sensitivity analyses
        sensitivity_lt = sensitivity_by_lead_time(df_matrix)
        sensitivity_sl = sensitivity_by_service_level(df_matrix)

        # Step 5: Write all results
        write_results(
            df_results, policy_summary, cell_comparison,
            sensitivity_lt, sensitivity_sl, conn
        )

        log.info("\n✅  Safety stock analysis complete!")
        log.info("\n   Output files in data/cleaned/:")
        log.info("     safety_stock_results.csv  — per-SKU SS, ROP, HC for all 3 policies")
        log.info("     policy_comparison.csv      — aggregate cost comparison")
        log.info("     cell_comparison.csv        — cost by ABC-XYZ cell")
        log.info("     sensitivity_lead_time.csv  — cost at 3/5/7/10 day LT")
        log.info("     sensitivity_service_level.csv — cost curve vs SL (for paper chart)")
        log.info("\n   Next step: python scripts/04_promotion_impact.py")
        log.info("   Or: launch the Streamlit app: streamlit run agent/app.py")

    except Exception as e:
        log.error(f"❌  Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
