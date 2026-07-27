"""
tools.py
--------
Inventory optimization computation functions for the LLM agent.
Each function:
  - Queries SQLite directly
  - Returns a dict with numerical results AND a human-readable summary string
  - Is independently testable with known inputs
  - Validates inputs before execution

Design constraint: each function returns a COMPLETE answer.
The agent should need at most 1-2 tool calls per user query.
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Optional
import json

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "inventory.db"

HOLDING_COST_RATE = 0.22
BASE_LEAD_TIME = 7

CELL_SERVICE_LEVELS = {
    "AX": 0.99, "AY": 0.97, "AZ": 0.95,
    "BX": 0.95, "BY": 0.93, "BZ": 0.90,
    "CX": 0.90, "CY": 0.88, "CZ": 0.85,
}

CELL_Z_SCORES = {
    "AX": 2.326, "AY": 1.881, "AZ": 1.645,
    "BX": 1.645, "BY": 1.476, "BZ": 1.282,
    "CX": 1.282, "CY": 1.175, "CZ": 1.036,
}


def _get_conn() -> sqlite3.Connection:
    """Return a SQLite connection with performance settings."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA cache_size = -32768")
    return conn


def _validate_store(store_id: int, conn: sqlite3.Connection) -> bool:
    """Check if store_id exists in the database."""
    row = conn.execute(
        "SELECT COUNT(*) FROM abc_xyz_matrix WHERE store_nbr = ?",
        (store_id,)
    ).fetchone()
    return row[0] > 0


# ---------------------------------------------------------------------------
# Tool 1: ABC-XYZ Classification Matrix
# ---------------------------------------------------------------------------
def get_classification(store_id: int) -> dict:
    """
    Returns the 9-cell ABC-XYZ matrix for a given store with:
    - SKU count per cell
    - Total revenue per cell
    - Revenue share %
    - Current service level policy
    - Total holding cost per cell (9-cell policy)

    Args:
        store_id: Store number (integer)

    Returns:
        dict with 'matrix', 'totals', 'summary' keys
    """
    conn = _get_conn()
    try:
        if not _validate_store(store_id, conn):
            return {
                "error": f"Store {store_id} not found in database.",
                "summary": f"Store {store_id} does not exist. Available stores: "
                           + str([r[0] for r in conn.execute("SELECT DISTINCT store_nbr FROM abc_xyz_matrix ORDER BY store_nbr").fetchall()])
            }

        df = pd.read_sql("""
            SELECT
                abc_class, xyz_class, cell,
                COUNT(*) AS num_skus,
                SUM(revenue_proxy) AS total_revenue,
                AVG(cv) AS avg_cv,
                service_level_9cell,
                z_score_9cell
            FROM abc_xyz_matrix
            WHERE store_nbr = ?
            GROUP BY abc_class, xyz_class, cell, service_level_9cell, z_score_9cell
            ORDER BY abc_class, xyz_class
        """, conn, params=(store_id,))

        total_skus = df["num_skus"].sum()
        total_rev  = df["total_revenue"].sum()
        df["revenue_share_pct"] = (df["total_revenue"] / total_rev * 100).round(2)
        df["sku_share_pct"]     = (df["num_skus"] / total_skus * 100).round(2)

        # Also pull holding cost from safety_stock_results if available
        try:
            hc_df = pd.read_sql("""
                SELECT cell, SUM(hc_9cell) AS total_hc_9cell
                FROM safety_stock_results
                WHERE store_nbr = ?
                GROUP BY cell
            """, conn, params=(store_id,))
            df = df.merge(hc_df, on="cell", how="left")
        except Exception:
            df["total_hc_9cell"] = None

        matrix_dict = df.to_dict(orient="records")

        # Build human-readable summary
        top_cell = df.sort_values("total_revenue", ascending=False).iloc[0]
        bottom_cell = df.sort_values("revenue_proxy" if "revenue_proxy" in df else "total_revenue").iloc[0]

        a_skus = df[df["abc_class"]=="A"]["num_skus"].sum()
        a_rev  = df[df["abc_class"]=="A"]["revenue_share_pct"].sum()

        summary = (
            f"Store {store_id} ABC-XYZ Classification:\n"
            f"  Total SKUs analyzed: {total_skus:,}\n"
            f"  A-class: {int(a_skus)} SKUs ({a_rev:.1f}% of revenue)\n"
            f"  Highest revenue cell: {top_cell['cell']} "
            f"({int(top_cell['num_skus'])} SKUs, {top_cell['revenue_share_pct']:.1f}% of revenue)\n"
            f"  Policy: AX=99%, AY=97%, AZ=95%, BX=95%, BY=93%, BZ=90%, CX=90%, CY=88%, CZ=85%"
        )

        return {
            "store_id": store_id,
            "total_skus": int(total_skus),
            "total_revenue_proxy": round(float(total_rev), 2),
            "matrix": matrix_dict,
            "summary": summary,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 2: Safety Stock Calculator
# ---------------------------------------------------------------------------
def calculate_safety_stock(
    store_id: int,
    family: Optional[str] = None,
    service_level: Optional[float] = None,
    lead_time: int = BASE_LEAD_TIME,
) -> dict:
    """
    Returns safety stock, ROP, and annual holding cost for a store.
    Optionally filter by product family and/or override service level.

    Args:
        store_id:      Store number
        family:        Product family filter (optional, e.g. 'BEVERAGES')
        service_level: Override service level 0.0-1.0 (optional)
        lead_time:     Lead time in days (default 7)

    Returns:
        dict with per-SKU results and aggregate totals
    """
    conn = _get_conn()
    try:
        if not _validate_store(store_id, conn):
            return {"error": f"Store {store_id} not found.", "summary": f"Store {store_id} does not exist."}

        if lead_time < 1 or lead_time > 90:
            return {"error": "lead_time must be between 1 and 90 days.", "summary": "Invalid lead time."}

        if service_level is not None and (service_level < 0.5 or service_level > 0.999):
            return {"error": "service_level must be between 0.5 and 0.999.", "summary": "Invalid service level."}

        # Load matrix data
        query = "SELECT * FROM abc_xyz_matrix WHERE store_nbr = ?"
        params = [store_id]
        if family:
            query += " AND UPPER(family) = UPPER(?)"
            params.append(family)

        df = pd.read_sql(query, conn, params=params)

        if df.empty:
            return {
                "error": f"No data found for store {store_id}" + (f", family {family}" if family else ""),
                "summary": f"No matching SKUs found."
            }

        # Compute SS/ROP using provided or cell-specific service level
        if service_level is not None:
            z = float(stats.norm.ppf(service_level))
            df["z_used"] = z
            df["sl_used"] = service_level
        else:
            df["z_used"] = df["z_score_9cell"]
            df["sl_used"] = df["service_level_9cell"]

        df["std_during_lt"]  = df["std_daily_demand"] * np.sqrt(lead_time)
        df["safety_stock"]   = df["z_used"] * df["std_during_lt"]
        df["reorder_point"]  = df["mean_daily_demand"] * lead_time + df["safety_stock"]
        df["annual_holding"] = df["safety_stock"] * df.get("unit_cost_proxy", 1.0) * HOLDING_COST_RATE

        totals = {
            "total_safety_stock_units": round(df["safety_stock"].sum(), 2),
            "total_annual_holding_cost": round(df["annual_holding"].sum(), 2),
            "num_skus": len(df),
            "avg_safety_stock_per_sku": round(df["safety_stock"].mean(), 2),
        }

        # Top 10 by holding cost
        top10 = df.nlargest(10, "annual_holding")[
            ["item_nbr", "family", "cell", "mean_daily_demand",
             "safety_stock", "reorder_point", "annual_holding"]
        ].round(3).to_dict(orient="records")

        summary = (
            f"Safety Stock Analysis — Store {store_id}"
            + (f", Family: {family}" if family else "")
            + f" (Lead Time: {lead_time} days):\n"
            f"  SKUs analyzed: {totals['num_skus']:,}\n"
            f"  Total safety stock: {totals['total_safety_stock_units']:,.1f} units\n"
            f"  Total annual holding cost: {totals['total_annual_holding_cost']:,.2f} (unit-cost proxy)\n"
            f"  Avg SS per SKU: {totals['avg_safety_stock_per_sku']:.2f} units\n"
            f"  Service level: {'custom ' + str(service_level) if service_level else '9-cell differentiated'}\n"
        )

        return {
            "store_id": store_id,
            "family_filter": family,
            "lead_time": lead_time,
            "service_level_used": float(service_level) if service_level else "differentiated",
            "totals": totals,
            "top10_by_holding_cost": top10,
            "summary": summary,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 3: Policy Change Simulator
# ---------------------------------------------------------------------------
def simulate_policy_change(
    store_id: int,
    segment: str,
    new_service_level: float,
) -> dict:
    """
    Returns cost delta vs. current 9-cell policy if service level for
    a specific ABC-XYZ cell (segment) is changed.

    Args:
        store_id:          Store number
        segment:           ABC-XYZ cell (e.g., 'AX', 'CZ', 'BY')
        new_service_level: New service level (0.5 to 0.999)

    Returns:
        dict with current_cost, new_cost, delta, and human-readable summary
    """
    conn = _get_conn()
    try:
        segment = segment.upper()
        if segment not in CELL_SERVICE_LEVELS:
            return {
                "error": f"Invalid segment '{segment}'. Must be one of: {list(CELL_SERVICE_LEVELS.keys())}",
                "summary": f"'{segment}' is not a valid ABC-XYZ cell."
            }

        if new_service_level < 0.5 or new_service_level > 0.999:
            return {"error": "new_service_level must be between 0.5 and 0.999.", "summary": "Invalid service level."}

        # Load SKUs in that cell
        df = pd.read_sql("""
            SELECT * FROM safety_stock_results
            WHERE store_nbr = ? AND cell = ?
        """, conn, params=(store_id, segment))

        if df.empty:
            # Fall back to abc_xyz_matrix
            df = pd.read_sql("""
                SELECT * FROM abc_xyz_matrix
                WHERE store_nbr = ? AND cell = ?
            """, conn, params=(store_id, segment))
            if df.empty:
                return {
                    "error": f"No SKUs found for store {store_id}, cell {segment}.",
                    "summary": f"Store {store_id} has no SKUs in cell {segment}."
                }
            # Compute SS from scratch
            z_current = CELL_Z_SCORES[segment]
            df["ss_9cell"] = z_current * df["std_daily_demand"] * np.sqrt(BASE_LEAD_TIME)
            df["hc_9cell"] = df["ss_9cell"] * HOLDING_COST_RATE

        current_hc   = float(df["hc_9cell"].sum())
        current_ss   = float(df["ss_9cell"].sum())
        current_sl   = CELL_SERVICE_LEVELS[segment]
        current_z    = CELL_Z_SCORES[segment]

        # Compute new SS with changed service level
        new_z        = float(stats.norm.ppf(new_service_level))
        std_col      = "std_demand" if "std_demand" in df.columns else "std_daily_demand"
        new_ss       = (new_z * df[std_col] * np.sqrt(BASE_LEAD_TIME)).sum()
        new_hc       = float(new_ss * HOLDING_COST_RATE)

        delta_hc     = new_hc - current_hc
        delta_ss     = float(new_ss) - current_ss
        delta_hc_pct = (delta_hc / current_hc * 100) if current_hc > 0 else 0

        direction = "increase" if delta_hc > 0 else "decrease"
        risk_note = (
            "⚠️  Higher service level increases holding cost but reduces stockout risk."
            if new_service_level > current_sl else
            "⚠️  Lower service level reduces holding cost but increases stockout risk."
        )

        summary = (
            f"Policy Change Simulation — Store {store_id}, Cell {segment}:\n"
            f"  Current service level: {current_sl:.0%} (z={current_z:.3f})\n"
            f"  Proposed service level: {new_service_level:.0%} (z={new_z:.3f})\n"
            f"  SKUs affected: {len(df):,}\n"
            f"  Current annual holding cost: {current_hc:,.2f}\n"
            f"  New annual holding cost: {new_hc:,.2f}\n"
            f"  Delta: {delta_hc:+,.2f} ({delta_hc_pct:+.1f}%)\n"
            f"  Safety stock change: {delta_ss:+,.1f} units\n"
            f"  {risk_note}"
        )

        return {
            "store_id": store_id,
            "segment": segment,
            "num_skus_affected": len(df),
            "current_service_level": current_sl,
            "new_service_level": new_service_level,
            "current_annual_holding_cost": round(current_hc, 2),
            "new_annual_holding_cost": round(new_hc, 2),
            "delta_holding_cost": round(delta_hc, 2),
            "delta_holding_cost_pct": round(delta_hc_pct, 2),
            "current_safety_stock_units": round(current_ss, 2),
            "new_safety_stock_units": round(float(new_ss), 2),
            "delta_safety_stock_units": round(delta_ss, 2),
            "summary": summary,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 4: Overstock Alert
# ---------------------------------------------------------------------------
def get_overstock_alerts(store_id: int, top_n: int = 10) -> dict:
    """
    Flags SKUs where holding cost is disproportionate to revenue contribution.
    Specifically: C-class items with high safety stock (high CV × high demand).

    Args:
        store_id: Store number
        top_n:    Number of top overstocked SKUs to return (default 10)

    Returns:
        dict with list of overstock candidates and summary
    """
    conn = _get_conn()
    try:
        if not _validate_store(store_id, conn):
            return {"error": f"Store {store_id} not found.", "summary": f"Store {store_id} does not exist."}

        try:
            df = pd.read_sql("""
                SELECT
                    m.store_nbr, m.item_nbr, m.family, m.abc_class, m.xyz_class, m.cell,
                    m.revenue_proxy, m.cv, m.mean_daily_demand, m.std_daily_demand,
                    s.ss_9cell, s.hc_9cell, s.rop_9cell
                FROM abc_xyz_matrix m
                LEFT JOIN safety_stock_results s
                    ON m.store_nbr = s.store_nbr AND m.item_nbr = s.item_nbr
                WHERE m.store_nbr = ?
            """, conn, params=(store_id,))
        except Exception:
            df = pd.read_sql("""
                SELECT *, 0 AS ss_9cell, 0 AS hc_9cell, 0 AS rop_9cell
                FROM abc_xyz_matrix WHERE store_nbr = ?
            """, conn, params=(store_id,))

        # Overstock score: holding_cost / revenue_proxy ratio
        df["hc_to_rev_ratio"] = (
            df["hc_9cell"].fillna(0) /
            df["revenue_proxy"].replace(0, np.nan).fillna(1)
        )

        # Focus on C-class or high-CV items with disproportionate holding cost
        overstock_candidates = df[
            (df["abc_class"] == "C") |
            (df["xyz_class"] == "Z")
        ].nlargest(top_n, "hc_to_rev_ratio")

        alerts = overstock_candidates[
            ["item_nbr", "family", "cell", "abc_class", "xyz_class",
             "revenue_proxy", "hc_9cell", "hc_to_rev_ratio", "cv"]
        ].round(4).to_dict(orient="records")

        total_c_hc = df[df["abc_class"]=="C"]["hc_9cell"].fillna(0).sum()
        total_c_rev = df[df["abc_class"]=="C"]["revenue_proxy"].sum()

        summary = (
            f"Overstock Alert — Store {store_id}:\n"
            f"  C-class items: {len(df[df['abc_class']=='C']):,} SKUs\n"
            f"  C-class total holding cost: {total_c_hc:,.2f}\n"
            f"  C-class total revenue: {total_c_rev:,.2f}\n"
            f"  HC/Revenue ratio for C-class: {total_c_hc/max(total_c_rev,1):.4f}\n"
            f"  Top overstock candidates (by HC/Revenue ratio):\n"
            + "\n".join(
                f"    {a['item_nbr']} ({a['family']}, {a['cell']}): "
                f"HC={a['hc_9cell']:.2f}, Rev={a['revenue_proxy']:.2f}, Ratio={a['hc_to_rev_ratio']:.4f}"
                for a in alerts[:5]
            )
        )

        return {
            "store_id": store_id,
            "total_c_class_skus": len(df[df["abc_class"]=="C"]),
            "total_z_class_skus": len(df[df["xyz_class"]=="Z"]),
            "top_overstock_candidates": alerts,
            "summary": summary,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 5: Promotion Adjustment
# ---------------------------------------------------------------------------
def get_promotion_adjustment(store_id: int, family: str, lead_time: int = BASE_LEAD_TIME) -> dict:
    """
    Returns recommended temporary ROP increase during promotions for a
    specific product family, based on historical promotion lift.

    Args:
        store_id:  Store number
        family:    Product family (e.g., 'BEVERAGES')
        lead_time: Lead time in days (default 7)

    Returns:
        dict with baseline vs. promo demand, recommended buffer, and summary
    """
    conn = _get_conn()
    try:
        # Load daily demand with promo flag
        df = pd.read_sql("""
            SELECT
                item_nbr,
                on_promotion,
                AVG(unit_sales) AS avg_demand,
                COUNT(*) AS num_days
            FROM daily_demand
            WHERE store_nbr = ? AND UPPER(family) = UPPER(?)
            GROUP BY item_nbr, on_promotion
        """, conn, params=(store_id, family))

        if df.empty:
            return {
                "error": f"No data for store {store_id}, family {family}.",
                "summary": f"No demand data found. Check store_id and family name."
            }

        # Pivot to compare promo vs. non-promo
        baseline = df[df["on_promotion"]==0].groupby("item_nbr")["avg_demand"].mean()
        promo    = df[df["on_promotion"]==1].groupby("item_nbr")["avg_demand"].mean()

        # Items with both baseline and promo data
        common = baseline.index.intersection(promo.index)
        if len(common) == 0:
            lift_pct = 0.0
        else:
            lift_pct = float((promo[common] / baseline[common] - 1).mean() * 100)

        # Aggregate stats
        avg_baseline_demand = float(df[df["on_promotion"]==0]["avg_demand"].mean())
        avg_promo_demand    = float(df[df["on_promotion"]==1]["avg_demand"].mean()) if (df["on_promotion"]==1).any() else avg_baseline_demand
        num_items           = df["item_nbr"].nunique()
        promo_items         = int((df["on_promotion"]==1).any())

        # Recommended ROP increase during promotion
        # Use demand during promo lead time as new demand estimate
        promo_demand_during_lt = avg_promo_demand * lead_time
        baseline_demand_during_lt = avg_baseline_demand * lead_time
        recommended_buffer_increase = promo_demand_during_lt - baseline_demand_during_lt

        summary = (
            f"Promotion Adjustment — Store {store_id}, Family: {family.upper()}:\n"
            f"  Items analyzed: {num_items:,}\n"
            f"  Baseline avg daily demand: {avg_baseline_demand:.2f} units/day\n"
            f"  Promo avg daily demand: {avg_promo_demand:.2f} units/day\n"
            f"  Demand lift during promotion: {lift_pct:+.1f}%\n"
            f"  Lead time: {lead_time} days\n"
            f"  Baseline ROP demand component: {baseline_demand_during_lt:.1f} units\n"
            f"  Promo ROP demand component: {promo_demand_during_lt:.1f} units\n"
            f"  Recommended ROP buffer increase: +{recommended_buffer_increase:.1f} units per SKU\n"
            + ("  ⚠️  Promotions significantly increase demand — pre-position stock." if lift_pct > 20 else "")
        )

        return {
            "store_id": store_id,
            "family": family.upper(),
            "num_items_in_family": num_items,
            "avg_baseline_daily_demand": round(avg_baseline_demand, 4),
            "avg_promo_daily_demand": round(avg_promo_demand, 4),
            "demand_lift_pct": round(lift_pct, 2),
            "lead_time_days": lead_time,
            "baseline_demand_during_lt": round(baseline_demand_during_lt, 2),
            "promo_demand_during_lt": round(promo_demand_during_lt, 2),
            "recommended_rop_increase": round(recommended_buffer_increase, 2),
            "summary": summary,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 6: Policy Comparison (summary)
# ---------------------------------------------------------------------------
def compare_policies_summary(store_id: int) -> dict:
    """
    Returns aggregate cost comparison between uniform, 3-tier, and 9-cell
    policies for a given store.

    Args:
        store_id: Store number

    Returns:
        dict with cost comparison table and summary
    """
    conn = _get_conn()
    try:
        if not _validate_store(store_id, conn):
            return {"error": f"Store {store_id} not found.", "summary": f"Store {store_id} does not exist."}

        try:
            df = pd.read_sql("""
                SELECT
                    SUM(hc_uniform) AS hc_uniform,
                    SUM(hc_3tier)   AS hc_3tier,
                    SUM(hc_9cell)   AS hc_9cell,
                    SUM(ss_uniform) AS ss_uniform,
                    SUM(ss_3tier)   AS ss_3tier,
                    SUM(ss_9cell)   AS ss_9cell,
                    COUNT(*)        AS num_skus
                FROM safety_stock_results
                WHERE store_nbr = ?
            """, conn, params=(store_id,))
        except Exception:
            return {
                "error": "Safety stock results not found. Please run scripts/03_safety_stock.py first.",
                "summary": "Run the safety stock analysis first."
            }

        row = df.iloc[0]
        uniform = float(row["hc_uniform"] or 0)
        tier3   = float(row["hc_3tier"] or 0)
        cell9   = float(row["hc_9cell"] or 0)

        results = [
            {"policy": "Uniform (95%)",     "total_hc": round(uniform, 2), "vs_uniform_pct": 0.0},
            {"policy": "3-Tier (ABC)",       "total_hc": round(tier3, 2),  "vs_uniform_pct": round((tier3-uniform)/max(uniform,1)*100, 2)},
            {"policy": "9-Cell (ABC-XYZ)",   "total_hc": round(cell9, 2),  "vs_uniform_pct": round((cell9-uniform)/max(uniform,1)*100, 2)},
        ]

        best = min(results, key=lambda x: x["total_hc"])

        summary = (
            f"Policy Comparison — Store {store_id} ({int(row['num_skus']):,} SKUs):\n"
            f"  Uniform policy:   HC = {uniform:,.2f}\n"
            f"  3-Tier policy:    HC = {tier3:,.2f} ({(tier3-uniform)/max(uniform,1)*100:+.1f}% vs Uniform)\n"
            f"  9-Cell policy:    HC = {cell9:,.2f} ({(cell9-uniform)/max(uniform,1)*100:+.1f}% vs Uniform)\n"
            f"  Recommended: {best['policy']} (lowest holding cost)\n"
            f"  Note: Lower HC from 9-cell policy comes from right-sizing stock by segment,\n"
            f"  not from reducing service on A-class items."
        )

        return {
            "store_id": store_id,
            "num_skus": int(row["num_skus"]),
            "policy_comparison": results,
            "recommended_policy": best["policy"],
            "summary": summary,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 7: Calculate Math
# ---------------------------------------------------------------------------
def calculate_math(expression: str) -> dict:
    """
    Evaluates a mathematical expression.
    
    Args:
        expression: A mathematical expression string (e.g., '2 + 2', '25 * 4.5')
        
    Returns:
        dict with result and summary
    """
    try:
        # Restrict eval for safety
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Only basic math operators allowed.", "summary": "Invalid characters in expression."}
        
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "summary": f"The result of {expression} is {result}."
        }
    except Exception as e:
        return {"error": f"Failed to evaluate expression: {e}", "summary": f"Could not calculate {expression}."}


# ---------------------------------------------------------------------------
# Tool registry for Ollama
# ---------------------------------------------------------------------------
TOOL_REGISTRY = [
    get_classification,
    calculate_safety_stock,
    simulate_policy_change,
    get_overstock_alerts,
    get_promotion_adjustment,
    compare_policies_summary,
    calculate_math,
]


def dispatch_tool(tool_name: str, args: dict) -> dict:
    """Dispatch a tool call by name with given arguments."""
    tool_map = {f.__name__: f for f in TOOL_REGISTRY}
    if tool_name not in tool_map:
        return {"error": f"Unknown tool: {tool_name}", "summary": f"Tool '{tool_name}' does not exist."}
    try:
        return tool_map[tool_name](**args)
    except TypeError as e:
        return {"error": f"Invalid arguments for {tool_name}: {e}", "summary": str(e)}
    except Exception as e:
        return {"error": f"Tool execution failed: {e}", "summary": str(e)}
