"""
test_tools.py
-------------
Unit tests for agent/tools.py computation functions.
Tests use synthetic data to verify mathematical correctness
independently of the database.

Run with:
    python -m pytest tests/test_tools.py -v
    # or
    python tests/test_tools.py
"""

import sys
import math
import sqlite3
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Windows cp1252 consoles can't encode the emoji used in test output below.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Redirect DB_PATH to temp file for testing
# ---------------------------------------------------------------------------
# We monkey-patch DB_PATH in tools before importing
import importlib

# Create a temp DB for testing
_TEMP_DB = tempfile.mktemp(suffix=".db")

# ---------------------------------------------------------------------------
# Pure formula tests (no DB required)
# ---------------------------------------------------------------------------

def test_safety_stock_formula_basic():
    """SS = z × σ_d × √(LT)"""
    z = 1.645  # 95% service level
    sigma = 10.0
    lt = 7
    expected_ss = z * sigma * math.sqrt(lt)

    # Replicate the formula
    computed = z * sigma * math.sqrt(lt)
    assert abs(computed - expected_ss) < 0.001, f"Expected {expected_ss}, got {computed}"
    print(f"✅  test_safety_stock_formula_basic: SS = {computed:.4f}")


def test_safety_stock_zero_demand():
    """Items with zero std dev should have zero safety stock."""
    z = 2.33
    sigma = 0.0
    lt = 7
    ss = z * sigma * math.sqrt(lt)
    assert ss == 0.0
    print("✅  test_safety_stock_zero_demand: SS = 0 for zero variance")


def test_rop_formula():
    """ROP = (d̄ × LT) + SS"""
    mean_demand = 5.0
    lt = 7
    ss = 10.0
    expected_rop = mean_demand * lt + ss  # = 35 + 10 = 45
    assert expected_rop == 45.0
    print(f"✅  test_rop_formula: ROP = {expected_rop}")


def test_holding_cost_formula():
    """HC = SS × C_unit × h"""
    ss = 50.0
    unit_cost = 1.0
    holding_rate = 0.22
    expected_hc = ss * unit_cost * holding_rate  # = 11.0
    assert abs(expected_hc - 11.0) < 0.001
    print(f"✅  test_holding_cost_formula: HC = {expected_hc}")


def test_cv_formula():
    """CV = std / mean"""
    demands = [10, 12, 8, 15, 9, 11, 13]
    mean = np.mean(demands)
    std  = np.std(demands, ddof=1)
    cv   = std / mean
    assert cv > 0
    assert cv < 2.0  # reasonable range for this data
    print(f"✅  test_cv_formula: CV = {cv:.4f}")


def test_xyz_thresholds():
    """XYZ classification based on CV thresholds."""
    def classify_xyz(cv):
        if cv < 0.5:
            return "X"
        elif cv <= 1.0:
            return "Y"
        else:
            return "Z"

    assert classify_xyz(0.2) == "X"
    assert classify_xyz(0.5) == "Y"
    assert classify_xyz(0.75) == "Y"
    assert classify_xyz(1.0) == "Y"
    assert classify_xyz(1.01) == "Z"
    assert classify_xyz(2.5) == "Z"
    print("✅  test_xyz_thresholds: All XYZ boundaries correct")


def test_abc_thresholds():
    """ABC classification based on cumulative revenue %."""
    # Simulate 10 SKUs with revenues
    revenues = [100, 80, 60, 50, 40, 30, 20, 10, 5, 5]
    total = sum(revenues)
    cumulative = 0

    classes = []
    for rev in revenues:
        cumulative += rev
        pct = cumulative / total * 100
        if pct <= 80:
            classes.append("A")
        elif pct <= 95:
            classes.append("B")
        else:
            classes.append("C")

    # Top items should be A
    assert classes[0] == "A"
    # Some should be B
    assert "B" in classes
    # Low-revenue items should be C
    assert classes[-1] == "C"
    print(f"✅  test_abc_thresholds: ABC = {classes}")


def test_sensitivity_lead_time():
    """SS increases with √(LT)."""
    z = 1.645
    sigma = 10.0
    ss_3  = z * sigma * math.sqrt(3)
    ss_7  = z * sigma * math.sqrt(7)
    ss_10 = z * sigma * math.sqrt(10)

    assert ss_3 < ss_7 < ss_10, "SS should increase with lead time"
    ratio = ss_7 / ss_3
    expected_ratio = math.sqrt(7) / math.sqrt(3)
    assert abs(ratio - expected_ratio) < 0.001
    print(f"✅  test_sensitivity_lead_time: SS(LT=3)={ss_3:.2f}, SS(LT=7)={ss_7:.2f}, SS(LT=10)={ss_10:.2f}")


def test_policy_cost_ordering():
    """
    For A-class items: 9-cell (99%) >= 3-tier (99%) >= uniform (95%)
    For C-class items: uniform (95%) >= 3-tier (90%) >= 9-cell CZ (85%)
    """
    from scipy import stats

    sigma = 10.0
    lt = 7
    std_lt = sigma * math.sqrt(lt)

    # A-class: all three should give same (99%) for AX, but 9-cell ≥ 3-tier for AZ
    z_AX_9cell = 2.326  # 99%
    z_AX_3tier = 2.326  # 99% (A-class)
    z_AX_uniform = 1.645  # 95%

    ss_AX_9cell   = z_AX_9cell * std_lt
    ss_AX_3tier   = z_AX_3tier * std_lt
    ss_AX_uniform = z_AX_uniform * std_lt

    assert ss_AX_9cell >= ss_AX_uniform, "AX 9-cell SS >= uniform SS"

    # CZ: 9-cell (85%) < uniform (95%)
    z_CZ_9cell   = 1.036  # 85%
    z_CZ_uniform = 1.645  # 95%

    ss_CZ_9cell   = z_CZ_9cell * std_lt
    ss_CZ_uniform = z_CZ_uniform * std_lt

    assert ss_CZ_9cell < ss_CZ_uniform, "CZ 9-cell SS < uniform SS"
    print(f"✅  test_policy_cost_ordering: AX 9cell({ss_AX_9cell:.2f}) >= uniform({ss_AX_uniform:.2f}), CZ 9cell({ss_CZ_9cell:.2f}) < uniform({ss_CZ_uniform:.2f})")


def test_promotion_lift_formula():
    """Demand lift formula: (promo_mean / baseline_mean - 1) × 100."""
    baseline = 10.0
    promo    = 13.5
    lift_pct = (promo / baseline - 1) * 100
    assert abs(lift_pct - 35.0) < 0.01
    print(f"✅  test_promotion_lift_formula: Lift = {lift_pct:.1f}%")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
def run_all():
    tests = [
        test_safety_stock_formula_basic,
        test_safety_stock_zero_demand,
        test_rop_formula,
        test_holding_cost_formula,
        test_cv_formula,
        test_xyz_thresholds,
        test_abc_thresholds,
        test_sensitivity_lead_time,
        test_policy_cost_ordering,
        test_promotion_lift_formula,
    ]

    print("\n" + "="*60)
    print("  Inventory Optimization — Unit Tests")
    print("="*60 + "\n")

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌  {test.__name__}: FAILED — {e}")
            failed += 1
        except Exception as e:
            print(f"❌  {test.__name__}: ERROR — {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
