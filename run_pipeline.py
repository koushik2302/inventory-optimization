"""
run_pipeline.py
---------------
Master pipeline runner. Executes all analysis steps in order.
Use this to run the full pipeline from scratch.

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --skip-download    # Skip data download (if already done)
    python run_pipeline.py --from-step 2      # Start from step 2
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Try to find the correct python executable
import shutil
_possible_pythons = [
    r"C:\Users\chunc\anaconda3\envs\inv-opt\python.exe",
    r"C:\Users\chunc\anaconda3\python.exe",
]
PYTHON = shutil.which("python") or shutil.which("python3") or sys.executable
for p in _possible_pythons:
    if Path(p).exists():
        PYTHON = p
        break


STEPS = [
    (1, "Download Dataset",        "scripts/download_data.py"),
    (2, "Load Data into SQLite",   "scripts/01_load_data.py"),
    (3, "ABC-XYZ Classification",  "scripts/02_classify_skus.py"),
    (4, "Safety Stock Model",      "scripts/03_safety_stock.py"),
    (5, "Promotion Impact",        "scripts/04_promotion_impact.py"),
]


def run_step(step_num: int, name: str, script: str) -> bool:
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        print(f"  ❌  Script not found: {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {name}")
    print(f"  Script: {script}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  ✅  Step {step_num} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌  Step {step_num} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


def run_tests() -> bool:
    print(f"\n{'='*60}")
    print(f"  Running Unit Tests")
    print(f"{'='*60}")
    result = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "tests" / "test_tools.py")],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Inventory Optimization Pipeline Runner")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the data download step (if data is already in data/raw/)")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Start from this step number (1-5)")
    parser.add_argument("--test-only", action="store_true",
                        help="Run unit tests only")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Inventory Optimization — Full Pipeline")
    print("="*60)
    print(f"  Python: {PYTHON}")
    print(f"  Project: {PROJECT_ROOT}")

    if args.test_only:
        success = run_tests()
        sys.exit(0 if success else 1)

    # Run unit tests first
    if not run_tests():
        print("\n❌  Unit tests failed. Fix errors before running pipeline.")
        sys.exit(1)

    t_total = time.time()

    for step_num, name, script in STEPS:
        if step_num < args.from_step:
            print(f"  [Skip] Step {step_num}: {name}")
            continue
        if step_num == 1 and args.skip_download:
            print(f"  [Skip] Step 1: Download (--skip-download flag set)")
            continue

        success = run_step(step_num, name, script)
        if not success:
            print(f"\n❌  Pipeline stopped at step {step_num}.")
            print(f"   Check the logs above for details.")
            sys.exit(1)

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ✅  PIPELINE COMPLETE in {total_time/60:.1f} minutes!")
    print(f"{'='*60}")
    print()
    print("  Next steps:")
    print("  1. Launch Streamlit app:")
    print(f"     {PYTHON} -m streamlit run agent/app.py")
    print()
    print("  2. Or start the interactive agent directly:")
    print(f"     {PYTHON} agent/agent.py")


if __name__ == "__main__":
    main()
