"""
download_data.py
----------------
Downloads the Corporación Favorita Grocery Sales dataset from Kaggle.

Requirements:
    - kaggle.json API key placed at C:\\Users\\<user>\\.kaggle\\kaggle.json
      (get yours at https://www.kaggle.com/settings → API → Create New Token)
    - kaggle package installed: pip install kaggle

Usage:
    python scripts/download_data.py
"""

import os
import sys
import zipfile
import hashlib
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
KAGGLE_COMPETITION = "store-sales-time-series-forecasting"

# Expected files after extraction
EXPECTED_FILES = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "items.csv",
    "transactions.csv",
    "oil.csv",
    "holidays_events.csv",
    "sample_submission.csv",
]

# Approximate size checks (in MB) - helps detect corrupt/incomplete downloads
MIN_FILE_SIZES_MB = {
    "train.csv": 200,    # ~450 MB actual
    "stores.csv": 0.001,
    "items.csv": 0.1,
    "transactions.csv": 1,
    "oil.csv": 0.01,
    "holidays_events.csv": 0.01,
}


def check_kaggle_auth() -> bool:
    """Verify Kaggle API credentials exist."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("❌  ERROR: Kaggle API key not found.")
        print(f"    Expected at: {kaggle_json}")
        print()
        print("    To fix:")
        print("    1. Go to https://www.kaggle.com/settings")
        print("    2. Scroll to 'API' section")
        print("    3. Click 'Create New Token'")
        print("    4. Save the downloaded kaggle.json to your ~/.kaggle/ folder")
        print(f"    5. Run: mkdir {kaggle_dir} && move kaggle.json {kaggle_dir}")
        return False

    # Check file permissions (should be 600 on Unix; on Windows just check it's readable)
    print(f"✅  Kaggle API key found at {kaggle_json}")
    return True


def check_already_downloaded() -> bool:
    """Check if all expected files already exist and are valid size."""
    missing = []
    too_small = []

    for fname in EXPECTED_FILES:
        fpath = RAW_DATA_DIR / fname
        if not fpath.exists():
            missing.append(fname)
            continue
        size_mb = fpath.stat().st_size / (1024 * 1024)
        min_size = MIN_FILE_SIZES_MB.get(fname, 0)
        if size_mb < min_size:
            too_small.append(f"{fname} ({size_mb:.1f} MB, expected >{min_size} MB)")

    if missing:
        print(f"⚠️   Missing files: {missing}")
        return False
    if too_small:
        print(f"⚠️   Files too small (possibly corrupt): {too_small}")
        return False

    print("✅  All expected files already present in data/raw/")
    return True


def download_dataset() -> bool:
    """Download the dataset from Kaggle."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("❌  kaggle package not found. Install with: pip install kaggle")
        return False

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n📥  Downloading '{KAGGLE_COMPETITION}' dataset...")
    print(f"     Destination: {RAW_DATA_DIR}")
    print("     This may take a while (~1-2 GB total)...\n")

    try:
        os.system(
            f'"{sys.executable}" -m kaggle competitions download '
            f'-c {KAGGLE_COMPETITION} -p "{RAW_DATA_DIR}"'
        )
        return True
    except Exception as e:
        print(f"❌  Download failed: {e}")
        return False


def extract_zip() -> bool:
    """Extract downloaded zip file."""
    zip_files = list(RAW_DATA_DIR.glob("*.zip"))
    if not zip_files:
        print("ℹ️   No zip file found — files may already be extracted.")
        return True

    for zip_path in zip_files:
        print(f"\n📦  Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # List contents
                contents = zf.namelist()
                print(f"     Files in archive: {contents}")
                zf.extractall(RAW_DATA_DIR)
            print(f"✅  Extracted to {RAW_DATA_DIR}")
            # Remove zip to save disk space
            zip_path.unlink()
            print(f"🗑️   Removed {zip_path.name}")
        except zipfile.BadZipFile:
            print(f"❌  {zip_path.name} is corrupted. Please re-download.")
            return False

    return True


def verify_files() -> bool:
    """Verify all expected files are present and plausibly sized."""
    print("\n🔍  Verifying downloaded files...")
    all_ok = True

    for fname in EXPECTED_FILES:
        fpath = RAW_DATA_DIR / fname
        if not fpath.exists():
            print(f"   ❌  MISSING: {fname}")
            all_ok = False
            continue

        size_mb = fpath.stat().st_size / (1024 * 1024)
        min_size = MIN_FILE_SIZES_MB.get(fname, 0)

        if size_mb < min_size:
            print(f"   ⚠️   {fname}: {size_mb:.1f} MB (expected >{min_size} MB) — possibly corrupt")
            all_ok = False
        else:
            print(f"   ✅  {fname}: {size_mb:.1f} MB")

    return all_ok


def print_next_steps():
    print("\n" + "=" * 60)
    print("✅  Dataset ready! Next steps:")
    print("=" * 60)
    print()
    print("  1. Load data into SQLite:")
    print("     python scripts/01_load_data.py")
    print()
    print("  2. Run ABC-XYZ classification:")
    print("     python scripts/02_classify_skus.py")
    print()
    print("  3. Run safety stock model:")
    print("     python scripts/03_safety_stock.py")
    print()


def main():
    print("=" * 60)
    print("  Corporación Favorita Dataset Downloader")
    print("=" * 60)
    print()

    # Step 1: Check authentication
    if not check_kaggle_auth():
        sys.exit(1)

    # Step 2: Check if already downloaded
    if check_already_downloaded():
        print_next_steps()
        return

    # Step 3: Download
    if not download_dataset():
        sys.exit(1)

    # Step 4: Extract
    if not extract_zip():
        sys.exit(1)

    # Step 5: Verify
    if not verify_files():
        print("\n❌  Some files are missing or corrupt. Please re-run this script.")
        sys.exit(1)

    print_next_steps()


if __name__ == "__main__":
    main()
