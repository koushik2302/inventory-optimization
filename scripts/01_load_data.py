"""
01_load_data.py
---------------
Loads the Corporación Favorita "Grocery Sales Forecasting" CSVs (the
ORIGINAL, item-level 125M-row Kaggle competition) into a SQLite database
and creates the daily_demand table. Currently configured for full-data
mode: all 54 stores, full 2013-2017 date range, no sales filtering (see
DATASET NOTE and Scope configuration below for why).

DATASET NOTE:
    data/raw_125m/favorita-grocery-sales-forecasting.zip is the original
    item-level competition (train.csv.7z -> train.csv):
        Columns: id, date, store_nbr, item_nbr, unit_sales, onpromotion
        125,497,040 rows, 54 stores, 4,100 items, 2013-01-01 to 2017-08-15
    Item family/class/perishable metadata comes from items.csv (real,
    not a synthetic store*1000+family mapping like the older 3M-row
    aggregated competition this script used to read).

    The raw train.csv (4.65 GB) does not fit comfortably in this
    machine's ~8 GB RAM, so it is streamed in chunks straight into
    SQLite (stg_train) — never materialized whole in pandas.

    Also builds data/inventory.duckdb — a native DuckDB copy of
    daily_demand, built directly from the raw CSVs (not from SQLite).
    This exists purely as an internal acceleration layer: 02_classify_skus.py
    and 04_promotion_impact.py's heavy full-table aggregations read from
    it instead of SQLite (benchmarked ~10-65x faster — DuckDB's columnar
    engine only gets its speed advantage over a native columnar store, not
    when bridging through SQLite's row-oriented storage). Everything else
    (agent, notebooks, Excel) still reads data/inventory.db unchanged —
    results computed via DuckDB are written back into the same SQLite
    tables/CSVs the rest of the pipeline expects.

Run AFTER: scripts/download_data.py (or manual extraction, see
    data/raw_125m/extracted/)
Run BEFORE: scripts/02_classify_skus.py

Usage:
    python scripts/01_load_data.py            # skips reload if data/inventory.db
                                                # already has the expected row count
    python scripts/01_load_data.py --force     # reload unconditionally
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import logging

import duckdb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_125m" / "extracted"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
DB_PATH = PROJECT_ROOT / "data" / "inventory.db"
DUCKDB_PATH = PROJECT_ROOT / "data" / "inventory.duckdb"

# Scope configuration: full available date range and ALL 54 stores / 22
# cities, to maximize use of the 125M-row item-level dataset. Only genuine
# data-quality filtering (zero/negative sales) is applied beyond this — no
# scoping filter removes rows for convenience. NOTE: this is a significant
# widening vs. the previously-published README/notebooks/paper scope
# (18 Quito stores, 2015-2016) — their numbers will need regenerating.
DATE_START = "2013-01-01"
DATE_END   = "2017-12-31"      # covers the full data span (last actual date 2017-08-15)
TARGET_CITY = None             # None = all 54 stores / 22 cities; or set a city name to scope
MIN_OBSERVATIONS = 30          # Minimum demand days per item-store pair (data-quality filter)
TRAIN_CHUNKSIZE = 1_000_000    # Rows per streamed chunk (~8GB RAM machine)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "data" / "load_log.txt", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def timer(msg: str):
    """Context manager to log elapsed time."""
    class _Timer:
        def __enter__(self):
            self._start = time.time()
            log.info(f">>  {msg}...")
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self._start
            log.info(f"   Done ({elapsed:.1f}s)")
    return _Timer()


def check_required_files() -> bool:
    """Verify all required CSV files exist before proceeding."""
    required = ["train.csv", "items.csv", "stores.csv", "oil.csv", "holidays_events.csv", "transactions.csv"]
    missing = [f for f in required if not (RAW_DATA_DIR / f).exists()]
    if missing:
        log.error(f"Missing files in {RAW_DATA_DIR}: {missing}")
        log.error("Extract data/raw_125m/favorita-grocery-sales-forecasting.zip (nested .7z members) first.")
        return False
    log.info(f"[OK]  All required CSV files found in {RAW_DATA_DIR}")
    return True


def count_raw_rows(csv_path: Path) -> int:
    """Fast newline count (no CSV parsing) — data rows only, header excluded."""
    count = 0
    with open(csv_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            count += chunk.count(b"\n")
    return count - 1  # header row


def already_loaded(expected_rows: int) -> bool:
    """
    Check whether data/inventory.db already has daily_demand loaded with the
    expected row count (i.e. a prior run of this script, with the current
    scope config, already completed). Used to skip the ~90-minute full
    reload when nothing has changed — pass --force to reload anyway.
    """
    if not DB_PATH.exists():
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='daily_demand'"
        ).fetchone()
        if row[0] == 0:
            return False
        actual_rows = conn.execute("SELECT COUNT(*) FROM daily_demand").fetchone()[0]
        conn.close()
        return actual_rows == expected_rows
    except sqlite3.Error:
        return False


def load_items(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load items.csv — real item-level metadata (family, class, perishable)."""
    with timer("Loading items.csv"):
        df = pd.read_csv(RAW_DATA_DIR / "items.csv")
        log.info(f"   Items: {len(df)} rows, families: {df['family'].nunique()}")
        df.to_sql("stg_items", conn, if_exists="replace", index=False)
    return df


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------
def load_stores(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load stores.csv."""
    with timer("Loading stores.csv"):
        df = pd.read_csv(RAW_DATA_DIR / "stores.csv")
        log.info(f"   Stores: {len(df)} rows, columns: {list(df.columns)}")
        df.to_sql("stg_stores", conn, if_exists="replace", index=False)
    return df


def load_oil(conn: sqlite3.Connection):
    """Load oil.csv."""
    with timer("Loading oil.csv"):
        df = pd.read_csv(RAW_DATA_DIR / "oil.csv", parse_dates=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df.to_sql("stg_oil", conn, if_exists="replace", index=False)
        log.info(f"   Oil prices: {len(df)} rows")


def load_holidays(conn: sqlite3.Connection):
    """Load holidays_events.csv."""
    with timer("Loading holidays_events.csv"):
        df = pd.read_csv(RAW_DATA_DIR / "holidays_events.csv", parse_dates=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df.to_sql("stg_holidays", conn, if_exists="replace", index=False)
        log.info(f"   Holidays: {len(df)} rows, types: {df['type'].value_counts().to_dict()}")


def load_transactions(conn: sqlite3.Connection):
    """Load transactions.csv."""
    with timer("Loading transactions.csv"):
        df = pd.read_csv(RAW_DATA_DIR / "transactions.csv", parse_dates=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df.to_sql("stg_transactions", conn, if_exists="replace", index=False)
        log.info(f"   Transactions: {len(df)} rows")


def load_train(conn: sqlite3.Connection, stores_df: pd.DataFrame) -> dict:
    """
    Stream the 125M-row item-level train.csv into SQLite in chunks, applying
    scope filters per chunk so the full file is never held in memory at once.

    Dataset schema: id, date, store_nbr, item_nbr, unit_sales, onpromotion
    Granularity: store x item x date (true item-level, not family-aggregated)

    Filters applied:
      - Date range: DATE_START to DATE_END
      - Stores: TARGET_CITY stores (or all, if TARGET_CITY is None)

    NOTE: zero/negative unit_sales (returns, no-sale days) are kept, not
    filtered here. daily_demand retains every raw row so the LLM agent can
    see the full history (including returns). The zero/negative-sales
    filter is instead applied only inside 02_classify_skus.py's demand-stats
    query, where it is statistically required (mean/std/CV formulas assume
    positive demand observations) — documented there.
    """
    if TARGET_CITY is None:
        scoped_stores = set(stores_df["store_nbr"].tolist())
        log.info(f"   FULL-DATA MODE: all {len(scoped_stores)} stores across {stores_df['city'].nunique()} cities")
    else:
        scoped_stores = set(stores_df[stores_df["city"] == TARGET_CITY]["store_nbr"].tolist())
        log.info(f"   Scoping to {len(scoped_stores)} {TARGET_CITY} stores: {sorted(scoped_stores)}")

    total_rows = 0
    rows_accepted = 0
    removed_date = 0
    removed_store = 0
    first_chunk = True

    with timer("Streaming train.csv (125M rows, chunked)"):
        reader = pd.read_csv(
            RAW_DATA_DIR / "train.csv",
            dtype={
                "id": "int64",
                "store_nbr": "int16",
                "item_nbr": "int32",
                "unit_sales": "float32",
                "onpromotion": "object",  # has NaNs in the original 2017 competition file
            },
            chunksize=TRAIN_CHUNKSIZE,
        )

        for i, chunk in enumerate(reader):
            total_rows += len(chunk)

            # date stays a string; DATE_START/END are ISO so string comparison
            # sorts identically to a real date comparison and avoids parsing
            # 125M timestamps we mostly discard.
            mask = (chunk["date"] >= DATE_START) & (chunk["date"] <= DATE_END)
            removed_date += int((~mask).sum())
            chunk = chunk[mask]

            mask = chunk["store_nbr"].isin(scoped_stores)
            removed_store += int((~mask).sum())
            chunk = chunk[mask]

            if len(chunk):
                chunk = chunk.rename(columns={"unit_sales": "unit_sales_val"})
                chunk["on_promotion"] = (
                    chunk["onpromotion"].map({"True": 1, "False": 0}).fillna(0).astype(int)
                )
                chunk = chunk.rename(columns={"unit_sales_val": "unit_sales"})
                chunk = chunk.drop(columns=["onpromotion"])

                chunk.to_sql(
                    "stg_train", conn,
                    if_exists="replace" if first_chunk else "append",
                    index=False, chunksize=5_000,
                )
                rows_accepted += len(chunk)
                first_chunk = False

            if (i + 1) % 20 == 0:
                log.info(f"     ...{total_rows:,} rows scanned, {rows_accepted:,} accepted so far")

        log.info(f"   Raw rows scanned: {total_rows:,}")
        log.info(f"   After filtering:")
        log.info(f"     Removed by date: {removed_date:,}")
        log.info(f"     Removed by store: {removed_store:,}")
        log.info(f"     Remaining (incl. zero/negative sales, unfiltered): {rows_accepted:,} rows")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_stgtrain_store_item ON stg_train(store_nbr, item_nbr)")
        conn.commit()

    stats = {
        "total_rows_read": total_rows,
        "rows_accepted": rows_accepted,
        "rows_removed_date": removed_date,
        "rows_removed_store": removed_store,
    }
    log.info(f"\n   === Train Load Summary ===")
    for k, v in stats.items():
        log.info(f"   {k}: {v:,}")
    return stats


# ---------------------------------------------------------------------------
# Build clean daily_demand table
# ---------------------------------------------------------------------------
def build_daily_demand(conn: sqlite3.Connection):
    """
    Create the clean daily_demand table at true item level, joining stg_train
    (store_nbr, item_nbr, date, unit_sales, on_promotion) with stg_items for
    real family/class/perishable metadata. Each (store_nbr, item_nbr) is a
    genuine SKU — no synthetic ID mapping needed now that item-level data
    from the 125M-row dataset is used.
    """
    with timer("Building daily_demand table"):
        conn.execute("DROP TABLE IF EXISTS daily_demand")
        conn.execute("""
            CREATE TABLE daily_demand AS
            SELECT
                t.id,
                t.date,
                t.store_nbr,
                i.family,
                t.item_nbr,
                i.class,
                i.perishable,
                CAST(t.unit_sales AS REAL) AS unit_sales,
                CAST(t.on_promotion AS INTEGER) AS on_promotion
            FROM stg_train t
            JOIN stg_items i ON t.item_nbr = i.item_nbr
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_store_item ON daily_demand(store_nbr, item_nbr)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_store_family ON daily_demand(store_nbr, family)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_date ON daily_demand(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_family ON daily_demand(family)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_item ON daily_demand(item_nbr)")
        # Covering index for the "GROUP BY family, on_promotion" aggregations used by
        # 04_promotion_impact.py (promo lift) and the notebooks — lets SQLite satisfy
        # those queries from the index alone instead of touching the main table.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_family_promo "
                      "ON daily_demand(family, on_promotion, unit_sales)")
        conn.commit()

        # Verify
        row = conn.execute("""
            SELECT COUNT(*) as total, MIN(date), MAX(date),
                   COUNT(DISTINCT store_nbr), COUNT(DISTINCT family),
                   COUNT(DISTINCT item_nbr)
            FROM daily_demand
        """).fetchone()
        log.info(f"\n   === daily_demand Summary ===")
        log.info(f"   Rows:             {row[0]:,}")
        log.info(f"   Date range:       {row[1]} to {row[2]}")
        log.info(f"   Stores:           {row[3]}")
        log.info(f"   Families:         {row[4]}")
        log.info(f"   Unique SKU IDs:   {row[5]:,}")


# ---------------------------------------------------------------------------
# Build native DuckDB daily_demand (acceleration layer — see module docstring)
# ---------------------------------------------------------------------------
def duckdb_already_loaded(expected_rows: int) -> bool:
    """Mirrors already_loaded() but for data/inventory.duckdb."""
    if not DUCKDB_PATH.exists():
        return False
    try:
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'daily_demand'"
        ).fetchone()
        if row[0] == 0:
            con.close()
            return False
        actual_rows = con.execute("SELECT COUNT(*) FROM daily_demand").fetchone()[0]
        con.close()
        return actual_rows == expected_rows
    except duckdb.Error:
        return False


def build_duckdb_daily_demand(force: bool = False):
    """
    Build data/inventory.duckdb's daily_demand table directly from the raw
    CSVs (not from SQLite — DuckDB only gets its columnar speed advantage
    reading its own native storage, not bridging through SQLite; measured
    ~10-65x slower via the sqlite_scanner extension, see CHANGELOG/PIPELINE_LOG).

    Applies the same scope filters (DATE_START/END, TARGET_CITY) as the
    SQLite build, so the two stay equivalent — currently both are no-ops
    (full-data mode) but this keeps them from silently diverging if the
    scope config changes later.
    """
    expected_rows = count_raw_rows(RAW_DATA_DIR / "train.csv")

    if not force and duckdb_already_loaded(expected_rows):
        log.info(f"[SKIP]  data/inventory.duckdb already has daily_demand with "
                  f"{expected_rows:,} rows. Nothing to do.")
        return

    with timer(f"Building DuckDB daily_demand (native, {DUCKDB_PATH.name})"):
        if DUCKDB_PATH.exists():
            DUCKDB_PATH.unlink()
        con = duckdb.connect(str(DUCKDB_PATH))
        con.execute("PRAGMA memory_limit='4GB'")
        con.execute("PRAGMA threads=2")
        con.execute("SET preserve_insertion_order=false")

        store_filter = ""
        if TARGET_CITY is not None:
            store_filter = f"""
                AND t.store_nbr IN (
                    SELECT store_nbr FROM read_csv('{(RAW_DATA_DIR / "stores.csv").as_posix()}')
                    WHERE city = '{TARGET_CITY}'
                )"""

        con.execute(f"""
            CREATE OR REPLACE TABLE daily_demand AS
            SELECT
                t.id, t.date, t.store_nbr, i.family, t.item_nbr, i.class, i.perishable,
                CAST(t.unit_sales AS DOUBLE) AS unit_sales,
                CASE WHEN t.onpromotion = 'True' THEN 1 ELSE 0 END AS on_promotion
            FROM read_csv('{(RAW_DATA_DIR / "train.csv").as_posix()}',
                           columns={{'id':'BIGINT','date':'VARCHAR','store_nbr':'SMALLINT',
                                     'item_nbr':'INTEGER','unit_sales':'DOUBLE','onpromotion':'VARCHAR'}}) t
            JOIN read_csv('{(RAW_DATA_DIR / "items.csv").as_posix()}') i ON t.item_nbr = i.item_nbr
            WHERE t.date >= '{DATE_START}' AND t.date <= '{DATE_END}'{store_filter}
        """)

        n = con.execute("SELECT COUNT(*) FROM daily_demand").fetchone()[0]
        log.info(f"   DuckDB daily_demand: {n:,} rows")
        con.close()


# ---------------------------------------------------------------------------
# Export cleaned data to CSV
# ---------------------------------------------------------------------------
def export_cleaned_csv(conn: sqlite3.Connection):
    """Export scoped cleaned data to CSV for inspection."""
    with timer("Exporting cleaned data to CSV"):
        CLEANED_DATA_DIR.mkdir(exist_ok=True)

        # Store list
        stores_scoped = pd.read_sql(
            "SELECT DISTINCT store_nbr FROM daily_demand ORDER BY store_nbr", conn
        )
        stores_full = pd.read_sql("SELECT * FROM stg_stores ORDER BY store_nbr", conn)
        stores_scoped = stores_full[stores_full["store_nbr"].isin(stores_scoped["store_nbr"])]
        stores_scoped.to_csv(CLEANED_DATA_DIR / "stores_scoped.csv", index=False)
        log.info(f"   Exported stores_scoped.csv ({len(stores_scoped)} rows)")

        # Demand summary
        summary = pd.read_sql("""
            SELECT store_nbr, family, item_nbr,
                   COUNT(*) as demand_days,
                   SUM(unit_sales) as total_units,
                   ROUND(AVG(unit_sales), 2) as mean_daily,
                   MIN(date) as first_date,
                   MAX(date) as last_date
            FROM daily_demand
            GROUP BY store_nbr, family, item_nbr
            ORDER BY total_units DESC
        """, conn)
        summary.to_csv(CLEANED_DATA_DIR / "demand_summary.csv", index=False)
        log.info(f"   Exported demand_summary.csv ({len(summary)} rows)")

        # Family overview
        family_overview = pd.read_sql("""
            SELECT family,
                   COUNT(DISTINCT store_nbr) as num_stores,
                   COUNT(*) as total_obs,
                   ROUND(SUM(unit_sales), 0) as total_units,
                   ROUND(AVG(unit_sales), 2) as avg_daily_sales
            FROM daily_demand
            GROUP BY family
            ORDER BY total_units DESC
        """, conn)
        family_overview.to_csv(CLEANED_DATA_DIR / "family_overview.csv", index=False)
        log.info(f"   Exported family_overview.csv ({len(family_overview)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Reload even if data/inventory.db already has the expected row count.")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  Inventory Optimization - Data Loader (Week 1)")
    log.info("=" * 60)
    log.info(f"  Scope: {'ALL stores (full data)' if TARGET_CITY is None else TARGET_CITY + ' stores'} | {DATE_START} to {DATE_END}")
    log.info(f"  Database: {DB_PATH}")
    log.info(f"  Source: 125M-row item-level dataset ({RAW_DATA_DIR})")
    log.info("")

    if not check_required_files():
        sys.exit(1)

    with timer("Checking whether data is already loaded (pass --force to skip this check)"):
        expected_rows = count_raw_rows(RAW_DATA_DIR / "train.csv")

    sqlite_skip = not args.force and already_loaded(expected_rows)
    if sqlite_skip:
        log.info(f"[SKIP]  data/inventory.db already has daily_demand with {expected_rows:,} rows "
                  f"(matches raw train.csv). Nothing to do — pass --force to reload anyway.")
    else:
        CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -65536")
        conn.execute("PRAGMA temp_store = MEMORY")

        try:
            stores_df = load_stores(conn)
            load_items(conn)
            load_oil(conn)
            load_holidays(conn)
            load_transactions(conn)
            load_train(conn, stores_df)
            build_daily_demand(conn)
            export_cleaned_csv(conn)

            log.info("\n[OK]  SQLite data loading complete!")
            log.info(f"   Database saved to: {DB_PATH}")

        except Exception as e:
            log.error(f"[FAIL]  Error during SQLite loading: {e}", exc_info=True)
            sys.exit(1)
        finally:
            conn.close()

    # DuckDB acceleration-layer table — independent guard, built/skipped
    # regardless of whether the SQLite load above ran or was skipped.
    try:
        build_duckdb_daily_demand(force=args.force)
    except Exception as e:
        log.error(f"[FAIL]  Error building DuckDB daily_demand: {e}", exc_info=True)
        sys.exit(1)

    log.info("\n[OK]  Data loading complete (SQLite + DuckDB)!")
    log.info("\n   Next step: python scripts/02_classify_skus.py")


if __name__ == "__main__":
    main()
