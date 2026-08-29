import sqlite3
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
db_path = PROJECT_ROOT / "data" / "inventory.db"
print("Connecting to database...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def create_index(name, table, cols):
    try:
        print(f"Creating index {name} on {table}({cols})...")
        t0 = time.time()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({cols})")
        print(f"Done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Error creating index {name}: {e}")

# Indexes for abc_xyz_matrix
create_index("idx_abc_store_cell", "abc_xyz_matrix", "store_nbr, cell")
create_index("idx_abc_store_family", "abc_xyz_matrix", "store_nbr, family")

# Indexes for safety_stock_results
create_index("idx_ss_store_cell", "safety_stock_results", "store_nbr, cell")

# Indexes for daily_demand
create_index("idx_dd_store_family", "daily_demand", "store_nbr, family, item_nbr, on_promotion")

conn.commit()
conn.close()
print("Optimization complete.")
