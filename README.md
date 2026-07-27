# Inventory Optimization with ABC-XYZ Classification & LLM-Powered Decision Support

> **Lead with the finding, not the method.**
>
> Textbook wisdom says differentiated (ABC-XYZ) safety-stock policies cost *more* than a uniform policy, because protecting best-sellers extra well is expensive. We tested this on the full Corporación Favorita dataset — 125.5M transaction rows, all 54 stores, the full 2013–2017 span, 166,720 product-store combinations — and found the **opposite**: the differentiated policy is **1.14% cheaper** than uniform. A smaller-scale test (575 combos, 1 city, 2 years) reproduces the textbook result (+20.7% more expensive), showing the direction of the conclusion flips once the long tail of low-value, erratic-demand items is represented at full scale. This project quantifies that reversal using real retail transaction data and provides an LLM-powered advisory agent for store-level inventory decisions.

## Project Overview

A prescriptive decision support system for differentiated inventory policy, combining ABC-XYZ segmentation with an LLM-powered advisory agent (Qwen 3 8B via Ollama), validated end-to-end on the full-scale real retail data from Corporación Favorita (all 125.5M rows, not a sample).

**Three deliverables:**
1. A working inventory optimization tool (portfolio project)
2. A research paper targeting OPSEARCH / MDPI journals
3. A deployable Streamlit app with a conversational agent

## Quick Start

### Prerequisites
- Python 3.11+
- Anaconda / conda environment
- Ollama with Qwen 3 8B model

### Installation

```bash
# Clone and enter the repo
git clone https://github.com/koushik2302/inventory-optimization.git
cd inventory-optimization

# Create conda environment
conda env create -f environment.yml
conda activate inv-opt

# Download dataset (requires Kaggle API key)
python scripts/download_data.py

# Load and clean data
python scripts/01_load_data.py

# Run the analysis pipeline
python scripts/02_classify_skus.py
python scripts/03_safety_stock.py

# Launch Streamlit app
streamlit run agent/app.py
```

## Dataset

**Corporación Favorita Grocery Sales** (Kaggle)
- Source: Kaggle competition "Store Sales - Time Series Forecasting"
- **125,497,040 rows** (full dataset, no sampling), 2013-01-01 to 2017-08-15 (~4.5 years), all 54 stores across 22 cities, 33 product families, 4,036 real item-store SKUs
- **166,720 product-store combinations** classified into ABC-XYZ cells (≥30 days of positive-sales observations required); 166,702 fed into the safety-stock model
- An earlier, smaller-scale pass (18 Quito stores, 2015–2016, 575 SKU-store pairs) is retained in the paper as a deliberate contrast case — see Key Results below

## Tech Stack

| Component | Tool |
|---|---|
| Database | SQLite (canonical storage), DuckDB (acceleration layer for heavy aggregations over `daily_demand`) |
| Analysis | Python (pandas, NumPy) |
| LLM | Qwen 3 8B via Ollama |
| Frontend | Streamlit |

The full pipeline (load → classify → safety stock → promotion/holiday
analysis) re-runs against the full 125.5M-row dataset in **under a
minute** end-to-end (down from ~90 min pre-optimization), via early-exit
guards on unchanged inputs, added indexes, and DuckDB for the heaviest
aggregations — see `CHANGELOG.md` for details.

## Project Structure

```
inventory-optimization/
├── data/
│   ├── raw/              # Original Kaggle CSVs
│   └── cleaned/          # Filtered, processed CSVs
├── sql/                  # SQL analysis scripts
├── notebooks/            # Jupyter notebooks
├── agent/                # LLM agent (tools.py, agent.py, app.py)
├── scripts/              # ETL and pipeline scripts
├── tests/                # Unit tests
├── paper/                # Research paper draft
└── docs/                 # Methodology documentation
```

## Key Results

**Full dataset (166,702 SKU-scenarios, all 54 stores, full 4.5-year span):**

| Policy | Total Holding Cost | Total Safety Stock (units) | vs. Uniform |
|---|---|---|---|
| Uniform (95% for all) | 1,007,197 | 4,578,170 | — |
| 3-tier (ABC only) | 1,233,282 | 5,605,827 | +22.45% |
| 9-cell (ABC-XYZ) | 995,678 | 4,525,810 | **-1.14%** |

**Small-scale contrast (575 SKU-store pairs, 18 Quito stores, 2015–2016 only):**

| Policy | Total Holding Cost | Total Safety Stock (units) | vs. Uniform |
|---|---|---|---|
| Uniform (95% for all) | 153,674 | 698,518 | — |
| 3-tier (ABC only) | 200,053 | 909,330 | +30.2% |
| 9-cell (ABC-XYZ) | 185,408 | 842,764 | +20.7% |

> **Key Finding:** The direction of the result *flips* with scale. At small
> scale, differentiated policies cost more, confirming the textbook
> expectation that extra protection on A-class items outweighs savings
> elsewhere. At full scale, the 9-cell policy is **1.14% cheaper than
> uniform** (and 19.3% cheaper than 3-tier) — the long tail of low-value,
> erratic-demand (C/Z-class) items is large enough at full-catalog,
> all-store scale that the safety-stock savings there outweigh the extra
> buffer spent on A-class SKUs. This holds consistently across lead times
> (3/5/7/10 days) in both regimes. **The practical lesson:** conclusions
> about differentiated inventory policy drawn from a small sample can be
> wrong in direction, not just magnitude, if that sample doesn't reflect
> the true shape of the whole business — the central contribution of the
> research paper.

## Other Findings (Full Dataset)

- **Promotions:** demand lift varies sharply by category — Home & Kitchen II
  (+357.4%) and School & Office Supplies (+280.8%) spike hardest;
  Beverages sees a smaller but still substantial +76.7% lift. Extra safety
  stock needed during promotions was computed per category.
- **Holidays:** Frozen Foods sees a 67.3% demand spike around holidays;
  Liquor/Wine/Beer sees 31.6%; Grocery II sees 17.6%.

## ABC-XYZ Service Level Policy

| Cell | Service Level | z-score |
|---|---|---|
| AX | 99% | 2.33 |
| AY | 97% | 1.88 |
| AZ | 95% | 1.65 |
| BX | 95% | 1.65 |
| BY | 93% | 1.48 |
| BZ | 90% | 1.28 |
| CX | 90% | 1.28 |
| CY | 88% | 1.17 |
| CZ | 85% | 1.04 |

## Key Formulas

**Safety Stock:** `SS = z × σ_d × √(LT)`

**Reorder Point:** `ROP = (d̄ × LT) + SS`

**Annual Holding Cost:** `HC = SS × C_unit × 0.22`

**Coefficient of Variation:** `CV = σ_d / d̄`
- X: CV < 0.5 | Y: 0.5 ≤ CV ≤ 1.0 | Z: CV > 1.0

## Research Contribution

Three things no single paper has done together:
1. ABC-XYZ policy differentiation on a large-scale public retail dataset (Favorita) instead of a single-company case study
2. Empirical cost comparison (uniform vs. tiered vs. 9-cell) with sensitivity analysis
3. An LLM-agent-based decision support layer with real tool-use orchestration

**Target journals:** OPSEARCH, MDPI Operations Research Forum, MDPI Logistics

## License

MIT
