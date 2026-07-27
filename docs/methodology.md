# Methodology Documentation

## Inventory Optimization with ABC-XYZ Classification

### 1. Dataset Description and Scoping

**Source:** Corporación Favorita Grocery Sales (Kaggle)

- Original size: ~125M rows, 4 years (2013–2017), 54 stores, 33 product families
- **Scoped to:** 6 Quito stores, 2 years (2015–2016)
- **Rationale for scope:**
  - Quito stores have the highest transaction volume and product diversity
  - 2 years (730 days) provides sufficient data for robust CV estimation
  - Reduces compute requirements while maintaining representativeness

**Cleaning rules applied:**
1. Removed negative unit sales (returns): `unit_sales < 0`
2. Removed zero-sales records: `unit_sales = 0`
3. Applied minimum observation filter: SKUs with `< 30` demand days excluded
4. Joined with `items.csv` for product family metadata

---

### 2. ABC Classification

**Method:** Cumulative revenue contribution per store

**Steps:**
1. Compute total unit sales per SKU per store (revenue proxy, unit price = 1.0)
2. Rank SKUs in descending order of revenue within each store
3. Compute cumulative revenue percentage
4. Assign class:
   - **A:** Cumulative revenue ≤ 80%
   - **B:** Cumulative revenue 80–95%
   - **C:** Cumulative revenue > 95%

**Limitation:** No actual unit prices available. Unit sales used as revenue proxy.
All relative comparisons within a store remain valid. Cross-store absolute comparisons should be interpreted cautiously.

---

### 3. XYZ Classification

**Method:** Coefficient of Variation (CV) of daily demand

$$CV = \frac{\sigma_d}{\bar{d}}$$

Where:
- $\sigma_d$ = standard deviation of daily demand (ddof=1)
- $\bar{d}$ = mean daily demand

**Thresholds (standard literature):**
- **X:** CV < 0.5 (stable, predictable demand)
- **Y:** 0.5 ≤ CV ≤ 1.0 (moderate variability)
- **Z:** CV > 1.0 (highly erratic, intermittent demand)

---

### 4. Safety Stock Formulas

#### Safety Stock

$$SS = z \times \sigma_d \times \sqrt{LT}$$

Where:
- $z$ = z-score for target service level (from standard normal distribution)
- $\sigma_d$ = standard deviation of daily demand
- $LT$ = lead time in days

**Note:** This formula assumes independent, normally distributed daily demand.
For items with high CV (Z-class), this is an approximation. The Syntetos-Boylan method
(for intermittent demand) is noted as a robustness check in future work.

#### Reorder Point

$$ROP = (\bar{d} \times LT) + SS$$

#### Annual Holding Cost

$$HC = SS \times C_{unit} \times h$$

Where:
- $C_{unit}$ = unit cost (proxy = 1.0 in this analysis)
- $h$ = annual holding cost rate = **22%**

---

### 5. Policy Scenarios

| Policy | Description | Service Level |
|---|---|---|
| **Uniform** | Single SL for all SKUs | 95% for all |
| **3-Tier** | ABC-differentiated | A=99%, B=95%, C=90% |
| **9-Cell** | ABC-XYZ differentiated | See table below |

#### 9-Cell Service Level Policy

| Cell | Description | Service Level | z-score |
|---|---|---|---|
| AX | High-value, stable | 99% | 2.326 |
| AY | High-value, moderate var | 97% | 1.881 |
| AZ | High-value, erratic | 95% | 1.645 |
| BX | Mid-value, stable | 95% | 1.645 |
| BY | Mid-value, moderate var | 93% | 1.476 |
| BZ | Mid-value, erratic | 90% | 1.282 |
| CX | Low-value, stable | 90% | 1.282 |
| CY | Low-value, moderate var | 88% | 1.175 |
| CZ | Low-value, erratic | 85% | 1.036 |

---

### 6. Assumptions

Stated explicitly for reproducibility and sensitivity testing:

| Assumption | Value | Sensitivity Range |
|---|---|---|
| Lead time | 7 days | Tested at 3, 5, 7, 10 days |
| Holding cost rate | 22% annually | Standard supply chain literature |
| Unit cost | 1.0 (proxy) | No actual prices in dataset |
| Demand distribution | Normal | Approximation for high-CV items |
| Stockout cost | Not modeled | Expressed as fill-rate target |

---

### 7. Sensitivity Analysis

Two sensitivity analyses are conducted:

1. **Lead time sensitivity** (3, 5, 7, 10 days): How do total holding costs
   and safety stock quantities change under each policy as lead time varies?

2. **Service level curve** (80% to 99.5%): For a uniform policy, what is the
   marginal cost of each additional percentage point of service level?
   This is the "money chart" — showing the nonlinear cost-service tradeoff.

---

### 8. Promotion Adjustment

**Method:** Historical demand lift estimation

1. Compare mean daily demand: promo days vs. non-promo days, per family
2. Compute lift percentage: `lift_pct = (promo_mean / baseline_mean - 1) × 100`
3. Adjust ROP demand component: `promo_ROP = promo_mean_demand × LT + SS`
4. Recommended buffer increase: `promo_ROP - baseline_ROP`

**Limitation:** Promotions are binary (on/off). Promotion intensity, discount depth,
and advertising are not captured. Lift estimates are averages and will vary by SKU.

---

### 9. LLM Agent Architecture

**Model:** Qwen 3 8B via Ollama (local, no API cost)

**Why Qwen 3 8B:**
- Native tool-use support
- Fits in 8GB VRAM (RTX 4060)
- ~30-40 tokens/second on local hardware
- Extended thinking (`think=True`) for reasoning transparency

**Tool design principles:**
1. Each tool returns a **complete answer** (numbers + human-readable summary)
2. Maximum **2 tool calls per query** (Qwen 3 8B coherence constraint)
3. Input validation inside each tool (not in the agent loop)
4. Tools query SQLite directly (no intermediate data structures needed)

**Tools:**
| Tool | Description |
|---|---|
| `get_classification` | ABC-XYZ matrix for a store |
| `calculate_safety_stock` | SS, ROP, HC per store/family |
| `simulate_policy_change` | Cost delta for SL change in a cell |
| `get_overstock_alerts` | High HC/Revenue ratio SKUs |
| `get_promotion_adjustment` | ROP buffer for promotional periods |
| `compare_policies_summary` | Aggregate policy cost comparison |

---

### 10. Limitations

1. **No actual unit prices** — revenue proxy based on unit sales only
2. **Assumed lead time** — 7 days not from real logistics data; sensitivity tested
3. **No actual stockout cost data** — analysis uses fill-rate targets, not penalty multipliers
4. **Normal demand assumption** — approximate for Z-class items (CV > 1.0)
5. **LLM reliability** — Qwen 3 8B may hallucinate tool arguments; all inputs are validated
6. **Static classification** — ABC-XYZ assigned once; no seasonal reclassification
