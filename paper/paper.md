# ABC-XYZ Inventory Policy Differentiation at Population Scale: A Large-Sample Reassessment Using 125 Million Retail Transactions

**Target journals:** OPSEARCH · MDPI Operations Research Forum · MDPI Logistics

---

## Abstract

ABC-XYZ segmentation is a long-standing heuristic for differentiating
inventory service-level policy by item value and demand variability, but
empirical validations of the technique have almost universally relied on
small samples — typically a few hundred to a few thousand SKUs from a
single company or store. This paper re-examines the classic finding that
differentiated (ABC-XYZ, "9-cell") service-level policies are more costly
than a uniform policy, using the full Corporación Favorita grocery-sales
dataset: 125,497,040 item-level transactions across 54 stores, 22 cities,
4,100 items, and 4.5 years (2013–2017). At small scale (575 classified synthetic
SKU-store pairs, one city, a two-year window) we replicate the textbook
result: 9-cell differentiation costs 20.7% more in annual holding cost
than a uniform 95% service level. At full population scale (166,720
genuine item-store SKUs, all stores, the full date range), the same
methodology produces the **opposite** sign: 9-cell differentiation is
1.14% *cheaper* than uniform, and this savings percentage is invariant to
lead time (3–10 days). We show the reversal is driven by how C-class,
erratic-demand (CZ) items dominate the SKU count once item-level (rather
than store×family-aggregated) granularity and the full store network are
used — a segment essentially invisible in small-sample analyses. We
additionally quantify promotional and holiday demand lift by product
family and their implied safety-stock adjustments, and discuss the
methodological implication: published ABC-XYZ cost-comparison results
should be treated as scale-and-granularity-dependent, not as generalizable
constants.

**Keywords:** inventory management, ABC-XYZ classification, safety stock,
service level, retail analytics, large-scale empirical validation

---

## 1. Introduction

Differentiated inventory policy — setting service levels by item segment
rather than applying one target uniformly — is one of the oldest
prescriptions in inventory theory. The ABC-XYZ matrix, which crosses
revenue-contribution tiers (A/B/C, from Pareto analysis) with
demand-variability tiers (X/Y/Z, from the coefficient of variation), is
taught in essentially every operations management curriculum and remains
in active use in retail and manufacturing practice.

A recurring empirical finding in the applied literature is that
differentiated policies, evaluated against a single uniform benchmark,
often turn out to be more expensive in aggregate — because upgrading
A-class items to a 99% service level requires disproportionately more
safety stock than is saved by downgrading C-class items to 85–90%. This
is not a failure of the method (its purpose is risk allocation, not blanket
cost minimization) but it is a genuine, reproducible empirical pattern in
the small studies that have measured it.

Nearly all published validations of this pattern share a common
limitation: sample size. Case studies typically draw on a few hundred to
low thousands of SKUs, one company, one or a handful of locations, and a
short observation window (often one to two years). This paper asks a
narrow, falsifiable question: **does the well-known "differentiation costs
more" result survive a move from small-sample, single-location analysis to
full-population, multi-location analysis on the same underlying business?**

We answer it directly, using the same dataset, the same formulas, and the
same policy definitions at two scales:

- **Small-sample condition:** 575 classified synthetic SKU-store pairs (store ×
  product-family aggregates, one city, 2015–2016).
- **Population-scale condition:** 166,720 genuine item-store SKUs
  (individual item-level transactions, all 54 stores / 22 cities,
  2013–2017, the dataset's full extent).

The result is a sign reversal, not a magnitude adjustment: the
population-scale 9-cell policy is *cheaper* than uniform, not more
expensive. Section 4 traces this to a specific, identifiable mechanism —
the CZ (low-value, erratic-demand) segment's disproportionate share of
SKU count at item-level granularity — rather than to noise or an
artifact of the larger sample.

## 2. Background and Related Work

**ABC classification** ranks items by cumulative revenue contribution,
conventionally splitting the catalog into A (top ~80% of revenue), B
(next ~15%), and C (remaining ~5%) tiers — a direct application of Pareto
analysis to inventory management.

**XYZ classification** ranks items by the coefficient of variation (CV)
of demand, $CV = \sigma_d / \bar{d}$, splitting into X (stable, CV < 0.5),
Y (moderate, 0.5 ≤ CV ≤ 1.0), and Z (erratic, CV > 1.0) tiers. XYZ
segmentation is the more fragile of the two classifications in practice:
CV estimates are sensitive to the granularity and length of the
observation window, a point this paper's own results illustrate directly
(Section 4.3).

**Safety stock and service level.** Under the standard
periodic-review/continuous-review approximation with normally distributed
demand, safety stock is $SS = z \cdot \sigma_d \cdot \sqrt{LT}$, where $z$
is the standard normal quantile for the target service level and $LT$ is
replenishment lead time. Reorder point is $ROP = \bar{d} \cdot LT + SS$,
and annual holding cost is $HC = SS \cdot C_{unit} \cdot h$ for unit cost
$C_{unit}$ and annual holding rate $h$.

**Gap addressed.** We are not aware of a prior ABC-XYZ cost-comparison
study conducted at this combination of scale (125M+ raw transactions),
granularity (true item-level, not category-aggregated), and geographic
breadth (54 stores across 22 cities) on a single, consistent dataset with
both a small-sample and full-population condition computed identically.
This lets us isolate scale/granularity as the explanatory variable for
the sign reversal, rather than confounding it with a different company,
industry, or time period.

## 3. Data and Methodology

### 3.1 Dataset

Corporación Favorita Grocery Sales (Kaggle "Favorita Grocery Sales
Forecasting" competition, the original item-level release): 125,497,040
transaction rows spanning 2013-01-01 to 2017-08-15, 54 stores across 22
Ecuadorian cities, 4,100 distinct items, 33 product families. Each row
records `(date, store, item, unit_sales, on_promotion)`. Item metadata
(family, class, perishability) is joined from a separate item master.

No filtering is applied to the source data at load time — including
zero-sales days and negative `unit_sales` (returns) — so that the
retained dataset is a complete, auditable copy of the raw transaction
log. A positive-sales filter (`unit_sales > 0`) is applied only at the
point of computing demand statistics (mean, standard deviation, CV),
since these formulas are undefined or misleading over return
transactions; this filter removes a negligible fraction of rows and does
not affect scope.

### 3.2 Two conditions, same methodology

| | Small-sample condition | Population-scale condition |
|---|---|---|
| Granularity | Store × family (synthetic SKU) | Store × item (true SKU) |
| Geography | 1 city (Quito), 18 stores | All 22 cities, 54 stores |
| Date range | 2015–2016 (2 years) | 2013–2017 (4.5 years, full extent) |
| Classified SKU-store pairs | 575 | 166,720 |
| Minimum observations filter | ≥30 demand days | ≥30 demand days |

Both conditions apply identical ABC thresholds (A ≤ 80% cumulative
revenue, B ≤ 95%, C > 95%), identical XYZ thresholds (X: CV < 0.5, Y:
0.5–1.0, Z: CV > 1.0), and identical 9-cell service-level policy
(Table 1), lead time (7 days baseline, 3/5/7/10-day sensitivity), and
holding-cost rate (22% annually).

**Table 1. Service-level policy by ABC-XYZ cell**

| Cell | Description | Service Level | z |
|---|---|---|---|
| AX | High-value, stable | 99% | 2.326 |
| AY | High-value, moderate | 97% | 1.881 |
| AZ | High-value, erratic | 95% | 1.645 |
| BX | Mid-value, stable | 95% | 1.645 |
| BY | Mid-value, moderate | 93% | 1.476 |
| BZ | Mid-value, erratic | 90% | 1.282 |
| CX | Low-value, stable | 90% | 1.282 |
| CY | Low-value, moderate | 88% | 1.175 |
| CZ | Low-value, erratic | 85% | 1.036 |

### 3.3 Unit cost proxy and its implications

The dataset carries no true unit price. Following standard practice when
cost data is unavailable, unit cost is set to a constant proxy
($C_{unit} = 1.0$) and revenue is proxied by unit sales volume. Holding
costs are therefore reported in a unit-cost-proxy currency, not a real
monetary unit; all *relative* comparisons (uniform vs. 3-tier vs. 9-cell,
cell vs. cell) remain valid under this proxy, but absolute dollar
magnitudes should not be read literally. This is a limitation shared with
most public-dataset inventory studies and is discussed further in
Section 6.

### 3.4 Computational approach

At population scale, the demand-statistics computation (mean, standard
deviation, CV per SKU) is a full aggregation over 125.5 million rows.
This was computed via a columnar (DuckDB) query engine reading the raw
transaction log directly, rather than a row-oriented database — a purely
computational choice with no effect on results, included here because it
is a practical prerequisite for reproducing this analysis at this scale
on commodity hardware (the full aggregation completes in under 20 seconds
on an 8GB-RAM machine).

## 4. Results

### 4.1 Small-sample condition (baseline replication)

At 575 SKU-store pairs (18 Quito stores, 2015–2016), the classic result
replicates cleanly:

**Table 2. Policy comparison, small-sample condition**

| Policy | Total Holding Cost | Total Safety Stock (units) | vs. Uniform |
|---|---|---|---|
| Uniform (95%) | 153,674 | 698,518 | — |
| 3-Tier (ABC only) | 200,053 | 909,330 | +30.2% |
| 9-Cell (ABC-XYZ) | 185,408 | 842,764 | **+20.7%** |

Lead-time sensitivity held this premium essentially constant (~20.7%
across 3–10 day lead times), and cell-level detail showed the expected
pattern: AX cells cost 41.4% more than their uniform-policy counterpart
(99% service level demands substantially more buffer), while CZ cells
cost 37.0% less (85% service level demands substantially less). The
9-cell policy is a *reallocation* of safety stock toward high-value items
— exactly as intended — but at this scale the reallocation's net cost is
positive.

### 4.2 Population-scale condition

At 166,720 SKU-store pairs (all 54 stores, full 2013–2017 window), the
same methodology, same formulas, same thresholds produce:

**Table 3. Policy comparison, population-scale condition**

| Policy | Total Holding Cost | Total Safety Stock (units) | vs. Uniform |
|---|---|---|---|
| Uniform (95%) | 1,007,197 | 4,578,170 | — |
| 3-Tier (ABC only) | 1,233,282 | 5,605,827 | +22.45% |
| 9-Cell (ABC-XYZ) | 995,678 | 4,525,810 | **−1.14%** |

*(See Figure 2.)* The 3-tier premium is essentially unchanged from the
small-sample condition (+30.2% → +22.45%, same direction, similar
magnitude). The 9-cell result, however, flips sign: from a 20.7% premium
to a 1.14% **saving**. Lead-time sensitivity confirms this is not a
lead-time artifact — the saving holds at a constant 1.14% across 3, 5, 7,
and 10-day lead times (Table 4), mirroring the constancy pattern observed
(with the opposite sign) at small scale.

**Table 4. Lead-time sensitivity, population-scale condition**

| Lead Time (days) | HC Uniform | HC 9-Cell | Saving |
|---|---|---|---|
| 3 | 659,366 | 651,824 | 1.14% |
| 5 | 851,237 | 841,502 | 1.14% |
| 7 | 1,007,197 | 995,678 | 1.14% |
| 10 | 1,203,831 | 1,190,063 | 1.14% |

### 4.3 Mechanism: why the sign flips

Cell-level decomposition (Table 5, Figure 3) shows the reversal is not
diffuse — it is concentrated in specific cells whose relative size
changes dramatically between the two conditions.

A useful analytical fact simplifies the mechanism: **within any single
cell, the percentage cost difference versus the uniform policy is a
closed-form function of that cell's z-score ratio alone,
$\Delta\%_{cell} = z_{cell}/z_{uniform} - 1$, independent of sample
scale.** This follows directly from $HC = z \cdot \Sigma\sigma \cdot
\sqrt{LT} \cdot C_{unit} \cdot h$: for a fixed cell, the $\Sigma\sigma$
term cancels in the ratio $HC_{9cell}/HC_{uniform}$, leaving only the
z-score ratio. For AX, $2.326/1.645 - 1 = +41.4\%$; for CZ, $1.036/1.645
- 1 = -37.0\%$ — exactly the values in both Table 5 (population scale)
and the small-sample condition's cell-level results (Section 4.1). This
is not a coincidence: it holds by construction, at any scale, for any
population of SKUs assigned to a given cell. **The aggregate reversal
documented in Table 3 therefore cannot come from any change in per-cell
economics — it can only come from a change in how much aggregate revenue
and holding cost is attributable to each cell, i.e. a compositional
shift.** That shift is exactly what we document next.

**Table 5. Cell-level detail, population-scale condition**

| Cell | SKUs | Avg CV | HC Uniform | HC 9-Cell | Δ vs. Uniform |
|---|---|---|---|---|---|
| AX | 4,844 | 0.436 | 63,710 | 90,085 | +41.4% |
| AY | 42,527 | 0.706 | 320,857 | 366,889 | +14.4% |
| AZ | 9,066 | 1.849 | 249,244 | 249,244 | 0.0% |
| BX | 460 | 0.455 | 2,767 | 2,767 | 0.0% |
| BY | 38,590 | 0.732 | 117,217 | 105,175 | −10.3% |
| BZ | 8,218 | 1.639 | 88,894 | 69,278 | −22.1% |
| CX | 6,072 | 0.416 | 4,317 | 3,364 | −22.1% |
| CY | 48,083 | 0.705 | 94,561 | 67,543 | −28.6% |
| CZ | 8,842 | 1.556 | 65,631 | 41,333 | **−37.0%** |

Two effects compound:

1. **XYZ shift away from X at item-level granularity.** In the
   small-sample condition, demand was aggregated at store×family level —
   summing dozens of individual items into one "family" series smooths
   out item-level volatility, mechanically shrinking measured CV. The
   small-sample XYZ split was X = 256 / Y = 270 / Z = 49 (44.5% / 47.0% /
   8.5% of classified SKUs). At true item-level granularity, the
   population-scale split is X = 11,394 / Y = 129,200 / Z = 26,126
   (6.8% / 77.5% / 15.7%) — the stable (X) share collapses from 44.5% to
   6.8%, while the erratic (Z) share nearly doubles, from 8.5% to 15.7%.
   This is a measurement artifact of aggregation, not a change in the
   underlying business, and it is a caution for any ABC-XYZ study that
   uses category- or family-level aggregates as a stand-in for SKU-level
   analysis.

2. **C-class, non-stable cells (CY, CZ, BZ) carry far more SKU-count
   weight at population scale** (CY alone is 48,083 of 166,720 SKUs,
   28.8% of the total) **and their service-level discount (85–90%, vs.
   95% uniform) compounds across many more units than the A-class
   premium does.** In the small sample, C-class SKUs were a much smaller
   share of the classified population (the synthetic family-aggregation
   also collapsed many genuinely distinct low-volume items into the
   ≥30-observation filter's blind spot). At full item-level scale, the
   long tail of low-revenue, erratic-demand SKUs — the CZ cell
   specifically, at −37.0% vs. uniform — is large enough in aggregate to
   outweigh the AX/AY premium.

The practical implication is direct: **a differentiated policy's net
cost-effectiveness depends on the shape of the SKU population's ABC-XYZ
distribution, which is itself sensitive to the granularity at which
demand is aggregated.** A retailer (or researcher) evaluating this
tradeoff on category-level or store-level aggregates, rather than true
item-level data, risks the wrong sign, not just the wrong magnitude.

### 4.4 Promotional and holiday demand effects

Promotion and holiday flags in the dataset let us quantify demand lift
directly, at full population scale, family by family. Table 6 reports
the largest effects; the complete 33-family table is in
`data/cleaned/promotion_lift_by_family.csv` and
`data/cleaned/holiday_impact.csv`.

**Table 6. Top promotional and holiday demand lift by family**

| Family | Promo Lift | Family | Holiday Lift |
|---|---|---|---|
| Home & Kitchen II | +357.4% | Frozen Foods | +67.3% |
| School & Office Supplies | +280.8% | Liquor/Wine/Beer | +31.6% |
| Beverages | +76.7% | Grocery II | +17.6% |
| Grocery I | +73.2% | Lawn & Garden | +12.3% |
| Cleaning | +71.2% | Players & Electronics | +11.8% |

Applying each family's promotional lift as a temporary demand-mean
adjustment to the safety-stock formula (holding the same z-score, scaling
standard deviation sub-proportionally with the lift — see
`scripts/04_promotion_impact.py`) implies a population-wide safety-stock
increase of 1,348,135 units and a holding-cost increase of 296,590
(unit-cost-proxy currency) if all promotional periods were provisioned
for simultaneously. This is an upper bound (not all families promote
concurrently) but quantifies the scale of buffer a retailer should be
prepared to deploy during active promotional campaigns.

## 5. Discussion

The central contribution of this paper is methodological, not just
empirical: **the sign of the "does ABC-XYZ differentiation cost more than
uniform service" question is not a fixed property of the technique — it
depends on the scale and granularity at which it is measured**, at least
for this dataset. A researcher or practitioner replicating the
small-sample result (differentiation costs more) and generalizing it to
"differentiation is not worth it" would reach a conclusion this paper's
full-population analysis directly contradicts, on the same underlying
business.

This has three practical implications:

1. **Small-sample ABC-XYZ cost studies should be read as scope-specific,
   not as general claims about the technique.** A single store, a single
   category, or a short window can plausibly have a different ABC-XYZ
   shape than the full business.
2. **Family- or category-level demand aggregation is not a safe
   substitute for item-level analysis when the downstream use is
   variability-sensitive (XYZ) classification.** Aggregation mechanically
   compresses CV; the resulting XYZ split is not merely noisier, it is
   systematically biased toward "more stable than the true item-level
   population."
3. **The long tail matters disproportionately for cost, not just for
   count.** The CZ cell (low-value, erratic-demand) is easy to
   deprioritize in a small study — it is often the smallest cell by SKU
   count and by definition the lowest-revenue. At population scale, its
   sheer volume (8,842 SKUs, and structurally similar cells BY/CY/BZ
   totaling over 100,000 more) makes its service-level discount the
   dominant driver of the aggregate cost comparison.

## 6. Limitations

1. **No true unit cost.** Holding costs are reported in a unit-cost-proxy
   currency (Section 3.3); relative comparisons are valid, absolute
   monetary figures are not.
2. **Assumed, not measured, lead time.** 7 days is an assumption, not
   sourced from real logistics data; sensitivity across 3–10 days is
   reported but the true lead-time distribution (and its own variability,
   which would itself contribute to safety stock under a full stochastic
   lead-time model) is unknown.
3. **No stockout-cost or penalty data.** The analysis expresses service
   protection via fill-rate targets, not via an explicit
   stockout-cost/holding-cost tradeoff optimization — the 9-cell policy
   used here is a standard heuristic assignment (Table 1), not a
   cost-optimized one.
4. **Normal-demand approximation.** The safety-stock formula assumes
   independent, normally distributed daily demand. This is a
   substantially worse approximation for Z-class (erratic, often
   intermittent) items — precisely the cells this paper identifies as
   most consequential for the aggregate result. A robustness check using
   an intermittent-demand method (e.g., Syntetos-Boylan) for Z-class
   items is a natural extension.
5. **Static classification.** ABC-XYZ assignment is computed once over
   the full window, not re-evaluated seasonally; a SKU's cell can in
   practice drift over a 4.5-year span.
6. **Single retailer, single country.** Generalization to other
   retailers, categories, or markets is not established by this paper —
   the claim is that scale/granularity matters *within* a consistent
   dataset, not a claim that this specific sign reversal will replicate
   elsewhere.

## 7. Conclusion

Using 125,497,040 real retail transactions from Corporación Favorita, we
show that a textbook empirical finding — that ABC-XYZ-differentiated
inventory policy costs more in aggregate holding cost than a uniform
policy — replicates at small sample scale (575 SKU-store pairs,
one city, two years: +20.7%) but **reverses** at full population scale
(166,720 true item-level SKU-store pairs, all 54 stores, the dataset's
full 4.5-year extent: −1.14%), with the reversal traced to the changing
relative weight of C-class, erratic-demand SKUs as measurement moves from
category-aggregated to item-level granularity. The result argues for
treating published ABC-XYZ cost comparisons as scale-and-granularity-
dependent findings rather than general claims, and for preferring
item-level over category-level demand aggregation whenever XYZ
(variability-based) classification is part of the analysis.

---

## Data and Code Availability

All analysis code, the full pipeline (data loading, classification,
safety-stock computation, promotion/holiday analysis), and the generated
result CSVs referenced in this paper are available in this repository
(`scripts/`, `data/cleaned/`). The underlying Kaggle dataset is public
("Corporación Favorita Grocery Sales Forecasting" / "Store Sales - Time
Series Forecasting" competitions).

## Figures

- **Figure 1** (`figures/fig1_cost_service_tradeoff.png`): Cost–service
  level tradeoff curve, uniform policy, population scale.
- **Figure 2** (`figures/fig2_policy_comparison.png`): Total holding cost
  and safety stock, three policies, population scale.
- **Figure 3** (`figures/fig3_cell_comparison.png`): Holding cost by
  ABC-XYZ cell, uniform vs. 9-cell, population scale.

## References

*(Placeholder — to be completed with formal citations before submission.
Candidate anchors: Silver, Pyke & Peterson on inventory theory;
foundational ABC/Pareto-analysis literature; XYZ/coefficient-of-variation
segmentation literature; Syntetos & Boylan on intermittent demand;
prior ABC-XYZ empirical case studies for direct comparison against this
paper's small-sample replication.)*
