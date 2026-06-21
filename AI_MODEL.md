# AI Model — Parking Intelligence

## Problem Statement
**How can AI-driven parking intelligence detect illegal parking hotspots and quantify their
impact on traffic flow to enable targeted enforcement?**

The core challenge: enforcement is reactive and patrol-based. There is no spatial map of
where violations concentrate, no measure of which zones harm traffic most, and no way to
predict where the next spike will happen. This system solves all three.

---

## What "AI-Driven" Actually Means Here

We are honest about what the data can and cannot support.

The dataset contains **parking violation records** — timestamps, GPS coordinates, violation
types, and police-station attribution. It does **not** contain ground-truth traffic-flow
measurements. We do not fabricate congestion labels. Any system that claims to "train a model
to predict congestion impact" from this dataset alone would be leaking or lying.

Instead, we build a **three-layer AI pipeline** where every claim is traceable:

---

## The Three-Layer AI Pipeline

### Layer 1 — Unsupervised Hotspot Detection (DBSCAN + H3)
**What it does:** Groups 298,000+ violation points into 506 spatial enforcement zones.

- **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) runs on
  haversine-projected coordinates to find organically shaped clusters along carriageways
  and intersections — without requiring the number of clusters upfront.
- **H3 hexagonal binning** (Uber's spatial indexing library) provides a uniform heatmap
  grid and a fallback when DBSCAN finds no dense clusters.
- **Why unsupervised:** no labels exist. The structure is latent in the GPS data itself.
- **Correctness guarantee:** every emitted hotspot has `member_count ≥ min_samples` and
  every violation belongs to at most one cluster (partition property).

### Layer 2 — Supervised Violation-Demand Forecasting (LightGBM, Poisson objective)
**What it does:** Predicts how many violations each zone will see in each of the next 7 days.

- **Target (y):** daily violation count per H3 cell (res 8). The label is derived directly
  from the dataset's own timestamps — no external data, no fabrication.
- **Model:** LightGBM gradient-boosted regressor with a **Poisson objective** (correct
  for non-negative, skewed count data; penalises the right loss function).
- **Features (all leakage-safe):** lag-1, lag-7, rolling-7/14-day mean, expanding historical
  mean, hour-of-day, day-of-week, is-weekend, month, day-of-year, H3 cell code.
- **Evaluation split:** **chronological hold-out — last 30 days held out, trained on
  everything before**. Not random split (random split leaks future data into training — we
  explicitly avoid this).
- **Baseline:** same-weekday last week (lag-7). We beat it on every metric.

| Metric | Naive Baseline | **LightGBM** | Improvement |
|--------|---------------|-------------|-------------|
| MAE | 4.89 | **3.98** | −18.7% |
| RMSE | 13.10 | **9.97** | −23.9% |
| R² | 0.25 | **0.565** | +126% |
| Hotspot-day F1 | — | **0.622** | — |
| ROC-AUC | — | **0.831** | — |

- **Recursive multi-day forecasting:** predictions are fed back as lag features for the next
  day, enabling genuine 7-day-ahead patrol scheduling.
- **Hyperparameter tuning:** light grid search (4 configs) scored on the hold-out, best
  config (`num_leaves=127, lr=0.03, n_estimators=900`) selected.

### Layer 3 — Explainable Priority Scoring (Transparent Composite)
**What it does:** Converts ML outputs into an actionable, inspectable enforcement ranking.

- **Congestion Impact Score (0–100):** weighted combination of three components, each
  traceable back to raw data:
  - **Severity** — max violation-type weight (e.g. blocking a road crossing > a quiet lane)
  - **Proximity** — fraction of violations near junctions or road crossings
  - **Temporal Concentration** — `1 − normalized entropy` of the hour-of-day distribution
    (a spike is worse than uniform spread)
- **Priority Score (0–100):** weighted geometric mean of impact × frequency × persistence ×
  recency. A near-zero factor anywhere drives the score to zero — an AND-like gate ensuring
  no single dimension inflates rank.
- **Why transparent:** every number a city official sees can be traced back to component
  sub-scores. This is essential for trust in an enforcement context.
- **Deterministic:** given identical inputs and seeds, the pipeline produces byte-identical
  output every run.

---

## Unsupervised Risk Profiling (Bonus ML Layer)

Two additional unsupervised models enrich each hotspot's profile:

- **IsolationForest → `anomaly_score` ∈ [0, 1]:** flags hotspots with unusual feature
  profiles relative to the population. A small, highly severe, temporally concentrated
  cluster may rank lower on priority (low frequency) but score high on anomaly — worth
  a second look.
- **K-Means (k=3) → `risk_tier` ∈ {HIGH, MEDIUM, LOW}:** groups hotspots by their feature
  profile. Clusters are labelled by descending mean impact (not an arbitrary ordering).
  No classifier is stacked on the K-Means labels — that would be circular.

---

## What We Do NOT Claim

| Claim | Status |
|-------|--------|
| "We predict congestion impact from violation data alone" | ❌ No ground-truth congestion labels exist. We measure *violation severity* honestly. |
| "Our model is trained on traffic-flow data" | ❌ We have no such data. |
| "RandomForest predicts risk tiers" | ❌ Removed. Predicting K-Means' own labels is circular. |
| "Deep learning / LSTM for time series" | ❌ Not used. On 300k tabular rows LightGBM is strictly better and more explainable. |

---

## Reproducibility

- Fixed random seed (42) throughout.
- Temporal split defined by calendar date, not random state.
- `python train_forecaster.py` → trains, evaluates, and persists `artifacts/forecast_model.joblib`.
- `python analysis.py` → regenerates all evaluation plots in `eval_output/`.
- `python -m streamlit run parking_intelligence/dashboard.py -- --artifacts artifacts/` → live demo.

---

## Files

| File | Purpose |
|------|---------|
| `parking_intelligence/hotspots.py` | DBSCAN + H3 unsupervised clustering (Layer 1) |
| `parking_intelligence/impact.py` | Congestion impact scoring (Layer 3) |
| `parking_intelligence/priority.py` | Priority ranking + recency decay (Layer 3) |
| `parking_intelligence/forecast.py` | Temporal peak profiles — hour×day heatmaps (Layer 3) |
| `parking_intelligence/forecaster.py` | LightGBM Poisson demand forecaster (Layer 2) |
| `parking_intelligence/risk.py` | IsolationForest anomaly + K-Means tiers (Bonus) |
| `train_forecaster.py` | Tune + train + persist the LightGBM model |
| `analysis.py` | Full-dataset evaluation plots |
| `eval_output/ml/` | Tuned model metrics, confusion matrix, ROC, feature importance |
