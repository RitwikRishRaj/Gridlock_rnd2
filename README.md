# Parking Intelligence

**AI-driven parking enforcement analytics for Bengaluru** — detects illegal parking
hotspots, quantifies congestion impact, ranks enforcement zones per police station, and
**forecasts where violations will spike next week** so patrols can be proactive instead
of reactive.

> Full AI methodology details: [`AI_MODEL.md`](AI_MODEL.md)

---

## What's inside

A three-layer AI pipeline, all running **100% offline** — no API keys, no cloud, no connectivity required:

| Layer | Technology | Output |
|-------|-----------|--------|
| **Hotspot Detection** | DBSCAN + H3 hexagonal binning (unsupervised) | 506 spatial enforcement zones |
| **Demand Forecasting** | LightGBM (Poisson, temporal hold-out eval) | Next-7-day patrol priority |
| **Priority Ranking** | Explainable composite scoring | Ranked zones per police station |
| **Risk Profiling** | IsolationForest + K-Means (unsupervised) | Anomaly scores + risk tiers |

**Key metrics (LightGBM, chronological hold-out):**
- MAE improvement over naive baseline: **−18.7%**
- R²: **0.565** | ROC-AUC: **0.831** | Hotspot-day F1: **0.622**
- Data: Nov 2023 – Apr 2024, 298,277 violation records

---

## Project layout (flat)

```
parking_intelligence/
  models.py       # dataclasses + config
  ingest.py       # chunked CSV parsing, geo validation
  hotspots.py     # DBSCAN + H3 clustering
  impact.py       # congestion-impact scoring
  priority.py     # per-station priority ranking
  forecast.py     # temporal peak profiles (hour × day heatmaps)
  forecaster.py   # LightGBM demand forecaster (train + recursive predict)
  risk.py         # IsolationForest anomaly + K-Means risk tiers
  export.py       # hotspots.geojson + priority_zones.csv
  pipeline.py     # orchestrator
  dashboard.py    # Streamlit UI
train_forecaster.py   # tune + train + persist LightGBM model
analysis.py           # full-dataset evaluation plots
ml_forecast.py        # standalone eval (superseded by train_forecaster.py)
```

---

## Quick start

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Train + persist the LightGBM forecaster
python train_forecaster.py

# 3. Run the full analytics pipeline (generates artifacts/)
python -c "
from parking_intelligence import pipeline
pipeline.run('jan to may police violation_anonymized791b166.csv', out_dir='artifacts/')
"

# 4. Launch the dashboard
python -m streamlit run parking_intelligence/dashboard.py -- --artifacts artifacts/
```

Dashboard opens at **http://localhost:8501**

---

## Evaluation plots

```bash
python analysis.py          # generates eval_output/*.png (pipeline analytics)
python train_forecaster.py  # generates eval_output/ml/*.png (ML model evaluation)
```

Plots include: impact distribution, cluster sizes, geographic scatter, temporal heat
matrix, predicted-vs-actual, residuals, feature importance, confusion matrix, ROC curve.

---

## Offline design

Every stage — ingestion, clustering, scoring, forecasting, export, dashboard — runs on
local computation only. No API keys. No outbound network calls. The demo cannot be
blocked by connectivity.

Optional live-traffic augmentation is a pluggable bonus with a 5-second timeout; the
pipeline completes without it if unavailable.

---

## Reproducibility

Fixed seed (42) throughout. Two runs on identical inputs produce byte-identical
`priority_zones.csv` and `hotspots.geojson`. The LightGBM temporal split is a
**chronological hold-out** (last 30 days), not a random split — no future data leaks
into training.
