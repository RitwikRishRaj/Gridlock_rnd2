# Running Parking Intelligence — Step-by-Step

## Prerequisites
- Python 3.10 or later
- The violation CSV in the project root:
  `jan to may police violation_anonymized791b166.csv`

---

## 1 · Install dependencies

```bash
pip install -r requirements.txt
```

This installs pandas, numpy, h3, scikit-learn, LightGBM, Streamlit, pydeck, hypothesis,
and pytest. Everything runs offline — no API keys needed.

---

## 2 · Train the LightGBM forecaster (one-time)

```bash
python train_forecaster.py
```

What this does:
- Loads and cleans the full dataset (~298 k rows)
- Runs a hyperparameter sweep on a leakage-free chronological hold-out (last 30 days)
- Fits the best config on **all** data
- Saves `artifacts/forecast_model.joblib`
- Writes tuned evaluation plots + metrics to `eval_output/ml/`

Expected runtime: ~25 seconds.

---

## 3 · Run the full analytics pipeline

```bash
python -c "
from parking_intelligence import pipeline
pipeline.run(
    'jan to may police violation_anonymized791b166.csv',
    out_dir='artifacts/',
)
print('Done — artifacts/ folder is ready.')
"
```

What this produces in `artifacts/`:
- `hotspots.geojson` — 506 enforcement zones (GeoJSON FeatureCollection)
- `priority_zones.csv` — ranked zones with impact, priority, risk tier, anomaly score

Expected runtime: ~40 seconds.

> **Note:** if you already ran `train_forecaster.py`, the `forecast_model.joblib` is
> already in `artifacts/`. Re-running the pipeline will not overwrite it.

---

## 4 · Launch the dashboard

```bash
python -m streamlit run parking_intelligence/dashboard.py -- --artifacts artifacts/
```

Opens at **http://localhost:8501**

Dashboard panels:
| Panel | What you see |
|-------|-------------|
| Violation Heatmap | 3-D HexagonLayer of all hotspots, filterable by hour / day |
| Top 20 Priority Zones | Ranked enforcement table, filterable by risk tier |
| Station Drilldown | Zones, peak hours, sample violations for a chosen station |
| ML Model Insights | Risk tier distribution (K-Means) + anomaly leaderboard (IsolationForest) |
| Predicted Hotspots — Next 7 Days | LightGBM 7-day demand forecast + patrol-priority map |

---

## 5 · Generate evaluation plots (optional)

```bash
# Core pipeline analytics (impact distribution, geographic scatter, temporal matrix…)
python analysis.py

# ML model evaluation (pred-vs-actual, residuals, feature importance, confusion matrix, ROC)
python train_forecaster.py   # already done in step 2; re-run to refresh plots
```

All plots land in `eval_output/` and `eval_output/ml/`.

---

## 6 · Run tests (optional)

```bash
pytest
```

---

## Quick-reference commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Train ML model | `python train_forecaster.py` |
| Run pipeline | `python -c "from parking_intelligence import pipeline; pipeline.run('jan to may police violation_anonymized791b166.csv', out_dir='artifacts/')"` |
| Launch dashboard | `python -m streamlit run parking_intelligence/dashboard.py -- --artifacts artifacts/` |
| Eval plots | `python analysis.py` |
| Tests | `pytest` |

---

## Troubleshooting

**`streamlit` not found**
Add the Python scripts folder to your PATH, or use `python -m streamlit run …`.

**Dashboard shows "Forecast model not available"**
Run `python train_forecaster.py` first — it creates `artifacts/forecast_model.joblib`.

**Dashboard shows "Risk tier data not available"**
Re-run the pipeline (step 3) to regenerate `artifacts/priority_zones.csv` with the
`risk_tier` and `anomaly_score` columns.

**Memory pressure on the CSV**
Lower the chunk size: `pipeline.run(…, chunksize=50_000)`.
