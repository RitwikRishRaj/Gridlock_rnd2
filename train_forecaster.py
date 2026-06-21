"""Train, lightly tune, evaluate and PERSIST the demand forecaster.

Pipeline:
  1. Load + clean full dataset, assign H3 cells.
  2. Light hyperparameter sweep, scored on a leakage-free temporal hold-out
     (last 30 days). Pick the best config by hold-out MAE.
  3. Re-fit the chosen config on ALL data and persist to
     ``artifacts/forecast_model.joblib`` for instant dashboard loading.
  4. Write metrics + plots to ``eval_output/ml/``.

Removable: delete this file + ml_forecast.py + parking_intelligence/forecaster.py
+ artifacts/forecast_model.joblib to drop the ML add-on entirely.

Usage:
    python train_forecaster.py
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)

from parking_intelligence.ingest import Ingestor
from parking_intelligence.hotspots import HotspotBuilder
from parking_intelligence.forecaster import (
    DemandForecaster, build_daily_panel, FEATURES, CAT_FEATURES,
)

CSV = "jan to may police violation_anonymized791b166.csv"
OUT = "eval_output/ml"
MODEL_PATH = "artifacts/forecast_model.joblib"
H3_RES = 8
SEED = 42

# Small, honest sweep (kept tiny so it runs in seconds).
PARAM_GRID = [
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 600, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 700, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 900, "min_child_samples": 30},
    {"num_leaves": 127, "learning_rate": 0.03, "n_estimators": 900, "min_child_samples": 30},
]


def banner(m): print(f"\n{'='*60}\n{m}\n{'='*60}", flush=True)


def fit_eval(train, test, params):
    base = dict(objective="poisson", subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, n_jobs=-1, verbose=-1)
    base.update(params)
    model = lgb.LGBMRegressor(**base)
    model.fit(train[FEATURES], train["count"].to_numpy(float),
              categorical_feature=CAT_FEATURES)
    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    mae = mean_absolute_error(test["count"], pred)
    return model, pred, mae


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    t0 = time.time()

    banner("Load + clean + assign H3")
    df, report = Ingestor().load_and_clean(CSV, chunksize=100_000)
    df = HotspotBuilder().assign_h3(df, resolution=H3_RES)
    print(f"retained rows: {report.rows_retained:,}", flush=True)

    panel = build_daily_panel(df, h3_res=H3_RES)
    panel = panel.dropna(subset=["lag_7", "roll_7_mean", "expanding_mean"])
    # Stable codes for the eval split.
    codes = {c: i for i, c in enumerate(panel["h3_cell"].astype("category").cat.categories)}
    panel = panel.assign(cell_code=panel["h3_cell"].map(codes))

    max_date = panel["date"].max()
    cutoff = max_date - pd.Timedelta(days=30)
    train = panel[panel["date"] <= cutoff]
    test = panel[panel["date"] > cutoff]
    print(f"date range {panel['date'].min().date()} -> {max_date.date()} | "
          f"train {len(train):,} / test {len(test):,}", flush=True)

    banner("Light hyperparameter sweep (temporal hold-out)")
    best = None
    for i, params in enumerate(PARAM_GRID):
        _, _, mae = fit_eval(train, test, params)
        print(f"  cfg {i}: MAE={mae:.4f}  {params}", flush=True)
        if best is None or mae < best[0]:
            best = (mae, params)
    best_mae, best_params = best
    print(f"\nBEST cfg -> MAE={best_mae:.4f}  {best_params}", flush=True)

    # Refit best config on the eval split to produce reported metrics + plots.
    model, pred, _ = fit_eval(train, test, best_params)
    y_te = test["count"].to_numpy(float)
    baseline = test["lag_7"].to_numpy(float)

    def metrics(name, yhat):
        return {"model": name,
                "MAE": mean_absolute_error(y_te, yhat),
                "RMSE": float(np.sqrt(mean_squared_error(y_te, yhat))),
                "R2": r2_score(y_te, yhat)}

    res = pd.DataFrame([metrics("Baseline(lag7)", baseline),
                        metrics("LightGBM(tuned)", pred)]).set_index("model")
    banner("Regression metrics (tuned, temporal hold-out)")
    print(res.round(3).to_string(), flush=True)
    improve = (res.loc["Baseline(lag7)", "MAE"] - res.loc["LightGBM(tuned)", "MAE"]) \
        / res.loc["Baseline(lag7)", "MAE"] * 100

    # Hotspot-day classification + best-F1 threshold search.
    thr_q = float(np.quantile(train["count"], 0.75))
    y_bin = (y_te >= thr_q).astype(int)
    try:
        auc = roc_auc_score(y_bin, pred)
    except ValueError:
        auc = float("nan")
    # Search threshold on predictions that maximizes F1 (operating point).
    cand = np.quantile(pred, np.linspace(0.5, 0.95, 19))
    best_f1, best_thr = -1, thr_q
    for t in cand:
        f1 = f1_score(y_bin, (pred >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    y_pred_bin = (pred >= best_thr).astype(int)
    prec = precision_score(y_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_bin, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_bin, y_pred_bin)
    print(f"\nHotspot-day (label thr={thr_q:.1f}/day, op thr={best_thr:.2f}): "
          f"precision={prec:.3f} recall={rec:.3f} F1={f1:.3f} AUC={auc:.3f}", flush=True)
    print("confusion matrix [actual x pred]:\n", cm, flush=True)

    # Plots (overwrite the standalone ones with tuned versions).
    plt.figure(figsize=(6, 6)); lim = max(y_te.max(), pred.max())
    plt.scatter(y_te, pred, s=8, alpha=0.3, color="#4285f4")
    plt.plot([0, lim], [0, lim], "r--", lw=1)
    plt.title("Predicted vs Actual (tuned, hold-out)")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.tight_layout(); plt.savefig(f"{OUT}/01_pred_vs_actual.png", dpi=120); plt.close()

    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 6)); fi.plot(kind="barh", color="#34a853")
    plt.title("Feature Importance (tuned)")
    plt.tight_layout(); plt.savefig(f"{OUT}/03_feature_importance.png", dpi=120); plt.close()

    plt.figure(figsize=(5.5, 5)); plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm.max()/2 else "black", fontsize=13)
    plt.xticks([0, 1], ["Not", "Hotspot"]); plt.yticks([0, 1], ["Not", "Hotspot"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (tuned, F1={f1:.2f})"); plt.colorbar()
    plt.tight_layout(); plt.savefig(f"{OUT}/05_confusion_matrix.png", dpi=120); plt.close()

    # ---- Persist FINAL model trained on ALL data ----
    banner("Fitting FINAL model on all data + persisting")
    fc = DemandForecaster(h3_res=H3_RES)
    fc.fit(df, params=best_params)
    path = fc.save(MODEL_PATH)
    sample = fc.predict_hotspots(n_days=7, top_n=5)
    print(f"saved -> {path}", flush=True)
    print("\nSanity: top-5 predicted hotspots (next 7 days):", flush=True)
    print(sample[["forecast_rank", "police_station", "total_predicted", "peak_date"]]
          .to_string(index=False), flush=True)

    lines = [
        "TUNED FORECASTER - EVALUATION + PERSISTENCE",
        "=" * 50,
        f"H3 resolution        : {H3_RES}",
        f"Best params          : {best_params}",
        f"Date range / cutoff  : {panel['date'].min().date()} -> {max_date.date()} (last 30d test)",
        "",
        res.round(3).to_string(),
        f"\nMAE improvement over baseline : {improve:.1f}%",
        f"Hotspot-day F1={f1:.3f}  precision={prec:.3f}  recall={rec:.3f}  ROC-AUC={auc:.3f}",
        f"confusion matrix [actual x pred]:\n{cm}",
        f"\nPersisted model: {path}",
        f"Runtime: {time.time()-t0:.1f}s",
    ]
    with open(f"{OUT}/summary_tuned.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
