"""Supervised violation-demand forecaster (OPTIONAL / REMOVABLE add-on).

A LightGBM (Poisson) model that forecasts daily violation counts per H3 cell and
supports recursive multi-day forecasting for the dashboard's "next 7 days"
prediction panel.

This module is intentionally decoupled from the core pipeline: nothing in
``pipeline.py`` imports it. The dashboard imports it inside a try/except so the
app still runs if the model or LightGBM is absent. To remove the feature
entirely, delete this file, ``train_forecaster.py``, the persisted
``artifacts/forecast_model.joblib`` and the dashboard's guarded section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

DEFAULT_H3_RES = 8
DEFAULT_MIN_CELL_TOTAL = 50
SEED = 42

FEATURES = [
    "lag_1", "lag_7", "roll_7_mean", "roll_14_mean", "expanding_mean",
    "dow", "is_weekend", "month", "day", "day_of_year", "cell_code",
]
CAT_FEATURES = ["cell_code", "dow", "month", "is_weekend"]


# ---------------------------------------------------------------------------
# Panel construction (shared by training + forecasting)
# ---------------------------------------------------------------------------
def build_daily_panel(
    df: pd.DataFrame,
    *,
    h3_res: int = DEFAULT_H3_RES,
    min_cell_total: int = DEFAULT_MIN_CELL_TOTAL,
) -> pd.DataFrame:
    """Build a zero-filled (cell x date) daily-count panel with leakage-safe features.

    Requires ``df`` to already contain an ``h3_cell`` column and a tz-aware
    ``created_at`` column.
    """
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at", "h3_cell"])
    df["date"] = df["created_at"].dt.normalize().dt.tz_localize(None)

    totals = df.groupby("h3_cell").size()
    keep = totals[totals >= min_cell_total].index
    df = df[df["h3_cell"].isin(keep)]

    daily = df.groupby(["h3_cell", "date"]).size().rename("count").reset_index()

    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    cells = daily["h3_cell"].unique()
    full_idx = pd.MultiIndex.from_product(
        [cells, all_dates], names=["h3_cell", "date"]
    )
    panel = (
        daily.set_index(["h3_cell", "date"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
        .sort_values(["h3_cell", "date"])
        .reset_index(drop=True)
    )
    return _add_features(panel)


def _add_features(panel: pd.DataFrame, cell_codes: dict | None = None) -> pd.DataFrame:
    """Attach calendar + lag + rolling features (all shift-before-use)."""
    panel = panel.sort_values(["h3_cell", "date"]).reset_index(drop=True)
    g = panel.groupby("h3_cell")["count"]
    panel["lag_1"] = g.shift(1)
    panel["lag_7"] = g.shift(7)
    panel["roll_7_mean"] = g.shift(1).rolling(7, min_periods=1).mean()
    panel["roll_14_mean"] = g.shift(1).rolling(14, min_periods=1).mean()
    panel["expanding_mean"] = g.shift(1).expanding(min_periods=1).mean()

    panel["dow"] = panel["date"].dt.weekday
    panel["is_weekend"] = (panel["dow"] >= 5).astype(int)
    panel["month"] = panel["date"].dt.month
    panel["day"] = panel["date"].dt.day
    panel["day_of_year"] = panel["date"].dt.dayofyear

    if cell_codes is None:
        panel["cell_code"] = panel["h3_cell"].astype("category").cat.codes
    else:
        panel["cell_code"] = panel["h3_cell"].map(cell_codes).fillna(-1).astype(int)
    return panel


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------
@dataclass
class DemandForecaster:
    """Train, persist, and run recursive multi-day violation-demand forecasts."""

    h3_res: int = DEFAULT_H3_RES
    min_cell_total: int = DEFAULT_MIN_CELL_TOTAL
    model: object = None
    cell_codes: dict = field(default_factory=dict)            # h3_cell -> int
    cell_centroids: dict = field(default_factory=dict)        # h3_cell -> (lat, lon)
    cell_station: dict = field(default_factory=dict)          # h3_cell -> station
    tail_history: pd.DataFrame | None = None                  # recent counts per cell
    last_date: pd.Timestamp | None = None

    # -- Training ---------------------------------------------------------
    def fit(self, df: pd.DataFrame, *, params: dict | None = None) -> "DemandForecaster":
        import lightgbm as lgb

        panel = build_daily_panel(
            df, h3_res=self.h3_res, min_cell_total=self.min_cell_total
        )
        train = panel.dropna(subset=["lag_7", "roll_7_mean", "expanding_mean"])

        # Stable cell codes for inference.
        cats = train["h3_cell"].astype("category")
        self.cell_codes = {c: i for i, c in enumerate(cats.cat.categories)}
        train = train.assign(cell_code=train["h3_cell"].map(self.cell_codes))

        p = dict(
            objective="poisson", n_estimators=700, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        if params:
            p.update(params)
        self.model = lgb.LGBMRegressor(**p)
        self.model.fit(train[FEATURES], train["count"].to_numpy(float),
                       categorical_feature=CAT_FEATURES)

        # Cell metadata for the dashboard map.
        self._capture_cell_metadata(df)
        # Keep the last 21 days per cell so recursive lags can be computed.
        self.last_date = panel["date"].max()
        cutoff = self.last_date - timedelta(days=21)
        self.tail_history = (
            panel[panel["date"] > cutoff][["h3_cell", "date", "count"]]
            .reset_index(drop=True)
        )
        return self

    def _capture_cell_metadata(self, df: pd.DataFrame) -> None:
        df = df.dropna(subset=["h3_cell"])
        for cell, grp in df.groupby("h3_cell"):
            if cell not in self.cell_codes:
                continue
            self.cell_centroids[cell] = (
                float(grp["latitude"].mean()),
                float(grp["longitude"].mean()),
            )
            if "police_station" in grp.columns:
                mode = grp["police_station"].mode()
                self.cell_station[cell] = (
                    str(mode.iloc[0]) if not mode.empty else None
                )

    # -- Recursive multi-day forecast ------------------------------------
    def predict_next_days(self, n_days: int = 7) -> pd.DataFrame:
        """Forecast the next ``n_days`` daily counts for every known cell.

        Returns a tidy DataFrame: ``[h3_cell, date, predicted_count,
        centroid_lat, centroid_lon, police_station]``.
        """
        if self.model is None or self.tail_history is None:
            raise RuntimeError("Forecaster is not fitted/loaded.")

        hist = self.tail_history.copy()
        cells = list(self.cell_codes.keys())
        out_rows: list[dict] = []

        for step in range(1, n_days + 1):
            d = self.last_date + timedelta(days=step)
            feat_rows = []
            for cell in cells:
                ch = hist[hist["h3_cell"] == cell].sort_values("date")
                counts = ch["count"].to_numpy(float)
                if counts.size == 0:
                    continue
                lag_1 = counts[-1]
                lag_7 = counts[-7] if counts.size >= 7 else counts.mean()
                roll_7 = counts[-7:].mean()
                roll_14 = counts[-14:].mean()
                expanding = counts.mean()
                feat_rows.append({
                    "h3_cell": cell,
                    "lag_1": lag_1, "lag_7": lag_7,
                    "roll_7_mean": roll_7, "roll_14_mean": roll_14,
                    "expanding_mean": expanding,
                    "dow": d.weekday(), "is_weekend": int(d.weekday() >= 5),
                    "month": d.month, "day": d.day, "day_of_year": d.dayofyear,
                    "cell_code": self.cell_codes[cell],
                })
            fdf = pd.DataFrame(feat_rows)
            yhat = np.clip(self.model.predict(fdf[FEATURES]), 0, None)
            fdf = fdf.assign(date=d, predicted_count=yhat)

            # Feed predictions back as history for the next step (recursive).
            hist = pd.concat(
                [hist, fdf[["h3_cell", "date"]].assign(count=yhat)],
                ignore_index=True,
            )
            for _, r in fdf.iterrows():
                lat, lon = self.cell_centroids.get(r["h3_cell"], (np.nan, np.nan))
                out_rows.append({
                    "h3_cell": r["h3_cell"],
                    "date": d,
                    "predicted_count": float(r["predicted_count"]),
                    "centroid_lat": lat,
                    "centroid_lon": lon,
                    "police_station": self.cell_station.get(r["h3_cell"]),
                })
        return pd.DataFrame(out_rows)

    def predict_hotspots(self, n_days: int = 7, top_n: int = 20) -> pd.DataFrame:
        """Aggregate the next-``n_days`` forecast into a ranked hotspot table."""
        fc = self.predict_next_days(n_days=n_days)
        if fc.empty:
            return fc
        agg = (
            fc.groupby(["h3_cell", "centroid_lat", "centroid_lon", "police_station"])
            .agg(total_predicted=("predicted_count", "sum"),
                 peak_day=("predicted_count", "idxmax"))
            .reset_index()
        )
        # Resolve the peak date for each cell.
        peak_dates = fc.loc[agg["peak_day"], "date"].dt.date.to_numpy()
        agg["peak_date"] = peak_dates
        agg = agg.drop(columns=["peak_day"])
        agg = agg.sort_values("total_predicted", ascending=False).reset_index(drop=True)
        agg.insert(0, "forecast_rank", range(1, len(agg) + 1))
        return agg.head(top_n)

    # -- Persistence ------------------------------------------------------
    def save(self, path: str) -> str:
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # protocol=4 ensures the file is loadable by Python 3.4+ regardless of
        # which exact Python version the deployment server uses (3.11, 3.12, 3.13…)
        joblib.dump(self, path, protocol=4)
        return os.path.abspath(path)

    @staticmethod
    def load(path: str) -> "DemandForecaster":
        import joblib
        return joblib.load(path)
