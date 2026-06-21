"""Unsupervised risk profiling for scored hotspots.

Two honest, deterministic unsupervised models that enrich the priority output:

* **IsolationForest** -> ``anomaly_score`` in [0, 1]: how unusual a hotspot's
  feature profile is relative to the population (1 = most anomalous). Useful for
  surfacing hotspots that don't look like the others (e.g. a small but extremely
  severe/concentrated cluster).
* **K-Means (k=3)** -> ``risk_tier`` in {HIGH, MEDIUM, LOW}: groups hotspots by
  their feature profile, then labels the clusters by descending mean impact.

Both are fit on the same engineered features and seeded for reproducibility.
No supervised labels are invented. (We intentionally do NOT stack a classifier
on the K-Means labels — that would be circular.)

The pipeline imports this inside a try/except, so removing this file simply
drops the two extra columns without breaking the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .models import ScoredHotspot

SEED = 42
_FEATURE_NAMES = [
    "impact_score", "severity_component", "proximity_component",
    "concentration_component", "log_member_count",
]


def _feature_matrix(scored: list["ScoredHotspot"]) -> pd.DataFrame:
    rows = []
    for s in scored:
        rows.append({
            "hotspot_id": s.hotspot.hotspot_id,
            "impact_score": s.impact_score,
            "severity_component": s.severity_component,
            "proximity_component": s.proximity_component,
            "concentration_component": s.concentration_component,
            "log_member_count": float(np.log1p(s.hotspot.member_count)),
        })
    return pd.DataFrame(rows)


def compute_risk(scored: list["ScoredHotspot"]) -> pd.DataFrame:
    """Return a DataFrame ``[hotspot_id, anomaly_score, risk_tier]``.

    Deterministic (fixed seeds). Degrades gracefully for tiny inputs.
    """
    if not scored:
        return pd.DataFrame(columns=["hotspot_id", "anomaly_score", "risk_tier"])

    feats = _feature_matrix(scored)
    X = feats[_FEATURE_NAMES].to_numpy(dtype=float)
    n = len(feats)

    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)

    # ---- IsolationForest anomaly score (0..1, higher = more anomalous) ----
    try:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(
            n_estimators=200, contamination="auto", random_state=SEED,
        )
        iso.fit(Xs)
        raw = -iso.score_samples(Xs)  # higher = more anomalous
        lo, hi = raw.min(), raw.max()
        anomaly = (raw - lo) / (hi - lo) if hi > lo else np.zeros(n)
    except Exception:
        anomaly = np.zeros(n)
    feats["anomaly_score"] = np.round(anomaly, 4)

    # ---- K-Means risk tiers (HIGH / MEDIUM / LOW by mean impact) ----
    tier = np.array(["MEDIUM"] * n, dtype=object)
    if n >= 3:
        try:
            from sklearn.cluster import KMeans
            k = 3
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labels = km.fit_predict(Xs)
            # Rank clusters by mean impact_score -> HIGH/MEDIUM/LOW.
            order = (
                pd.DataFrame({"label": labels, "impact": feats["impact_score"]})
                .groupby("label")["impact"].mean()
                .sort_values(ascending=False).index.tolist()
            )
            mapping = {order[0]: "HIGH", order[1]: "MEDIUM", order[2]: "LOW"}
            tier = np.array([mapping[l] for l in labels], dtype=object)
        except Exception:
            tier = _quantile_tiers(feats["impact_score"].to_numpy())
    else:
        tier = _quantile_tiers(feats["impact_score"].to_numpy())
    feats["risk_tier"] = tier

    return feats[["hotspot_id", "anomaly_score", "risk_tier"]]


def _quantile_tiers(impact: np.ndarray) -> np.ndarray:
    """Fallback tiering by impact quantiles when clustering isn't viable."""
    if len(impact) == 0:
        return np.array([], dtype=object)
    hi = np.quantile(impact, 2 / 3)
    lo = np.quantile(impact, 1 / 3)
    out = np.where(impact >= hi, "HIGH", np.where(impact >= lo, "MEDIUM", "LOW"))
    return out.astype(object)
