"""Compatibility shim — ml_model.py is retired.

All ML logic now lives in :mod:`parking_intelligence.risk`:
  - IsolationForest anomaly detection → ``anomaly_score``
  - K-Means (k=3) risk tiers → ``risk_tier``

This module re-exports the minimal symbols that pipeline.py previously
used so that any external code that imported MLScores or run_ml_models
continues to work without changes.  New code should import from
:mod:`parking_intelligence.risk` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MLScores:
    """Lightweight result holder kept for backward compatibility.

    The fields ``rf_intensity_matrices`` and ``feature_importances`` previously
    contained output from the (now-removed) RandomForest-based components.
    They are retained here as empty dicts so that any code that reads them
    degrades gracefully rather than raising ``AttributeError``.
    """

    # Populated by risk.compute_risk (via pipeline.py).
    anomaly_scores: dict[str, float] = field(default_factory=dict)
    risk_tiers: dict[str, str] = field(default_factory=dict)

    # Removed components — kept as empty stubs.
    rf_intensity_matrices: dict = field(default_factory=dict)
    feature_importances: dict = field(default_factory=dict)


def run_ml_models(*args, **kwargs) -> MLScores:
    """Deprecated entry-point.  pipeline.py no longer calls this."""
    raise NotImplementedError(
        "run_ml_models() has been removed. "
        "Use parking_intelligence.risk.compute_risk(scored) instead."
    )
