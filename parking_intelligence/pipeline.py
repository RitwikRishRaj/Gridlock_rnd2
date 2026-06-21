"""Pipeline orchestrator for the Parking Intelligence system.

Stages
------
1. Ingest    — chunked CSV, parse, geo-validate, normalise.
2. Hotspots  — DBSCAN density clustering (H3-cell fallback).
3. Impact    — per-hotspot congestion-impact score (0–100).
4. Forecast  — Laplace-smoothed 7×24 peak profiles + contiguous-window coalescing.
5. Priority  — weighted geometric-mean rank with recency decay.
5a. Risk     — two honest unsupervised ML models (IsolationForest + K-Means).
6. Export    — atomic GeoJSON + CSV with ML columns embedded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from .export import Exporter
from .forecast import PeakForecaster
from .hotspots import HotspotBuilder
from .impact import ImpactScorer
from .ingest import Ingestor
from .ml_model import MLScores
from .models import (
    ImpactWeights,
    IngestionReport,
    PriorityConfig,
    PriorityZone,
    ScoredHotspot,
)
from .priority import PriorityRanker
from . import risk as _risk

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Outcome of a :func:`run` call.

    Attributes
    ----------
    geojson_path:
        Absolute path to the written ``hotspots.geojson``.
    priority_csv_path:
        Absolute path to the written ``priority_zones.csv``.
    priority_zones:
        In-memory list of all ranked :class:`~parking_intelligence.models.PriorityZone`
        records.
    scored_hotspots:
        The scored hotspot list produced by the impact stage.
    ingestion_report:
        Drop accounting from the ingestor.
    used_dbscan:
        ``True`` if DBSCAN produced clusters; ``False`` if H3-cell fallback was used.
    ml_scores:
        :class:`~parking_intelligence.ml_model.MLScores` with
        ``anomaly_score`` (IsolationForest) and ``risk_tier`` (K-Means)
        per hotspot.  Both are embedded in the exported artifacts.
    """

    geojson_path: str = ""
    priority_csv_path: str = ""
    priority_zones: list[PriorityZone] = field(default_factory=list)
    scored_hotspots: list[ScoredHotspot] = field(default_factory=list)
    ingestion_report: IngestionReport = field(default_factory=IngestionReport)
    used_dbscan: bool = True
    ml_scores: MLScores = field(default_factory=MLScores)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
_DEFAULT_SEED: int = 42
# Deterministic fallback recency reference used only when the cleaned data has
# no usable timestamps (keeps runs reproducible — Requirement 14.4).
_DEFAULT_AS_OF: datetime = datetime(2024, 6, 1, tzinfo=timezone.utc)
_DEFAULT_H3_RES: int = 9
_DEFAULT_EPS_M: float = 75.0
_DEFAULT_MIN_SAMPLES: int = 15
_DEFAULT_TOP_K_PEAKS: int = 3


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(
    csv_path: str,
    out_dir: str = "artifacts/",
    *,
    # Ingestion
    chunksize: int = 50_000,
    # Spatial
    h3_res: int = _DEFAULT_H3_RES,
    dbscan_eps_m: float = _DEFAULT_EPS_M,
    min_samples: int = _DEFAULT_MIN_SAMPLES,
    # Impact scoring
    impact_weights: ImpactWeights | None = None,
    # Priority ranking
    priority_config: PriorityConfig | None = None,
    # Forecasting
    top_k_peaks: int = _DEFAULT_TOP_K_PEAKS,
    # Reproducibility
    seed: int = _DEFAULT_SEED,
    # Optional live-traffic (bonus)
    live_traffic_connector: object | None = None,
) -> PipelineResult:
    """Run the full Parking Intelligence pipeline and return a :class:`PipelineResult`.

    Parameters
    ----------
    csv_path:
        Path to the violation CSV file.
    out_dir:
        Directory where ``hotspots.geojson`` and ``priority_zones.csv`` will
        be written.  Created if it does not exist.
    chunksize:
        Rows per CSV chunk passed to the ingestor.
    h3_res:
        H3 resolution for spatial binning.
    dbscan_eps_m:
        DBSCAN neighbourhood radius in metres.
    min_samples:
        DBSCAN minimum cluster density.
    impact_weights:
        Custom :class:`~parking_intelligence.models.ImpactWeights`.  Defaults
        are used when ``None``.
    priority_config:
        Custom :class:`~parking_intelligence.models.PriorityConfig` (sets the
        ``as_of`` reference time for recency decay).  Defaults to the current
        UTC time when ``None``.
    top_k_peaks:
        Number of peak windows attached to each priority zone.
    seed:
        Random seed for DBSCAN reproducibility.  A fixed default is applied
        rather than an entropy-derived seed (Requirement 14.4).
    live_traffic_connector:
        Optional connector object exposing a ``fetch(hotspot_ids, timeout=5)``
        method.  ``None`` (the default) disables live-traffic augmentation
        (Requirements 17.2–17.4).

    Returns
    -------
    PipelineResult
    """
    # Apply the seed deterministically (Requirement 14.4).
    np.random.seed(seed)

    # Defaults.
    if impact_weights is None:
        impact_weights = ImpactWeights()

    # ------------------------------------------------------------------
    # Stage 1: Ingestion
    # ------------------------------------------------------------------
    ingestor = Ingestor()
    df, report = ingestor.load_and_clean(csv_path, chunksize=chunksize)

    # Derive a DETERMINISTIC default recency reference (Requirement 14.4):
    # use the latest event timestamp in the data rather than wall-clock now(),
    # so repeated runs on identical input produce byte-identical artifacts.
    if priority_config is None:
        as_of = _DEFAULT_AS_OF
        if "created_at" in df.columns:
            import pandas as pd  # local import; pandas is a hard dependency
            max_ts = pd.to_datetime(df["created_at"], errors="coerce").max()
            if pd.notna(max_ts):
                as_of = max_ts.to_pydatetime()
        priority_config = PriorityConfig(as_of=as_of)

    # ------------------------------------------------------------------
    # Stage 2: Spatial aggregation (DBSCAN → H3 fallback)
    # ------------------------------------------------------------------
    builder = HotspotBuilder()
    hotspots = builder.build_hotspots(
        df,
        h3_res=h3_res,
        eps_m=dbscan_eps_m,
        min_samples=min_samples,
    )

    # Detect whether DBSCAN was used or the H3 fallback was triggered.
    used_dbscan = bool(hotspots) and any(
        hs.hotspot_id.startswith("dbscan-") for hs in hotspots
    )

    # If completely empty (e.g., DataFrame had too few rows), force H3 fallback.
    if not hotspots and not df.empty:
        hotspots = builder._hotspots_from_h3(builder.assign_h3(df, resolution=h3_res))
        used_dbscan = False

    # ------------------------------------------------------------------
    # Stage 3: Congestion impact scoring
    # ------------------------------------------------------------------
    # Optional live-traffic signal (Requirement 17.1–17.4).
    live_signal: dict[str, float] | None = None
    if live_traffic_connector is not None:
        live_signal = _fetch_live_traffic(
            live_traffic_connector,
            [hs.hotspot_id for hs in hotspots],
        )

    scorer = ImpactScorer()
    scored = scorer.score_impact(hotspots, df, impact_weights, live_traffic_signal=live_signal)

    # ------------------------------------------------------------------
    # Stage 4: Temporal peak forecasting
    # ------------------------------------------------------------------
    forecaster = PeakForecaster()
    profiles = forecaster.build_peak_profiles(df, hotspots)

    # ------------------------------------------------------------------
    # Stage 5: Priority ranking
    # ------------------------------------------------------------------
    ranker = PriorityRanker()
    zones = ranker.rank_zones(
        scored,
        profiles,
        df,
        priority_config,
        top_k_peaks=top_k_peaks,
    )

    # ------------------------------------------------------------------
    # Stage 5a: Unsupervised ML risk profiling
    #   - IsolationForest  → anomaly_score ∈ [0,1]  (1 = most anomalous)
    #   - K-Means (k=3)    → risk_tier ∈ {HIGH, MEDIUM, LOW}
    # Both fit on the hotspot feature matrix; no labels are invented.
    # ------------------------------------------------------------------
    risk_df = _risk.compute_risk(scored)   # DataFrame[hotspot_id, anomaly_score, risk_tier]
    ml_scores = MLScores(
        anomaly_scores=dict(zip(risk_df["hotspot_id"], risk_df["anomaly_score"])),
        risk_tiers=dict(zip(risk_df["hotspot_id"], risk_df["risk_tier"])),
    )

    # ------------------------------------------------------------------
    # Stage 6: Export (with ML scores attached)
    # ------------------------------------------------------------------
    exporter = Exporter()
    paths = exporter.export_all(scored, zones, out_dir, ml_scores=ml_scores)

    return PipelineResult(
        geojson_path=paths["geojson"],
        priority_csv_path=paths["csv"],
        priority_zones=zones,
        scored_hotspots=scored,
        ingestion_report=report,
        used_dbscan=used_dbscan,
        ml_scores=ml_scores,
    )


# ---------------------------------------------------------------------------
# Live-traffic helper (timeout-bounded, offline fallback on failure)
# ---------------------------------------------------------------------------
def _fetch_live_traffic(
    connector: object,
    hotspot_ids: list[str],
    timeout: float = 5.0,
) -> dict[str, float] | None:
    """Call the live-traffic connector with a timeout; return ``None`` on failure.

    Requirements 17.2–17.4: any exception (network error, timeout, bad data)
    returns ``None`` so the pipeline continues with offline scoring only.
    """
    try:
        result = connector.fetch(hotspot_ids, timeout=timeout)  # type: ignore[attr-defined]
        if isinstance(result, dict):
            return {str(k): float(v) for k, v in result.items()}
        return None
    except Exception:
        # Any failure → offline fallback; never abort the pipeline.
        return None
