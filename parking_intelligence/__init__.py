"""Parking Intelligence: offline, AI-driven parking-violation analytics pipeline.

This package turns raw Bengaluru traffic-police parking-violation CSV records into a
prioritized, map-based enforcement plan. Every stage runs locally with no required API
keys or outbound network calls (Requirements 16.1, 16.2).

Pipeline stages (one module each):
    ingest      - chunked CSV reading, parsing, geo validation, normalization
    hotspots    - H3 binning + DBSCAN density clustering
    impact      - congestion-impact scoring
    priority    - per-station priority ranking
    forecast    - temporal peak prediction
    export      - hotspots.geojson + priority_zones.csv
    pipeline    - orchestration
    dashboard   - Streamlit UI

Quickstart::

    from parking_intelligence import pipeline

    result = pipeline.run(
        csv_path="jan to may police violation_anonymized791b166.csv",
        out_dir="artifacts/",
    )
    print(result.geojson_path)        # artifacts/hotspots.geojson
    print(result.priority_csv_path)   # artifacts/priority_zones.csv
"""

__version__ = "0.1.0"

# Re-export the most commonly used public surface.
from .pipeline import PipelineResult, run  # noqa: F401
from .models import (  # noqa: F401
    ImpactWeights,
    PriorityConfig,
    PriorityZone,
    ScoredHotspot,
    Hotspot,
    PeakProfile,
    PeakWindow,
    IngestionReport,
    TimeWindow,
)
