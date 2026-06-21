"""Core data models and config objects for the Parking Intelligence pipeline.

This module defines the immutable (frozen) dataclasses that flow between pipeline
stages, the configuration objects that tune scoring/ranking, the Bengaluru
bounding-box constants used for geo validation, and small helper structures
(``TimeWindow`` for dashboard filtering, ``IngestionReport`` for dropped-row
accounting).

All records are pure dataclasses with no heavy dependencies, so this module
imports cleanly in any environment with a standard Python install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Bengaluru bounding-box constants
# ---------------------------------------------------------------------------
# Used by ingestion/geo validation: a row is retained only when its coordinates
# fall inside this inclusive box (lat in [LAT_MIN, LAT_MAX], lon in [LON_MIN, LON_MAX]).
LAT_MIN: float = 12.7
LAT_MAX: float = 13.2
LON_MIN: float = 77.3
LON_MAX: float = 77.9


# ---------------------------------------------------------------------------
# Model 1: ViolationRecord (a cleaned row)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ViolationRecord:
    """A single cleaned, typed, geo-validated parking-violation row."""

    id: str
    latitude: float                 # validated within Bengaluru bbox
    longitude: float
    location: str | None
    vehicle_type: str | None        # normalized lowercase token
    violation_types: list[str]      # parsed from JSON array, upper-cased tokens
    offence_codes: list[int]        # parsed from JSON array
    created_at: datetime            # tz-aware (Asia/Kolkata)
    closed_at: datetime | None
    police_station: str | None
    junction_name: str | None
    center_code: str | None
    validation_status: str | None


# ---------------------------------------------------------------------------
# Model 2: Hotspot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hotspot:
    """A spatial cluster of violations (H3 cell and/or DBSCAN cluster)."""

    hotspot_id: str                 # stable id, e.g. h3 cell or "dbscan-<n>"
    centroid_lat: float
    centroid_lon: float
    h3_cell: str | None
    cluster_label: int | None       # -1 == DBSCAN noise (excluded from hotspots)
    member_count: int               # number of violations in the hotspot
    police_station: str | None      # modal station for member violations
    member_ids: list[str]


# ---------------------------------------------------------------------------
# Model 3: ScoredHotspot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoredHotspot:
    """A hotspot annotated with a congestion-impact score and breakdown."""

    hotspot: Hotspot
    impact_score: float             # 0..100
    severity_component: float       # 0..1
    proximity_component: float      # 0..1
    concentration_component: float  # 0..1
    breakdown: dict[str, float]     # explainability: factor -> contribution


# ---------------------------------------------------------------------------
# Model 5: PeakProfile & PeakWindow
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeakProfile:
    """A hotspot's temporal signature as a normalized 7x24 intensity matrix."""

    hotspot_id: str
    hour_dow_matrix: list[list[float]]   # 7 x 24 normalized intensity
    total_events: int


@dataclass(frozen=True)
class PeakWindow:
    """A forecasted recurring peak window for a hotspot."""

    day_of_week: int                # 0 == Monday
    start_hour: int                 # 0..23
    end_hour: int                   # inclusive, 0..23
    expected_intensity: float       # 0..1 relative to hotspot max


# ---------------------------------------------------------------------------
# Model 4: PriorityZone
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PriorityZone:
    """A ranked enforcement zone with explainable priority factors."""

    hotspot_id: str
    centroid_lat: float
    centroid_lon: float
    police_station: str | None
    impact: float                   # 0..1 normalized
    frequency: float                # 0..1
    persistence: float              # 0..1
    recency: float                  # 0..1
    priority_score: float           # 0..100
    global_rank: int                # 1 == highest priority
    station_rank: int               # rank within its police_station
    peak_windows: list[PeakWindow]


# ---------------------------------------------------------------------------
# Model 6: Config objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImpactWeights:
    """Weights for the congestion-impact score. Must sum to 1.0."""

    w_severity: float = 0.45
    w_proximity: float = 0.35
    w_concentration: float = 0.20   # defaults sum to 1.0


@dataclass(frozen=True)
class PriorityConfig:
    """Configuration for priority ranking's recency component."""

    as_of: datetime                 # reference "now" for recency
    recency_halflife_days: float = 21.0


# ---------------------------------------------------------------------------
# Helper structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeWindow:
    """A dashboard time filter by day-of-week and/or hour-of-day.

    ``None`` on either field means "do not constrain on this axis".
    """

    day_of_week: int | None = None   # 0 == Monday, 0..6
    hour: int | None = None          # 0..23


@dataclass(frozen=True)
class IngestionReport:
    """Accounting of rows dropped during ingestion, keyed by reason.

    ``dropped_by_reason`` maps a human-readable reason (e.g. "invalid_geo",
    "unparseable_created_at", "duplicate_id") to the number of rows dropped for
    that reason. ``total_rows_read`` and ``rows_retained`` give an at-a-glance
    overview that the dashboard surfaces.
    """

    total_rows_read: int = 0
    rows_retained: int = 0
    dropped_by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def total_dropped(self) -> int:
        """Total number of rows dropped across all reasons."""
        return sum(self.dropped_by_reason.values())
