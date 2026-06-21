"""Priority ranking for the Parking Intelligence pipeline.

Implements Tasks 11.1 and 11.2:

* :func:`recency_decay` — exponential half-life decay in ``(0, 1]``
  (Requirements 10.1–10.6).
* :class:`PriorityRanker` with:
  - :meth:`frequency_score` — raw violation count for a hotspot.
  - :meth:`persistence_score` — number of distinct active days.
  - :meth:`rank_zones` — min-max normalise frequency/persistence, compute the
    weighted geometric mean priority score, and assign global + per-station
    ranks (Requirements 9.1–9.9, 10.1–10.6).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .models import PeakProfile, PeakWindow, PriorityConfig, PriorityZone

if TYPE_CHECKING:
    from .models import ScoredHotspot

# Priority score weights for the weighted geometric mean (Requirement 9.3).
_W_IMPACT: float = 0.40
_W_FREQ: float = 0.25
_W_PERS: float = 0.20
_W_REC: float = 0.15


# ---------------------------------------------------------------------------
# Task 11.1a: Recency decay
# ---------------------------------------------------------------------------
def recency_decay(last_event: datetime, config: PriorityConfig) -> float:
    """Compute the exponential half-life recency decay factor.

    Returns a value in ``(0, 1]``:
    * exactly ``1.0`` when *last_event* == ``config.as_of`` (zero elapsed days,
      Requirement 10.4).
    * exactly ``0.5`` when elapsed days == ``config.recency_halflife_days``
      (Requirement 10.5).
    * strictly non-increasing as elapsed days grow (Requirement 10.6).

    Parameters
    ----------
    last_event:
        The most recent event timestamp for a hotspot.  Must be ≤ ``config.as_of``
        (Requirement 10.1).
    config:
        :class:`~parking_intelligence.models.PriorityConfig` carrying ``as_of``
        and ``recency_halflife_days > 0`` (Requirement 10.1).

    Raises
    ------
    ValueError
        * If ``config.recency_halflife_days ≤ 0`` (Requirement 10.2).
        * If ``last_event > config.as_of`` (Requirement 10.2).
    """
    if config.recency_halflife_days <= 0:
        raise ValueError(
            f"recency_halflife_days must be > 0, got {config.recency_halflife_days!r}"
        )

    # Normalise both timestamps to UTC for comparison.
    as_of = _to_utc(config.as_of)
    last = _to_utc(last_event)

    if last > as_of:
        raise ValueError(
            f"last_event ({last}) must be ≤ as_of ({as_of})"
        )

    elapsed_days = (as_of - last).total_seconds() / 86_400.0
    # recency = 2^(−Δd / H)  →  exactly 0.5 at Δd == H, exactly 1.0 at Δd == 0.
    decay = 2.0 ** (-elapsed_days / config.recency_halflife_days)
    # Clamp strictly into (0, 1] for floating-point safety.
    return float(np.clip(decay, 1e-12, 1.0))


def _to_utc(dt: datetime) -> datetime:
    """Return *dt* as a UTC-aware datetime (naive → assumed UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Task 11.1b: Raw factor helpers
# ---------------------------------------------------------------------------
def frequency_score(hotspot: "ScoredHotspot") -> int:
    """Raw frequency = member count of the underlying hotspot."""
    return hotspot.hotspot.member_count


def persistence_score(events: pd.DataFrame) -> int:
    """Number of distinct calendar dates on which violations occurred."""
    if events.empty:
        return 0
    ts_col = "created_at" if "created_at" in events.columns else "created_datetime"
    if ts_col not in events.columns:
        return 0
    dates = pd.to_datetime(events[ts_col], errors="coerce").dropna().dt.date
    return int(dates.nunique())


# ---------------------------------------------------------------------------
# Task 11.2: PriorityRanker
# ---------------------------------------------------------------------------
class PriorityRanker:
    """Rank enforcement zones globally and within each police station."""

    # Expose helpers as methods for testability.
    recency_decay = staticmethod(recency_decay)
    frequency_score = staticmethod(frequency_score)
    persistence_score = staticmethod(persistence_score)

    def rank_zones(
        self,
        scored: list["ScoredHotspot"],
        profiles: dict[str, PeakProfile],
        df: pd.DataFrame,
        config: PriorityConfig,
        *,
        top_k_peaks: int = 3,
    ) -> list[PriorityZone]:
        """Compute priority scores and assign global + per-station ranks.

        Steps:
        1. Compute raw frequency and persistence per hotspot.
        2. Min-max normalise frequency and persistence to ``[0, 1]``
           (Requirement 9.1, 9.2).
        3. Compute recency decay (Requirement 10).
        4. Priority = 100 × weighted geometric mean of normalised factors
           (Requirement 9.3).  Any factor == 0 forces priority == 0
           (Requirement 9.5).
        5. Assign global ranks ``1..N`` (Requirement 9.6).
        6. Assign per-station ranks ``1..M`` (Requirement 9.7).
        7. Attach peak windows from *profiles* (Requirements 9.8, 9.9).

        Parameters
        ----------
        scored:
            :class:`~parking_intelligence.models.ScoredHotspot` list from
            :class:`~parking_intelligence.impact.ImpactScorer`.
        profiles:
            Peak profiles from
            :class:`~parking_intelligence.forecast.PeakForecaster`.
        df:
            Cleaned DataFrame (used to compute persistence and recency).
        config:
            Recency configuration with ``as_of`` and ``recency_halflife_days``.
        top_k_peaks:
            How many peak windows to attach to each zone.

        Returns
        -------
        list[PriorityZone]
            Zones ordered by ascending ``global_rank``.
        """
        if not scored:
            return []

        from .forecast import PeakForecaster
        forecaster = PeakForecaster()

        rows: list[dict] = []
        for s in scored:
            hs = s.hotspot
            # Get member events.
            if hs.member_ids and "id" in df.columns:
                events = df[df["id"].isin(hs.member_ids)]
            else:
                events = df.iloc[0:0]

            freq_raw = frequency_score(s)
            pers_raw = persistence_score(events)

            # Recency decay: use the most recent event timestamp.
            ts_col = "created_at" if "created_at" in events.columns else "created_datetime"
            if not events.empty and ts_col in events.columns:
                ts = pd.to_datetime(events[ts_col], errors="coerce").dropna()
                last_event = ts.max().to_pydatetime() if not ts.empty else config.as_of
            else:
                last_event = config.as_of

            rec = recency_decay(last_event, config)

            rows.append({
                "scored": s,
                "freq_raw": freq_raw,
                "pers_raw": pers_raw,
                "rec": rec,
                "events": events,
            })

        # Min-max normalise frequency and persistence (Requirements 9.1, 9.2).
        freq_vals = [r["freq_raw"] for r in rows]
        pers_vals = [r["pers_raw"] for r in rows]
        freq_norm = _minmax_normalize(freq_vals)
        pers_norm = _minmax_normalize(pers_vals)

        zones: list[PriorityZone] = []
        for row, fn, pn in zip(rows, freq_norm, pers_norm):
            s = row["scored"]
            hs = s.hotspot
            rec = row["rec"]
            impact_n = s.impact_score / 100.0  # normalise to [0, 1]

            priority = _weighted_geomean(
                [impact_n, fn, pn, rec],
                [_W_IMPACT, _W_FREQ, _W_PERS, _W_REC],
            )
            priority_score = float(np.clip(priority * 100.0, 0.0, 100.0))

            # Attach peak windows (Requirements 9.8, 9.9).
            profile = profiles.get(hs.hotspot_id)
            if profile is not None:
                peak_windows = forecaster.predict_next_peaks(profile, top_k=top_k_peaks)
            else:
                peak_windows = []

            zones.append(
                PriorityZone(
                    hotspot_id=hs.hotspot_id,
                    centroid_lat=hs.centroid_lat,
                    centroid_lon=hs.centroid_lon,
                    police_station=hs.police_station,
                    impact=impact_n,
                    frequency=fn,
                    persistence=pn,
                    recency=rec,
                    priority_score=priority_score,
                    global_rank=0,       # assigned below
                    station_rank=0,      # assigned below
                    peak_windows=peak_windows,
                )
            )

        # Assign global ranks (Requirement 9.6): descending priority, ascending id on tie.
        zones.sort(
            key=lambda z: (-z.priority_score, z.hotspot_id)
        )
        zones = [
            _replace_zone(z, global_rank=i + 1)
            for i, z in enumerate(zones)
        ]

        # Assign per-station ranks (Requirement 9.7).
        zones = _assign_station_ranks(zones)

        return zones


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _minmax_normalize(values: list[float]) -> list[float]:
    """Min-max normalise *values* to ``[0, 1]``.

    Collapsed range (all equal) → all ``1.0`` (Requirement 9.2).
    """
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [1.0] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def _weighted_geomean(factors: list[float], weights: list[float]) -> float:
    """Weighted geometric mean: ``∏ f_i^w_i``.

    Returns ``0.0`` when any factor is ``0`` (Requirement 9.5 — AND-like behaviour).
    """
    if any(f == 0.0 for f in factors):
        return 0.0
    log_sum = sum(w * math.log(f) for f, w in zip(factors, weights) if f > 0)
    return math.exp(log_sum)


def _replace_zone(zone: PriorityZone, **kwargs) -> PriorityZone:
    """Return a new PriorityZone with the given fields replaced (frozen dataclass)."""
    return PriorityZone(
        hotspot_id=zone.hotspot_id,
        centroid_lat=zone.centroid_lat,
        centroid_lon=zone.centroid_lon,
        police_station=zone.police_station,
        impact=zone.impact,
        frequency=zone.frequency,
        persistence=zone.persistence,
        recency=zone.recency,
        priority_score=zone.priority_score,
        global_rank=kwargs.get("global_rank", zone.global_rank),
        station_rank=kwargs.get("station_rank", zone.station_rank),
        peak_windows=zone.peak_windows,
    )


def _assign_station_ranks(zones: list[PriorityZone]) -> list[PriorityZone]:
    """Assign per-station ranks in-place on frozen copies."""
    # Group by police_station.
    from collections import defaultdict
    by_station: dict[str | None, list[tuple[int, PriorityZone]]] = defaultdict(list)
    for idx, z in enumerate(zones):
        by_station[z.police_station].append((idx, z))

    result = list(zones)
    for station, station_zones in by_station.items():
        # Sort within station by descending priority, ascending id on tie.
        station_zones.sort(key=lambda t: (-t[1].priority_score, t[1].hotspot_id))
        for rank, (orig_idx, z) in enumerate(station_zones, start=1):
            result[orig_idx] = _replace_zone(z, station_rank=rank)

    return result
