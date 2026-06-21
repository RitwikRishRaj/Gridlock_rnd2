"""Temporal peak prediction for the Parking Intelligence pipeline.

Implements Tasks 9.1 and 9.2:

* :class:`PeakForecaster` with:
  - :meth:`build_peak_profiles` — build a normalised ``7×24`` hour-of-day ×
    day-of-week intensity matrix per hotspot, with Laplace smoothing to avoid
    zero cells (Requirements 11.1, 11.2).
  - :meth:`predict_next_peaks` — identify cells ≥ 0.5, coalesce contiguous
    hours within the same day-of-week, and return the top-k peak windows
    ordered by descending intensity (Requirements 11.3–11.8).
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .models import PeakProfile, PeakWindow

if TYPE_CHECKING:
    from .models import Hotspot

# Default top-k when none is provided (Requirement 11.5).
_DEFAULT_TOP_K: int = 3

# Peak threshold: cells with normalised value ≥ this are "peaks" (Req 11.6).
_PEAK_THRESHOLD: float = 0.5

# Laplace smoothing alpha; keeps every cell > 0 (Requirement 11.2).
_LAPLACE_ALPHA: float = 0.5


class PeakForecaster:
    """Build temporal peak profiles and predict peak windows for hotspots."""

    # ------------------------------------------------------------------
    # Task 9.1: Build peak profiles
    # ------------------------------------------------------------------
    def build_peak_profiles(
        self,
        df: pd.DataFrame,
        hotspots: list["Hotspot"],
    ) -> dict[str, PeakProfile]:
        """Build a :class:`PeakProfile` for each hotspot.

        Parameters
        ----------
        df:
            Cleaned DataFrame produced by the ingestor.  Must contain a
            ``created_at`` (or ``created_datetime``) column with tz-aware
            timestamps.
        hotspots:
            Non-noise hotspot records from
            :class:`~parking_intelligence.hotspots.HotspotBuilder`.

        Returns
        -------
        dict[str, PeakProfile]
            Mapping of ``hotspot_id → PeakProfile``.  Hotspots with no
            member events receive a uniform (smoothed) profile.
        """
        # Resolve the timestamp column.
        ts_col = "created_at" if "created_at" in df.columns else "created_datetime"

        profiles: dict[str, PeakProfile] = {}
        for hs in hotspots:
            if hs.member_ids and "id" in df.columns:
                events = df[df["id"].isin(hs.member_ids)]
            else:
                events = df.iloc[0:0]  # empty

            profile = self._build_single_profile(hs.hotspot_id, events, ts_col)
            profiles[hs.hotspot_id] = profile

        return profiles

    def _build_single_profile(
        self,
        hotspot_id: str,
        events: pd.DataFrame,
        ts_col: str,
    ) -> PeakProfile:
        """Build a single ``PeakProfile`` from *events*."""
        # 7 days × 24 hours raw count matrix.
        M = np.zeros((7, 24), dtype=float)

        if not events.empty and ts_col in events.columns:
            ts = pd.to_datetime(events[ts_col], errors="coerce")
            for t in ts:
                if pd.isna(t):
                    continue
                dow = t.weekday()   # 0=Monday … 6=Sunday
                hour = t.hour       # 0..23
                M[dow, hour] += 1

        # Laplace smoothing so every cell > 0 (Requirement 11.2).
        M += _LAPLACE_ALPHA

        # Normalise so maximum value is exactly 1.0 (Requirement 11.1).
        max_val = M.max()
        if max_val > 0:
            M = M / max_val

        # Clip to [0, 1] for safety.
        M = np.clip(M, 0.0, 1.0)

        return PeakProfile(
            hotspot_id=hotspot_id,
            hour_dow_matrix=M.tolist(),
            total_events=len(events),
        )

    # ------------------------------------------------------------------
    # Task 9.2: Predict next peak windows
    # ------------------------------------------------------------------
    def predict_next_peaks(
        self,
        profile: PeakProfile,
        *,
        top_k: int = _DEFAULT_TOP_K,
    ) -> list[PeakWindow]:
        """Return at most *top_k* peak windows from *profile*.

        Parameters
        ----------
        profile:
            A :class:`PeakProfile` produced by :meth:`build_peak_profiles`.
        top_k:
            Maximum number of peak windows to return.  Must be an integer in
            ``[1, 168]`` (Requirements 11.3, 11.4, 11.5).

        Returns
        -------
        list[PeakWindow]
            At most *top_k* windows ordered by descending intensity.  Each
            window specifies a ``day_of_week ∈ [0, 6]`` and hours satisfying
            ``0 ≤ start_hour ≤ end_hour ≤ 23`` (Requirement 11.7).

        Raises
        ------
        ValueError
            If *top_k* is non-integer, < 1, or > 168 (Requirement 11.4).
        """
        # Validate top_k (Requirements 11.3, 11.4).
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError(
                f"top_k must be an integer in [1, 168], got {top_k!r}"
            )
        if top_k < 1 or top_k > 168:
            raise ValueError(
                f"top_k must be in [1, 168], got {top_k!r}"
            )

        matrix = profile.hour_dow_matrix  # list[list[float]], 7×24

        # Collect all (dow, hour, intensity) cells and sort by descending intensity.
        cells = [
            (dow, hour, matrix[dow][hour])
            for dow in range(7)
            for hour in range(24)
        ]
        cells.sort(key=lambda c: c[2], reverse=True)

        # Identify peak cells: normalised value ≥ threshold (Requirement 11.6).
        peak_cells = [(dow, hour, intensity) for dow, hour, intensity in cells
                      if intensity >= _PEAK_THRESHOLD]

        # Coalesce contiguous hours within the same day-of-week (Requirement 11.8).
        windows = _coalesce_peak_windows(peak_cells)

        # Sort windows by descending intensity and take at most top_k.
        windows.sort(key=lambda w: w.expected_intensity, reverse=True)
        return windows[:top_k]


# ---------------------------------------------------------------------------
# Helper: coalesce contiguous hours into PeakWindow records
# ---------------------------------------------------------------------------
def _coalesce_peak_windows(
    peak_cells: list[tuple[int, int, float]],
) -> list[PeakWindow]:
    """Merge contiguous hours in the same day-of-week into single PeakWindows.

    *peak_cells* is a list of ``(dow, hour, intensity)`` tuples.  The function
    groups by ``dow``, sorts hours within each group, then merges runs of
    consecutive hours.  The window intensity is the *maximum* cell intensity in
    the merged run (most representative of the peak).
    """
    if not peak_cells:
        return []

    # Group by day-of-week.
    by_dow: dict[int, list[tuple[int, float]]] = {}
    for dow, hour, intensity in peak_cells:
        by_dow.setdefault(dow, []).append((hour, intensity))

    windows: list[PeakWindow] = []
    for dow in sorted(by_dow):
        hour_intensity = sorted(by_dow[dow], key=lambda x: x[0])  # sort by hour

        # Merge consecutive hours.
        run_start: int = hour_intensity[0][0]
        run_end: int = hour_intensity[0][0]
        run_max: float = hour_intensity[0][1]

        for i in range(1, len(hour_intensity)):
            h, intensity = hour_intensity[i]
            if h == run_end + 1:
                # Extend the current run.
                run_end = h
                run_max = max(run_max, intensity)
            else:
                # Emit the completed run.
                windows.append(PeakWindow(
                    day_of_week=dow,
                    start_hour=run_start,
                    end_hour=run_end,
                    expected_intensity=run_max,
                ))
                run_start = h
                run_end = h
                run_max = intensity

        # Emit the final run for this day.
        windows.append(PeakWindow(
            day_of_week=dow,
            start_hour=run_start,
            end_hour=run_end,
            expected_intensity=run_max,
        ))

    return windows
