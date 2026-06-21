"""Spatial aggregation for the Parking Intelligence pipeline.

Implements Tasks 7.1 and 7.2:

* :class:`HotspotBuilder` with three public methods:
  - :meth:`assign_h3` — add an ``h3_cell`` column to a cleaned DataFrame using
    H3 hexagonal binning at a configurable resolution (Requirements 6.1–6.4).
  - :meth:`cluster_dbscan` — add a ``cluster_label`` column using DBSCAN on
    haversine-projected radian coordinates (Requirements 7.1–7.5).
  - :meth:`build_hotspots` — wire both methods together and emit a list of
    :class:`~parking_intelligence.models.Hotspot` records, excluding DBSCAN
    noise points, with the H3-cell fallback when no clusters are found
    (Requirements 6.5, 7.6–7.8, 18.2).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import h3
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .models import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    Hotspot,
)

if TYPE_CHECKING:
    pass

# Earth radius used to convert metres → radians for the haversine DBSCAN metric.
_EARTH_RADIUS_M: float = 6_371_000.0

# H3 resolution bounds (Requirements 6.2, 6.4).
_H3_RES_MIN: int = 0
_H3_RES_MAX: int = 15

# DBSCAN parameter bounds (Requirements 7.2, 7.3).
_EPS_M_MAX: float = 5_000.0


class HotspotBuilder:
    """Build spatial hotspot records from a cleaned violation DataFrame.

    All methods are stateless; the same instance may be reused across calls.
    """

    # ------------------------------------------------------------------
    # Task 7.1: H3 hexagonal binning
    # ------------------------------------------------------------------
    def assign_h3(
        self,
        df: pd.DataFrame,
        resolution: int = 9,
    ) -> pd.DataFrame:
        """Add an ``h3_cell`` column to *df* at the given H3 *resolution*.

        Each retained row is assigned exactly one H3 cell index computed from
        its ``latitude`` and ``longitude`` (Requirement 6.1).

        Parameters
        ----------
        df:
            Cleaned DataFrame with numeric ``latitude`` and ``longitude``
            columns (guaranteed valid by :func:`validate_geo`).
        resolution:
            H3 resolution integer in ``[0, 15]`` (Requirements 6.2, 6.3).

        Returns
        -------
        pd.DataFrame
            A copy of *df* with an additional ``h3_cell`` column (string H3
            index).

        Raises
        ------
        ValueError
            If *resolution* is not an integer within ``[0, 15]`` (Req 6.4).
        """
        if not isinstance(resolution, int) or isinstance(resolution, bool):
            raise ValueError(
                f"H3 resolution must be an integer in [0, 15], got {resolution!r}"
            )
        if resolution < _H3_RES_MIN or resolution > _H3_RES_MAX:
            raise ValueError(
                f"H3 resolution must be in [0, 15], got {resolution!r}"
            )

        out = df.copy()
        # h3.latlng_to_cell (h3-py ≥ 4.x API).
        out["h3_cell"] = [
            h3.latlng_to_cell(lat, lon, resolution)
            for lat, lon in zip(out["latitude"], out["longitude"])
        ]
        return out

    # ------------------------------------------------------------------
    # Task 7.2a: DBSCAN density clustering
    # ------------------------------------------------------------------
    def cluster_dbscan(
        self,
        df: pd.DataFrame,
        *,
        eps_m: float = 75.0,
        min_samples: int = 15,
    ) -> pd.DataFrame:
        """Add a ``cluster_label`` column using DBSCAN on haversine coordinates.

        DBSCAN runs on latitude/longitude expressed in **radians** with the
        haversine metric.  ``eps_m`` (meters) is converted to radians by
        dividing by the Earth radius (Requirement 7.1).

        Noise points receive label ``-1`` and are *not* suppressed here; callers
        decide how to handle them.

        Parameters
        ----------
        df:
            Cleaned DataFrame with numeric ``latitude`` and ``longitude``.
        eps_m:
            Neighbourhood radius in metres (``0 < eps_m ≤ 5000``, Req 7.2).
        min_samples:
            Minimum cluster density (integer ``≥ 1``, Requirement 7.2).

        Returns
        -------
        pd.DataFrame
            A copy of *df* with an additional ``cluster_label`` column.

        Raises
        ------
        ValueError
            If ``eps_m`` or ``min_samples`` are out of range (Req 7.3).
        """
        if not isinstance(eps_m, (int, float)) or isinstance(eps_m, bool):
            raise ValueError(
                f"eps_m must be a positive float ≤ 5000, got {eps_m!r}"
            )
        if eps_m <= 0 or eps_m > _EPS_M_MAX:
            raise ValueError(
                f"eps_m must be > 0 and ≤ 5000 m, got {eps_m!r}"
            )
        if not isinstance(min_samples, int) or isinstance(min_samples, bool):
            raise ValueError(
                f"min_samples must be an integer ≥ 1, got {min_samples!r}"
            )
        if min_samples < 1:
            raise ValueError(
                f"min_samples must be ≥ 1, got {min_samples!r}"
            )

        if df.empty:
            out = df.copy()
            out["cluster_label"] = pd.array([], dtype="int64")
            return out

        coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy())
        eps_rad = eps_m / _EARTH_RADIUS_M

        labels = DBSCAN(
            eps=eps_rad,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
            n_jobs=1,  # deterministic (Requirement 14.4)
        ).fit_predict(coords_rad)

        out = df.copy()
        out["cluster_label"] = labels
        return out

    # ------------------------------------------------------------------
    # Task 7.2b: Build Hotspot records
    # ------------------------------------------------------------------
    def build_hotspots(
        self,
        df: pd.DataFrame,
        *,
        h3_res: int = 9,
        eps_m: float = 75.0,
        min_samples: int = 15,
    ) -> list[Hotspot]:
        """Build :class:`Hotspot` records via DBSCAN (H3-cell fallback on no clusters).

        Pipeline:
        1. Assign H3 cells.
        2. Run DBSCAN.
        3. If DBSCAN produces at least one non-noise cluster, build hotspots
           from those clusters (Requirements 7.4–7.8).
        4. If DBSCAN labels every point as noise (empty result), fall back to
           H3-cell aggregation — each occupied H3 cell becomes one hotspot
           (Requirements 6.5, 18.2, 18.3).

        Parameters
        ----------
        df:
            Cleaned DataFrame from :func:`~parking_intelligence.ingest.Ingestor.load_and_clean`.
        h3_res:
            H3 resolution (validated by :meth:`assign_h3`).
        eps_m:
            DBSCAN neighbourhood radius in metres (validated by :meth:`cluster_dbscan`).
        min_samples:
            DBSCAN minimum density (validated by :meth:`cluster_dbscan`).

        Returns
        -------
        list[Hotspot]
            Non-noise hotspot records.  An empty list is returned (without
            raising) when DBSCAN finds only noise *and* the DataFrame is empty
            (Requirement 18.2).
        """
        if df.empty:
            return []

        # Step 1: H3 binning.
        df_h3 = self.assign_h3(df, resolution=h3_res)

        # Step 2: DBSCAN clustering.
        df_cl = self.cluster_dbscan(df_h3, eps_m=eps_m, min_samples=min_samples)

        # Step 3: Build hotspots from DBSCAN clusters (excluding noise).
        cluster_labels = df_cl["cluster_label"].unique()
        non_noise_labels = [lbl for lbl in cluster_labels if lbl != -1]

        if non_noise_labels:
            return self._hotspots_from_dbscan(df_cl, min_samples=min_samples)

        # Step 4: Fallback — H3-cell aggregation (Requirement 6.5).
        return self._hotspots_from_h3(df_h3)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------
    def _hotspots_from_dbscan(
        self,
        df: pd.DataFrame,
        *,
        min_samples: int,
    ) -> list[Hotspot]:
        """Build Hotspot records from non-noise DBSCAN clusters."""
        hotspots: list[Hotspot] = []
        for label, group in df.groupby("cluster_label"):
            if label == -1:
                continue  # DBSCAN noise — excluded (Requirement 7.5)

            member_count = len(group)
            # Requirement 7.4: member_count ≥ min_samples (guaranteed by DBSCAN).
            centroid_lat = float(group["latitude"].mean())
            centroid_lon = float(group["longitude"].mean())

            # Clamp centroid within bbox for safety (Requirement 7.7).
            centroid_lat = max(LAT_MIN, min(LAT_MAX, centroid_lat))
            centroid_lon = max(LON_MIN, min(LON_MAX, centroid_lon))

            modal_station = _modal_value(group.get("police_station"))
            modal_h3 = _modal_value(group.get("h3_cell"))

            hotspots.append(
                Hotspot(
                    hotspot_id=f"dbscan-{int(label)}",
                    centroid_lat=centroid_lat,
                    centroid_lon=centroid_lon,
                    h3_cell=modal_h3,
                    cluster_label=int(label),
                    member_count=member_count,
                    police_station=modal_station,
                    member_ids=group["id"].tolist() if "id" in group.columns else [],
                )
            )
        return hotspots

    def _hotspots_from_h3(self, df: pd.DataFrame) -> list[Hotspot]:
        """Build Hotspot records from H3-cell aggregation (DBSCAN fallback)."""
        hotspots: list[Hotspot] = []
        for h3_cell, group in df.groupby("h3_cell"):
            member_count = len(group)
            centroid_lat = float(group["latitude"].mean())
            centroid_lon = float(group["longitude"].mean())

            centroid_lat = max(LAT_MIN, min(LAT_MAX, centroid_lat))
            centroid_lon = max(LON_MIN, min(LON_MAX, centroid_lon))

            modal_station = _modal_value(group.get("police_station"))

            hotspots.append(
                Hotspot(
                    hotspot_id=f"h3-{h3_cell}",
                    centroid_lat=centroid_lat,
                    centroid_lon=centroid_lon,
                    h3_cell=str(h3_cell),
                    cluster_label=None,
                    member_count=member_count,
                    police_station=modal_station,
                    member_ids=group["id"].tolist() if "id" in group.columns else [],
                )
            )
        return hotspots


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _modal_value(series: pd.Series | None) -> str | None:
    """Return the modal (most frequent) non-null value from *series*.

    Tie-breaking rule: if two or more values share the maximum count, the one
    that sorts first in ascending lexicographic order is chosen (Requirement 7.8).

    Returns ``None`` if *series* is ``None``, empty, or all-null.
    """
    if series is None or series.empty:
        return None
    counts = Counter(
        v for v in series if v is not None and not _is_null_scalar(v)
    )
    if not counts:
        return None
    max_count = max(counts.values())
    # All candidates with the maximum count, sorted for determinism.
    candidates = sorted(k for k, v in counts.items() if v == max_count)
    return candidates[0]


def _is_null_scalar(value: object) -> bool:
    """Light null check (avoids importing from ingest to prevent circular deps)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "null", "nan", "none", "na"}:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
