"""Export layer for the Parking Intelligence pipeline.

Implements Task 12.1:

* :class:`Exporter` with:
  - :meth:`to_geojson` — write a valid GeoJSON ``FeatureCollection`` to disk,
    one ``Point`` Feature per :class:`~parking_intelligence.models.ScoredHotspot`
    (Requirements 12.1–12.5).
  - :meth:`to_priority_csv` — write ``priority_zones.csv`` ordered by ascending
    ``global_rank`` with the documented columns (Requirements 13.1–13.6).
  - :meth:`export_all` — call both writers and return their paths
    (Requirement 13.5).

Privacy rules (Requirements 16.3, 16.4):
  Neither export contains ``vehicle_number``, ``updated_vehicle_number``, ``id``,
  or any other per-individual identifier.  Only aggregate, score, and
  centroid-location data is emitted.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .ml_model import MLScores
    from .models import PriorityZone, ScoredHotspot

# Columns excluded from every exported artifact (Requirements 12.3, 13.3, 16.3).
_EXCLUDED_FIELDS: frozenset[str] = frozenset({
    "vehicle_number",
    "updated_vehicle_number",
    "id",
})

# Base columns always written to priority_zones.csv (Requirements 13.1).
_CSV_COLUMNS_BASE: list[str] = [
    "hotspot_id",
    "centroid_lat",
    "centroid_lon",
    "police_station",
    "impact",
    "frequency",
    "persistence",
    "recency",
    "priority_score",
    "global_rank",
    "station_rank",
]
# ML columns appended when anomaly / risk data is available.
_CSV_COLUMNS_ML: list[str] = ["risk_tier", "anomaly_score"]


class Exporter:
    """Serialize pipeline results into portable, standards-compliant artifacts."""

    # ------------------------------------------------------------------
    # Task 12.1a: GeoJSON export
    # ------------------------------------------------------------------
    def to_geojson(
        self,
        hotspots: list["ScoredHotspot"],
        path: str,
        *,
        ml_scores: "MLScores | None" = None,
    ) -> str:
        """Write *hotspots* as a GeoJSON ``FeatureCollection`` to *path``.

        Parameters
        ----------
        hotspots:
            Scored hotspot records.  An empty list produces an empty features
            array (Requirement 12.4).
        path:
            Absolute or relative file path to write.  Parent directory must
            exist.  The write is atomic (temp-then-rename) so an error never
            leaves a partial file at *path* (Requirement 12.5).
        ml_scores:
            Optional :class:`~parking_intelligence.ml_model.MLScores` whose
            ``risk_tiers`` and ``anomaly_scores`` are added to Feature
            properties when supplied.

        Returns
        -------
        str
            The resolved absolute path of the written file.

        Raises
        ------
        OSError
            If the parent directory does not exist or is not writable.
        """
        features = []
        for sh in hotspots:
            hs = sh.hotspot
            # GeoJSON uses [longitude, latitude] order (Requirement 12.1).
            geometry = {
                "type": "Point",
                "coordinates": [hs.centroid_lon, hs.centroid_lat],
            }
            properties = {
                "hotspot_id": hs.hotspot_id,
                "member_count": hs.member_count,
                "police_station": hs.police_station,
                "Congestion_Impact_Score": sh.impact_score,
                "severity_component": sh.severity_component,
                "proximity_component": sh.proximity_component,
                "concentration_component": sh.concentration_component,
                "h3_cell": hs.h3_cell,
                "cluster_label": hs.cluster_label,
            }
            # Attach ML scores when available.
            if ml_scores is not None:
                properties["risk_tier"] = ml_scores.risk_tiers.get(hs.hotspot_id, "UNKNOWN")
                properties["anomaly_score"] = ml_scores.anomaly_scores.get(hs.hotspot_id, 0.5)
            # Strip any excluded fields (belt-and-suspenders, Req 12.3).
            properties = {k: v for k, v in properties.items()
                          if k not in _EXCLUDED_FIELDS}

            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }

        _atomic_write_json(geojson, path)
        return os.path.abspath(path)

    # ------------------------------------------------------------------
    # Task 12.1b: Priority CSV export
    # ------------------------------------------------------------------
    def to_priority_csv(
        self,
        zones: list["PriorityZone"],
        path: str,
        *,
        ml_scores: "MLScores | None" = None,
    ) -> str:
        """Write *zones* as ``priority_zones.csv`` ordered by ``global_rank``.

        Parameters
        ----------
        zones:
            Priority zone records.  An empty list produces a header-only file
            (Requirement 13.4).
        path:
            File path to write.
        ml_scores:
            Optional ML scores to include as extra columns.

        Returns
        -------
        str
            Resolved absolute path of the written file.

        Raises
        ------
        OSError
            If *path* cannot be written.
        """
        rows = []
        for z in zones:
            row = {
                "hotspot_id": z.hotspot_id,
                "centroid_lat": z.centroid_lat,
                "centroid_lon": z.centroid_lon,
                "police_station": z.police_station,
                "impact": z.impact,
                "frequency": z.frequency,
                "persistence": z.persistence,
                "recency": z.recency,
                "priority_score": z.priority_score,
                "global_rank": z.global_rank,
                "station_rank": z.station_rank,
            }
            # Attach ML scores when available.
            if ml_scores is not None:
                row["risk_tier"] = ml_scores.risk_tiers.get(z.hotspot_id, "UNKNOWN")
                row["anomaly_score"] = ml_scores.anomaly_scores.get(z.hotspot_id, 0.5)
            rows.append(row)

        cols = _CSV_COLUMNS_BASE + (_CSV_COLUMNS_ML if ml_scores is not None else [])
        df = pd.DataFrame(rows, columns=cols)
        # Order by ascending global_rank (Requirement 13.2).
        df = df.sort_values("global_rank", ascending=True)

        _atomic_write_csv(df, path)
        return os.path.abspath(path)

    # ------------------------------------------------------------------
    # Task 12.1c: Export all artifacts
    # ------------------------------------------------------------------
    def export_all(
        self,
        hotspots: list["ScoredHotspot"],
        zones: list["PriorityZone"],
        out_dir: str,
        *,
        ml_scores: "MLScores | None" = None,
    ) -> dict[str, str]:
        """Write both artifacts to *out_dir* and return their paths.

        Parameters
        ----------
        hotspots:
            Scored hotspot list for ``hotspots.geojson``.
        zones:
            Priority zone list for ``priority_zones.csv``.
        out_dir:
            Directory that will contain both artifacts.  Created if absent.
        ml_scores:
            Optional ML scores to embed in both artifacts.

        Returns
        -------
        dict[str, str]
            ``{"geojson": "<abs-path>", "csv": "<abs-path>"}`` (Requirement 13.5).
        """
        os.makedirs(out_dir, exist_ok=True)
        geojson_path = self.to_geojson(
            hotspots, os.path.join(out_dir, "hotspots.geojson"), ml_scores=ml_scores
        )
        csv_path = self.to_priority_csv(
            zones, os.path.join(out_dir, "priority_zones.csv"), ml_scores=ml_scores
        )
        return {"geojson": geojson_path, "csv": csv_path}


# ---------------------------------------------------------------------------
# Atomic write helpers (write-to-temp-then-rename, Requirements 12.5, 13.6)
# ---------------------------------------------------------------------------
def _atomic_write_json(data: dict, path: str) -> None:
    """Write *data* as JSON to *path* atomically."""
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    _assert_writable_dir(dir_name, path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write *df* as CSV to *path* atomically."""
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    _assert_writable_dir(dir_name, path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _assert_writable_dir(dir_name: str, target_path: str) -> None:
    """Raise ``OSError`` early if *dir_name* does not exist or is not writable."""
    if not os.path.isdir(dir_name):
        raise OSError(
            f"Cannot write {target_path!r}: parent directory {dir_name!r} does not exist"
        )
    if not os.access(dir_name, os.W_OK):
        raise OSError(
            f"Cannot write {target_path!r}: parent directory {dir_name!r} is not writable"
        )


def _json_default(obj: object) -> object:
    """Fallback JSON serialiser for numpy scalars etc."""
    try:
        import numpy as np  # noqa: PLC0415
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    return str(obj)
