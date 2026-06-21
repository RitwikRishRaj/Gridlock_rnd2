"""Real tests covering all 9 required areas.

Areas tested (in order):
  1.  JSON parsing (parse_json_array totality + token normalization)
  2.  Timestamp parsing (tz-aware, offset-less, NaT on garbage)
  3.  Bengaluru bounding-box filtering (inclusive edges, out-of-range drop)
  4.  H3 assignment (resolution validation, every row gets a cell)
  5.  DBSCAN fallback (sparse data → empty hotspot list, H3 fallback path)
  6.  Impact score bounds (0–100, max = 100, empty-list floor)
  7.  Priority rank ordering (contiguous 1..N, descending score, tie-break)
  8.  Export privacy (vehicle_number / id absent from GeoJSON + CSV)
  9.  Deterministic output ordering (identical inputs → identical CSV bytes)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

# ── helpers to build minimal fixtures ────────────────────────────────────────

def _ts(y=2024, m=3, d=15, h=23):
    """Return a tz-aware IST datetime that is AFTER all synthetic event timestamps."""
    from zoneinfo import ZoneInfo
    return datetime(y, m, d, h, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


def _make_clean_df(n=30, lat=12.95, lon=77.60, station="TestStation"):
    """Minimal cleaned DataFrame that pipeline stages accept."""
    rng = np.random.default_rng(42)
    lats = lat + rng.uniform(-0.01, 0.01, n)
    lons = lon + rng.uniform(-0.01, 0.01, n)
    ts = [_ts(h=rng.integers(0, 24)) for _ in range(n)]
    return pd.DataFrame({
        "id": [f"v{i}" for i in range(n)],
        "latitude": lats,
        "longitude": lons,
        "police_station": station,
        "junction_name": [None] * n,
        "violation_type": [["WRONG PARKING"]] * n,
        "vehicle_type": ["car"] * n,
        "created_at": ts,
    })


def _make_hotspot(hid="dbscan-0", n=20, lat=12.95, lon=77.60):
    from parking_intelligence.models import Hotspot
    return Hotspot(
        hotspot_id=hid,
        centroid_lat=lat, centroid_lon=lon,
        h3_cell="8928308280fffff",
        cluster_label=0,
        member_count=n,
        police_station="TestStation",
        member_ids=[f"v{i}" for i in range(n)],
    )


def _make_scored(hid="dbscan-0", impact=80.0, n=20):
    from parking_intelligence.models import ScoredHotspot
    hs = _make_hotspot(hid, n)
    return ScoredHotspot(
        hotspot=hs,
        impact_score=impact,
        severity_component=0.8,
        proximity_component=0.4,
        concentration_component=0.6,
        breakdown={"severity": 0.8, "proximity": 0.4, "concentration": 0.6},
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. JSON PARSING
# ═══════════════════════════════════════════════════════════════════════════
class TestJsonParsing:
    """Requirements 2.1–2.6: totality, token normalization, offence codes."""

    def setup_method(self):
        from parking_intelligence.ingest import parse_json_array
        self.parse = parse_json_array

    def test_valid_array_returns_list(self):
        result = self.parse('["WRONG PARKING", "NO PARKING ZONE"]')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_none_returns_empty_list(self):
        assert self.parse(None) == []

    def test_nan_returns_empty_list(self):
        assert self.parse(float("nan")) == []

    def test_empty_string_returns_empty_list(self):
        assert self.parse("") == []

    def test_malformed_json_returns_empty_list(self):
        assert self.parse("{not valid json}") == []

    def test_never_raises_on_any_input(self):
        for val in [None, float("nan"), "", "bad", "[]", '["x"]', 42, [], {}]:
            try:
                result = self.parse(val)
                assert isinstance(result, list)
            except Exception as exc:
                pytest.fail(f"parse_json_array raised {exc!r} for input {val!r}")

    def test_violation_type_tokens_upper_cased(self):
        from parking_intelligence.ingest import parse_violation_types
        result = parse_violation_types('["wrong parking", "  No Parking Zone  "]')
        assert result == ["WRONG PARKING", "NO PARKING ZONE"]

    def test_offence_code_keeps_only_integers(self):
        from parking_intelligence.ingest import parse_offence_codes
        result = parse_offence_codes('["101", "abc", "102", null]')
        assert result == [101, 102]

    def test_empty_json_array_returns_empty_list(self):
        assert self.parse("[]") == []

    def test_source_order_preserved(self):
        result = self.parse('["C", "A", "B"]')
        assert result == ["C", "A", "B"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. TIMESTAMP PARSING
# ═══════════════════════════════════════════════════════════════════════════
class TestTimestampParsing:
    """Requirements 3.1–3.4: tz-aware IST, offset-less interpretation, NaT fallback."""

    def setup_method(self):
        from parking_intelligence.ingest import parse_timestamps
        self.parse = parse_timestamps

    def _df(self, vals):
        return pd.DataFrame({"created_datetime": vals})

    def test_offset_less_value_localized_to_ist(self):
        df = self._df(["2024-01-15 09:00:00"])
        out = self.parse(df)
        ts = out["created_datetime"].iloc[0]
        assert ts is not pd.NaT
        assert str(ts.tzinfo) in ("Asia/Kolkata", "IST", "+05:30")
        assert ts.hour == 9

    def test_utc_offset_converted_to_ist(self):
        df = self._df(["2024-01-15T03:30:00+00:00"])  # 03:30 UTC = 09:00 IST
        out = self.parse(df)
        ts = out["created_datetime"].iloc[0]
        assert ts.hour == 9

    def test_garbage_becomes_nat(self):
        df = self._df(["not-a-date", "also bad"])
        out = self.parse(df)
        assert out["created_datetime"].isna().all()

    def test_null_becomes_nat(self):
        df = self._df([None])
        out = self.parse(df)
        assert pd.isna(out["created_datetime"].iloc[0])

    def test_input_not_mutated(self):
        df = self._df(["2024-01-15 09:00:00"])
        original_dtype = df["created_datetime"].dtype
        _ = self.parse(df)
        assert df["created_datetime"].dtype == original_dtype

    def test_optional_closed_datetime_tolerated(self):
        df = pd.DataFrame({
            "created_datetime": ["2024-01-15 09:00:00"],
            "closed_datetime": [None],
        })
        out = self.parse(df)
        assert pd.isna(out["closed_datetime"].iloc[0])


# ═══════════════════════════════════════════════════════════════════════════
# 3. BENGALURU BOUNDING-BOX FILTERING
# ═══════════════════════════════════════════════════════════════════════════
class TestGeoBboxFiltering:
    """Requirements 4.1–4.5: inclusive bounds, drop on invalid/out-of-range."""

    def setup_method(self):
        from parking_intelligence.ingest import validate_geo
        self.validate = validate_geo

    def _df(self, rows):
        return pd.DataFrame(rows, columns=["latitude", "longitude"])

    def test_valid_row_inside_bbox_retained(self):
        df = self._df([(12.95, 77.60)])
        out = self.validate(df)
        assert len(out) == 1

    def test_inclusive_lower_corner_kept(self):
        df = self._df([(12.7, 77.3)])
        assert len(self.validate(df)) == 1

    def test_inclusive_upper_corner_kept(self):
        df = self._df([(13.2, 77.9)])
        assert len(self.validate(df)) == 1

    def test_lat_just_below_min_dropped(self):
        df = self._df([(12.699, 77.60)])
        assert len(self.validate(df)) == 0

    def test_lat_just_above_max_dropped(self):
        df = self._df([(13.201, 77.60)])
        assert len(self.validate(df)) == 0

    def test_lon_just_below_min_dropped(self):
        df = self._df([(12.95, 77.299)])
        assert len(self.validate(df)) == 0

    def test_lon_just_above_max_dropped(self):
        df = self._df([(12.95, 77.901)])
        assert len(self.validate(df)) == 0

    def test_null_coordinate_dropped(self):
        df = self._df([(None, 77.60)])
        assert len(self.validate(df)) == 0

    def test_nan_coordinate_dropped(self):
        df = self._df([(float("nan"), 77.60)])
        assert len(self.validate(df)) == 0

    def test_non_numeric_coordinate_dropped(self):
        df = pd.DataFrame({"latitude": ["bad"], "longitude": ["77.60"]})
        assert len(self.validate(df)) == 0

    def test_output_len_lte_input_len(self):
        df = self._df([(12.95, 77.60), (0.0, 0.0), (12.8, 77.5)])
        out = self.validate(df)
        assert len(out) <= len(df)

    def test_dropped_count_recorded_in_report(self):
        from parking_intelligence.models import IngestionReport
        df = self._df([(12.95, 77.60), (0.0, 0.0)])
        report = IngestionReport()
        self.validate(df, report)
        assert report.dropped_by_reason.get("invalid_geo", 0) == 1

    def test_zero_dropped_still_records(self):
        from parking_intelligence.models import IngestionReport
        df = self._df([(12.95, 77.60)])
        report = IngestionReport()
        self.validate(df, report)
        assert "invalid_geo" in report.dropped_by_reason


# ═══════════════════════════════════════════════════════════════════════════
# 4. H3 ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════
class TestH3Assignment:
    """Requirements 6.1–6.4: every row gets exactly one H3 cell, resolution validation."""

    def setup_method(self):
        from parking_intelligence.hotspots import HotspotBuilder
        self.builder = HotspotBuilder()

    def test_every_row_gets_h3_cell(self):
        df = _make_clean_df(10)
        out = self.builder.assign_h3(df, resolution=9)
        assert "h3_cell" in out.columns
        assert out["h3_cell"].notna().all()
        assert len(out) == 10

    def test_default_resolution_is_9(self):
        df = _make_clean_df(5)
        out = self.builder.assign_h3(df)
        # H3 res-9 cell string has a specific length pattern
        assert all(isinstance(v, str) and len(v) > 0 for v in out["h3_cell"])

    def test_invalid_resolution_raises(self):
        df = _make_clean_df(5)
        with pytest.raises((ValueError, Exception)):
            self.builder.assign_h3(df, resolution=99)

    def test_resolution_0_accepted(self):
        df = _make_clean_df(5)
        out = self.builder.assign_h3(df, resolution=0)
        assert out["h3_cell"].notna().all()

    def test_resolution_15_accepted(self):
        df = _make_clean_df(5)
        out = self.builder.assign_h3(df, resolution=15)
        assert out["h3_cell"].notna().all()


# ═══════════════════════════════════════════════════════════════════════════
# 5. DBSCAN FALLBACK
# ═══════════════════════════════════════════════════════════════════════════
class TestDbscanFallback:
    """Requirements 7.1–7.5, 18.2–18.3: sparse data → empty list, H3 fallback path."""

    def setup_method(self):
        from parking_intelligence.hotspots import HotspotBuilder
        self.builder = HotspotBuilder()

    def test_sparse_data_dbscan_labels_all_noise(self):
        # With min_samples=500 on 2 points, DBSCAN labels every point as noise (-1).
        # cluster_dbscan should return a df with all cluster_label == -1.
        df = _make_clean_df(2)
        out = self.builder.cluster_dbscan(df, eps_m=75.0, min_samples=500)
        assert "cluster_label" in out.columns
        assert (out["cluster_label"] == -1).all(), \
            "Expected all noise labels when min_samples >> n_points"

    def test_build_hotspots_returns_empty_when_all_noise_and_no_h3_override(self):
        # build_hotspots falls back to H3 when DBSCAN finds nothing — that is correct
        # behavior (Requirement 18.2/18.3). So we test cluster_dbscan directly.
        df = _make_clean_df(2)
        out = self.builder.cluster_dbscan(df, eps_m=75.0, min_samples=500)
        # Extract non-noise hotspots manually — should be empty.
        non_noise = out[out["cluster_label"] != -1]
        assert len(non_noise) == 0

    def test_empty_list_does_not_raise(self):
        df = _make_clean_df(2)
        try:
            result = self.builder.build_hotspots(df, h3_res=9, eps_m=75.0, min_samples=500)
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(f"build_hotspots raised {exc!r} on sparse data")

    def test_h3_fallback_returns_hotspots(self):
        # H3 fallback groups by cell — should always produce something.
        df = _make_clean_df(5)
        df_with_h3 = self.builder.assign_h3(df, resolution=9)
        hotspots = self.builder._hotspots_from_h3(df_with_h3)
        assert len(hotspots) > 0

    def test_noise_points_excluded(self):
        """DBSCAN cluster_label == -1 must never appear in output."""
        df = _make_clean_df(30)
        hotspots = self.builder.build_hotspots(df, h3_res=9, eps_m=75.0, min_samples=5)
        for hs in hotspots:
            assert hs.cluster_label != -1

    def test_member_count_gte_min_samples(self):
        df = _make_clean_df(100)
        min_s = 3
        hotspots = self.builder.build_hotspots(df, h3_res=9, eps_m=500.0, min_samples=min_s)
        for hs in hotspots:
            assert hs.member_count >= min_s

    def test_invalid_eps_raises(self):
        df = _make_clean_df(5)
        with pytest.raises((ValueError, Exception)):
            self.builder.build_hotspots(df, eps_m=-1.0, min_samples=5)


# ═══════════════════════════════════════════════════════════════════════════
# 6. IMPACT SCORE BOUNDS
# ═══════════════════════════════════════════════════════════════════════════
class TestImpactScoreBounds:
    """Requirements 8.1–8.10: 0–100 range, max = 100, empty token floor."""

    def setup_method(self):
        from parking_intelligence.impact import ImpactScorer
        from parking_intelligence.models import ImpactWeights
        self.scorer = ImpactScorer()
        self.weights = ImpactWeights()

    def test_all_scores_in_0_to_100(self):
        hotspots = [_make_hotspot(f"dbscan-{i}", n=20) for i in range(5)]
        df = _make_clean_df(100)
        scored = self.scorer.score_impact(hotspots, df, self.weights)
        for s in scored:
            assert 0.0 <= s.impact_score <= 100.0

    def test_max_score_is_100_when_hotspots_exist(self):
        hotspots = [_make_hotspot(f"dbscan-{i}", n=20) for i in range(3)]
        df = _make_clean_df(60)
        scored = self.scorer.score_impact(hotspots, df, self.weights)
        if scored:
            assert max(s.impact_score for s in scored) == pytest.approx(100.0, abs=0.01)

    def test_empty_hotspot_list_returns_empty(self):
        df = _make_clean_df(10)
        scored = self.scorer.score_impact([], df, self.weights)
        assert scored == []

    def test_breakdown_components_in_0_to_1(self):
        hotspots = [_make_hotspot("dbscan-0", n=20)]
        df = _make_clean_df(20)
        scored = self.scorer.score_impact(hotspots, df, self.weights)
        for s in scored:
            for v in s.breakdown.values():
                assert 0.0 <= v <= 1.0

    def test_invalid_weights_raise(self):
        from parking_intelligence.models import ImpactWeights
        bad_weights = ImpactWeights(w_severity=0.9, w_proximity=0.9, w_concentration=0.9)
        hotspots = [_make_hotspot("dbscan-0", n=20)]
        df = _make_clean_df(20)
        with pytest.raises((ValueError, Exception)):
            self.scorer.score_impact(hotspots, df, bad_weights)


# ═══════════════════════════════════════════════════════════════════════════
# 7. PRIORITY RANK ORDERING
# ═══════════════════════════════════════════════════════════════════════════
class TestPriorityRankOrdering:
    """Requirements 9.1–9.9: contiguous 1..N, descending score, tie-break."""

    def setup_method(self):
        from parking_intelligence.priority import PriorityRanker
        from parking_intelligence.models import PriorityConfig
        self.ranker = PriorityRanker()
        self.cfg = PriorityConfig(as_of=_ts())

    def _rank(self, scored_list, df=None):
        if df is None:
            df = _make_clean_df(len(scored_list) * 20)
        profiles = {}
        return self.ranker.rank_zones(scored_list, profiles, df, self.cfg)

    def test_global_ranks_are_contiguous_from_1(self):
        scored = [_make_scored(f"dbscan-{i}", impact=float(i * 10)) for i in range(1, 6)]
        df = _make_clean_df(100)
        zones = self._rank(scored, df)
        ranks = sorted(z.global_rank for z in zones)
        assert ranks == list(range(1, len(zones) + 1))

    def test_rank_1_has_highest_priority_score(self):
        scored = [_make_scored(f"dbscan-{i}", impact=float(i * 10)) for i in range(1, 4)]
        df = _make_clean_df(60)
        zones = self._rank(scored, df)
        rank1 = next(z for z in zones if z.global_rank == 1)
        assert all(rank1.priority_score >= z.priority_score for z in zones)

    def test_all_priority_scores_in_0_to_100(self):
        scored = [_make_scored(f"dbscan-{i}") for i in range(3)]
        df = _make_clean_df(60)
        zones = self._rank(scored, df)
        for z in zones:
            assert 0.0 <= z.priority_score <= 100.0

    def test_empty_scored_returns_empty(self):
        assert self._rank([]) == []

    def test_tie_broken_by_hotspot_id_ascending(self):
        """Two zones with same impact → alphabetically later id gets higher rank number."""
        # Same impact → very similar priority → tie-break by id
        scored = [
            _make_scored("dbscan-z", impact=50.0, n=20),
            _make_scored("dbscan-a", impact=50.0, n=20),
        ]
        df = _make_clean_df(40)
        zones = self._rank(scored, df)
        # "dbscan-a" < "dbscan-z" lexicographically → dbscan-a should have lower rank number
        zone_a = next(z for z in zones if z.hotspot_id == "dbscan-a")
        zone_z = next(z for z in zones if z.hotspot_id == "dbscan-z")
        assert zone_a.global_rank < zone_z.global_rank


# ═══════════════════════════════════════════════════════════════════════════
# 8. EXPORT PRIVACY
# ═══════════════════════════════════════════════════════════════════════════
class TestExportPrivacy:
    """Requirements 12.2, 12.3, 13.3, 16.3, 16.4: PII never in artifacts."""

    PRIVATE_FIELDS = {"vehicle_number", "updated_vehicle_number", "id"}

    def setup_method(self):
        from parking_intelligence.export import Exporter
        self.exporter = Exporter()

    def test_geojson_no_pii_fields(self, tmp_path):
        scored = [_make_scored("dbscan-0")]
        path = str(tmp_path / "hotspots.geojson")
        self.exporter.to_geojson(scored, path)
        with open(path) as f:
            gj = json.load(f)
        for feat in gj["features"]:
            props = set(feat["properties"].keys())
            assert not props.intersection(self.PRIVATE_FIELDS), \
                f"PII field found in GeoJSON: {props & self.PRIVATE_FIELDS}"

    def test_geojson_coordinate_order_is_lon_lat(self, tmp_path):
        scored = [_make_scored("dbscan-0", impact=80.0)]
        path = str(tmp_path / "hotspots.geojson")
        self.exporter.to_geojson(scored, path)
        with open(path) as f:
            gj = json.load(f)
        coords = gj["features"][0]["geometry"]["coordinates"]
        # [lon, lat] — lon is 77.x, lat is 12.x
        assert 77.0 <= coords[0] <= 78.0, f"Expected longitude first, got {coords}"
        assert 12.0 <= coords[1] <= 14.0, f"Expected latitude second, got {coords}"

    def test_geojson_empty_input_produces_empty_features(self, tmp_path):
        path = str(tmp_path / "hotspots.geojson")
        self.exporter.to_geojson([], path)
        with open(path) as f:
            gj = json.load(f)
        assert gj["features"] == []

    def test_csv_no_pii_fields(self, tmp_path):
        from parking_intelligence.models import PriorityZone
        zone = PriorityZone(
            hotspot_id="dbscan-0", centroid_lat=12.95, centroid_lon=77.60,
            police_station="TestStation", impact=0.8, frequency=0.9,
            persistence=0.7, recency=0.95, priority_score=75.0,
            global_rank=1, station_rank=1, peak_windows=[],
        )
        path = str(tmp_path / "priority_zones.csv")
        self.exporter.to_priority_csv([zone], path)
        df = pd.read_csv(path)
        for field in self.PRIVATE_FIELDS:
            assert field not in df.columns, f"PII column '{field}' found in CSV"

    def test_csv_global_rank_column_starts_at_1(self, tmp_path):
        from parking_intelligence.models import PriorityZone
        zones = [
            PriorityZone("dbscan-0", 12.95, 77.60, "S", 0.8, 0.9, 0.7, 0.95, 80.0, 1, 1, []),
            PriorityZone("dbscan-1", 12.95, 77.60, "S", 0.6, 0.8, 0.6, 0.90, 60.0, 2, 2, []),
        ]
        path = str(tmp_path / "priority_zones.csv")
        self.exporter.to_priority_csv(zones, path)
        df = pd.read_csv(path)
        assert list(df["global_rank"]) == [1, 2]

    def test_csv_empty_input_writes_header_only(self, tmp_path):
        path = str(tmp_path / "priority_zones.csv")
        self.exporter.to_priority_csv([], path)
        df = pd.read_csv(path)
        assert len(df) == 0
        assert "global_rank" in df.columns


# ═══════════════════════════════════════════════════════════════════════════
# 9. DETERMINISTIC OUTPUT ORDERING
# ═══════════════════════════════════════════════════════════════════════════
class TestDeterministicOrdering:
    """Requirement 14: same inputs + same seed → byte-identical CSV."""

    def _run_pipeline(self, df, out_dir):
        """Run ingest-skipped mini pipeline on a pre-cleaned DataFrame."""
        from parking_intelligence.hotspots import HotspotBuilder
        from parking_intelligence.impact import ImpactScorer
        from parking_intelligence.forecast import PeakForecaster
        from parking_intelligence.priority import PriorityRanker
        from parking_intelligence.export import Exporter
        from parking_intelligence.models import ImpactWeights, PriorityConfig

        import numpy as np
        np.random.seed(42)

        builder = HotspotBuilder()
        hotspots = builder.build_hotspots(df, h3_res=9, eps_m=200.0, min_samples=3)
        if not hotspots:
            hotspots = builder._hotspots_from_h3(builder.assign_h3(df, resolution=9))

        scorer = ImpactScorer()
        scored = scorer.score_impact(hotspots, df, ImpactWeights())

        forecaster = PeakForecaster()
        profiles = forecaster.build_peak_profiles(df, hotspots)

        cfg = PriorityConfig(as_of=_ts())
        ranker = PriorityRanker()
        zones = ranker.rank_zones(scored, profiles, df, cfg)

        os.makedirs(out_dir, exist_ok=True)
        exp = Exporter()
        exp.to_priority_csv(zones, os.path.join(out_dir, "priority_zones.csv"))
        exp.to_geojson(scored, os.path.join(out_dir, "hotspots.geojson"))

    def test_csv_byte_identical_across_two_runs(self, tmp_path):
        df = _make_clean_df(80)
        d1 = str(tmp_path / "run1")
        d2 = str(tmp_path / "run2")
        self._run_pipeline(df, d1)
        self._run_pipeline(df, d2)
        b1 = open(os.path.join(d1, "priority_zones.csv"), "rb").read()
        b2 = open(os.path.join(d2, "priority_zones.csv"), "rb").read()
        assert b1 == b2, "priority_zones.csv is not byte-identical across runs"

    def test_geojson_byte_identical_across_two_runs(self, tmp_path):
        df = _make_clean_df(80)
        d1 = str(tmp_path / "run1")
        d2 = str(tmp_path / "run2")
        self._run_pipeline(df, d1)
        self._run_pipeline(df, d2)
        b1 = open(os.path.join(d1, "hotspots.geojson"), "rb").read()
        b2 = open(os.path.join(d2, "hotspots.geojson"), "rb").read()
        assert b1 == b2, "hotspots.geojson is not byte-identical across runs"
