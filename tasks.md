# Implementation Plan: Parking Intelligence

## Overview

This plan implements the offline Parking Intelligence pipeline in Python using a simple, flat
package layout (`parking_intelligence/` with one module per pipeline stage). Tasks are ordered
by data-flow dependency: data models first, then ingestion, spatial aggregation, impact scoring,
peak forecasting, priority ranking, export, the orchestrator that wires everything together, and
finally the Streamlit dashboard. Each stage is implemented as a pure, independently testable unit
followed by its unit and property-based tests. Property-based tests (using `hypothesis`) encode
the design's Correctness Properties 1–10 and live next to the code they validate so regressions
surface early.

Target file tree (flat, no deep nesting):

```
parking_intelligence/
  __init__.py
  models.py
  ingest.py
  hotspots.py
  impact.py
  priority.py
  forecast.py
  export.py
  pipeline.py
  dashboard.py
tests/
requirements.txt
README.md
```

## Tasks

- [x] 1. Set up project scaffolding and dependencies
  - Create the flat `parking_intelligence/` package with an empty `__init__.py` and a `tests/` directory
  - Create `requirements.txt` pinning `pandas`, `numpy`, `h3`, `scikit-learn`, `shapely`/`geojson`, `streamlit`, `pydeck`, `pytest`, `hypothesis`
  - Add a `pytest.ini`/`conftest.py` so `pytest` and `hypothesis` run against the package; add a short `README.md` describing offline usage
  - _Requirements: 16.1, 16.2_

- [x] 2. Define core data models and config objects
  - [x] 2.1 Implement all dataclasses in `models.py`
    - Define `ViolationRecord`, `Hotspot`, `ScoredHotspot`, `PriorityZone`, `PeakProfile`, `PeakWindow`
    - Define config objects `ImpactWeights` (defaults summing to 1.0) and `PriorityConfig` (`as_of`, `recency_halflife_days`), plus the `Bengaluru_Bbox` constants `lat ∈ [12.7, 13.2]`, `lon ∈ [77.3, 77.9]`
    - Define a `TimeWindow` helper and an `IngestionReport` structure (dropped-row counts by reason)
    - _Requirements: 8.5, 10.1, 4.1_
  - [ ]* 2.2 Write unit tests for model construction and config validation
    - Assert frozen dataclasses, default `ImpactWeights` sum to 1.0, and field types match the design
    - _Requirements: 8.5, 10.1_

- [x] 3. Implement JSON-array and timestamp parsing primitives
  - [x] 3.1 Implement `parse_json_array` and timestamp parsing in `ingest.py`
    - `parse_json_array(raw)` returns a list for every input (null/NaN/empty/malformed → `[]`), never raises; trims and upper-cases `violation_type` tokens; keeps only integer-convertible `offence_code` tokens (defaulting to `[]`)
    - `parse_timestamps(df)` converts `created_datetime`/`closed_datetime` to tz-aware Asia/Kolkata datetimes; interprets offset-less values as Asia/Kolkata; sets unparseable optional columns to null
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.3, 3.4_
  - [ ]* 3.2 Write property test for JSON-array parsing totality
    - **Property 2: JSON parse totality**
    - **Validates: Requirements 2.2, 2.3**
  - [ ]* 3.3 Write unit tests for token normalization and timestamp parsing
    - Cover upper-casing/trim of `violation_type`, integer filtering of `offence_code`, empty-array inputs, tz-aware conversion, and null optional timestamps
    - _Requirements: 2.1, 2.4, 2.5, 2.6, 3.1, 3.3, 3.4_

- [x] 4. Implement geographic validation and categorical normalization
  - [x] 4.1 Implement `validate_geo` and `normalize_categoricals` in `ingest.py`
    - `validate_geo(df)` retains rows only when lat/lon are numeric, non-null, non-NaN and inside the inclusive Bengaluru bbox; drops all others; guarantees `len(output) ≤ len(input)`
    - Normalize `vehicle_type` (trim + lowercase), `police_station` (trim + collapse internal whitespace), and `violation_type` tokens (trim + uppercase); set null/empty values to empty without raising
    - Record dropped-coordinate counts into the ingestion report
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4_
  - [ ]* 4.2 Write property test for geo validity
    - **Property 1: Geo validity**
    - **Validates: Requirements 4.1, 4.4**
  - [ ]* 4.3 Write unit tests for bbox edges and categorical normalization
    - Test coordinates just inside/outside each bound, NaN/non-numeric coordinates, and whitespace/case normalization of categoricals
    - _Requirements: 4.2, 4.3, 5.1, 5.2, 5.3, 5.4_

- [x] 5. Implement chunked CSV ingestion orchestration
  - [x] 5.1 Implement `Ingestor.load_and_clean` in `ingest.py`
    - Validate `chunksize` is a positive integer ≥ 1 (default 50,000) before reading; reject invalid chunk size and missing path/required columns with errors and no partial DataFrame
    - Stream the CSV chunk-by-chunk (one chunk in memory at a time), applying parse → timestamp → geo → normalize per chunk, dropping rows with unparseable `created_datetime` and recording drops
    - Concatenate cleaned chunks, drop duplicate `id` rows keeping first occurrence (idempotent re-ingest), and return the ingestion report alongside the DataFrame
    - Guard per-row cleaning failures so a single bad row is dropped and logged without aborting the chunk
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 3.2, 18.1, 18.5, 18.6_
  - [ ]* 5.2 Write unit tests for ingestion behavior and error handling
    - Test chunked reads produce the same result as a single pass, idempotent re-ingestion (no duplicate `id`), default vs invalid chunk size, missing-column/missing-path rejection, and dropped-row reporting
    - _Requirements: 1.1, 1.5, 1.6, 1.7, 1.8, 3.2, 18.5, 18.6_

- [x] 6. Checkpoint - ingestion stage
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement H3 binning and DBSCAN hotspot construction
  - [x] 7.1 Implement `HotspotBuilder.assign_h3` in `hotspots.py`
    - Assign exactly one H3 cell index per retained row at the configured resolution (default 9); validate resolution is an integer in `[0, 15]` and reject out-of-range values without modifying the input DataFrame
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [x] 7.2 Implement `HotspotBuilder.cluster_dbscan` and `build_hotspots` in `hotspots.py`
    - Run DBSCAN on haversine-projected radian coordinates using `eps_m` (meters) and integer `min_samples`; validate `0 < eps_m ≤ 5000` and `min_samples ≥ 1`, rejecting invalid parameters with no hotspots emitted
    - Build `Hotspot` records: exclude noise (`label == -1`), enforce `member_count ≥ min_samples`, assign each violation to at most one hotspot, set centroid to mean member coordinates (within bbox), set `police_station` to the modal station with ascending-lexicographic tie-break
    - Return an empty hotspot list (no error) when DBSCAN finds only noise; provide H3-cell aggregation grouping as the documented fallback source
    - _Requirements: 6.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 18.2_
  - [ ]* 7.3 Write property test for hotspot minimality
    - **Property 3: Hotspot minimality**
    - **Validates: Requirements 7.4, 7.5**
  - [ ]* 7.4 Write property test for hotspot partition
    - **Property 4: Partition**
    - **Validates: Requirements 7.6**
  - [ ]* 7.5 Write unit tests for resolution/parameter validation and centroid/station rules
    - Test out-of-range H3 resolution and DBSCAN parameter rejection, modal-station lexicographic tie-break, centroid bounds, and the all-noise empty-list case
    - _Requirements: 6.2, 6.4, 7.2, 7.3, 7.7, 7.8, 18.2_

- [x] 8. Implement congestion impact scoring
  - [x] 8.1 Implement impact component functions in `impact.py`
    - `severity_weight` returns the max token weight in `[0, 1]`, applying the configured severity floor for empty token lists; `proximity_factor` returns the junction/road-crossing fraction in `[0, 1]`; `temporal_concentration` returns `1 − normalized hour-of-day entropy` in `[0, 1]` (single bin → 1.0, uniform → 0.0)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 18.1_
  - [x] 8.2 Implement `ImpactScorer.score_impact` in `impact.py`
    - Validate impact weights each lie in `[0, 1]` and sum to 1.0 within ±0.001, rejecting invalid weights with no scores; combine components into a raw score and normalize so the max-raw hotspot scores exactly 100 (all-zero raw → all scores 0); attach component breakdown that reconstructs the raw score within ±0.001
    - Keep scores within `[0, 100]`; ensure the augmented-score path stays in `[0, 100]` when a live-traffic signal is supplied
    - _Requirements: 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 17.1_
  - [ ]* 8.3 Write property test for impact bounds
    - **Property 5: Impact bounds**
    - **Validates: Requirements 8.7, 8.8**
  - [ ]* 8.4 Write unit tests for components, weight validation, and breakdown reconstruction
    - Test severity floor on empty tokens, concentration extremes, weight-sum rejection, all-zero normalization, and breakdown reconstruction tolerance
    - _Requirements: 8.1, 8.2, 8.4, 8.6, 8.9, 8.10_

- [x] 9. Implement temporal peak forecasting
  - [x] 9.1 Implement `PeakForecaster.build_peak_profiles` in `forecast.py`
    - Build a 7×24 hour-of-day × day-of-week matrix per hotspot, apply Laplace smoothing so every cell is `> 0` and `≤ 1` before normalization, then normalize so values lie in `[0, 1]` with maximum exactly 1.0
    - _Requirements: 11.1, 11.2_
  - [x] 9.2 Implement `PeakForecaster.predict_next_peaks` in `forecast.py`
    - Validate `top_k` is an integer in `[1, 168]` (default 3), rejecting invalid values with no windows; identify peak cells with normalized value `≥ 0.5`, coalesce contiguous hours within the same day-of-week, and return at most `top_k` windows ordered by descending intensity with `0 ≤ start_hour ≤ end_hour ≤ 23` and `day_of_week ∈ [0, 6]`
    - _Requirements: 11.3, 11.4, 11.5, 11.6, 11.7, 11.8_
  - [ ]* 9.3 Write property test for peak profile normalization
    - **Property 9: Peak profile normalization**
    - **Validates: Requirements 11.1, 11.2**
  - [ ]* 9.4 Write unit tests for peak prediction
    - Test `top_k` validation, threshold/coalescing of contiguous hours on synthetic data with a known injected peak, and window ordering/bounds
    - _Requirements: 11.3, 11.4, 11.6, 11.7, 11.8_

- [x] 10. Checkpoint - scoring and forecasting stages
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement recency decay and priority ranking
  - [x] 11.1 Implement recency and factor functions in `priority.py`
    - `recency_decay` validates `recency_halflife_days > 0` and `last_event ≤ as_of` (rejecting invalid config without computing, leaving input unchanged); returns a value in `(0, 1]`, exactly 1.0 at zero elapsed days, exactly 0.5 at one half-life, non-increasing as elapsed days grow
    - Implement raw `frequency_score` (member count) and `persistence_score` (distinct active days)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  - [x] 11.2 Implement `PriorityRanker.rank_zones` in `priority.py`
    - Min-max normalize frequency and persistence to `[0, 1]` (collapsed range → 1.0 for all); compute `priority_score` as 100 × weighted geometric mean of normalized impact/frequency/persistence/recency with weights 0.4/0.25/0.2/0.15; force score to 0 when any factor is 0; keep scores in `[0, 100]`
    - Assign contiguous global ranks `1..N` by descending score and contiguous per-station ranks `1..M`, breaking ties by ascending hotspot identifier; attach forecaster peak windows (empty set when none)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9_
  - [ ]* 11.3 Write property test for recency monotonicity
    - **Property 8: Recency monotonicity**
    - **Validates: Requirements 10.3, 10.4, 10.6**
  - [ ]* 11.4 Write property test for priority bounds and ordering
    - **Property 6: Priority bounds and ordering**
    - **Validates: Requirements 9.4, 9.6**
  - [ ]* 11.5 Write property test for zero-factor annihilation
    - **Property 7: Zero-factor annihilation**
    - **Validates: Requirements 9.5**
  - [ ]* 11.6 Write unit tests for normalization edges and ranking tie-breaks
    - Test collapsed-range normalization, half-life and zero-elapsed recency values, recency config rejection, global/station rank contiguity, and ascending-id tie-breaks
    - _Requirements: 9.1, 9.2, 9.7, 9.9, 10.2, 10.5_

- [x] 12. Implement export layer
  - [x] 12.1 Implement GeoJSON and CSV export in `export.py`
    - `to_geojson` writes a FeatureCollection with one Point Feature per scored hotspot (lon-then-lat, within bbox), populating aggregate properties (`member_count`, `police_station`, `Congestion_Impact_Score`) and excluding `vehicle_number`, `updated_vehicle_number`, and `id`; empty input → empty features array
    - `to_priority_csv` writes a header plus one row per zone ordered by ascending `global_rank` with the documented columns, excluding all per-individual identifiers; empty input → header only
    - `export_all` returns both file paths; on unwritable paths raise an error and leave no partial file (write-to-temp-then-rename)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 16.3, 16.4_
  - [ ]* 12.2 Write unit tests for export artifacts and privacy exclusions
    - Validate GeoJSON structure and coordinate order, CSV columns/rank ordering, empty-input cases, identifier exclusion, and atomic failure on unwritable paths
    - _Requirements: 12.1, 12.2, 12.4, 13.1, 13.4, 16.3, 16.4_

- [x] 13. Implement pipeline orchestrator
  - [x] 13.1 Implement `pipeline.run` in `pipeline.py`
    - Wire ingest → hotspots → impact → forecast → priority → export, threading config and applying a configured default seed when clustering/random seeds are not provided (no time/entropy seeds)
    - Fall back to H3-cell aggregation as the hotspot source when `build_hotspots` returns empty, completing all remaining stages without aborting; return a `PipelineResult` with artifact paths and the ingestion report
    - Ensure the full run uses only local computation and makes no outbound network calls; isolate optional live-traffic augmentation behind a disabled-by-default, timeout-bounded connector that falls back to offline scoring on failure
    - _Requirements: 14.4, 18.3, 16.1, 16.2, 17.2, 17.3, 17.4_
  - [ ]* 13.2 Write property test for pipeline determinism
    - **Property 10: Determinism**
    - **Validates: Requirements 14.1, 14.2, 14.3**
  - [ ]* 13.3 Write integration tests for end-to-end pipeline
    - Run `pipeline.run` on a small fixture CSV (~1k rows); assert valid GeoJSON, CSV with expected columns and contiguous ranks, byte-identical artifacts across two runs with a fixed seed, and the H3 fallback path
    - _Requirements: 14.1, 14.2, 18.3, 16.2_

- [x] 14. Checkpoint - full batch pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Implement Streamlit dashboard
  - [x] 15.1 Implement artifact loading and core views in `dashboard.py`
    - Load pre-computed `priority_zones.csv` and `hotspots.geojson` (cached, no pipeline recompute), render a violation heatmap and a top-20 priority-zone table ordered by ascending `global_rank`, and surface the ingestion dropped-row count
    - Display an error message and stay running if a required artifact is missing/unreadable/malformed
    - _Requirements: 15.1, 15.2, 15.6, 15.7_
  - [x] 15.2 Implement time-slider filtering and per-station drilldown in `dashboard.py`
    - Filter hotspot events by selected hour `[0, 23]`/day-of-week `[0, 6]` and re-render the heatmap (empty heatmap + message when no events match); render a per-station drilldown with zones, peak hours, and up to 10 sample violations excluding `vehicle_number`/`updated_vehicle_number`
    - Show the H3-cell heatmap with a warning recommending relaxed `eps_m`/`min_samples` when no density clusters were found
    - _Requirements: 15.3, 15.4, 15.5, 18.4, 16.3, 16.4_
  - [ ]* 15.3 Write unit tests for dashboard data helpers
    - Test artifact-loading error handling, time-window filtering logic, top-20 ordering, and drilldown identifier exclusion using pure helper functions extracted from the UI layer
    - _Requirements: 15.2, 15.3, 15.4, 15.5, 16.4_

- [x] 16. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP.
- Each task references specific requirement sub-clauses for traceability.
- Property tests (Properties 1–10) validate universal correctness invariants; unit tests cover
  specific examples and edge cases; integration tests cover end-to-end determinism and artifacts.
- The file tree is intentionally flat: one module per pipeline stage under `parking_intelligence/`,
  with all tests under `tests/`.
- Checkpoints provide incremental validation points between major stages.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "4.1"] },
    { "id": 4, "tasks": ["4.2", "4.3", "5.1"] },
    { "id": 5, "tasks": ["5.2", "7.1"] },
    { "id": 6, "tasks": ["7.2"] },
    { "id": 7, "tasks": ["7.3", "7.4", "7.5", "8.1", "9.1"] },
    { "id": 8, "tasks": ["8.2", "9.2"] },
    { "id": 9, "tasks": ["8.3", "8.4", "9.3", "9.4", "11.1"] },
    { "id": 10, "tasks": ["11.2"] },
    { "id": 11, "tasks": ["11.3", "11.4", "11.5", "11.6", "12.1"] },
    { "id": 12, "tasks": ["12.2", "13.1"] },
    { "id": 13, "tasks": ["13.2", "13.3", "15.1"] },
    { "id": 14, "tasks": ["15.2"] },
    { "id": 15, "tasks": ["15.3"] }
  ]
}
```
