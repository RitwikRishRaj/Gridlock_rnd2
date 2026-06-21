# Requirements Document

## Introduction

Parking Intelligence is an offline-friendly, AI-driven analytics system that transforms raw
Bengaluru traffic-police parking-violation CSV records into a prioritized, map-based enforcement
plan. The system ingests and cleans a large (>50MB) violation CSV, detects illegal-parking
hotspots through spatial aggregation (H3 hexagonal binning and DBSCAN density clustering),
quantifies each hotspot's impact on traffic flow with a transparent congestion-impact score,
and ranks enforcement zones per police station using an explainable priority formula
(impact × frequency × persistence × recency). A temporal model forecasts when each hotspot
peaks by hour-of-day and day-of-week so patrols can be scheduled proactively.

The system runs entirely locally with no required API keys or network calls so a live demo
cannot be blocked by connectivity. It emits two portable artifacts — `hotspots.geojson` and
`priority_zones.csv` — and renders an interactive Streamlit dashboard (heatmap, time slider,
top-N priority zones, per-station drilldown). Optional pluggable live-traffic integration is a
bonus that augments, but is never required by, the impact score.

These requirements are derived from the approved design document and are organized to cover
every pipeline stage, the correctness properties, and the documented error-handling scenarios.

## Glossary

- **Parking_Intelligence_System**: The complete offline analytics pipeline and dashboard.
- **Ingestor**: The ingestion and cleaning component that converts the raw CSV into a clean, typed, geo-validated DataFrame.
- **JSON_Array_Parser**: The function that parses JSON-array string fields (`violation_type`, `offence_code`) into Python lists.
- **Timestamp_Parser**: The component that parses `*_datetime` columns into timezone-aware datetimes.
- **Geo_Validator**: The component that validates coordinates against the Bengaluru bounding box.
- **Hotspot_Builder**: The spatial-aggregation component that assigns H3 cells and runs DBSCAN to build hotspot records.
- **Impact_Scorer**: The component that computes the congestion-impact score per hotspot.
- **Priority_Ranker**: The component that combines impact, frequency, persistence, and recency into a priority score and ranks zones.
- **Peak_Forecaster**: The component that builds hour-of-day × day-of-week peak profiles and predicts peak windows.
- **Exporter**: The component that serializes results into `hotspots.geojson` and `priority_zones.csv`.
- **Dashboard**: The Streamlit user interface presenting the heatmap, time slider, top-N zones, and per-station drilldown.
- **Pipeline_Orchestrator**: The `pipeline.run` orchestrator that wires stages and caches intermediate artifacts.
- **Live_Traffic_Connector**: The optional, pluggable component that augments the impact score with external live-traffic data.
- **Bengaluru_Bbox**: The geographic bounding box where `latitude ∈ [12.7, 13.2]` and `longitude ∈ [77.3, 77.9]`.
- **Hotspot**: A spatial grouping of violation points with a centroid, member count, and police-station attribution.
- **Scored_Hotspot**: A hotspot annotated with a normalized 0–100 impact score and component breakdown.
- **Priority_Zone**: A scored hotspot annotated with priority score, global rank, station rank, and peak windows.
- **Peak_Profile**: A 7×24 normalized hour-of-day × day-of-week intensity matrix for a hotspot.
- **Peak_Window**: A predicted peak time range (day-of-week, start hour, end hour, expected intensity).
- **Congestion_Impact_Score**: A 0–100 normalized score combining severity, proximity, and temporal concentration.
- **Priority_Score**: A 0–100 weighted geometric-mean blend of impact, frequency, persistence, and recency.
- **Recency_Decay**: The exponential half-life decay factor in `(0, 1]` measuring how recent a hotspot's last event is.

## Requirements

### Requirement 1: CSV Ingestion and Memory-Bounded Reading

**User Story:** As an enforcement data operator, I want the system to ingest a large violation CSV without exhausting memory, so that I can process the full Jan–May dataset on a laptop during a demo.

#### Acceptance Criteria

1. WHEN a readable CSV path containing all documented source columns is provided, THE Ingestor SHALL read the file sequentially in chunks of the configured chunk size and return a single cleaned DataFrame containing the cleaned rows from every chunk.
2. WHERE a chunk size is provided, THE Ingestor SHALL require it to be a positive integer of at least `1`.
3. WHERE no chunk size is provided, THE Ingestor SHALL apply a default chunk size of `50,000` rows.
4. WHILE reading a CSV whose full materialization would exceed available memory, THE Ingestor SHALL hold at most one chunk of rows in memory at a time before concatenation, such that peak additional memory scales with the configured chunk size rather than the total file row count.
5. WHEN ingestion completes, THE Ingestor SHALL remove duplicate rows that share the same `id` value, retaining the first occurrence in read order and discarding subsequent occurrences.
6. WHEN the same CSV is ingested more than once, THE Ingestor SHALL produce a row set containing no duplicate `id` values, making re-ingestion idempotent.
7. IF the provided CSV path is missing, unreadable, or does not contain all documented source columns, THEN THE Ingestor SHALL reject the input and return an error indicating the missing path or absent required columns, without producing a partial DataFrame.
8. IF a provided chunk size is not a positive integer of at least `1`, THEN THE Ingestor SHALL reject the configuration and return an error indicating the invalid chunk size, without reading the file.

### Requirement 2: JSON-Array Field Parsing

**User Story:** As a data operator, I want JSON-array fields parsed into typed lists, so that downstream scoring can read violation tokens and offence codes reliably.

#### Acceptance Criteria

1. WHEN a syntactically valid, non-empty JSON-array string is provided for `violation_type` or `offence_code`, THE JSON_Array_Parser SHALL return a Python list containing one element for each array element, preserving the source order of the elements.
2. THE JSON_Array_Parser SHALL return a value of type list for every input, including null, NaN, empty, and malformed inputs.
3. IF the input is null, NaN, an empty string, or not a syntactically valid JSON array, THEN THE JSON_Array_Parser SHALL return an empty list and SHALL NOT raise an exception.
4. WHEN parsing `violation_type` tokens, THE JSON_Array_Parser SHALL trim leading and trailing whitespace from each token and convert each token to upper case before returning it.
5. WHEN parsing `offence_code` values, THE Ingestor SHALL retain only tokens that are convertible to integers (returning them as integer values), discard every token that is not convertible to an integer, and default the field to an empty list when no integer-convertible token remains.
6. WHEN a syntactically valid but empty JSON array is provided for `violation_type` or `offence_code`, THE JSON_Array_Parser SHALL return an empty list without raising an exception.

### Requirement 3: Timestamp Parsing

**User Story:** As a data operator, I want timestamps parsed into timezone-aware datetimes, so that temporal analysis uses consistent local time.

#### Acceptance Criteria

1. WHEN a `created_datetime` value that conforms to a recognized datetime format is present, THE Timestamp_Parser SHALL convert it into a timezone-aware datetime expressed in the Asia/Kolkata timezone, interpreting any value lacking an explicit timezone offset as Asia/Kolkata local time.
2. IF the `created_datetime` value of a row is null, empty, or cannot be converted into a valid datetime, THEN THE Ingestor SHALL drop that row, retain all other rows, and record the drop in the ingestion report.
3. WHERE an optional timestamp column such as `closed_datetime` holds a value that is null or cannot be converted into a valid datetime, THE Timestamp_Parser SHALL retain the row and set that column to a null value.
4. WHEN an optional timestamp column such as `closed_datetime` holds a value that conforms to a recognized datetime format, THE Timestamp_Parser SHALL convert it into a timezone-aware datetime expressed in the Asia/Kolkata timezone.

### Requirement 4: Geographic Validation Against the Bengaluru Bounding Box

**User Story:** As a data operator, I want out-of-area and invalid coordinates removed, so that hotspots reflect only valid Bengaluru locations.

#### Acceptance Criteria

1. THE Geo_Validator SHALL retain a row only when its `latitude` is greater than or equal to 12.7 and less than or equal to 13.2 (inclusive of both bounds) and its `longitude` is greater than or equal to 77.3 and less than or equal to 77.9 (inclusive of both bounds).
2. IF a row has a `latitude` or `longitude` value that is null, empty, non-numeric, NaN, or numerically outside the bounds defined in Criterion 1, THEN THE Geo_Validator SHALL drop that row and retain all remaining valid rows unchanged.
3. THE Geo_Validator SHALL produce an output row count that is greater than or equal to zero and less than or equal to its input row count.
4. WHEN geo validation completes, THE Geo_Validator SHALL ensure that every row in the output contains a numeric, non-null, non-NaN `latitude` and `longitude` that satisfy the bounds defined in Criterion 1.
5. WHEN geo validation completes, THE Ingestor SHALL record in the ingestion report the count of rows dropped due to invalid or out-of-range coordinates, recording a value of zero when no rows are dropped.

### Requirement 5: Categorical Normalization

**User Story:** As a data operator, I want categorical fields normalized, so that grouping by station and violation type is consistent.

#### Acceptance Criteria

1. WHEN cleaning a chunk, THE Ingestor SHALL set `vehicle_type` to its value with leading and trailing whitespace removed and all characters converted to lowercase.
2. WHEN cleaning a chunk, THE Ingestor SHALL set `police_station` to its value with leading and trailing whitespace removed and every run of consecutive internal whitespace characters collapsed to a single space character.
3. WHEN cleaning a chunk, THE Ingestor SHALL normalize each `violation_type` token to its value with leading and trailing whitespace removed and all characters converted to uppercase, such that two tokens differing only in letter case or surrounding whitespace map to an identical token.
4. IF a `vehicle_type`, `police_station`, or `violation_type` value is null, NaN, or empty after whitespace is removed, THEN THE Ingestor SHALL retain the row and set that field to an empty value without raising an error.

### Requirement 6: H3 Spatial Binning

**User Story:** As an analyst, I want each violation assigned to an H3 hexagonal cell, so that I can produce a uniform, responsive heatmap.

#### Acceptance Criteria

1. WHEN a cleaned DataFrame is provided, THE Hotspot_Builder SHALL assign every retained violation row exactly one H3 cell index computed from that row's `latitude` and `longitude` at the configured resolution.
2. WHERE an H3 resolution is provided, THE Hotspot_Builder SHALL require the resolution to be an integer within `[0, 15]`.
3. WHERE no H3 resolution is provided, THE Hotspot_Builder SHALL apply a default resolution of `9`.
4. IF the provided H3 resolution is not an integer within `[0, 15]`, THEN THE Hotspot_Builder SHALL reject the configuration before any assignment, leave the input DataFrame unmodified, and return an error indicating the resolution is out of range.
5. WHEN DBSCAN produces no clusters, THE Hotspot_Builder SHALL fall back to H3-cell aggregation as the hotspot source by grouping all retained violations by their assigned H3 cell index.

### Requirement 7: DBSCAN Density Clustering and Hotspot Construction

**User Story:** As an analyst, I want dense violation clusters detected as hotspots, so that enforcement focuses on organically shaped problem areas.

#### Acceptance Criteria

1. WHEN building hotspots, THE Hotspot_Builder SHALL run DBSCAN on haversine-projected radian coordinates using the configured `eps_m` expressed in meters and the configured integer `min_samples`.
2. WHERE clustering parameters are provided, THE Hotspot_Builder SHALL require `eps_m` to be greater than zero and less than or equal to `5000` meters, and `min_samples` to be an integer greater than or equal to one.
3. IF `eps_m` is less than or equal to zero or greater than `5000`, or `min_samples` is non-integer or less than one, THEN THE Hotspot_Builder SHALL reject the parameters with an error indicating the invalid clustering parameter and SHALL NOT emit any hotspot.
4. THE Hotspot_Builder SHALL emit each hotspot with a `member_count` greater than or equal to the configured `min_samples`.
5. THE Hotspot_Builder SHALL exclude DBSCAN noise points whose cluster label equals `-1` from the emitted hotspots.
6. THE Hotspot_Builder SHALL assign each violation to at most one emitted hotspot.
7. WHEN constructing a hotspot, THE Hotspot_Builder SHALL set its centroid latitude and longitude to the arithmetic mean of its member coordinates, with both values lying within the Bengaluru_Bbox.
8. WHEN constructing a hotspot, THE Hotspot_Builder SHALL set its `police_station` to the modal station among its members, and IF two or more stations share the maximum member count, THEN THE Hotspot_Builder SHALL select the station whose name is first in ascending lexicographic order.

### Requirement 8: Congestion Impact Scoring

**User Story:** As a city official, I want each hotspot scored for its impact on traffic flow, so that I can compare hotspots on a transparent, explainable scale.

#### Acceptance Criteria

1. WHEN scoring a hotspot, THE Impact_Scorer SHALL compute a severity component as the maximum severity weight (each weight in `[0, 1]`) among the hotspot's violation tokens, with the result in `[0, 1]`.
2. IF a hotspot has an empty violation-token list, THEN THE Impact_Scorer SHALL assign the configured severity floor, which lies in `[0, 1]`, as the severity component.
3. WHEN scoring a hotspot, THE Impact_Scorer SHALL compute a proximity component as the fraction of member events tied to a junction or road-crossing context, with the result in `[0, 1]`.
4. WHEN scoring a hotspot, THE Impact_Scorer SHALL compute a temporal-concentration component as one minus the entropy of the hour-of-day distribution normalized by the entropy of a uniform distribution over 24 hour-of-day bins, such that a single occupied bin yields `1.0`, a uniform distribution yields `0.0`, and the result lies in `[0, 1]`.
5. WHERE impact weights are provided, THE Impact_Scorer SHALL require each of the severity, proximity, and concentration weights to lie in `[0, 1]` and their sum to equal `1.0` within a tolerance of `±0.001`.
6. IF the provided impact weights do not each lie in `[0, 1]` or do not sum to `1.0` within `±0.001`, THEN THE Impact_Scorer SHALL reject the weights with an error indicating the invalid weighting and SHALL NOT produce scores.
7. THE Impact_Scorer SHALL produce a `Congestion_Impact_Score` within `[0, 100]` for every scored hotspot.
8. WHEN at least one hotspot exists and the maximum raw score is greater than `0`, THE Impact_Scorer SHALL normalize scores so that the highest-raw hotspot receives an impact score of exactly `100`.
9. IF every hotspot's raw score equals `0`, THEN THE Impact_Scorer SHALL assign a `Congestion_Impact_Score` of `0` to every hotspot.
10. WHEN scoring a hotspot, THE Impact_Scorer SHALL attach a component breakdown whose severity, proximity, and concentration values each lie in `[0, 1]` and reconstruct the raw score under the configured weights within a tolerance of `±0.001`.

### Requirement 9: Priority Ranking per Police Station

**User Story:** As an enforcement planner, I want hotspots ranked globally and within each police station, so that I can allocate patrols where they matter most.

#### Acceptance Criteria

1. WHEN ranking zones, THE Priority_Ranker SHALL min-max normalize the frequency and persistence factors to the closed interval `[0, 1]` across all hotspots, where the hotspot with the minimum raw value maps to `0` and the hotspot with the maximum raw value maps to `1`.
2. IF the maximum raw value equals the minimum raw value for the frequency or persistence factor across all hotspots, THEN THE Priority_Ranker SHALL set that normalized factor to `1.0` for every hotspot.
3. WHEN ranking zones, THE Priority_Ranker SHALL compute `Priority_Score` as `100` times the weighted geometric mean of the normalized impact, frequency, persistence, and recency factors using the respective weights `0.4`, `0.25`, `0.2`, and `0.15`.
4. THE Priority_Ranker SHALL produce a `Priority_Score` within the closed interval `[0, 100]` for every zone.
5. IF any of the impact, frequency, persistence, or recency factors equals `0` for a zone, THEN THE Priority_Ranker SHALL set that zone's `Priority_Score` to `0`.
6. WHEN ranking zones, THE Priority_Ranker SHALL assign global ranks as a contiguous integer sequence `1..N` with no gaps, ordered by descending `Priority_Score`, where rank `1` has the maximum score, and SHALL break ties between equal `Priority_Score` values by ascending hotspot identifier.
7. WHEN ranking zones, THE Priority_Ranker SHALL assign, within each police station, a station rank as a contiguous integer sequence `1..M` with no gaps, ordered by descending `Priority_Score`, and SHALL break ties between equal `Priority_Score` values by ascending hotspot identifier.
8. WHEN ranking a zone, THE Priority_Ranker SHALL attach the predicted peak windows provided by the Peak_Forecaster for that hotspot.
9. IF the Peak_Forecaster provides no predicted peak window for a hotspot, THEN THE Priority_Ranker SHALL attach an empty set of peak windows for that zone and retain the zone in the ranking.

### Requirement 10: Recency Decay

**User Story:** As an enforcement planner, I want recent hotspots weighted more heavily, so that current problem areas rise in priority.

#### Acceptance Criteria

1. WHERE a priority configuration is provided, THE Priority_Ranker SHALL require `recency_halflife_days` to be greater than zero and `last_event` to be less than or equal to `as_of`.
2. IF `recency_halflife_days` is less than or equal to zero, or `last_event` is greater than `as_of`, THEN THE Priority_Ranker SHALL reject the configuration with an error indicating the invalid recency parameter, SHALL NOT compute a `Recency_Decay`, and SHALL leave the input data unchanged.
3. THE Priority_Ranker SHALL compute `Recency_Decay` as a value within `(0, 1]`.
4. WHEN the elapsed time between `last_event` and `as_of` equals zero days, THE Priority_Ranker SHALL compute a `Recency_Decay` of exactly `1.0`.
5. WHEN the elapsed days between `last_event` and `as_of` equal `recency_halflife_days`, THE Priority_Ranker SHALL compute a `Recency_Decay` of exactly `0.5`.
6. WHILE the reference `as_of` is held fixed, THE Priority_Ranker SHALL compute a `Recency_Decay` that is non-increasing as the elapsed days since `last_event` increase.

### Requirement 11: Temporal Peak Prediction

**User Story:** As an enforcement planner, I want each hotspot's peak hours and days forecast, so that I can schedule patrols before violations occur.

#### Acceptance Criteria

1. WHEN building a peak profile, THE Peak_Forecaster SHALL produce a `hour_dow_matrix` of shape `7×24` (7 days-of-week × 24 hours) whose values all lie in `[0, 1]` with a maximum value of exactly `1.0`.
2. WHEN building a peak profile, THE Peak_Forecaster SHALL apply smoothing such that every cell of `hour_dow_matrix` is strictly greater than `0` and less than or equal to `1` before normalization.
3. WHERE a `top_k` value is provided, THE Peak_Forecaster SHALL require `top_k` to be an integer greater than or equal to `1` and less than or equal to `168`.
4. IF a provided `top_k` is non-integer, less than `1`, or greater than `168`, THEN THE Peak_Forecaster SHALL reject the request with an error indication identifying the invalid `top_k` and SHALL NOT return any peak windows.
5. WHERE no `top_k` value is provided, THE Peak_Forecaster SHALL default `top_k` to `3`.
6. WHEN predicting peaks, THE Peak_Forecaster SHALL identify peak hours as matrix cells whose normalized value is greater than or equal to `0.5`.
7. WHEN predicting peaks, THE Peak_Forecaster SHALL return at most `top_k` peak windows ordered by descending peak intensity, each window specifying a `day_of_week` in `[0, 6]` and hours satisfying `0 ≤ start_hour ≤ end_hour ≤ 23`.
8. WHEN predicting peaks, THE Peak_Forecaster SHALL coalesce contiguous peak hours within the same `day_of_week` into a single merged peak window.

### Requirement 12: GeoJSON Export

**User Story:** As a GIS user, I want hotspots exported as valid GeoJSON, so that downstream mapping tools can consume them directly.

#### Acceptance Criteria

1. WHEN exporting scored hotspots, THE Exporter SHALL write a GeoJSON FeatureCollection to the configured path in which each Scored_Hotspot is represented as one Feature whose geometry is a Point set to the hotspot centroid in longitude-then-latitude order with both coordinates within Bengaluru_Bbox.
2. WHEN writing each hotspot Feature, THE Exporter SHALL populate the Feature properties with the hotspot's aggregate attributes (including `member_count`, `police_station`, and `Congestion_Impact_Score`) and SHALL exclude the `vehicle_number` and `updated_vehicle_number` fields.
3. THE Exporter SHALL include only aggregate, score, and centroid-location data in the GeoJSON output and SHALL exclude every per-individual identifier field, including `vehicle_number`, `updated_vehicle_number`, and `id`.
4. WHEN exporting scored hotspots while no scored hotspots exist, THE Exporter SHALL write a GeoJSON FeatureCollection whose features array is empty.
5. IF the configured path cannot be written (for example, the parent directory does not exist or is not writable), THEN THE Exporter SHALL raise an error indicating the write failure and SHALL NOT leave a partial GeoJSON file at the configured path.

### Requirement 13: Priority CSV Export

**User Story:** As an enforcement planner, I want priority zones exported as CSV, so that teams can work from a portable ranked list.

#### Acceptance Criteria

1. WHEN exporting priority zones, THE Exporter SHALL write a `priority_zones.csv` file to the configured path containing a header row followed by one row per Priority_Zone, with columns including `hotspot_id`, `centroid_lat`, `centroid_lon`, `police_station`, `impact`, `frequency`, `persistence`, `recency`, `priority_score`, `global_rank`, and `station_rank`.
2. THE Exporter SHALL write priority-zone rows ordered by ascending `global_rank` as a contiguous `1..N` sequence, with the rank-1 row appearing first.
3. THE Exporter SHALL restrict the priority CSV output to aggregate, ranking, and centroid-location data, and SHALL exclude every per-individual identifier field, including `vehicle_number` and `updated_vehicle_number`.
4. WHEN exporting priority zones while no zones exist, THE Exporter SHALL write a `priority_zones.csv` file containing only the header row.
5. WHEN exporting all artifacts, THE Exporter SHALL return the file paths of the produced `hotspots.geojson` and `priority_zones.csv`.
6. IF the configured path cannot be written (for example, the parent directory does not exist or is not writable), THEN THE Exporter SHALL raise an error indicating the write failure and SHALL NOT leave a partial `priority_zones.csv` file at the configured path.

### Requirement 14: Pipeline Determinism

**User Story:** As an analyst, I want repeated runs to produce identical output, so that results are reproducible and trustworthy in an enforcement context.

#### Acceptance Criteria

1. WHEN the pipeline is run two or more times with identical inputs, identical configuration, and fixed clustering and random seeds, THE Parking_Intelligence_System SHALL produce a byte-identical `priority_zones.csv` (identical row order, column order, and field values) on every run.
2. WHEN the pipeline is run two or more times with identical inputs, identical configuration, and fixed seeds, THE Parking_Intelligence_System SHALL produce a byte-identical `hotspots.geojson` on every run.
3. WHEN two or more zones share an equal `Priority_Score`, THE Priority_Ranker SHALL order them by a fixed deterministic sort key (ascending hotspot identifier) so that output ordering is reproducible.
4. WHERE clustering or random seeds are not explicitly provided, THE Pipeline_Orchestrator SHALL apply a configured default seed rather than a time- or entropy-derived seed, so that reproducibility is preserved.

### Requirement 15: Streamlit Dashboard

**User Story:** As a city official, I want an interactive dashboard, so that I can explore hotspots, peaks, and priority zones visually.

#### Acceptance Criteria

1. WHEN the dashboard opens, THE Dashboard SHALL load the pre-computed `priority_zones.csv` and `hotspots.geojson` artifacts and render, within 5 seconds, a violation heatmap and a table of the top 20 priority zones ordered by ascending `global_rank`.
2. IF a required artifact (`priority_zones.csv` or `hotspots.geojson`) is missing, unreadable, or malformed, THEN THE Dashboard SHALL display an error message identifying the affected artifact and SHALL remain running rather than crash.
3. WHEN a viewer moves the time slider to a selected hour in `[0, 23]` or day-of-week in `[0, 6]`, THE Dashboard SHALL filter hotspot events to that time window and re-render the heatmap for the window within 2 seconds.
4. WHEN the selected time window matches no hotspot events, THE Dashboard SHALL render an empty heatmap and display a message indicating that no violations match the selected window.
5. WHEN a viewer selects a police station, THE Dashboard SHALL render a drilldown showing that station's zones, peak hours, and up to 10 sample violations, excluding `vehicle_number` and `updated_vehicle_number`.
6. THE Dashboard SHALL render visualizations from pre-computed artifacts and SHALL NOT recompute the pipeline on any interaction.
7. THE Dashboard SHALL surface the count of rows dropped during ingestion as recorded in the ingestion report.

### Requirement 16: Offline Operation and Data Privacy

**User Story:** As a demo presenter, I want the system to run fully offline and protect anonymized data, so that connectivity cannot block the demo and no re-identification is possible.

#### Acceptance Criteria

1. THE Parking_Intelligence_System SHALL complete the full pipeline — ingestion, spatial aggregation, impact scoring, priority ranking, peak forecasting, artifact export, and dashboard rendering — using only local computation, requiring no API keys and initiating no outbound network calls for any required stage.
2. WHILE the host has no network connectivity, THE Parking_Intelligence_System SHALL complete the full pipeline and produce both `hotspots.geojson` and `priority_zones.csv` without raising a connectivity-related error.
3. THE Parking_Intelligence_System SHALL treat `vehicle_number` and `updated_vehicle_number` as opaque identifiers, using them for no purpose other than duplicate detection, and SHALL exclude both fields from all exported artifacts (`hotspots.geojson` and `priority_zones.csv`) and from every dashboard view.
4. THE Parking_Intelligence_System SHALL ensure that no exported artifact and no dashboard view contains `vehicle_number`, `updated_vehicle_number`, or any other field that maps an aggregated record to an individual vehicle or person.

### Requirement 17: Optional Live-Traffic Integration (Bonus)

**User Story:** As an analyst, I want optional live-traffic data to augment scoring when available, so that impact scores can reflect current conditions without ever blocking the offline pipeline.

#### Acceptance Criteria

1. WHERE the Live_Traffic_Connector is enabled, THE Impact_Scorer SHALL augment the Congestion_Impact_Score with the external live-traffic signal and SHALL keep the resulting augmented score within `[0, 100]`.
2. WHERE the Live_Traffic_Connector is enabled, THE Live_Traffic_Connector SHALL bound each external live-traffic request to a timeout of 5 seconds.
3. IF the Live_Traffic_Connector is enabled but the live-traffic request fails, returns invalid data, or exceeds its 5-second timeout, THEN THE Impact_Scorer SHALL compute the Congestion_Impact_Score using only offline data and THE Parking_Intelligence_System SHALL complete the full pipeline without aborting.
4. WHERE the Live_Traffic_Connector is disabled or unavailable, THE Parking_Intelligence_System SHALL complete the full pipeline without the live-traffic signal and SHALL make no outbound network calls.

### Requirement 18: Error Handling and Resilience

**User Story:** As a data operator, I want the pipeline to recover gracefully from messy data, so that a single bad row or sparse cluster never aborts the run.

#### Acceptance Criteria

1. IF a `violation_type` or `offence_code` cell is empty, NaN, or invalid JSON, THEN THE Ingestor SHALL retain the row, set its parsed token list to an empty list, apply the configured severity floor as the row's severity component, and include the row in spatial aggregation.
2. IF DBSCAN labels every point as noise, THEN THE Hotspot_Builder SHALL return an empty hotspot list without raising an error.
3. WHEN the Hotspot_Builder returns an empty hotspot list, THE Pipeline_Orchestrator SHALL fall back to H3-cell aggregation as the hotspot source and complete the remaining pipeline stages without aborting the run.
4. WHEN DBSCAN finds no clusters, THE Dashboard SHALL display the H3-cell heatmap and a warning message indicating that no density clusters were found and recommending relaxed `eps_m` or `min_samples` values.
5. IF a row's `created_datetime` cannot be parsed, THEN THE Ingestor SHALL drop the row, increment the dropped-row count in the ingestion report, and continue processing the remaining rows without aborting the run.
6. IF an unexpected error occurs while cleaning or parsing a single row, THEN THE Ingestor SHALL drop that row, record the failure in the ingestion report, and continue processing the remaining rows in the current chunk without aborting the run.
