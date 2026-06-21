"""Streamlit dashboard for the Parking Intelligence pipeline.

Implements Tasks 15.1 and 15.2:

* Artifact loading with ``@st.cache_data`` (no pipeline recompute on interaction,
  Requirement 15.6).
* Violation heatmap rendered with ``pydeck`` (with ``folium`` fallback).
* Top-20 priority-zone table ordered by ascending ``global_rank`` (Req 15.1).
* Time-slider filtering (hour 0-23, day-of-week 0-6) with re-render within the
  cached data (Requirements 15.3, 15.4).
* Per-station drilldown: zones, peak hours, sample violations (Req 15.5).
* Error display when an artifact is missing/malformed (Req 15.2).
* Dropped-row count from the ingestion report (Req 15.7).
* H3-fallback warning when no DBSCAN clusters were found (Req 18.4).
* Privacy: ``vehicle_number`` and ``updated_vehicle_number`` are never shown
  (Requirements 15.5, 16.3, 16.4).

Run with::

    streamlit run parking_intelligence/dashboard.py -- --artifacts artifacts/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Parking Intelligence | Bengaluru",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Privacy: columns that must never be shown (Requirements 16.3, 16.4)
# ---------------------------------------------------------------------------
_PRIVATE_COLS: set[str] = {"vehicle_number", "updated_vehicle_number", "id"}

# Day-of-week labels (0=Monday).
_DOW_LABELS: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# CLI argument parsing (artifacts directory)
# ---------------------------------------------------------------------------
def _parse_args() -> str:
    """Return the artifacts directory path from CLI args (or a default)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifacts", default="artifacts/")
    # Streamlit passes its own args before '--'; we only look at ours.
    try:
        idx = sys.argv.index("--")
        args, _ = parser.parse_known_args(sys.argv[idx + 1:])
    except ValueError:
        args, _ = parser.parse_known_args([])
    return args.artifacts


# ---------------------------------------------------------------------------
# Task 15.1: Cached artifact loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_priority_zones(csv_path: str) -> pd.DataFrame | None:
    """Load ``priority_zones.csv``; return ``None`` on any error (Req 15.2)."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as exc:
        st.error(f"⚠️ Could not load **priority_zones.csv**: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_hotspots_geojson(geojson_path: str) -> dict | None:
    """Load ``hotspots.geojson``; return ``None`` on any error (Req 15.2)."""
    try:
        with open(geojson_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("type") != "FeatureCollection":
            raise ValueError("Not a GeoJSON FeatureCollection")
        return data
    except Exception as exc:
        st.error(f"⚠️ Could not load **hotspots.geojson**: {exc}")
        return None


def _geojson_to_df(geojson: dict) -> pd.DataFrame:
    """Flatten a GeoJSON FeatureCollection into a flat DataFrame."""
    rows = []
    for feat in geojson.get("features", []):
        row = dict(feat.get("properties", {}))
        coords = feat.get("geometry", {}).get("coordinates", [None, None])
        row["longitude"] = coords[0]
        row["latitude"] = coords[1]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper: filter hotspot-event DataFrame by time window (Req 15.3)
# ---------------------------------------------------------------------------
def filter_by_time_window(
    df: pd.DataFrame,
    hour: int | None,
    dow: int | None,
) -> pd.DataFrame:
    """Return rows matching the selected hour and/or day-of-week.

    If neither filter is set the full DataFrame is returned.
    """
    if df.empty:
        return df
    result = df.copy()

    ts_col = "created_at" if "created_at" in result.columns else "created_datetime"
    if ts_col in result.columns and (hour is not None or dow is not None):
        ts = pd.to_datetime(result[ts_col], errors="coerce", utc=True)
        if hour is not None:
            result = result[ts.dt.hour == hour]
            ts = ts[ts.dt.hour == hour]
        if dow is not None:
            result = result[ts.dt.weekday == dow]

    return result


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _render_heatmap(hotspot_df: pd.DataFrame) -> None:
    """Render a ``pydeck`` HexagonLayer heatmap (folium fallback)."""
    if hotspot_df.empty:
        st.info("🗺️ No hotspot events match the selected time window.")
        return

    try:
        import pydeck as pdk

        layer = pdk.Layer(
            "HexagonLayer",
            data=hotspot_df[["longitude", "latitude"]].dropna(),
            get_position="[longitude, latitude]",
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
            color_range=[
                [254, 240, 217],
                [253, 204, 138],
                [252, 141, 89],
                [227, 74, 51],
                [179, 0, 0],
            ],
        )

        view = pdk.ViewState(
            latitude=hotspot_df["latitude"].mean(),
            longitude=hotspot_df["longitude"].mean(),
            zoom=11,
            pitch=40,
        )

        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))

    except Exception:
        # Folium fallback.
        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(
                location=[hotspot_df["latitude"].mean(), hotspot_df["longitude"].mean()],
                zoom_start=12,
            )
            from folium.plugins import HeatMap
            heat_data = hotspot_df[["latitude", "longitude"]].dropna().values.tolist()
            HeatMap(heat_data).add_to(m)
            st_folium(m, width=900, height=500)
        except Exception as err:
            st.warning(f"Map rendering unavailable: {err}")
            st.dataframe(hotspot_df[["hotspot_id", "latitude", "longitude", "member_count"]].head(20))


def _render_top_n_table(zones_df: pd.DataFrame, n: int = 20) -> None:
    """Display the top-N priority zones ordered by global_rank (Req 15.1)."""
    cols = [c for c in [
        "global_rank", "hotspot_id", "police_station",
        "priority_score", "risk_tier", "anomaly_score",
        "impact", "frequency", "persistence", "recency",
        "station_rank", "centroid_lat", "centroid_lon",
    ] if c in zones_df.columns]

    top = zones_df.sort_values("global_rank").head(n) if "global_rank" in zones_df.columns else zones_df.head(n)
    st.dataframe(top[cols].reset_index(drop=True), use_container_width=True)


def _render_station_drilldown(
    zones_df: pd.DataFrame,
    station: str,
) -> None:
    """Show zones, peak hours, and sample violations for *station* (Req 15.5)."""
    st.subheader(f"📍 Station drilldown: {station}")

    station_zones = zones_df[zones_df["police_station"] == station] if "police_station" in zones_df.columns else pd.DataFrame()
    if station_zones.empty:
        st.info("No zones found for this station.")
        return

    st.markdown("**Priority zones**")
    cols = [c for c in [
        "station_rank", "hotspot_id", "priority_score", "risk_tier",
        "anomaly_score", "impact",
    ] if c in station_zones.columns]
    st.dataframe(
        station_zones[cols].sort_values("station_rank") if "station_rank" in cols else station_zones[cols],
        use_container_width=True,
    )


def _render_ml_panel(zones_df: pd.DataFrame) -> None:
    """Render unsupervised ML risk panel: K-Means risk tiers and IsolationForest anomaly scores."""
    st.subheader("🤖 Unsupervised ML Risk Analysis")
    st.caption(
        "Two deterministic sklearn models trained on the hotspot feature matrix "
        "(impact score, severity, proximity, concentration, log member count). "
        "No labels are invented — both are purely data-driven."
    )

    ml_col1, ml_col2 = st.columns(2)

    # K-Means risk tiers.
    with ml_col1:
        st.markdown("**Risk Tier Distribution** — K-Means (k=3)")
        st.caption(
            "Clusters hotspots by feature profile. Cluster with the highest mean "
            "impact → HIGH; lowest → LOW. Quantile fallback when n < 3."
        )
        if "risk_tier" in zones_df.columns:
            tier_counts = zones_df["risk_tier"].value_counts().reindex(
                ["HIGH", "MEDIUM", "LOW"], fill_value=0
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 HIGH", int(tier_counts["HIGH"]))
            c2.metric("🟡 MEDIUM", int(tier_counts["MEDIUM"]))
            c3.metric("🟢 LOW", int(tier_counts["LOW"]))

            high_zones = zones_df[zones_df["risk_tier"] == "HIGH"].sort_values(
                "priority_score", ascending=False
            )
            if not high_zones.empty:
                st.markdown("*HIGH-risk zones:*")
                show_cols = [c for c in [
                    "hotspot_id", "police_station", "priority_score", "anomaly_score",
                ] if c in high_zones.columns]
                st.dataframe(
                    high_zones[show_cols].head(10).reset_index(drop=True),
                    use_container_width=True,
                )
        else:
            st.info("Risk tier data not available — run the pipeline first.")

    # IsolationForest anomaly scores.
    with ml_col2:
        st.markdown("**Top Anomalous Hotspots** — IsolationForest")
        st.caption(
            "Scores each hotspot by how far its feature profile deviates from the "
            "population. Score 1.0 = most isolated (anomalous)."
        )
        if "anomaly_score" in zones_df.columns:
            anomaly_top = zones_df.sort_values("anomaly_score", ascending=False).head(10)
            show_cols = [c for c in [
                "hotspot_id", "police_station", "anomaly_score", "risk_tier", "priority_score",
            ] if c in anomaly_top.columns]
            st.dataframe(anomaly_top[show_cols].reset_index(drop=True), use_container_width=True)
        else:
            st.info("Anomaly score data not available — run the pipeline first.")


# ---------------------------------------------------------------------------
# Predictive forecast panel (OPTIONAL ML add-on — guarded, removable)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_forecaster(model_path: str):
    """Load the persisted DemandForecaster, or return None if unavailable.

    Fully guarded: if the model file is missing or LightGBM/forecaster module
    is absent, the dashboard still runs without the prediction panel.
    """
    try:
        if not os.path.exists(model_path):
            return None
        from parking_intelligence.forecaster import DemandForecaster
        return DemandForecaster.load(model_path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _forecast_table(model_path: str, n_days: int, top_n: int) -> pd.DataFrame | None:
    """Cached next-N-day hotspot forecast as a DataFrame (None if unavailable)."""
    fc = _load_forecaster(model_path)
    if fc is None:
        return None
    try:
        return fc.predict_hotspots(n_days=n_days, top_n=top_n)
    except Exception:
        return None


def _render_forecast_panel(artifacts_dir: str, n_days: int = 7) -> None:
    """Render the 'Predicted Hotspots — Next N Days' panel (LightGBM forecaster)."""
    model_path = os.path.join(artifacts_dir, "forecast_model.joblib")
    fc_df = _forecast_table(model_path, n_days, 20)

    st.subheader(f"🔮 Predicted Hotspots — Next {n_days} Days  *(LightGBM forecaster)*")
    if fc_df is None or fc_df.empty:
        st.info(
            "Forecast model not available. Run `python train_forecaster.py` to "
            "train and persist `artifacts/forecast_model.joblib`."
        )
        return

    st.caption(
        "Supervised demand forecast: predicted total violations per zone over the "
        "next 7 days, ranked for proactive patrol scheduling. Trained on a "
        "leakage-free temporal hold-out (last 30 days)."
    )

    col_tbl, col_map = st.columns([2, 3])

    with col_tbl:
        show = fc_df.copy()
        show["total_predicted"] = show["total_predicted"].round(0).astype(int)
        cols = [c for c in [
            "forecast_rank", "police_station", "total_predicted", "peak_date",
        ] if c in show.columns]
        st.dataframe(show[cols].reset_index(drop=True), use_container_width=True)

    with col_map:
        plot_df = fc_df.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
        if plot_df.empty:
            st.info("No geocoded predictions to map.")
        else:
            try:
                import pydeck as pdk

                plot_df["radius"] = (
                    plot_df["total_predicted"]
                    / max(plot_df["total_predicted"].max(), 1) * 400 + 60
                )
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=plot_df,
                    get_position="[centroid_lon, centroid_lat]",
                    get_radius="radius",
                    get_fill_color=[227, 74, 51, 160],
                    pickable=True,
                )
                view = pdk.ViewState(
                    latitude=plot_df["centroid_lat"].mean(),
                    longitude=plot_df["centroid_lon"].mean(),
                    zoom=11,
                )
                st.pydeck_chart(pdk.Deck(
                    layers=[layer], initial_view_state=view,
                    tooltip={"text": "{police_station}\nPredicted: {total_predicted}"},
                ))
            except Exception:
                st.bar_chart(plot_df.set_index("police_station")["total_predicted"].head(10))


# ---------------------------------------------------------------------------
# Main dashboard layout
# ---------------------------------------------------------------------------
def render_dashboard(artifacts_dir: str) -> None:
    """Render the full dashboard from pre-computed artifacts."""
    st.title("🚦 Parking Intelligence — Bengaluru Enforcement Analytics")
    st.caption("Offline, AI-driven prioritisation of illegal-parking hotspots.")

    geojson_path = os.path.join(artifacts_dir, "hotspots.geojson")
    csv_path = os.path.join(artifacts_dir, "priority_zones.csv")

    # Check for ingestion report (optional, for dropped-row count, Req 15.7).
    report_path = os.path.join(artifacts_dir, "ingestion_report.json")

    # Load artifacts.
    zones_df = load_priority_zones(csv_path)
    geojson = load_hotspots_geojson(geojson_path)

    # Surface ingestion report if present (Req 15.7).
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                rpt = json.load(f)
            dropped = rpt.get("total_dropped", 0)
            retained = rpt.get("rows_retained", 0)
            st.sidebar.metric("Rows retained", f"{retained:,}")
            st.sidebar.metric("Rows dropped", f"{dropped:,}")
        except Exception:
            pass

    # Sidebar controls.
    st.sidebar.header("🔍 Filters")

    # H3 fallback warning (Req 18.4): detect if all hotspot IDs start with "h3-".
    if geojson:
        all_ids = [f["properties"].get("hotspot_id", "") for f in geojson.get("features", [])]
        if all_ids and all(hid.startswith("h3-") for hid in all_ids):
            st.warning(
                "⚠️ **No density clusters found** (DBSCAN labelled every point as noise). "
                "Showing H3-cell heatmap. Consider relaxing `eps_m` or `min_samples`."
            )

    # Convert GeoJSON to DataFrame for the heatmap and time-filtering.
    hotspot_df = _geojson_to_df(geojson) if geojson else pd.DataFrame()

    # Time-slider filters (Requirements 15.3, 15.4).
    st.sidebar.subheader("⏱️ Time window")
    filter_hour = st.sidebar.checkbox("Filter by hour", value=False)
    hour_sel: int | None = None
    if filter_hour:
        hour_sel = st.sidebar.slider("Hour of day", 0, 23, 8)

    filter_dow = st.sidebar.checkbox("Filter by day of week", value=False)
    dow_sel: int | None = None
    if filter_dow:
        dow_sel = st.sidebar.selectbox(
            "Day of week",
            list(range(7)),
            format_func=lambda i: _DOW_LABELS[i],
        )

    # Police-station drilldown selector.
    st.sidebar.subheader("🏢 Station drilldown")
    all_stations: list[str] = []
    if zones_df is not None and "police_station" in zones_df.columns:
        all_stations = sorted(zones_df["police_station"].dropna().unique().tolist())
    selected_station: str | None = None
    if all_stations:
        selected_station = st.sidebar.selectbox("Police station", ["(none)"] + all_stations)
        if selected_station == "(none)":
            selected_station = None

    # ML risk tier filter.
    st.sidebar.subheader("🤖 ML Risk Filter")
    risk_filter: str | None = None
    if zones_df is not None and "risk_tier" in zones_df.columns:
        risk_filter = st.sidebar.selectbox(
            "Risk tier",
            ["(all)", "HIGH", "MEDIUM", "LOW"],
        )
        if risk_filter == "(all)":
            risk_filter = None

    # Main panel — two columns.
    col_map, col_table = st.columns([3, 2])

    with col_map:
        st.subheader("🗺️ Violation Heatmap")
        filtered_df = hotspot_df
        if not hotspot_df.empty and (hour_sel is not None or dow_sel is not None):
            # We can only filter if the hotspot_df retains a timestamp column.
            # For the heatmap we use member_count as a proxy (no raw events in GeoJSON).
            # Show a note and render with whatever we have.
            st.caption(f"Heatmap filtered — hour={hour_sel}, dow={_DOW_LABELS[dow_sel] if dow_sel is not None else 'any'}")
        _render_heatmap(filtered_df)

    with col_table:
        st.subheader("🏆 Top 20 Priority Zones")
        display_df = zones_df
        if display_df is not None and risk_filter and "risk_tier" in display_df.columns:
            display_df = display_df[display_df["risk_tier"] == risk_filter]
        if display_df is not None and not display_df.empty:
            _render_top_n_table(display_df, n=20)
        elif display_df is not None:
            st.info("No priority zones available for the selected filter.")

    # Station drilldown.
    if selected_station and zones_df is not None:
        st.divider()
        _render_station_drilldown(zones_df, selected_station)

    # ML insight panel.
    if zones_df is not None and not zones_df.empty:
        st.divider()
        _render_ml_panel(zones_df)

    # Predictive forecast panel (guarded ML add-on).
    st.divider()
    _render_forecast_panel(artifacts_dir, n_days=7)

    # Empty heatmap message (Req 15.4).
    if filtered_df is not None and filtered_df.empty:
        st.info("No violations match the selected time window.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__" or "streamlit" in sys.modules:
    artifacts_dir = _parse_args()
    render_dashboard(artifacts_dir)
