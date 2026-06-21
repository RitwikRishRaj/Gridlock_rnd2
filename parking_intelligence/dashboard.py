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
# Modern Minimalistic Dark CSS Theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── Global Reset ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0a0f !important;
    color: #e2e8f0 !important;
}

/* ─── Streamlit Frame ───────────────────────────────────────────────────────── */
.stApp { background: #0a0a0f !important; }

/* Hide default Streamlit header decoration */
header[data-testid="stHeader"] { background: transparent !important; }

/* ─── Sidebar ────────────────────────────────────────────────────────────────  */
[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.8rem !important; }

/* ─── KPI Cards ──────────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.5);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #475569;
}
.kpi-high  { border-color: rgba(239,68,68,0.35) !important; }
.kpi-high::before  { background: linear-gradient(90deg,#ef4444,#f97316) !important; }
.kpi-med   { border-color: rgba(234,179,8,0.35) !important; }
.kpi-med::before   { background: linear-gradient(90deg,#eab308,#f59e0b) !important; }
.kpi-low   { border-color: rgba(34,197,94,0.35) !important; }
.kpi-low::before   { background: linear-gradient(90deg,#22c55e,#10b981) !important; }
.kpi-blue  { border-color: rgba(6,182,212,0.35) !important; }
.kpi-blue::before  { background: linear-gradient(90deg,#06b6d4,#3b82f6) !important; }

/* ─── Section Headers ────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 32px 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.section-header h3 {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0;
    letter-spacing: -0.01em;
}
.section-badge {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 20px;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.25);
}

/* ─── Page Title ─────────────────────────────────────────────────────────── */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.03em;
    margin-bottom: 2px;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 0;
}

/* ─── Tabs ───────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #e2e8f0 !important; background: rgba(255,255,255,0.04) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #818cf8 !important;
    background: rgba(99,102,241,0.1) !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* ─── DataFrames ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.dataframe thead th {
    background: #111118 !important;
    color: #64748b !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
.dataframe tbody tr:hover td { background: rgba(99,102,241,0.05) !important; }

/* ─── Metrics ────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #13131f;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px !important;
}

/* ─── Selectbox / Slider ─────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #13131f !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ─── Alert / Info / Warning ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* ─── Divider ────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.05) !important; }

/* ─── Risk badges ────────────────────────────────────────────────────────── */
.risk-high { color: #f87171; font-weight: 600; }
.risk-med  { color: #fbbf24; font-weight: 600; }
.risk-low  { color: #34d399; font-weight: 600; }

/* ─── Sidebar metric ─────────────────────────────────────────────────────── */
.sidebar-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-stat-label { font-size: 0.72rem; color: #64748b; }
.sidebar-stat-value { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Privacy: columns that must never be shown (Requirements 16.3, 16.4)
# ---------------------------------------------------------------------------
_PRIVATE_COLS: set[str] = {"vehicle_number", "updated_vehicle_number", "id"}

# Day-of-week labels (0=Monday).
_DOW_LABELS: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DOW_FULL: list[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# CLI argument parsing (artifacts directory)
# ---------------------------------------------------------------------------
def _parse_args() -> str:
    """Return the artifacts directory path from CLI args (or a default)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifacts", default="artifacts/")
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


@st.cache_data(show_spinner=False)
def load_peak_windows(json_path: str) -> dict | None:
    """Load ``peak_windows.json``; return ``None`` on any error."""
    try:
        with open(json_path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
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
# Helper: filter by peak-window activity (uses peak_windows.json)
# ---------------------------------------------------------------------------
def _active_hotspot_ids(
    peak_windows: dict,
    hour: int | None,
    dow: int | None,
) -> set[str] | None:
    """Return the set of hotspot IDs whose peak windows cover *hour* and/or *dow*.

    Returns None when no time filter is active (meaning: include everything).
    A hotspot is considered active if ANY of its windows satisfies ALL
    selected criteria simultaneously.

    peak_windows structure (from peak_windows.json)::

        {"dbscan-2": [{"day_of_week": 6, "start_hour": 8, "end_hour": 11,
                        "expected_intensity": 1.0}, ...]}
    """
    if not peak_windows or (hour is None and dow is None):
        return None   # no filter → caller keeps everything

    active: set[str] = set()
    for hspot_id, windows in peak_windows.items():
        for w in windows:
            hour_match = (
                hour is None
                or int(w.get("start_hour", 0)) <= hour <= int(w.get("end_hour", 23))
            )
            dow_match = (
                dow is None
                or int(w.get("day_of_week", -1)) == dow
            )
            if hour_match and dow_match:
                active.add(hspot_id)
                break   # one matching window is enough
    return active


def _filter_df_by_ids(
    df: pd.DataFrame,
    active_ids: set[str] | None,
    id_col: str = "hotspot_id",
) -> pd.DataFrame:
    """Keep only rows whose *id_col* is in *active_ids* (pass-through if None)."""
    if active_ids is None or df.empty or id_col not in df.columns:
        return df
    return df[df[id_col].isin(active_ids)].copy()


# ---------------------------------------------------------------------------
# KPI Card helper
# ---------------------------------------------------------------------------
def _kpi(label: str, value: str, sub: str = "", css_class: str = "") -> str:
    cls = f"kpi-card {css_class}".strip()
    return f"""
    <div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def _section(icon: str, title: str, badge: str = "") -> None:
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="section-header"><h3>{icon} {title}</h3>{badge_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _render_heatmap(hotspot_df: pd.DataFrame) -> None:
    """Render all hotspot centroids as visible circles on a dark map.

    Uses pydeck ScatterplotLayer — one circle per hotspot, sized by
    member_count, coloured blue→red by intensity.  Falls back to st.map()
    when pydeck is unavailable.
    """
    if hotspot_df.empty:
        st.info("No hotspots match the selected filters.")
        return

    try:
        import pydeck as pdk

        map_df = hotspot_df[["longitude", "latitude"]].dropna().copy()

        # Per-point size and colour derived from member_count
        if "member_count" in hotspot_df.columns:
            mc = pd.to_numeric(
                hotspot_df.loc[map_df.index, "member_count"], errors="coerce"
            ).fillna(1)
            max_mc = max(float(mc.max()), 1.0)
            norm = mc / max_mc
            map_df["member_count"] = mc.values
            map_df["radius"] = (norm * 420 + 80).astype(int).values   # 80–500 m
            map_df["r"] = (norm * 220 + 20).astype(int).values        # blue → red
            map_df["g"] = ((1 - norm) * 80 + 20).astype(int).values
            map_df["b"] = (150 - norm * 120).astype(int).values
            map_df["a"] = 200
        else:
            map_df["member_count"] = 1
            map_df["radius"] = 150
            map_df["r"] = 99
            map_df["g"] = 102
            map_df["b"] = 241
            map_df["a"] = 200

        # Copy tooltip columns
        for col in ("hotspot_id", "police_station", "risk_tier"):
            if col in hotspot_df.columns:
                map_df[col] = hotspot_df.loc[map_df.index, col].values

        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_radius="radius",
            get_fill_color="[r, g, b, a]",
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255, 40],
            line_width_min_pixels=1,
        )

        view = pdk.ViewState(
            latitude=float(map_df["latitude"].mean()),
            longitude=float(map_df["longitude"].mean()),
            zoom=11,
            pitch=30,
            bearing=0,
        )

        tooltip_lines = [
            f"{label}: {{{col}}}"
            for col, label in [
                ("hotspot_id", "Hotspot"),
                ("police_station", "Station"),
                ("risk_tier", "Risk"),
                ("member_count", "Violations"),
            ]
            if col in map_df.columns
        ]

        st.pydeck_chart(
            pdk.Deck(
                layers=[scatter],
                initial_view_state=view,
                map_style="mapbox://styles/mapbox/dark-v10",
                tooltip={"text": "\n".join(tooltip_lines)},
            ),
            use_container_width=True,
        )

    except Exception as pdk_err:
        # st.map() fallback — always works, no extra deps needed
        st.caption(f"⚠️ Interactive map unavailable ({pdk_err}). Showing basic map.")
        fb_df = hotspot_df[["latitude", "longitude"]].dropna().rename(
            columns={"latitude": "lat", "longitude": "lon"}
        )
        st.map(fb_df, zoom=11, use_container_width=True)




def _render_top_n_table(zones_df: pd.DataFrame, n: int = 20) -> None:
    """Display the top-N priority zones ordered by global_rank (Req 15.1)."""
    cols = [c for c in [
        "global_rank", "hotspot_id", "police_station",
        "priority_score", "risk_tier", "anomaly_score",
        "impact", "frequency", "persistence", "recency",
        "station_rank",
    ] if c in zones_df.columns]

    top = zones_df.sort_values("global_rank").head(n) if "global_rank" in zones_df.columns else zones_df.head(n)
    st.dataframe(
        top[cols].reset_index(drop=True),
        use_container_width=True,
        height=420,
        column_config={
            "priority_score": st.column_config.ProgressColumn(
                "Priority Score", min_value=0, max_value=100, format="%.1f"
            ),
            "anomaly_score": st.column_config.ProgressColumn(
                "Anomaly Score", min_value=0, max_value=1, format="%.3f"
            ),
            "risk_tier": st.column_config.TextColumn("Risk Tier"),
            "global_rank": st.column_config.NumberColumn("Rank", format="%d"),
        },
    )


def _render_station_drilldown(zones_df: pd.DataFrame, station: str) -> None:
    """Show zones and peak hours for *station* (Req 15.5)."""
    station_zones = (
        zones_df[zones_df["police_station"] == station]
        if "police_station" in zones_df.columns
        else pd.DataFrame()
    )
    if station_zones.empty:
        st.info("No zones found for this station.")
        return

    # Summary KPIs
    n_zones = len(station_zones)
    high_n = int((station_zones["risk_tier"] == "HIGH").sum()) if "risk_tier" in station_zones.columns else 0
    top_score = float(station_zones["priority_score"].max()) if "priority_score" in station_zones.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.markdown(_kpi("Zones", str(n_zones), "in this station", "kpi-blue"), unsafe_allow_html=True)
    c2.markdown(_kpi("HIGH Risk", str(high_n), "critical zones", "kpi-high"), unsafe_allow_html=True)
    c3.markdown(_kpi("Top Score", f"{top_score:.1f}", "max priority score"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cols = [c for c in [
        "station_rank", "hotspot_id", "priority_score", "risk_tier",
        "anomaly_score", "impact", "frequency", "persistence",
    ] if c in station_zones.columns]
    sort_col = "station_rank" if "station_rank" in cols else cols[0]
    st.dataframe(
        station_zones[cols].sort_values(sort_col).reset_index(drop=True),
        use_container_width=True,
        column_config={
            "priority_score": st.column_config.ProgressColumn(
                "Priority Score", min_value=0, max_value=100, format="%.1f"
            ),
            "anomaly_score": st.column_config.ProgressColumn(
                "Anomaly Score", min_value=0, max_value=1, format="%.3f"
            ),
        },
    )


def _render_peak_windows_panel(peak_windows: dict, zones_df: pd.DataFrame | None) -> None:
    """Render peak activity windows panel from peak_windows.json."""
    if not peak_windows:
        st.info("Peak windows data not available.")
        return

    # Flatten all windows into a DataFrame for analysis
    rows = []
    for hspot_id, windows in peak_windows.items():
        for w in windows:
            rows.append({
                "hotspot_id": hspot_id,
                "day_of_week": w.get("day_of_week", 0),
                "day_label": _DOW_LABELS[w.get("day_of_week", 0)],
                "start_hour": w.get("start_hour", 0),
                "end_hour": w.get("end_hour", 0),
                "expected_intensity": w.get("expected_intensity", 0),
            })

    if not rows:
        st.info("No peak window data to display.")
        return

    pw_df = pd.DataFrame(rows)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Peak Activity by Day of Week**")
        day_intensity = pw_df.groupby("day_label")["expected_intensity"].mean().reindex(_DOW_LABELS)
        # Build a simple bar chart using Streamlit native
        chart_data = pd.DataFrame({
            "Average Intensity": day_intensity.values,
        }, index=day_intensity.index)
        st.bar_chart(chart_data, height=260, use_container_width=True)

    with col_right:
        st.markdown("**Top Peak Windows**")
        top_pw = (
            pw_df.sort_values("expected_intensity", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        top_pw["window"] = top_pw.apply(
            lambda r: f"{int(r['start_hour']):02d}:00–{int(r['end_hour']):02d}:00", axis=1
        )
        show_cols = ["hotspot_id", "day_label", "window", "expected_intensity"]
        st.dataframe(
            top_pw[show_cols],
            use_container_width=True,
            height=280,
            column_config={
                "expected_intensity": st.column_config.ProgressColumn(
                    "Intensity", min_value=0, max_value=1, format="%.2f"
                ),
                "day_label": st.column_config.TextColumn("Day"),
                "window": st.column_config.TextColumn("Time Window"),
            },
        )


def _render_ml_panel(zones_df: pd.DataFrame) -> None:
    """Render unsupervised ML risk panel: K-Means risk tiers and IsolationForest anomaly scores."""
    if "risk_tier" in zones_df.columns:
        tier_counts = zones_df["risk_tier"].value_counts().reindex(
            ["HIGH", "MEDIUM", "LOW"], fill_value=0
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            _kpi("HIGH Risk Zones", str(int(tier_counts["HIGH"])), "K-Means cluster", "kpi-high"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            _kpi("MEDIUM Risk", str(int(tier_counts["MEDIUM"])), "K-Means cluster", "kpi-med"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            _kpi("LOW Risk", str(int(tier_counts["LOW"])), "K-Means cluster", "kpi-low"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_tiers, col_anomaly = st.columns(2)

    with col_tiers:
        st.markdown("**HIGH-Risk Zones** *(K-Means, k=3)*")
        if "risk_tier" in zones_df.columns:
            high_zones = zones_df[zones_df["risk_tier"] == "HIGH"].sort_values(
                "priority_score", ascending=False
            )
            show_cols = [c for c in [
                "hotspot_id", "police_station", "priority_score", "anomaly_score",
            ] if c in high_zones.columns]
            st.dataframe(
                high_zones[show_cols].head(12).reset_index(drop=True),
                use_container_width=True,
                height=340,
                column_config={
                    "priority_score": st.column_config.ProgressColumn(
                        "Priority", min_value=0, max_value=100, format="%.1f"
                    ),
                    "anomaly_score": st.column_config.ProgressColumn(
                        "Anomaly", min_value=0, max_value=1, format="%.3f"
                    ),
                },
            )
        else:
            st.info("Risk tier data not available.")

    with col_anomaly:
        st.markdown("**Top Anomalous Hotspots** *(IsolationForest)*")
        if "anomaly_score" in zones_df.columns:
            anomaly_top = zones_df.sort_values("anomaly_score", ascending=False).head(12)
            show_cols = [c for c in [
                "hotspot_id", "police_station", "anomaly_score", "risk_tier", "priority_score",
            ] if c in anomaly_top.columns]
            st.dataframe(
                anomaly_top[show_cols].reset_index(drop=True),
                use_container_width=True,
                height=340,
                column_config={
                    "priority_score": st.column_config.ProgressColumn(
                        "Priority", min_value=0, max_value=100, format="%.1f"
                    ),
                    "anomaly_score": st.column_config.ProgressColumn(
                        "Anomaly", min_value=0, max_value=1, format="%.3f"
                    ),
                },
            )
        else:
            st.info("Anomaly score data not available.")


# ---------------------------------------------------------------------------
# Predictive forecast panel
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_forecaster(model_path: str):
    """Load persisted DemandForecaster, auto-training it on first deploy if missing."""
    import traceback
    try:
        if not os.path.exists(model_path):
            # Auto-train: find the violation CSV in the working directory or parent.
            csv_candidates = [
                "jan to may police violation_anonymized791b166.csv",
                "violations.csv",
            ]
            csv_path = None
            for name in csv_candidates:
                if os.path.exists(name):
                    csv_path = name
                    break
            if csv_path is None:
                for name in csv_candidates:
                    up = os.path.join("..", name)
                    if os.path.exists(up):
                        csv_path = up
                        break
            if csv_path is None:
                return None  # no CSV found — cannot train

            with st.spinner("🤖 Training LightGBM forecaster on first run (~20 s)…"):
                from parking_intelligence.forecaster import DemandForecaster
                from parking_intelligence.ingest import Ingestor
                from parking_intelligence.hotspots import HotspotBuilder
                df, _ = Ingestor().load_and_clean(csv_path, chunksize=100_000)
                df = HotspotBuilder().assign_h3(df, resolution=8)
                fc = DemandForecaster(h3_res=8)
                fc.fit(df)
                os.makedirs(os.path.dirname(os.path.abspath(model_path)) or ".", exist_ok=True)
                fc.save(model_path)
                return fc

        from parking_intelligence.forecaster import DemandForecaster
        return DemandForecaster.load(model_path)
    except Exception:
        # Store the traceback so the panel can display it for debugging
        st.session_state["_forecaster_error"] = traceback.format_exc()
        return None


@st.cache_data(show_spinner=False)
def _forecast_table(model_path: str, n_days: int, top_n: int) -> pd.DataFrame | None:
    fc = _load_forecaster(model_path)
    if fc is None:
        return None
    try:
        return fc.predict_hotspots(n_days=n_days, top_n=top_n)
    except Exception:
        st.session_state["_forecaster_error"] = __import__("traceback").format_exc()
        return None


def _render_forecast_panel(artifacts_dir: str, n_days: int = 7) -> None:
    """Render the 'Predicted Hotspots — Next N Days' panel (LightGBM forecaster)."""
    model_path = os.path.join(artifacts_dir, "forecast_model.joblib")
    fc_df = _forecast_table(model_path, n_days, 20)

    if fc_df is None or fc_df.empty:
        st.warning(
            "⚠️ Forecast model not available and the violation CSV was not found "
            "in the working directory. Place the CSV in the project root and "
            "restart the dashboard — it will auto-train on first load (~20 s)."
        )
        # --- TEMP DEBUG: show actual error so we can diagnose on Render ---
        err = st.session_state.get("_forecaster_error")
        if err:
            st.error("**Debug — actual exception:**")
            st.code(err)
        else:
            st.info(f"model_path resolved to: `{os.path.abspath(model_path)}`  \n"
                    f"exists: `{os.path.exists(model_path)}`  \n"
                    f"cwd: `{os.getcwd()}`")
        # --- END TEMP DEBUG ---
        return


    st.caption(
        "LightGBM demand forecast: predicted total violations per zone over the "
        "next 7 days, ranked for proactive patrol scheduling."
    )

    col_tbl, col_map = st.columns([2, 3])

    with col_tbl:
        show = fc_df.copy()
        show["total_predicted"] = show["total_predicted"].round(0).astype(int)
        cols = [c for c in [
            "forecast_rank", "police_station", "total_predicted", "peak_date",
        ] if c in show.columns]
        st.dataframe(
            show[cols].reset_index(drop=True),
            use_container_width=True,
            height=380,
            column_config={
                "total_predicted": st.column_config.ProgressColumn(
                    "Predicted Violations",
                    min_value=0,
                    max_value=int(show["total_predicted"].max()) if "total_predicted" in show.columns else 100,
                    format="%d",
                ),
                "forecast_rank": st.column_config.NumberColumn("Rank", format="%d"),
            },
        )

    with col_map:
        plot_df = fc_df.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
        if plot_df.empty:
            st.info("No geocoded predictions to map.")
        else:
            try:
                import pydeck as pdk

                plot_df["radius"] = (
                    plot_df["total_predicted"]
                    / max(plot_df["total_predicted"].max(), 1) * 400 + 80
                )
                plot_df["r"] = 220
                plot_df["g"] = 50
                plot_df["b"] = 100
                plot_df["a"] = 180

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=plot_df,
                    get_position="[centroid_lon, centroid_lat]",
                    get_radius="radius",
                    get_fill_color="[r, g, b, a]",
                    pickable=True,
                    stroked=True,
                    get_line_color=[255, 100, 150, 200],
                    line_width_min_pixels=1,
                )
                view = pdk.ViewState(
                    latitude=float(plot_df["centroid_lat"].mean()),
                    longitude=float(plot_df["centroid_lon"].mean()),
                    zoom=11,
                    pitch=30,
                )
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer],
                        initial_view_state=view,
                        map_style="mapbox://styles/mapbox/dark-v10",
                        tooltip={"text": "{police_station}\nPredicted: {total_predicted}"},
                    ),
                    use_container_width=True,
                )
            except Exception:
                st.bar_chart(plot_df.set_index("police_station")["total_predicted"].head(10))


# ---------------------------------------------------------------------------
# Main dashboard layout
# ---------------------------------------------------------------------------
def render_dashboard(artifacts_dir: str) -> None:
    """Render the full dashboard from pre-computed artifacts."""

    geojson_path = os.path.join(artifacts_dir, "hotspots.geojson")
    csv_path = os.path.join(artifacts_dir, "priority_zones.csv")
    report_path = os.path.join(artifacts_dir, "ingestion_report.json")
    peaks_path = os.path.join(artifacts_dir, "peak_windows.json")

    # Load artifacts.
    zones_df = load_priority_zones(csv_path)
    geojson = load_hotspots_geojson(geojson_path)
    peak_windows = load_peak_windows(peaks_path) if os.path.exists(peaks_path) else None
    hotspot_df = _geojson_to_df(geojson) if geojson else pd.DataFrame()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 8px 0 20px;">
                <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em;">
                    🚦 ParkIntel
                </div>
                <div style="font-size:0.72rem;color:#475569;margin-top:2px;">
                    Bengaluru Enforcement Analytics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Ingestion stats (Req 15.7).
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    rpt = json.load(f)
                dropped = rpt.get("total_dropped", 0)
                retained = rpt.get("rows_retained", 0)
                st.markdown(
                    f"""
                    <div class="sidebar-stat">
                        <span class="sidebar-stat-label">Rows retained</span>
                        <span class="sidebar-stat-value">{retained:,}</span>
                    </div>
                    <div class="sidebar-stat">
                        <span class="sidebar-stat-label">Rows dropped</span>
                        <span class="sidebar-stat-value">{dropped:,}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#475569;margin-bottom:12px;">Time Window</div>',
            unsafe_allow_html=True,
        )

        # Time-slider filters (Requirements 15.3, 15.4).
        filter_hour = st.checkbox("Filter by hour", value=False, key="sb_filter_hour")
        hour_sel: int | None = None
        if filter_hour:
            hour_sel = st.slider("Hour of day", 0, 23, 8, format="%d:00", key="sb_hour_slider")

        filter_dow = st.checkbox("Filter by day of week", value=False, key="sb_filter_dow")
        dow_sel: int | None = None
        if filter_dow:
            dow_sel = st.selectbox(
                "Day of week",
                list(range(7)),
                format_func=lambda i: _DOW_FULL[i],
                key="sb_dow_select",
            )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#475569;margin-bottom:12px;">Station Drilldown</div>',
            unsafe_allow_html=True,
        )


        all_stations: list[str] = []
        if zones_df is not None and "police_station" in zones_df.columns:
            all_stations = sorted(zones_df["police_station"].dropna().unique().tolist())
        selected_station: str | None = None
        if all_stations:
            selected_station = st.selectbox("Police station", ["\u2014 select \u2014"] + all_stations, key="sb_station_select")
            if selected_station == "\u2014 select \u2014":
                selected_station = None

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#475569;margin-bottom:12px;">ML Risk Filter</div>',
            unsafe_allow_html=True,
        )

        risk_filter: str | None = None
        if zones_df is not None and "risk_tier" in zones_df.columns:
            risk_filter = st.selectbox("Risk tier", ["All", "HIGH", "MEDIUM", "LOW"], key="sb_risk_select")
            if risk_filter == "All":
                risk_filter = None

    # \u2500\u2500 H3 fallback warning (Req 18.4) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if geojson:
        all_ids = [f["properties"].get("hotspot_id", "") for f in geojson.get("features", [])]
        if all_ids and all(hid.startswith("h3-") for hid in all_ids):
            st.warning(
                "\u26a0\ufe0f **No DBSCAN clusters found** \u2014 showing H3-cell heatmap fallback. "
                "Consider relaxing `eps_m` or `min_samples`."
            )

    # \u2500\u2500 Page header \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    st.markdown(
        """
        <div style="padding: 16px 0 8px;">
            <div class="page-title">Bengaluru Parking Intelligence</div>
            <div class="page-subtitle">AI-driven hotspot prioritisation &amp; enforcement analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # \u2500\u2500 Top KPI row \u2014 reactive to ALL filters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # Compute active_ids early so KPI row can use them
    active_ids_kpi = _active_hotspot_ids(peak_windows, hour_sel, dow_sel)
    time_filter_active_kpi = active_ids_kpi is not None

    if zones_df is not None and not zones_df.empty:
        kpi_df = _filter_df_by_ids(zones_df.copy(), active_ids_kpi, "hotspot_id")
        if risk_filter and "risk_tier" in kpi_df.columns:
            kpi_df = kpi_df[kpi_df["risk_tier"] == risk_filter]
        if selected_station and "police_station" in kpi_df.columns:
            kpi_df = kpi_df[kpi_df["police_station"] == selected_station]

        total_zones = len(kpi_df)
        total_all   = len(zones_df)
        n_high     = int((kpi_df["risk_tier"] == "HIGH").sum()) if "risk_tier" in kpi_df.columns else 0
        n_stations = kpi_df["police_station"].nunique() if "police_station" in kpi_df.columns else 0
        avg_score  = kpi_df["priority_score"].mean() if "priority_score" in kpi_df.columns and not kpi_df.empty else 0.0

        filter_active = bool(risk_filter or selected_station or time_filter_active_kpi)
        kpi_sub_zones = f"of {total_all} total" if filter_active else "detected zones"
        kpi_sub_high  = f"{n_high/total_zones*100:.0f}% of filtered" if total_zones > 0 else "\u2014"

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(_kpi("Filtered Hotspots" if filter_active else "Total Hotspots", f"{total_zones:,}", kpi_sub_zones, "kpi-blue"), unsafe_allow_html=True)
        c2.markdown(_kpi("HIGH Risk Zones", str(n_high), kpi_sub_high, "kpi-high"), unsafe_allow_html=True)
        c3.markdown(_kpi("Police Stations", str(n_stations), "in selection" if filter_active else "coverage areas"), unsafe_allow_html=True)
        c4.markdown(_kpi("Avg Priority Score", f"{avg_score:.1f}" if not kpi_df.empty else "\u2014", "filtered avg" if filter_active else "across all zones", "kpi-med"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_map, tab_zones, tab_ml, tab_forecast, tab_peaks, tab_drilldown = st.tabs([
        "🗺️  Live Map",
        "🏆  Priority Zones",
        "🤖  ML Analysis",
        "🔮  Forecast",
        "⏱️  Peak Windows",
        "📍  Station Drilldown",
    ])

    # ── Compute filtered data (used across all tabs) ─────────────────────────
    # Reuse the same active_ids already computed for the KPI row
    active_ids = active_ids_kpi
    time_filter_active = time_filter_active_kpi

    # Apply time filter to map and zones data
    filtered_hotspot_df = _filter_df_by_ids(hotspot_df, active_ids, "hotspot_id")
    filtered_zones_df   = _filter_df_by_ids(zones_df,   active_ids, "hotspot_id") if zones_df is not None else None

    # Apply risk-tier filter on top
    if filtered_zones_df is not None and risk_filter and "risk_tier" in filtered_zones_df.columns:
        filtered_zones_df = filtered_zones_df[filtered_zones_df["risk_tier"] == risk_filter]

    # ── Tab 1: Live Map ────────────────────────────────────────────────────────
    with tab_map:
        _section("🗺️", "Violation Heatmap", "pydeck / folium")
        if time_filter_active:
            parts = []
            if hour_sel is not None:
                parts.append(f"{hour_sel:02d}:00")
            if dow_sel is not None:
                parts.append(_DOW_FULL[dow_sel])
            n_filt = len(filtered_hotspot_df)
            n_total = len(hotspot_df)
            st.caption(
                f"🔍 Peak-window filter active: **{', '.join(parts)}** — "
                f"**{n_filt}** of {n_total} hotspots have recorded activity at this time"
            )
        _render_heatmap(filtered_hotspot_df)

        if filtered_hotspot_df.empty and time_filter_active:
            st.info("No hotspots have recorded peak activity at the selected hour / day.")

    # ── Tab 2: Priority Zones ─────────────────────────────────────────────────
    with tab_zones:
        _section("🏆", "Top Priority Zones", "global rank")

        # Apply ALL filters: time (via peak windows) + risk tier + station
        display_df = _filter_df_by_ids(zones_df, active_ids, "hotspot_id") if zones_df is not None else None
        if display_df is not None and risk_filter and "risk_tier" in display_df.columns:
            display_df = display_df[display_df["risk_tier"] == risk_filter]
        if display_df is not None and selected_station and "police_station" in display_df.columns:
            display_df = display_df[display_df["police_station"] == selected_station]

        # Active filter summary badge
        active_filters = []
        if time_filter_active:
            time_parts = []
            if hour_sel is not None:
                time_parts.append(f"{hour_sel:02d}:00")
            if dow_sel is not None:
                time_parts.append(_DOW_FULL[dow_sel])
            active_filters.append(f"Time: **{', '.join(time_parts)}**")
        if risk_filter:
            active_filters.append(f"Risk: **{risk_filter}**")
        if selected_station:
            active_filters.append(f"Station: **{selected_station}**")
        if active_filters:
            n_shown = len(display_df) if display_df is not None else 0
            st.caption(f"🔍 {' · '.join(active_filters)} — **{n_shown}** zone(s) shown")

        if display_df is not None and not display_df.empty:
            _render_top_n_table(display_df, n=20)
        elif display_df is not None:
            st.info("No priority zones match the selected filters.")

    # ── Tab 3: ML Analysis ────────────────────────────────────────────────────
    with tab_ml:
        _section("🤖", "Unsupervised ML Risk Analysis", "K-Means + IsolationForest")
        st.caption(
            "Two deterministic sklearn models trained on the hotspot feature matrix "
            "(impact, severity, proximity, concentration, log member count). "
            "Purely data-driven — no invented labels."
        )
        # Use the fully filtered zones (time + risk)
        ml_data = filtered_zones_df
        any_ml_filter = time_filter_active or bool(risk_filter)
        if any_ml_filter and ml_data is not None:
            parts = []
            if time_filter_active:
                if hour_sel is not None:
                    parts.append(f"{hour_sel:02d}:00")
                if dow_sel is not None:
                    parts.append(_DOW_FULL[dow_sel])
            if risk_filter:
                parts.append(risk_filter)
            st.caption(f"🔍 Filtered — {', '.join(parts)} — {len(ml_data) if ml_data is not None else 0} zone(s)")
        if ml_data is not None and not ml_data.empty:
            _render_ml_panel(ml_data)
        elif ml_data is not None:
            st.info("No zones match the selected filters.")
        else:
            st.info("Run the pipeline first to generate ML analysis data.")

    # ── Tab 4: Forecast ───────────────────────────────────────────────────────
    with tab_forecast:
        _section("🔮", "Predicted Hotspots — Next 7 Days", "LightGBM")
        _render_forecast_panel(artifacts_dir, n_days=7)

    # ── Tab 5: Peak Windows ───────────────────────────────────────────────────
    with tab_peaks:
        _section("⏱️", "Peak Activity Windows", "patrol scheduling")
        st.caption(
            "Recurring peak windows extracted from historical violation data. "
            "Use these to plan patrol schedules for maximum coverage."
        )
        _render_peak_windows_panel(peak_windows, zones_df)

    # ── Tab 6: Station Drilldown ──────────────────────────────────────────────
    with tab_drilldown:
        _section("📍", "Station Drilldown", "per-station analysis")
        if selected_station and zones_df is not None:
            st.markdown(f"**{selected_station}** — detailed zone breakdown")
            # Apply risk filter within drilldown too
            drilldown_df = zones_df
            if risk_filter and "risk_tier" in drilldown_df.columns:
                drilldown_df = drilldown_df[drilldown_df["risk_tier"] == risk_filter]
                if not drilldown_df.empty:
                    st.caption(f"🔍 Showing only **{risk_filter}** risk zones for this station")
            _render_station_drilldown(drilldown_df, selected_station)
        elif zones_df is not None:
            st.markdown(
                """
                <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                            border-radius:12px;padding:20px 24px;margin-top:12px;">
                    <div style="font-size:0.85rem;color:#818cf8;font-weight:600;margin-bottom:6px;">No station selected</div>
                    <div style="font-size:0.8rem;color:#64748b;">Use the <strong style="color:#94a3b8;">Station Drilldown</strong> 
                    dropdown in the sidebar to pick a police station and view its zone breakdown here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error("Priority zones data not loaded.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__" or "streamlit" in sys.modules:
    artifacts_dir = _parse_args()
    render_dashboard(artifacts_dir)
