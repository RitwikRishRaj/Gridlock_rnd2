"""Full-dataset analysis & visualization for Parking Intelligence.

Runs the deterministic pipeline over the entire violation CSV and renders
evaluation plots into ``eval_output/``. This is NOT model training (the system
has no learned parameters) — it is an evaluation of the analytics pipeline's
behaviour at full scale.

Usage:
    python analysis.py
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parking_intelligence.ingest import Ingestor
from parking_intelligence.hotspots import HotspotBuilder
from parking_intelligence.impact import ImpactScorer
from parking_intelligence.forecast import PeakForecaster
from parking_intelligence.priority import PriorityRanker
from parking_intelligence.export import Exporter
from parking_intelligence.models import ImpactWeights, PriorityConfig

CSV = "jan to may police violation_anonymized791b166.csv"
OUT = "eval_output"
ARTIFACTS = "artifacts"

# DBSCAN params: at full scale use a slightly larger neighbourhood so clusters
# are meaningful enforcement zones rather than thousands of tiny ones.
EPS_M = 75.0
MIN_SAMPLES = 20
H3_RES = 9


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ARTIFACTS, exist_ok=True)
    t0 = time.time()

    # ---- Stage 1: Ingest full dataset -------------------------------
    banner("Stage 1/6  Ingesting full dataset")
    ing = Ingestor()
    df, report = ing.load_and_clean(CSV, chunksize=100_000)
    print(f"rows read={report.total_rows_read}  retained={report.rows_retained}", flush=True)
    print(f"dropped={report.dropped_by_reason}", flush=True)

    # ---- Stage 2: Hotspots ------------------------------------------
    banner("Stage 2/6  Building hotspots (DBSCAN + H3)")
    hb = HotspotBuilder()
    hotspots = hb.build_hotspots(df, h3_res=H3_RES, eps_m=EPS_M, min_samples=MIN_SAMPLES)
    used_dbscan = bool(hotspots) and any(h.hotspot_id.startswith("dbscan-") for h in hotspots)
    if not hotspots and not df.empty:
        hotspots = hb._hotspots_from_h3(hb.assign_h3(df, resolution=H3_RES))
        used_dbscan = False
    print(f"hotspots={len(hotspots)}  used_dbscan={used_dbscan}", flush=True)

    # ---- Stage 3: Impact --------------------------------------------
    banner("Stage 3/6  Scoring congestion impact")
    scorer = ImpactScorer()
    scored = scorer.score_impact(hotspots, df, ImpactWeights())

    # ---- Stage 4: Forecast ------------------------------------------
    banner("Stage 4/6  Building peak profiles")
    fc = PeakForecaster()
    profiles = fc.build_peak_profiles(df, hotspots)

    # ---- Stage 5: Priority ------------------------------------------
    banner("Stage 5/6  Ranking priority zones")
    cfg = PriorityConfig(as_of=pd.to_datetime(df["created_at"]).max().to_pydatetime())
    ranker = PriorityRanker()
    zones = ranker.rank_zones(scored, profiles, df, cfg, top_k_peaks=3)

    # ---- Stage 6: Export + Plots ------------------------------------
    banner("Stage 6/6  Exporting artifacts + rendering plots")
    Exporter().export_all(scored, zones, ARTIFACTS)

    impact_scores = np.array([s.impact_score for s in scored])
    priority_scores = np.array([z.priority_score for z in zones])
    member_counts = np.array([s.hotspot.member_count for s in scored])

    # 1. Impact score distribution
    plt.figure(figsize=(8, 5))
    plt.hist(impact_scores, bins=40, color="#4285f4", edgecolor="white")
    plt.title("Congestion Impact Score Distribution")
    plt.xlabel("Impact score (0-100)"); plt.ylabel("Hotspot count")
    plt.tight_layout(); plt.savefig(f"{OUT}/01_impact_distribution.png", dpi=120); plt.close()

    # 2. Priority score distribution
    plt.figure(figsize=(8, 5))
    plt.hist(priority_scores, bins=40, color="#34a853", edgecolor="white")
    plt.title("Priority Score Distribution")
    plt.xlabel("Priority score (0-100)"); plt.ylabel("Zone count")
    plt.tight_layout(); plt.savefig(f"{OUT}/02_priority_distribution.png", dpi=120); plt.close()

    # 3. Cluster size distribution (log y)
    plt.figure(figsize=(8, 5))
    plt.hist(member_counts, bins=50, color="#f9ab00", edgecolor="white")
    plt.yscale("log")
    plt.title("Hotspot Size Distribution (members per hotspot)")
    plt.xlabel("Member violations"); plt.ylabel("Hotspot count (log)")
    plt.tight_layout(); plt.savefig(f"{OUT}/03_cluster_sizes.png", dpi=120); plt.close()

    # 4. Hotspots per police station (top 15)
    zdf = pd.DataFrame([{"station": z.police_station or "(unknown)",
                         "priority": z.priority_score} for z in zones])
    top_st = zdf.groupby("station").size().sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    top_st[::-1].plot(kind="barh", color="#4285f4")
    plt.title("Top 15 Police Stations by Hotspot Count")
    plt.xlabel("Number of priority zones")
    plt.tight_layout(); plt.savefig(f"{OUT}/04_hotspots_per_station.png", dpi=120); plt.close()

    # 5. Top 20 priority zones
    top20 = sorted(zones, key=lambda z: z.global_rank)[:20]
    labels = [f"#{z.global_rank} {(z.police_station or '?')[:18]}" for z in top20]
    vals = [z.priority_score for z in top20]
    plt.figure(figsize=(9, 8))
    plt.barh(labels[::-1], vals[::-1], color="#34a853")
    plt.title("Top 20 Priority Enforcement Zones")
    plt.xlabel("Priority score")
    plt.tight_layout(); plt.savefig(f"{OUT}/05_top20_zones.png", dpi=120); plt.close()

    # 6. Geographic scatter colored by impact
    lat = np.array([s.hotspot.centroid_lat for s in scored])
    lon = np.array([s.hotspot.centroid_lon for s in scored])
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(lon, lat, c=impact_scores, cmap="YlOrRd", s=18, alpha=0.7)
    plt.colorbar(sc, label="Impact score")
    plt.title("Hotspot Geography (color = impact)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.tight_layout(); plt.savefig(f"{OUT}/06_geographic_scatter.png", dpi=120); plt.close()

    # 7. Aggregate peak matrix (hour x day-of-week) across ALL events
    ts = pd.to_datetime(df["created_at"], errors="coerce").dropna()
    agg = np.zeros((7, 24))
    for dow, hr in zip(ts.dt.weekday, ts.dt.hour):
        agg[dow, hr] += 1
    plt.figure(figsize=(11, 5))
    plt.imshow(agg, aspect="auto", cmap="viridis")
    plt.colorbar(label="Violation count")
    plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.xticks(range(0, 24, 2))
    plt.title("Temporal Pattern Matrix - All Violations (hour x day)")
    plt.xlabel("Hour of day"); plt.ylabel("Day of week")
    plt.tight_layout(); plt.savefig(f"{OUT}/07_temporal_matrix_all.png", dpi=120); plt.close()

    # 8. Peak matrix for the #1 priority hotspot
    top_zone = min(zones, key=lambda z: z.global_rank)
    prof = profiles.get(top_zone.hotspot_id)
    if prof is not None:
        M = np.array(prof.hour_dow_matrix)
        plt.figure(figsize=(11, 5))
        plt.imshow(M, aspect="auto", cmap="magma")
        plt.colorbar(label="Normalized intensity")
        plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        plt.xticks(range(0, 24, 2))
        plt.title(f"Peak Profile - #1 Zone ({top_zone.police_station})")
        plt.xlabel("Hour of day"); plt.ylabel("Day of week")
        plt.tight_layout(); plt.savefig(f"{OUT}/08_peak_matrix_top_zone.png", dpi=120); plt.close()

    # ---- Summary text -----------------------------------------------
    elapsed = time.time() - t0
    summary = [
        "PARKING INTELLIGENCE - FULL DATASET EVALUATION",
        "=" * 50,
        f"Rows read           : {report.total_rows_read:,}",
        f"Rows retained       : {report.rows_retained:,}",
        f"Rows dropped        : {report.total_dropped:,}  {dict(report.dropped_by_reason)}",
        f"Hotspots detected   : {len(hotspots):,}  (DBSCAN={used_dbscan})",
        f"Priority zones      : {len(zones):,}",
        f"Impact score  mean/max : {impact_scores.mean():.1f} / {impact_scores.max():.1f}",
        f"Priority score mean/max: {priority_scores.mean():.2f} / {priority_scores.max():.2f}",
        f"Largest hotspot     : {int(member_counts.max()):,} violations",
        f"Distinct stations   : {zdf['station'].nunique()}",
        f"Runtime             : {elapsed:.1f}s",
        "",
        "Top 5 enforcement zones:",
    ]
    for z in sorted(zones, key=lambda z: z.global_rank)[:5]:
        pk = z.peak_windows[0] if z.peak_windows else None
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        peak_s = f"{days[pk.day_of_week]} {pk.start_hour:02d}-{pk.end_hour:02d}h" if pk else "n/a"
        summary.append(
            f"  #{z.global_rank}  {z.police_station or '?':<22} "
            f"score={z.priority_score:5.2f}  peak={peak_s}"
        )
    text = "\n".join(summary)
    print("\n" + text, flush=True)
    with open(f"{OUT}/summary.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\nPlots written to ./{OUT}/", flush=True)


if __name__ == "__main__":
    main()
