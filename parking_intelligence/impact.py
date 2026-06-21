"""Congestion-impact scoring for the Parking Intelligence pipeline.

Implements Tasks 8.1 and 8.2:

* Component functions:
  - :func:`severity_weight` — max violation-type severity in ``[0, 1]``
    (Requirements 8.1, 8.2).
  - :func:`proximity_factor` — junction/road-crossing fraction in ``[0, 1]``
    (Requirement 8.3).
  - :func:`temporal_concentration` — ``1 − normalised entropy`` of the
    hour-of-day distribution, in ``[0, 1]`` (Requirement 8.4).

* :class:`ImpactScorer` with:
  - :meth:`score_impact` — combines components into a ``0–100`` normalised
    score with an explainability breakdown (Requirements 8.5–8.10).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .models import ImpactWeights, ScoredHotspot

if TYPE_CHECKING:
    from .models import Hotspot

# ---------------------------------------------------------------------------
# Severity weight table (configurable; token → weight in [0, 1])
# ---------------------------------------------------------------------------
# Higher weight = more severe impact on traffic flow.
# Tokens are matched case-insensitively after upper-casing (the pipeline
# normalises violation_type tokens to uppercase via parse_violation_types).
SEVERITY_TABLE: dict[str, float] = {
    "PARKING NEAR ROAD CROSSING": 1.0,
    "WRONG PARKING ON CARRIAGEWAY": 0.9,
    "NO STOPPING ON ROAD": 0.85,
    "OBSTRUCTION ON ROAD": 0.8,
    "PARKING ON FOOTPATH": 0.7,
    "WRONG PARKING": 0.6,
    "NO PARKING ZONE": 0.5,
    "PARKING WITHOUT PERMIT": 0.4,
    "WRONG SIDE PARKING": 0.45,
    "PARKING NEAR JUNCTION": 0.75,
    "PARKING NEAR BUS STOP": 0.55,
    "PARKING NEAR SCHOOL": 0.5,
    "PARKING NEAR HOSPITAL": 0.5,
    "DOUBLE PARKING": 0.7,
}

# Default floor applied when a hotspot has no recognised violation tokens
# (Requirement 8.2, 18.1).
SEVERITY_FLOOR: float = 0.3

# Tokens indicating a road-crossing / junction context (for proximity, Req 8.3).
_CROSSING_TOKENS: frozenset[str] = frozenset({
    "PARKING NEAR ROAD CROSSING",
    "PARKING NEAR JUNCTION",
    "NO STOPPING ON ROAD",
    "WRONG PARKING ON CARRIAGEWAY",
})


# ---------------------------------------------------------------------------
# Task 8.1: Component functions
# ---------------------------------------------------------------------------
def severity_weight(
    violation_types: list[str],
    *,
    floor: float = SEVERITY_FLOOR,
    table: dict[str, float] = SEVERITY_TABLE,
) -> float:
    """Return the maximum severity weight among *violation_types* tokens.

    Parameters
    ----------
    violation_types:
        Normalised (upper-cased, trimmed) violation token list for a hotspot.
    floor:
        Value returned when *violation_types* is empty or contains no
        recognised token (Requirement 8.2).  Must be in ``[0, 1]``.
    table:
        Mapping of token → weight.  Tokens absent from the table are ignored.

    Returns
    -------
    float
        Maximum weight in ``[0, 1]``; *floor* if no tokens matched.
    """
    if not violation_types:
        return floor
    weights = [table.get(tok, 0.0) for tok in violation_types]
    max_w = max(weights)
    return max_w if max_w > 0 else floor


def proximity_factor(
    hotspot: "Hotspot",
    events: pd.DataFrame,
    *,
    crossing_tokens: frozenset[str] = _CROSSING_TOKENS,
) -> float:
    """Fraction of events tied to a junction / road-crossing context.

    An event is counted if its ``junction_name`` is non-null **or** if any of
    its ``violation_type`` tokens appears in *crossing_tokens* (Requirement 8.3).

    Returns
    -------
    float
        Value in ``[0, 1]``; ``0.0`` if *events* is empty.
    """
    if events.empty:
        return 0.0

    n = len(events)
    junction_hit = (
        events["junction_name"].notna() & (events["junction_name"].str.strip() != "")
        if "junction_name" in events.columns
        else pd.Series(False, index=events.index)
    )

    if "violation_type" in events.columns:
        def _has_crossing(tokens: object) -> bool:
            if not isinstance(tokens, list):
                return False
            return any(tok in crossing_tokens for tok in tokens)
        token_hit = events["violation_type"].map(_has_crossing)
    else:
        token_hit = pd.Series(False, index=events.index)

    near_junction = junction_hit | token_hit
    return float(near_junction.sum()) / n


def temporal_concentration(events: pd.DataFrame) -> float:
    """``1 − normalised entropy`` of the hour-of-day distribution.

    A hotspot with all events in a single hour → ``1.0`` (fully concentrated).
    A hotspot with events uniformly spread across all 24 hours → ``0.0``.
    Intermediate values lie in ``[0, 1]`` (Requirement 8.4).

    Uses ``created_at`` (renamed from ``created_datetime`` by the ingestor) or
    falls back to ``created_datetime`` if present.
    """
    if events.empty:
        return 0.0

    ts_col = "created_at" if "created_at" in events.columns else "created_datetime"
    if ts_col not in events.columns:
        return 0.0

    hours = pd.to_datetime(events[ts_col], errors="coerce").dt.hour.dropna()
    if hours.empty:
        return 0.0

    counts = np.zeros(24, dtype=float)
    for h in hours:
        counts[int(h)] += 1

    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    # Shannon entropy (bits), ignoring zero-probability bins.
    nonzero = probs[probs > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero)))

    max_entropy = math.log2(24)  # uniform over 24 bins
    if max_entropy == 0:
        return 1.0  # only 1 bin possible

    conc = 1.0 - (entropy / max_entropy)
    return float(np.clip(conc, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Task 8.2: ImpactScorer
# ---------------------------------------------------------------------------
class ImpactScorer:
    """Compute per-hotspot congestion-impact scores.

    All scoring is deterministic and offline (Requirement 16.1).
    """

    # Expose component functions as methods for testability.
    severity_weight = staticmethod(severity_weight)
    proximity_factor = staticmethod(proximity_factor)
    temporal_concentration = staticmethod(temporal_concentration)

    def score_impact(
        self,
        hotspots: list["Hotspot"],
        df: pd.DataFrame,
        weights: ImpactWeights,
        *,
        live_traffic_signal: dict[str, float] | None = None,
    ) -> list[ScoredHotspot]:
        """Score each hotspot and return a list of :class:`ScoredHotspot`.

        Parameters
        ----------
        hotspots:
            Non-noise hotspot records from :class:`~parking_intelligence.hotspots.HotspotBuilder`.
        df:
            Cleaned, deduplicated DataFrame (provides per-event attributes).
        weights:
            :class:`~parking_intelligence.models.ImpactWeights` whose three
            weights must each lie in ``[0, 1]`` and sum to ``1.0 ± 0.001``
            (Requirements 8.5, 8.6).
        live_traffic_signal:
            Optional dict mapping ``hotspot_id → extra_boost`` in ``[0, 1]``.
            When provided, the raw score is augmented before normalisation,
            keeping the final score within ``[0, 100]`` (Requirement 17.1).

        Returns
        -------
        list[ScoredHotspot]
            Scored hotspots with ``impact_score ∈ [0, 100]`` and an
            explainability ``breakdown`` dict (Requirements 8.7–8.10).

        Raises
        ------
        ValueError
            If any weight is outside ``[0, 1]`` or the weights don't sum to
            ``1.0 ± 0.001`` (Requirements 8.5, 8.6).
        """
        # --- Validate weights (Requirements 8.5, 8.6) ---
        _validate_weights(weights)

        if not hotspots:
            return []

        # Build an id-indexed lookup for fast per-hotspot event retrieval.
        if "id" in df.columns:
            df_indexed = df.set_index("id", drop=False)
        else:
            df_indexed = df

        raw_scores: list[tuple[Hotspot, float, float, float, float]] = []

        for hs in hotspots:
            # Gather member events.
            if "id" in df.columns and hs.member_ids:
                events = df[df["id"].isin(hs.member_ids)]
            elif "id" in df.columns:
                events = df.iloc[0:0]  # empty
            else:
                events = df

            # Flatten violation types across all events.
            all_tokens: list[str] = []
            if "violation_type" in events.columns:
                for cell in events["violation_type"]:
                    if isinstance(cell, list):
                        all_tokens.extend(cell)

            sev = severity_weight(all_tokens)
            prox = proximity_factor(hs, events)
            conc = temporal_concentration(events)

            raw = (
                weights.w_severity * sev
                + weights.w_proximity * prox
                + weights.w_concentration * conc
            )

            # Optional live-traffic augmentation (Requirement 17.1).
            if live_traffic_signal and hs.hotspot_id in live_traffic_signal:
                boost = float(np.clip(live_traffic_signal[hs.hotspot_id], 0.0, 1.0))
                # Blend: augmented_raw = raw + (1 - raw) * boost  →  always ≤ 1.
                raw = raw + (1.0 - raw) * boost

            raw_scores.append((hs, raw, sev, prox, conc))

        # Normalise so the maximum-raw hotspot scores exactly 100
        # (Requirements 8.8, 8.9).
        max_raw = max(r for _, r, *_ in raw_scores) if raw_scores else 0.0
        scale = 100.0 / max_raw if max_raw > 0.0 else 0.0

        scored: list[ScoredHotspot] = []
        for hs, raw, sev, prox, conc in raw_scores:
            impact = float(np.clip(raw * scale, 0.0, 100.0))
            scored.append(
                ScoredHotspot(
                    hotspot=hs,
                    impact_score=impact,
                    severity_component=sev,
                    proximity_component=prox,
                    concentration_component=conc,
                    breakdown={
                        "severity": sev,
                        "proximity": prox,
                        "concentration": conc,
                    },
                )
            )
        return scored


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def _validate_weights(weights: ImpactWeights) -> None:
    """Raise ``ValueError`` if any weight is out of ``[0, 1]`` or they don't sum to 1."""
    for name, val in [
        ("w_severity", weights.w_severity),
        ("w_proximity", weights.w_proximity),
        ("w_concentration", weights.w_concentration),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(
                f"Impact weight {name}={val!r} must lie in [0, 1]"
            )
    total = weights.w_severity + weights.w_proximity + weights.w_concentration
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Impact weights must sum to 1.0 ± 0.001, got {total:.6f}"
        )
