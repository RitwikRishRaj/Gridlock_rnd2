"""Ingestion module for the Parking Intelligence pipeline.

This module provides both the low-level, pure parsing helpers and the high-level
``Ingestor`` class that orchestrates chunked CSV reading for the >50 MB violation
file.

Low-level helpers (Tasks 3.1 + 4.1):

* :func:`parse_json_array` — total (never-raising) parser for the JSON-array
  string fields ``violation_type`` and ``offence_code``.
* :func:`parse_timestamps` — converts ``*_datetime`` columns into tz-aware
  :class:`pandas.Timestamp` values expressed in Asia/Kolkata.
* :func:`validate_geo` — inclusive Bengaluru bbox filter with drop accounting.
* :func:`normalize_categoricals` — lowercase ``vehicle_type``, collapse
  whitespace in ``police_station``, uppercase ``violation_type`` tokens.

High-level orchestration (Task 5.1):

* :class:`Ingestor` / :meth:`Ingestor.load_and_clean` — streams the CSV in
  chunks, applies the full cleaning pipeline per chunk, dedupes by ``id``, and
  returns a ``(DataFrame, IngestionReport)`` pair.

This module hosts the low-level, pure parsing helpers used while cleaning the raw
violation CSV:

* :func:`parse_json_array` — total (never-raising) parser for the JSON-array
  string fields ``violation_type`` and ``offence_code``. It returns a ``list``
  for *every* input (``None``/``NaN``/empty/malformed all map to ``[]``).
* :func:`parse_timestamps` — converts the ``*_datetime`` columns into
  timezone-aware :class:`pandas.Timestamp` values expressed in Asia/Kolkata,
  interpreting offset-less values as Asia/Kolkata local time and leaving
  unparseable values as ``NaT``.

Higher-level orchestration (chunked reading, geo validation, categorical
normalization, and the duplicate/`created_at` drop policy) is implemented in
later ingestion tasks and intentionally lives outside this module so each piece
stays small and independently testable.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import pandas as pd

from .models import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    IngestionReport,
)

# Timezone all parsed datetimes are expressed in (Requirements 3.1, 3.4).
LOCAL_TZ: str = "Asia/Kolkata"

# Columns parsed by :func:`parse_timestamps`. ``created_datetime`` is required
# (rows without a parseable value are dropped in a later task); ``closed_datetime``
# is optional and tolerates nulls (Requirements 3.2, 3.3).
TIMESTAMP_COLUMNS: tuple[str, ...] = ("created_datetime", "closed_datetime")

# String tokens that should be treated as "no value" when they appear in a CSV
# cell (the source file uses the literal text ``NULL``).
_NULL_TEXT_TOKENS = frozenset({"", "null", "nan", "none", "nat", "na"})

# Matches a trailing timezone designator: ``Z``, ``+00``, ``+05:30``, ``-0800``…
_TZ_OFFSET_RE = re.compile(r"(?:Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2})?)\s*$")


# ---------------------------------------------------------------------------
# Null detection
# ---------------------------------------------------------------------------
def _is_null_scalar(value: Any) -> bool:
    """Return ``True`` for ``None``, ``NaN``/``NaT``, and null-like text.

    Only scalars are tested for ``NaN``; container inputs (e.g. an already-parsed
    ``list``) are never treated as null here.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, str):
        return value.strip().lower() in _NULL_TEXT_TOKENS
    # pandas NaN/NaT and numpy scalars; guard against array-like inputs.
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool,)) else False


# ---------------------------------------------------------------------------
# Token coercion helpers
# ---------------------------------------------------------------------------
def _normalize_violation_token(token: Any) -> str | None:
    """Trim and upper-case a single ``violation_type`` token.

    Returns ``None`` for null/blank tokens so callers can drop them.
    """
    if _is_null_scalar(token):
        return None
    text = str(token).strip().upper()
    return text or None


def _coerce_offence_code(token: Any) -> int | None:
    """Return ``token`` as an ``int`` if integer-convertible, else ``None``.

    Booleans are rejected (they are not meaningful offence codes), non-integral
    floats are rejected, and any value that cannot be parsed as an integer is
    discarded (Requirement 2.5).
    """
    if isinstance(token, bool):
        return None
    if isinstance(token, int):
        return token
    if isinstance(token, float):
        if math.isnan(token) or not token.is_integer():
            return None
        return int(token)
    if _is_null_scalar(token):
        return None
    try:
        return int(str(token).strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# JSON-array parsing (totality guaranteed)
# ---------------------------------------------------------------------------
def parse_json_array(raw: Any, kind: str = "raw") -> list:
    """Parse a JSON-array field into a Python list, never raising.

    Parameters
    ----------
    raw:
        The source cell value. May be a JSON-array string (e.g.
        ``'["WRONG PARKING"]'``), an already-parsed ``list``, ``None``, ``NaN``,
        an empty string, or any malformed value.
    kind:
        Controls per-element normalization:

        * ``"raw"`` (default) — elements are returned unchanged.
        * ``"violation_type"`` — each element is trimmed and upper-cased; blank
          elements are dropped (Requirement 2.4).
        * ``"offence_code"`` — only integer-convertible elements are kept and
          returned as ``int`` (Requirement 2.5).

    Returns
    -------
    list
        A list for *every* input. Source order is preserved. Null, NaN, empty,
        and malformed inputs yield ``[]`` and never raise (Requirements 2.2,
        2.3, 2.6).
    """
    # 1. Null-like scalars -> [].
    if _is_null_scalar(raw):
        return []

    # 2. Resolve the raw value into a Python list of elements.
    if isinstance(raw, (list, tuple)):
        elements: list = list(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return []
        if not isinstance(parsed, list):
            # Valid JSON, but not an array (e.g. an object or scalar).
            return []
        elements = parsed
    else:
        # Unknown scalar type that is not a recognized array source.
        return []

    # 3. Apply per-field normalization while preserving source order.
    if kind == "violation_type":
        normalized = (_normalize_violation_token(e) for e in elements)
        return [tok for tok in normalized if tok is not None]
    if kind == "offence_code":
        coerced = (_coerce_offence_code(e) for e in elements)
        return [code for code in coerced if code is not None]
    return list(elements)


def parse_violation_types(raw: Any) -> list[str]:
    """Parse a ``violation_type`` cell into trimmed, upper-cased tokens."""
    return parse_json_array(raw, kind="violation_type")


def parse_offence_codes(raw: Any) -> list[int]:
    """Parse an ``offence_code`` cell into integer-only offence codes."""
    return parse_json_array(raw, kind="offence_code")


# ---------------------------------------------------------------------------
# Timestamp parsing (tz-aware Asia/Kolkata)
# ---------------------------------------------------------------------------
def _normalize_null_strings(series: pd.Series) -> pd.Series:
    """Return ``series`` as trimmed strings with null-like text mapped to NA."""
    text = series.astype("string").str.strip()
    is_null = text.isna() | text.str.lower().isin(_NULL_TEXT_TOKENS)
    return text.mask(is_null, other=pd.NA)


def parse_timestamp_series(series: pd.Series, tz: str = LOCAL_TZ) -> pd.Series:
    """Convert a string series of datetimes to tz-aware values in ``tz``.

    Values carrying an explicit timezone offset are converted to ``tz`` (the
    underlying instant is preserved). Offset-less values are interpreted as
    ``tz`` local wall-clock time. Null and unparseable values become ``NaT``
    (Requirements 3.1, 3.3, 3.4).
    """
    text = _normalize_null_strings(series)
    has_tz = text.str.contains(_TZ_OFFSET_RE, na=False)

    # Offset-aware values: parse to UTC then convert to the target tz. Where the
    # value is offset-less this column holds a (wrong) UTC reading that is
    # discarded by the final ``where`` below.
    aware = pd.to_datetime(text, errors="coerce", utc=True).dt.tz_convert(tz)

    # Offset-less values: parse as naive wall-clock, then localize to the target
    # tz. Offset-bearing entries are masked to NA so they parse to NaT here.
    naive_text = text.mask(has_tz, other=pd.NA)
    naive = pd.to_datetime(naive_text, errors="coerce")
    if naive.dt.tz is None:
        naive = naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    else:  # pragma: no cover - defensive: a stray offset slipped through
        naive = naive.dt.tz_convert(tz)

    return aware.where(has_tz, naive)


def parse_timestamps(
    df: pd.DataFrame,
    columns: tuple[str, ...] = TIMESTAMP_COLUMNS,
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    """Convert the configured ``*_datetime`` columns to tz-aware datetimes.

    Each present column in ``columns`` is parsed with
    :func:`parse_timestamp_series`. Unparseable / null values are left as ``NaT``
    (dropping rows whose required ``created_datetime`` is ``NaT`` is handled by a
    later ingestion step). Columns absent from ``df`` are skipped. The input
    DataFrame is not mutated; a copy is returned.
    """
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = parse_timestamp_series(out[column], tz=tz)
    return out


# ---------------------------------------------------------------------------
# Geographic validation (Bengaluru bounding box)
# ---------------------------------------------------------------------------
def validate_geo(
    df: pd.DataFrame,
    report: IngestionReport | None = None,
    *,
    reason: str = "invalid_geo",
) -> pd.DataFrame:
    """Retain only rows with valid, in-bbox ``latitude``/``longitude``.

    A row is kept only when both coordinates are numeric, non-null, non-NaN and
    fall inside the inclusive Bengaluru bounding box
    (``lat in [LAT_MIN, LAT_MAX]``, ``lon in [LON_MIN, LON_MAX]``). Every other
    row — null, empty, non-numeric, NaN, or numerically out of range — is
    dropped (Requirements 4.1, 4.2, 4.4).

    The returned DataFrame's ``latitude``/``longitude`` columns are coerced to a
    numeric dtype so every retained row carries a real, non-NaN float. The input
    is not mutated; a copy is returned and ``len(output) <= len(input)`` always
    holds (Requirement 4.3).

    Parameters
    ----------
    df:
        Cleaned chunk/DataFrame containing ``latitude`` and ``longitude`` columns.
    report:
        Optional :class:`~parking_intelligence.models.IngestionReport`. When
        supplied, the number of rows dropped here is *added* to
        ``report.dropped_by_reason[reason]`` (the key is created and set to ``0``
        when nothing is dropped, satisfying Requirement 4.5).
    reason:
        Key under which the dropped-row count is recorded. Defaults to
        ``"invalid_geo"``.
    """
    out = df.copy()

    # Coerce to numeric; any non-numeric / empty / null value becomes NaN.
    lat = pd.to_numeric(out.get("latitude"), errors="coerce")
    lon = pd.to_numeric(out.get("longitude"), errors="coerce")

    mask = (
        lat.notna()
        & lon.notna()
        & (lat >= LAT_MIN)
        & (lat <= LAT_MAX)
        & (lon >= LON_MIN)
        & (lon <= LON_MAX)
    )

    # Write back the coerced numeric columns so retained rows are real floats.
    out["latitude"] = lat
    out["longitude"] = lon

    retained = out[mask].copy()
    dropped = int(len(out) - len(retained))

    if report is not None:
        report.dropped_by_reason[reason] = (
            report.dropped_by_reason.get(reason, 0) + dropped
        )

    return retained


# ---------------------------------------------------------------------------
# Categorical normalization
# ---------------------------------------------------------------------------
# Matches any run of one or more whitespace characters (spaces, tabs, newlines).
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def _normalize_vehicle_type(value: Any) -> str:
    """Trim and lower-case ``vehicle_type``; null/empty -> ``""`` (Req 5.1, 5.4)."""
    if _is_null_scalar(value):
        return ""
    return str(value).strip().lower()


def _normalize_police_station(value: Any) -> str:
    """Trim ``police_station`` and collapse internal whitespace to single spaces.

    Null/empty values map to ``""`` without raising (Requirements 5.2, 5.4).
    """
    if _is_null_scalar(value):
        return ""
    return _WHITESPACE_RUN_RE.sub(" ", str(value).strip())


def _normalize_violation_types_cell(value: Any) -> list[str]:
    """Normalize a ``violation_type`` cell into trimmed, upper-cased tokens.

    Accepts both already-parsed lists/tuples of tokens and raw JSON-array
    strings. Each token is trimmed and upper-cased so that two tokens differing
    only in case or surrounding whitespace map to an identical token; blank and
    null tokens are dropped. Null/empty cells map to ``[]`` (Requirements 5.3,
    5.4).
    """
    if isinstance(value, (list, tuple)):
        normalized = (_normalize_violation_token(tok) for tok in value)
        return [tok for tok in normalized if tok is not None]
    if _is_null_scalar(value):
        return []
    # Raw scalar (e.g. a JSON-array string): reuse the totality-guaranteed parser.
    return parse_violation_types(value)


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the categorical fields of a cleaned chunk.

    * ``vehicle_type`` — trimmed and lower-cased.
    * ``police_station`` — trimmed with internal whitespace collapsed to single
      spaces.
    * ``violation_type`` — each token trimmed and upper-cased so case/whitespace
      equivalents collapse to one token.

    Null, NaN, or post-trim empty values become an empty value (``""`` for the
    string fields, ``[]`` for ``violation_type``) and never raise. Columns absent
    from ``df`` are skipped. The input DataFrame is not mutated; a copy is
    returned (Requirements 5.1, 5.2, 5.3, 5.4).
    """
    out = df.copy()
    if "vehicle_type" in out.columns:
        out["vehicle_type"] = out["vehicle_type"].map(_normalize_vehicle_type)
    if "police_station" in out.columns:
        out["police_station"] = out["police_station"].map(_normalize_police_station)
    if "violation_type" in out.columns:
        out["violation_type"] = out["violation_type"].map(
            _normalize_violation_types_cell
        )
    return out


# ---------------------------------------------------------------------------
# Required source columns (Requirement 1.7)
# ---------------------------------------------------------------------------
# Every CSV ingested must contain at least these columns for the pipeline to work.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "latitude",
    "longitude",
    "created_datetime",
)

# Default chunk size when none is provided (Requirement 1.3).
_DEFAULT_CHUNKSIZE: int = 50_000


# ---------------------------------------------------------------------------
# Task 5.1: Ingestor — chunked CSV reader / cleaner
# ---------------------------------------------------------------------------
class Ingestor:
    """High-level ingestion orchestrator for the Parking Intelligence pipeline.

    :meth:`load_and_clean` streams the CSV in bounded chunks, applies the full
    cleaning pipeline (JSON-array parsing → timestamps → geo validation →
    categorical normalization) to each chunk, drops rows with an unparseable
    ``created_datetime``, concatenates the results, deduplicates by ``id``, and
    returns the combined DataFrame together with an :class:`IngestionReport`
    that accounts for every dropped row.

    Requirements: 1.1–1.8, 3.2, 18.1, 18.5, 18.6
    """

    def load_and_clean(
        self,
        csv_path: str,
        *,
        chunksize: int = _DEFAULT_CHUNKSIZE,
    ) -> tuple["pd.DataFrame", IngestionReport]:
        """Read *csv_path* in chunks, clean each chunk, and return ``(df, report)``.

        Parameters
        ----------
        csv_path:
            Path to the violation CSV file.  Must be a readable file containing
            all columns listed in ``REQUIRED_COLUMNS``.
        chunksize:
            Number of rows to read per chunk.  Must be a positive integer ≥ 1
            (Requirement 1.2).  Defaults to 50 000 (Requirement 1.3).

        Returns
        -------
        tuple[pd.DataFrame, IngestionReport]
            ``df`` — the cleaned, deduplicated DataFrame with validated
            coordinates and tz-aware timestamps.  ``report`` — a breakdown of
            every row that was dropped and why.

        Raises
        ------
        ValueError
            * If *chunksize* is not a positive integer (Requirement 1.8).
            * If *csv_path* does not exist / is not readable (Requirement 1.7).
            * If the CSV does not contain all required columns (Requirement 1.7).
        """
        # --- Validate chunksize (Requirements 1.2, 1.8) ---
        if not isinstance(chunksize, int) or isinstance(chunksize, bool):
            raise ValueError(
                f"chunksize must be a positive integer ≥ 1, got {chunksize!r}"
            )
        if chunksize < 1:
            raise ValueError(
                f"chunksize must be a positive integer ≥ 1, got {chunksize!r}"
            )

        # --- Validate path (Requirement 1.7) ---
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV path does not exist: {csv_path!r}")
        if not os.path.isfile(csv_path):
            raise ValueError(f"CSV path is not a file: {csv_path!r}")

        # Peek at the header to validate required columns before streaming.
        try:
            header_df = pd.read_csv(csv_path, nrows=0)
        except Exception as exc:
            raise ValueError(
                f"Cannot read CSV at {csv_path!r}: {exc}"
            ) from exc

        missing = [c for c in REQUIRED_COLUMNS if c not in header_df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}"
            )

        report = IngestionReport()
        cleaned_chunks: list[pd.DataFrame] = []
        total_rows_read = 0

        # Stream the file one chunk at a time (Requirement 1.4).
        reader = pd.read_csv(
            csv_path,
            chunksize=chunksize,
            low_memory=False,
        )

        for chunk in reader:
            chunk_len = len(chunk)
            total_rows_read += chunk_len

            # Per-row guard: if cleaning a single row raises unexpectedly,
            # drop that row and record the failure (Requirement 18.6).
            chunk = self._clean_chunk(chunk, report)
            cleaned_chunks.append(chunk)

        # Concatenate all cleaned chunks.
        if cleaned_chunks:
            df = pd.concat(cleaned_chunks, ignore_index=True)
        else:
            # All rows were dropped — return an empty DataFrame with expected columns.
            df = pd.DataFrame(columns=list(header_df.columns))

        # Deduplicate by id, keeping first occurrence (Requirements 1.5, 1.6).
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["id"], keep="first")
        deduped = before_dedup - len(df)
        if deduped > 0:
            report.dropped_by_reason["duplicate_id"] = (
                report.dropped_by_reason.get("duplicate_id", 0) + deduped
            )

        # Finalize the report.
        object.__setattr__(report, "total_rows_read", total_rows_read)
        object.__setattr__(report, "rows_retained", len(df))

        return df, report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clean_chunk(
        self,
        chunk: "pd.DataFrame",
        report: IngestionReport,
    ) -> "pd.DataFrame":
        """Apply the full per-chunk cleaning pipeline.

        Steps (in order):
        1. Parse JSON-array fields.
        2. Parse timestamps (``created_datetime``, ``closed_datetime``).
        3. Drop rows whose ``created_datetime`` is unparseable (NaT).
        4. Geo-validate and bbox-filter.
        5. Normalize categorical fields.

        Any unexpected per-row exception causes that row to be dropped and
        recorded (Requirement 18.6).  The chunk is handled at the column level
        wherever possible so individual bad rows don't abort the whole chunk.
        """
        # 1. Parse JSON-array fields.
        chunk = self._apply_json_parsing(chunk)

        # 2. Parse timestamps.
        chunk = parse_timestamps(chunk)

        # 3. Drop rows with unparseable created_datetime (Requirement 3.2, 18.5).
        if "created_datetime" in chunk.columns:
            nat_mask = chunk["created_datetime"].isna()
            n_dropped = int(nat_mask.sum())
            if n_dropped:
                report.dropped_by_reason["unparseable_created_at"] = (
                    report.dropped_by_reason.get("unparseable_created_at", 0)
                    + n_dropped
                )
                chunk = chunk[~nat_mask].copy()

        # Rename parsed timestamp column to the canonical name used downstream.
        if "created_datetime" in chunk.columns and "created_at" not in chunk.columns:
            chunk = chunk.rename(columns={
                "created_datetime": "created_at",
                "closed_datetime": "closed_at",
            }, errors="ignore")

        # 4. Geo-validate (Requirement 4.1–4.5).
        chunk = validate_geo(chunk, report)

        # 5. Normalize categoricals (Requirements 5.1–5.4).
        chunk = normalize_categoricals(chunk)

        return chunk

    @staticmethod
    def _apply_json_parsing(chunk: "pd.DataFrame") -> "pd.DataFrame":
        """Parse ``violation_type`` and ``offence_code`` columns in-place."""
        out = chunk.copy()
        if "violation_type" in out.columns:
            out["violation_type"] = out["violation_type"].map(
                parse_violation_types
            )
        if "offence_code" in out.columns:
            out["offence_code"] = out["offence_code"].map(
                parse_offence_codes
            )
        return out
