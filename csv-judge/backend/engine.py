"""
CSV Comparison Engine
Implements hybrid scoring: row matching + cell-level comparison + column alignment.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import io
import re
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_value(val: Any, lowercase: bool = True) -> str:
    """Normalize a single cell value."""
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    if lowercase:
        s = s.lower()
    # Normalize multiple whitespace
    s = re.sub(r'\s+', ' ', s)
    return s


def try_numeric(val: str) -> Optional[float]:
    """Try to parse a string as a number."""
    try:
        return float(val.replace(',', ''))
    except (ValueError, AttributeError):
        return None


# ─── Cell Similarity ─────────────────────────────────────────────────────────

def cell_similarity(a: str, b: str, numeric_tolerance: float = 1e-6) -> float:
    """
    Returns similarity [0.0, 1.0] between two normalized cell strings.
    - Exact match → 1.0
    - Numeric closeness → tolerance-based
    - String similarity → SequenceMatcher ratio
    """
    if a == b:
        return 1.0

    # Both empty
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0

    # Numeric comparison
    na, nb = try_numeric(a), try_numeric(b)
    if na is not None and nb is not None:
        if na == 0 and nb == 0:
            return 1.0
        denom = max(abs(na), abs(nb), 1e-12)
        rel_diff = abs(na - nb) / denom
        if rel_diff <= numeric_tolerance:
            return 1.0
        # Decay: close numbers still get partial credit
        return max(0.0, 1.0 - min(rel_diff, 1.0))

    # String similarity (SequenceMatcher is O(n) average)
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return ratio


# ─── Row Similarity ───────────────────────────────────────────────────────────

def row_similarity(ref_row: List[str], sub_row: List[str]) -> float:
    """Average cell similarity across aligned cells."""
    if not ref_row and not sub_row:
        return 1.0
    n = max(len(ref_row), len(sub_row))
    total = 0.0
    for i in range(n):
        a = ref_row[i] if i < len(ref_row) else ""
        b = sub_row[i] if i < len(sub_row) else ""
        total += cell_similarity(a, b)
    return total / n


# ─── Column Matching ──────────────────────────────────────────────────────────

def match_columns(ref_cols: List[str], sub_cols: List[str]) -> Dict[int, int]:
    """
    Returns mapping: ref_col_index → sub_col_index
    First tries exact header match, then content-similarity fallback.
    """
    mapping: Dict[int, int] = {}
    used_sub = set()

    # Exact name match
    sub_lower = {c.strip().lower(): i for i, c in enumerate(sub_cols)}
    for ri, rc in enumerate(ref_cols):
        key = rc.strip().lower()
        if key in sub_lower:
            si = sub_lower[key]
            if si not in used_sub:
                mapping[ri] = si
                used_sub.add(si)

    # Similarity fallback for unmatched ref columns
    unmatched_ref = [ri for ri in range(len(ref_cols)) if ri not in mapping]
    unmatched_sub = [si for si in range(len(sub_cols)) if si not in used_sub]

    if unmatched_ref and unmatched_sub:
        sim_matrix = np.zeros((len(unmatched_ref), len(unmatched_sub)))
        for i, ri in enumerate(unmatched_ref):
            for j, si in enumerate(unmatched_sub):
                sim_matrix[i, j] = SequenceMatcher(
                    None, ref_cols[ri].lower(), sub_cols[si].lower()
                ).ratio()
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        for i, j in zip(row_ind, col_ind):
            if sim_matrix[i, j] > 0.4:  # threshold
                mapping[unmatched_ref[i]] = unmatched_sub[j]

    return mapping


# ─── Row Matching (Hungarian Algorithm) ──────────────────────────────────────

MAX_HUNGARIAN_SIZE = 500  # use greedy above this

def match_rows_hungarian(
    ref_rows: List[List[str]],
    sub_rows: List[List[str]],
) -> List[Tuple[int, int, float]]:
    """
    Optimal row matching using Hungarian algorithm (for small datasets)
    or greedy approximation for large datasets.
    Returns list of (ref_idx, sub_idx, similarity).
    """
    n, m = len(ref_rows), len(sub_rows)
    if n == 0 or m == 0:
        return []

    use_greedy = (n * m) > (MAX_HUNGARIAN_SIZE ** 2)

    if use_greedy:
        return _greedy_row_match(ref_rows, sub_rows)

    # Build cost matrix
    cost = np.zeros((n, m))
    for i, rr in enumerate(ref_rows):
        for j, sr in enumerate(sub_rows):
            cost[i, j] = 1.0 - row_similarity(rr, sr)

    row_ind, col_ind = linear_sum_assignment(cost)
    results = []
    for ri, si in zip(row_ind, col_ind):
        sim = 1.0 - cost[ri, si]
        results.append((ri, si, sim))
    return results


def _greedy_row_match(
    ref_rows: List[List[str]],
    sub_rows: List[List[str]],
) -> List[Tuple[int, int, float]]:
    """Greedy O(n log n) approximation using row hash pre-filtering."""
    results = []
    used_sub = set()

    # First pass: exact hash matches (O(n))
    ref_hashes = [tuple(r) for r in ref_rows]
    sub_hash_map: Dict[tuple, List[int]] = {}
    for si, sr in enumerate(sub_rows):
        key = tuple(sr)
        sub_hash_map.setdefault(key, []).append(si)

    unmatched_ref = []
    for ri, rh in enumerate(ref_hashes):
        if rh in sub_hash_map:
            candidates = [s for s in sub_hash_map[rh] if s not in used_sub]
            if candidates:
                si = candidates[0]
                used_sub.add(si)
                results.append((ri, si, 1.0))
                continue
        unmatched_ref.append(ri)

    unmatched_sub = [si for si in range(len(sub_rows)) if si not in used_sub]

    # Second pass: similarity on remaining
    for ri in unmatched_ref:
        best_sim, best_si = -1.0, -1
        for si in unmatched_sub:
            if si in used_sub:
                continue
            sim = row_similarity(ref_rows[ri], sub_rows[si])
            if sim > best_sim:
                best_sim, best_si = sim, si
        if best_si >= 0:
            used_sub.add(best_si)
            results.append((ri, best_si, best_sim))

    return results


# ─── Main Scoring Function ────────────────────────────────────────────────────

def _sanitize(obj: Any) -> Any:
    """
    Recursively convert numpy scalar types to plain Python types so
    FastAPI's JSON encoder (which does not know about numpy.int64,
    numpy.float64, etc.) can serialise the response without errors.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    # numpy integers  →  int
    if hasattr(obj, 'item'):          # catches np.int64, np.float64, np.bool_, …
        return obj.item()
    # plain numpy bool (edge-case on older numpy)
    if type(obj).__module__ == 'numpy':
        return obj.item()
    return obj

def parse_csv(content: bytes) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame, trying multiple encodings."""
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc, dtype=str)
            return df
        except Exception:
            continue
    raise ValueError("Unable to parse CSV file. Check encoding and format.")


def compare_csvs(
    ref_bytes: bytes,
    sub_bytes: bytes,
    lowercase: bool = True,
    numeric_tolerance: float = 1e-6,
    missing_row_penalty: float = 0.5,
    extra_row_penalty: float = 0.2,
) -> Dict[str, Any]:
    """
    Full comparison pipeline. Returns rich result dict.
    """
    ref_df = parse_csv(ref_bytes)
    sub_df = parse_csv(sub_bytes)

    # ── Column alignment ──
    ref_cols = list(ref_df.columns)
    sub_cols = list(sub_df.columns)
    col_mapping = match_columns(ref_cols, sub_cols)  # ref_idx → sub_idx

    # Reorder sub_df columns to match ref order
    aligned_sub_cols = []
    for ri in range(len(ref_cols)):
        if ri in col_mapping:
            aligned_sub_cols.append(sub_df.columns[col_mapping[ri]])
        else:
            aligned_sub_cols.append(None)  # missing column

    # ── Normalize all cells ──
    def normalize_df(df: pd.DataFrame, cols: List[str]) -> List[List[str]]:
        rows = []
        for _, row in df.iterrows():
            rows.append([normalize_value(row[c], lowercase) if c in df.columns else "" for c in cols])
        return rows

    ref_data = normalize_df(ref_df, ref_cols)
    sub_aligned_cols = [c for c in aligned_sub_cols if c is not None]
    # Build sub rows aligned to ref columns
    sub_data_aligned = []
    for _, row in sub_df.iterrows():
        sub_row = []
        for ri in range(len(ref_cols)):
            if ri in col_mapping:
                sc = sub_df.columns[col_mapping[ri]]
                sub_row.append(normalize_value(row[sc], lowercase))
            else:
                sub_row.append("")
        sub_data_aligned.append(sub_row)

    # ── Row matching ──
    n_ref = len(ref_data)
    n_sub = len(sub_data_aligned)
    matches = match_rows_hungarian(ref_data, sub_data_aligned)

    # ── Score computation ──
    total_ref_cells = n_ref * len(ref_cols)
    if total_ref_cells == 0:
        return _empty_result()

    matched_cell_score = 0.0
    matched_rows = 0
    mismatched_rows = 0
    row_details = []  # for diff view

    matched_ref_indices = set()
    matched_sub_indices = set()

    MATCH_THRESHOLD = 0.5

    for ri, si, row_sim in matches:
        matched_ref_indices.add(ri)
        matched_sub_indices.add(si)

        ref_row = ref_data[ri]
        sub_row = sub_data_aligned[si]
        cell_scores = []
        for ci in range(len(ref_cols)):
            a = ref_row[ci] if ci < len(ref_row) else ""
            b = sub_row[ci] if ci < len(sub_row) else ""
            cs = cell_similarity(a, b)
            cell_scores.append(cs)
            matched_cell_score += cs

        avg_row_score = np.mean(cell_scores) if cell_scores else 0.0
        if avg_row_score >= MATCH_THRESHOLD:
            matched_rows += 1
        else:
            mismatched_rows += 1

        row_details.append({
            "ref_idx": ri,
            "sub_idx": si,
            "score": round(avg_row_score * 100, 2),
            "cells": [
                {
                    "col": ref_cols[ci],
                    "ref": ref_data[ri][ci] if ci < len(ref_data[ri]) else "",
                    "sub": sub_data_aligned[si][ci] if ci < len(sub_data_aligned[si]) else "",
                    "score": round(cell_scores[ci] * 100, 2),
                }
                for ci in range(len(ref_cols))
            ]
        })

    # Unmatched ref rows (missing rows)
    missing_rows = n_ref - len(matched_ref_indices)
    extra_rows = n_sub - len(matched_sub_indices)

    # Penalty for missing rows: deduct fraction of their cell budget
    penalty = (missing_rows * len(ref_cols) * missing_row_penalty)
    # Extra rows add small penalty to total budget
    extra_penalty = extra_rows * len(ref_cols) * extra_row_penalty

    adjusted_total = total_ref_cells + extra_penalty
    raw_score = matched_cell_score - penalty
    raw_score = max(0.0, raw_score)

    final_score = min(100.0, (raw_score / adjusted_total) * 100)
    final_score = round(final_score, 2)

    # Column match info
    matched_cols = len(col_mapping)
    missing_cols = len(ref_cols) - matched_cols

    result = {
        "score": float(final_score),
        "matched_rows": int(matched_rows),
        "mismatched_rows": int(mismatched_rows),
        "missing_rows": int(missing_rows),
        "extra_rows": int(extra_rows),
        "total_ref_rows": int(n_ref),
        "total_sub_rows": int(n_sub),
        "total_ref_cols": int(len(ref_cols)),
        "matched_cols": int(matched_cols),
        "missing_cols": int(missing_cols),
        "row_details": row_details[:200],
        "ref_columns": ref_cols,
        "sub_columns": sub_cols,
    }
    return _sanitize(result)



def _empty_result() -> Dict[str, Any]:
    return {
        "score": 0.0,
        "matched_rows": 0,
        "mismatched_rows": 0,
        "missing_rows": 0,
        "extra_rows": 0,
        "total_ref_rows": 0,
        "total_sub_rows": 0,
        "total_ref_cols": 0,
        "matched_cols": 0,
        "missing_cols": 0,
        "row_details": [],
        "ref_columns": [],
        "sub_columns": [],
    }
