# CSV Online Judge

A full-stack web application for comparing CSV files with a sophisticated hybrid scoring engine. Upload a reference CSV (ground truth) and score any number of submissions against it — with support for shuffled rows, shuffled columns, partial matches, numeric tolerance, and rich diff output.

---

## Features

- **Hybrid similarity scoring** (0–100) with cell-level breakdown
- **Row order independence** — Hungarian algorithm for optimal matching
- **Column order independence** — header-name matching with similarity fallback
- **Numeric tolerance** — configurable relative-difference scoring
- **String similarity** — SequenceMatcher partial credit
- **Diff viewer** — cell-level reference vs submission comparison
- **Leaderboard** — ranked submission history per session
- **Multi-problem support** — multiple reference CSVs per session
- **10 MB file support** — memory-efficient streaming
- **Dockerized** — one-command deployment

---

## Quick Start

### Option A: Docker (recommended)

```bash
git clone <repo>
cd csv-judge
docker-compose up --build
```

Open [http://localhost:8000](http://localhost:8000).

---

### Option B: Local Python

**Requirements**: Python 3.9+

```bash
cd csv-judge/backend
pip install -r requirements.txt
python main.py
```

Open [http://localhost:8000](http://localhost:8000).

---

### Option C: Development (hot reload)

```bash
cd csv-judge/backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The frontend is served statically from `../frontend/index.html`.

---

## Project Structure

```
csv-judge/
├── backend/
│   ├── main.py          # FastAPI app + static file serving
│   ├── api.py           # REST endpoints
│   ├── engine.py        # CSV comparison engine (core logic)
│   ├── storage.py       # Thread-safe in-memory session store
│   └── requirements.txt
├── frontend/
│   └── index.html       # Single-file SPA (no build step)
├── tests/
│   └── test_engine.py   # Pytest test suite
├── samples/
│   ├── reference.csv          # Example reference
│   ├── submit_perfect.csv     # Score = 100
│   ├── submit_shuffled.csv    # Shuffled rows+cols → high score
│   └── submit_partial.csv     # Partial match → ~65-80
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## API Reference

All endpoints are prefixed with `/api`.

### `POST /api/upload-reference`

Upload the ground truth CSV.

| Field | Type | Description |
|---|---|---|
| `file` | file | `.csv` file (multipart) |
| `session_id` | string (optional) | Resume existing session |
| `problem_name` | string (optional) | Problem identifier (default: `"default"`) |

**Response:**
```json
{
  "session_id": "abc123",
  "problem_name": "default",
  "filename": "ground_truth.csv",
  "rows": 100,
  "columns": ["id", "name", "score"],
  "message": "Reference CSV uploaded successfully."
}
```

---

### `POST /api/submit`

Submit a CSV for scoring.

| Field | Type | Description |
|---|---|---|
| `file` | file | `.csv` submission file |
| `session_id` | string | Session from reference upload |
| `submitter_name` | string (optional) | Display name |
| `problem_name` | string (optional) | Must match reference problem |
| `lowercase` | bool (optional) | Case-insensitive matching (default: `true`) |
| `numeric_tolerance` | float (optional) | Relative tolerance for floats (default: `1e-6`) |

**Response:**
```json
{
  "submission_id": "a1b2c3d4",
  "score": 87.35,
  "matched_rows": 18,
  "mismatched_rows": 2,
  "missing_rows": 0,
  "extra_rows": 1,
  "total_ref_rows": 20,
  "total_sub_rows": 21,
  "total_ref_cols": 5,
  "matched_cols": 5,
  "missing_cols": 0,
  "processing_time_s": 0.042,
  "row_details": [ ... ]
}
```

---

### `GET /api/results`

Retrieve leaderboard and all submissions.

| Param | Description |
|---|---|
| `session_id` | Your session ID |
| `problem_name` | Problem to query (default: `"default"`) |

---

### `DELETE /api/reset`

Clear reference and submissions for a session/problem.

---

## How Scoring Works

### Step 1 — Column Alignment

Before any row comparison, columns are aligned between reference and submission:

1. **Exact name match** — column headers are lowercased and trimmed, then matched by name.
2. **Similarity fallback** — unmatched columns are paired using `SequenceMatcher` similarity + Hungarian algorithm. Pairs below 40% similarity are left unmatched.

This means `Name`, `name`, and `  NAME  ` all match correctly, and column order doesn't matter.

---

### Step 2 — Row Matching

Rows are matched optimally across the two files regardless of order:

- **Small datasets (≤ 500×500 cells):** Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) finds the globally optimal assignment minimizing total dissimilarity.
- **Large datasets:** Two-phase greedy approach:
  1. **Hash pass** — exact row matches found in O(n) using dict lookup.
  2. **Similarity pass** — remaining rows matched greedily by similarity.

---

### Step 3 — Cell-Level Scoring

Each matched cell pair `(ref_value, sub_value)` is scored `[0.0, 1.0]`:

| Case | Score |
|---|---|
| Both empty | 1.0 |
| Exact string match (after normalization) | 1.0 |
| Numeric exact match | 1.0 |
| Numeric within relative tolerance | 1.0 |
| Numeric close (relative diff < 1.0) | `1 - rel_diff` |
| String partial match | `SequenceMatcher.ratio()` |
| One side empty, other not | 0.0 |

**Normalization** (configurable):
- Whitespace trimmed
- Optional lowercase
- Numeric strings parsed as floats (commas stripped)

---

### Step 4 — Final Score Calculation

```
matched_score  = Σ cell_similarity(ref[i][j], sub[match(i)][j])
penalty        = missing_rows × n_cols × 0.5
extra_penalty  = extra_rows  × n_cols × 0.2
adjusted_total = (n_ref_rows × n_cols) + extra_penalty

Score = clamp(0, (matched_score - penalty) / adjusted_total × 100, 100)
```

**Penalties:**
- **Missing rows** (in ref but not in submission): 50% cell budget deducted
- **Extra rows** (in submission but not in ref): 20% cell budget added to denominator

---

## Test Cases

Run the test suite:

```bash
cd csv-judge
python -m pytest tests/ -v
# or without pytest:
python3 tests/test_engine.py
```

| Test | Expected Score |
|---|---|
| Identical CSV | 100.00 |
| Shuffled rows | ≥ 95.00 |
| Shuffled columns | ≥ 95.00 |
| Completely different values | < 20.00 |
| Empty submission | 0.00 |
| Whitespace normalization | 100.00 |
| Case-insensitive matching | 100.00 |
| Partial match (1 wrong cell / 6) | 50–99 |
| Missing rows | 0–99 (penalized) |
| Numeric tolerance (1e-7 diff) | 100.00 |

---

## Sample Files

The `samples/` directory contains ready-to-use CSVs:

| File | Purpose |
|---|---|
| `reference.csv` | Upload this as reference first |
| `submit_perfect.csv` | Should score 100.0 |
| `submit_shuffled.csv` | Rows + columns shuffled → should score ≥ 95 |
| `submit_partial.csv` | Some errors + missing row → intermediate score |

---

## Configuration

| Setting | Default | Description |
|---|---|---|
| `lowercase` | `true` | Normalize text to lowercase before comparison |
| `numeric_tolerance` | `1e-6` | Relative tolerance for float comparisons |
| `missing_row_penalty` | `0.5` | Fraction of cell budget deducted per missing row |
| `extra_row_penalty` | `0.2` | Extra denominator weight per extra row |
| `MAX_FILE_SIZE` | 10 MB | Maximum upload size (backend) |
| `MAX_HUNGARIAN_SIZE` | 500 | Switch to greedy above this row count |

---

## Architecture

```
Browser (SPA)
    │  multipart/form-data
    ▼
FastAPI (main.py + api.py)
    │  bytes
    ▼
engine.py
    ├── parse_csv()         — pandas, multi-encoding
    ├── match_columns()     — name match + Hungarian similarity
    ├── match_rows_*()      — Hungarian or hash+greedy
    ├── cell_similarity()   — exact / numeric / string
    └── compare_csvs()      — orchestrates full pipeline
    │
    ▼
storage.py (SessionStore)
    — thread-safe in-memory dict
    — keyed by session_id + problem_name
```

---

## Limitations & Future Work

- **Authentication** — currently session-based via UUID; add JWT for multi-user
- **Persistence** — in-memory store resets on restart; swap `SessionStore` for Redis or SQLite
- **Weighted columns** — scoring can be extended to weight important columns higher
- **Streaming large files** — chunked upload for files > 10 MB
- **Async comparison** — queue heavy jobs with Celery for very large CSVs
