"""
API Routes for CSV Online Judge
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import uuid
import time
import json
from pathlib import Path
from datetime import datetime

from engine import compare_csvs, parse_csv
from storage import SessionStore

router = APIRouter()
store = SessionStore()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def validate_csv_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")


async def read_file_bytes(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File exceeds maximum size of 10 MB.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return content


# ─── POST /upload-reference ───────────────────────────────────────────────────

@router.post("/upload-reference")
async def upload_reference(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    problem_name: Optional[str] = Form("default"),
):
    """
    Upload the reference (ground truth) CSV.
    Returns a session_id to use for subsequent submissions.
    """
    validate_csv_upload(file)
    content = await read_file_bytes(file)

    # Validate it's a parseable CSV
    try:
        df = parse_csv(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Reference CSV has no rows.")

    session_id = session_id or str(uuid.uuid4())
    problem_name = problem_name or "default"

    store.save_reference(session_id, problem_name, content, file.filename, {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "uploaded_at": datetime.utcnow().isoformat(),
    })

    return {
        "session_id": session_id,
        "problem_name": problem_name,
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "message": "Reference CSV uploaded successfully.",
    }


# ─── POST /submit ─────────────────────────────────────────────────────────────

@router.post("/submit")
async def submit(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    submitter_name: Optional[str] = Form("Anonymous"),
    problem_name: Optional[str] = Form("default"),
    lowercase: Optional[bool] = Form(True),
    numeric_tolerance: Optional[float] = Form(1e-6),
):
    """
    Submit a CSV for comparison against the reference.
    Returns similarity score and detailed breakdown.
    """
    validate_csv_upload(file)
    content = await read_file_bytes(file)

    # Validate parseable
    try:
        sub_df = parse_csv(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {str(e)}")

    # Retrieve reference
    ref = store.get_reference(session_id, problem_name)
    if ref is None:
        raise HTTPException(
            status_code=404,
            detail="No reference CSV found for this session/problem. Upload a reference first.",
        )

    # Run comparison
    t0 = time.time()
    try:
        result = compare_csvs(
            ref["content"],
            content,
            lowercase=lowercase,
            numeric_tolerance=numeric_tolerance,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
    elapsed = round(time.time() - t0, 3)

    submission_id = str(uuid.uuid4())[:8]
    entry = {
        "submission_id": submission_id,
        "submitter_name": submitter_name or "Anonymous",
        "filename": file.filename,
        "problem_name": problem_name,
        "submitted_at": datetime.utcnow().isoformat(),
        "processing_time_s": elapsed,
        **result,
    }

    store.add_submission(session_id, problem_name, entry)

    return entry


# ─── GET /results ─────────────────────────────────────────────────────────────

@router.get("/results")
async def get_results(
    session_id: str,
    problem_name: Optional[str] = "default",
):
    """
    Get all submission results for this session, sorted by score descending.
    """
    submissions = store.get_submissions(session_id, problem_name)
    if submissions is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Sort by score desc, then by time asc
    sorted_subs = sorted(
        submissions,
        key=lambda x: (-x.get("score", 0), x.get("submitted_at", "")),
    )

    ref_info = store.get_reference_meta(session_id, problem_name)

    return {
        "session_id": session_id,
        "problem_name": problem_name,
        "reference": ref_info,
        "total_submissions": len(sorted_subs),
        "leaderboard": [
            {
                "rank": i + 1,
                "submission_id": s["submission_id"],
                "submitter_name": s["submitter_name"],
                "filename": s["filename"],
                "score": s["score"],
                "matched_rows": s["matched_rows"],
                "total_ref_rows": s["total_ref_rows"],
                "submitted_at": s["submitted_at"],
            }
            for i, s in enumerate(sorted_subs)
        ],
        "submissions": sorted_subs,
    }


# ─── GET /session-info ────────────────────────────────────────────────────────

@router.get("/session-info")
async def session_info(session_id: str):
    """Get all problems/references for a session."""
    problems = store.get_all_problems(session_id)
    return {"session_id": session_id, "problems": problems}


# ─── DELETE /reset ────────────────────────────────────────────────────────────

@router.delete("/reset")
async def reset_session(session_id: str, problem_name: Optional[str] = "default"):
    """Clear submissions and reference for a session/problem."""
    store.reset(session_id, problem_name)
    return {"message": "Session reset successfully."}
