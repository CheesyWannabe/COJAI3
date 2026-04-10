"""
Session Storage
In-memory store with optional JSON persistence.
Supports multiple sessions and multiple problems per session.
"""

import json
import os
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path


DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/csv_judge_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


class SessionStore:
    """
    Thread-safe in-memory store.
    Structure:
        sessions[session_id][problem_name] = {
            "reference": { "content": bytes, "meta": {...} },
            "submissions": [ {...}, ... ]
        }
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _ensure(self, session_id: str, problem_name: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = {}
        if problem_name not in self._sessions[session_id]:
            self._sessions[session_id][problem_name] = {
                "reference": None,
                "submissions": [],
            }

    def save_reference(
        self,
        session_id: str,
        problem_name: str,
        content: bytes,
        filename: str,
        meta: Dict[str, Any],
    ):
        with self._lock:
            self._ensure(session_id, problem_name)
            self._sessions[session_id][problem_name]["reference"] = {
                "content": content,
                "filename": filename,
                "meta": meta,
            }
            # Reset submissions when reference changes
            self._sessions[session_id][problem_name]["submissions"] = []

    def get_reference(self, session_id: str, problem_name: str) -> Optional[Dict]:
        with self._lock:
            try:
                return self._sessions[session_id][problem_name]["reference"]
            except KeyError:
                return None

    def get_reference_meta(self, session_id: str, problem_name: str) -> Optional[Dict]:
        with self._lock:
            try:
                ref = self._sessions[session_id][problem_name]["reference"]
                if ref is None:
                    return None
                return {"filename": ref["filename"], **ref["meta"]}
            except KeyError:
                return None

    def add_submission(self, session_id: str, problem_name: str, entry: Dict):
        with self._lock:
            self._ensure(session_id, problem_name)
            self._sessions[session_id][problem_name]["submissions"].append(entry)

    def get_submissions(self, session_id: str, problem_name: str) -> Optional[List[Dict]]:
        with self._lock:
            try:
                return list(self._sessions[session_id][problem_name]["submissions"])
            except KeyError:
                return None

    def get_all_problems(self, session_id: str) -> List[str]:
        with self._lock:
            if session_id not in self._sessions:
                return []
            return list(self._sessions[session_id].keys())

    def reset(self, session_id: str, problem_name: str):
        with self._lock:
            try:
                del self._sessions[session_id][problem_name]
            except KeyError:
                pass
