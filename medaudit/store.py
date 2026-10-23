"""Append-only, thread-safe JSONL store keyed by record id (makes runs resumable)."""
from __future__ import annotations

import json
import threading
from pathlib import Path


class JsonlStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self) -> dict[str, dict]:
        """Latest record per key (later lines override earlier ones, e.g. retried errors)."""
        out: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a torn final line after a crash
                    out[rec["key"]] = rec
        return out

    def append(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")
            f.flush()
