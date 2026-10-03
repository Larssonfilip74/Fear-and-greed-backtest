"""Append-only registry of every evaluated configuration (protocol rule R3).

The trial count feeds the Deflated Sharpe ratio and PBO. Rows are never edited or deleted.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fgbt.splits import HoldoutLockedError

FIELDS = ["timestamp_utc", "hypothesis", "segment", "config_hash", "config", "metrics"]


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


SEGMENTS = {"DEV", "VAL", "DEV+VAL", "HOLD"}


def log_trial(path, hypothesis: str, segment: str, config: dict, metrics: dict, unlock_holdout: bool = False) -> str:
    segment = segment.strip().upper()
    if segment not in SEGMENTS:
        raise ValueError(f"unknown segment {segment!r}; use one of {sorted(SEGMENTS)}")
    if segment == "HOLD" and not unlock_holdout:
        raise HoldoutLockedError("refusing to log a holdout trial while the holdout is locked")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    h = config_hash(config)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "hypothesis": hypothesis, "segment": segment, "config_hash": h,
                    "config": json.dumps(config, sort_keys=True, default=str),
                    "metrics": json.dumps(metrics, sort_keys=True, default=str)})
    return h


def count_trials(path, hypothesis: str | None = None, distinct: bool = False) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if hypothesis is None or r["hypothesis"] == hypothesis]
    return len({(r["hypothesis"], r["config_hash"]) for r in rows}) if distinct else len(rows)
