"""Fear & Greed loaders. OLD (money.cnn.com scrape) and NEW (CNN API) eras are kept distinct (protocol §2)."""
import csv
import io
import json
import subprocess
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from fgbt.data.align import fg_available_at

ERA_SPLIT = date(2021, 2, 1)  # first CNN API date; OLD strictly before, NEW from here on

# Pinned public mirror (MIT): github.com/whit3rabbit/fear-greed-data
MIRROR_URL = "https://github.com/whit3rabbit/fear-greed-data"
MIRROR_SHA = "6143a3bcea70c0dc156b2db24bae16cd78460f7a"
OLD_ERA_PATH = "datasets/archive/fear-greed-pre-2021.csv"
CNN_JSON_PATH = "json/cnn_output.json"


def parse_cnn_json(text: str) -> tuple[pd.Series, dict]:
    """Return (end-of-day history indexed by trading date, current block).

    Only points stamped exactly 00:00:00 UTC are end-of-day history; the trailing live point is dropped.
    Some saved snapshots carry a text line before the JSON, so parsing starts at the first '{'.
    """
    payload = json.loads(text[text.index("{"):])
    rows = {}
    for p in payload["fear_and_greed_historical"]["data"]:
        ts = datetime.fromtimestamp(p["x"] / 1000, timezone.utc)
        if (ts.hour, ts.minute, ts.second) != (0, 0, 0):
            continue
        d = ts.date()
        if d in rows:
            raise ValueError(f"duplicate history date {d}")
        rows[d] = float(p["y"])
    hist = pd.Series(rows, dtype=float, name="fg").sort_index()
    return hist, payload["fear_and_greed"]


def parse_old_csv(text: str) -> pd.Series:
    rows = {}
    for r in list(csv.reader(io.StringIO(text)))[1:]:
        if len(r) < 2 or not r[0].strip() or not r[1].strip():
            continue
        s = r[0].strip()
        if "/" in s:
            m, d, y = (int(x) for x in s.split("/"))
            dt_ = date(y, m, d)
        else:
            dt_ = date.fromisoformat(s)
        if dt_ in rows:
            raise ValueError(f"duplicate date {dt_}")
        rows[dt_] = float(r[1])
    return pd.Series(rows, dtype=float, name="fg").sort_index()


def build_series(old: pd.Series, new: pd.Series) -> pd.DataFrame:
    old = old[[d < ERA_SPLIT for d in old.index]]
    new = new[[d >= ERA_SPLIT for d in new.index]]
    s = pd.concat([old, new]).sort_index()
    bad = s[(s < 0) | (s > 100) | s.isna()]
    if len(bad):
        raise ValueError(f"F&G values out of range [0, 100]: {bad.to_dict()}")
    df = pd.DataFrame({"fg": s})
    df["era"] = ["OLD" if d < ERA_SPLIT else "NEW" for d in df.index]
    df["available_at"] = [pd.Timestamp(fg_available_at(d)) for d in df.index]
    return df


def implied_component_count(value: float, tol: float = 1e-6) -> int | None:
    """CNN NEW-era scores are means of k component scores each with one decimal; infer k (7 = complete)."""
    for k in (7, 6, 5, 4, 3, 2, 1):
        x = value * k * 10
        if abs(x - round(x)) < tol:
            return k
    return None


def git_show(repo_dir: str, sha: str, path: str) -> str:
    return subprocess.run(["git", "-C", repo_dir, "show", f"{sha}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def load_pinned(repo_dir: str, sha: str = MIRROR_SHA) -> pd.DataFrame:
    old = parse_old_csv(git_show(repo_dir, sha, OLD_ERA_PATH))
    new, _ = parse_cnn_json(git_show(repo_dir, sha, CNN_JSON_PATH))
    return build_series(old, new)
