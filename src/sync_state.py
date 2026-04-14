"""
sync_state.py — Persistent sync timestamp for incremental OTCS crawls

Stores the last successful sync time in a JSON file.
On next run, only documents with modify_date > last_sync are processed.

Full writeup: https://kapiluthra.github.io/blog-otcs-incremental-sync.html
"""

import json
import logging
import pathlib
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

SYNC_STATE_FILE = pathlib.Path("sync_state.json")


def get_last_sync() -> str:
    """Return ISO 8601 timestamp of the last successful sync.

    Returns a distant past date on first run, triggering a full initial crawl.
    """
    if SYNC_STATE_FILE.exists():
        try:
            data = json.loads(SYNC_STATE_FILE.read_text())
            return data.get("last_sync", "2000-01-01T00:00:00Z")
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt sync state file — resetting to full crawl")
    return "2000-01-01T00:00:00Z"


def save_sync_time(buffer_minutes: int = 5) -> None:
    """Persist sync completion time with a small lookback buffer.

    The buffer (default 5 min) provides safety against clock skew between
    the OTCS server and the sync process. Documents modified in the overlap
    window get re-checked on the next run — false positives are fine.
    Missed documents (false negatives) are not.

    Call ONLY after a successful sync with zero failures.
    """
    buffered_time = datetime.now(timezone.utc) - timedelta(minutes=buffer_minutes)
    ts = buffered_time.isoformat()
    SYNC_STATE_FILE.write_text(json.dumps({
        "last_sync": ts,
        "last_sync_human": ts,
        "buffer_minutes": buffer_minutes,
    }, indent=2))
    logger.info("Sync timestamp saved: %s (-%d min buffer)", ts, buffer_minutes)


def needs_update(node: dict, last_sync: str) -> bool:
    """Return True if this OTCS node was modified after last_sync.

    Uses string comparison — works correctly because OTCS returns
    modify_date in ISO 8601 format which sorts lexicographically.
    """
    modify_date = (
        node.get("data", {})
            .get("properties", {})
            .get("modify_date", "")
    )
    if not modify_date:
        # No modify_date = assume changed (fail safe)
        return True
    return modify_date > last_sync
