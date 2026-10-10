"""Thread-safe, persistent daily AI API budget and availability guard.

The daily counter is stored beside the configured database so restarts do not
reset free-tier usage. Counts are reserved before each outbound provider HTTP
request, including retries and tool-round requests.
"""
from __future__ import annotations

import json
import os
import threading
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from config import CFG

_lock = threading.RLock()
_calls: deque[float] = deque()  # compatibility/rolling diagnostics
_blocked_until = 0.0
_reason = ""
_usage_path = Path(f"{CFG.db_path}.ai_usage.json")
_usage_day = datetime.now(timezone.utc).date().isoformat()
_calls_today = 0


def _load_usage() -> None:
    global _usage_day, _calls_today
    try:
        payload = json.loads(_usage_path.read_text(encoding="utf-8"))
        today = datetime.now(timezone.utc).date().isoformat()
        if payload.get("day") == today:
            _usage_day = today
            _calls_today = max(0, int(payload.get("calls", 0)))
        else:
            _usage_day, _calls_today = today, 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        _usage_day = datetime.now(timezone.utc).date().isoformat()
        _calls_today = 0


def _persist_usage() -> None:
    _usage_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _usage_path.with_suffix(_usage_path.suffix + ".tmp")
    tmp.write_text(json.dumps({"day": _usage_day, "calls": _calls_today}), encoding="utf-8")
    os.replace(tmp, _usage_path)


_load_usage()


def reserve_shared_request_slot() -> bool:
    """Cross-process request budget: UTC daily cap plus rolling-minute cap in SQLite.

    Fails closed on storage errors. Each actual outbound HTTP request must call this.
    """
    now = time.time()
    day = datetime.now(timezone.utc).date().isoformat()
    path = Path(f"{CFG.db_path}.ai_budget.sqlite3")
    path.parent.mkdir(parents=True, exist_ok=True)
    daily_limit = max(1, int(CFG.max_ai_calls_per_day))
    # Provider limits differ; use an explicit configured cap or a conservative shared default.
    minute_limit = max(1, int(CFG.ai_minute_call_limit or 10))
    try:
        conn = sqlite3.connect(str(path), timeout=10, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("CREATE TABLE IF NOT EXISTS ai_budget_day (id INTEGER PRIMARY KEY CHECK(id=1), day TEXT NOT NULL, count INTEGER NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS ai_budget_events (event_id TEXT PRIMARY KEY, ts REAL NOT NULL)")
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT day,count FROM ai_budget_day WHERE id=1").fetchone()
            if row is None:
                conn.execute("INSERT INTO ai_budget_day(id,day,count) VALUES(1,?,0)", (day,)); count = 0
            elif row[0] != day:
                conn.execute("UPDATE ai_budget_day SET day=?,count=0 WHERE id=1", (day,)); count = 0
            else:
                count = int(row[1])
            conn.execute("DELETE FROM ai_budget_events WHERE ts <= ?", (now - 60.0,))
            minute_count = int(conn.execute("SELECT COUNT(*) FROM ai_budget_events").fetchone()[0])
            if count >= daily_limit or minute_count >= minute_limit:
                conn.rollback(); return False
            import uuid
            conn.execute("UPDATE ai_budget_day SET count=count+1 WHERE id=1")
            conn.execute("INSERT INTO ai_budget_events(event_id,ts) VALUES(?,?)", (uuid.uuid4().hex, now))
            conn.commit(); return True
        finally:
            conn.close()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("shared AI request budget unavailable; blocking provider call")
        return False


def configured() -> bool:
    return any((CFG.groq_api_key, CFG.openrouter_api_key, CFG.gemini_api_key,
                CFG.anthropic_api_key, CFG.openai_api_key))


def _roll_day() -> None:
    global _usage_day, _calls_today
    today = datetime.now(timezone.utc).date().isoformat()
    if today != _usage_day:
        _usage_day, _calls_today = today, 0
        _persist_usage()


def try_record_call() -> bool:
    """Reserve one outbound AI HTTP call; False means daily budget is exhausted."""
    global _calls_today
    with _lock:
        _roll_day()
        limit = max(1, int(CFG.max_ai_calls_per_day))
        if _calls_today >= limit:
            return False
        _calls_today += 1
        now = time.time()
        _calls.append(now)
        while _calls and _calls[0] < now - 86400:
            _calls.popleft()
        try:
            _persist_usage()
        except OSError:
            # Fail closed: do not make a provider call if its quota reservation
            # could not be persisted, because restart could otherwise reset usage.
            _calls_today -= 1
            logger_message = "AI usage counter could not be persisted; provider call blocked"
            import logging
            logging.getLogger(__name__).error(logger_message, exc_info=True)
            return False
        return True


def record_call() -> bool:
    """Compatibility alias for callers/tests; reserves a quota slot."""
    return try_record_call()


def usage_status() -> dict:
    with _lock:
        _roll_day()
        limit = max(1, int(CFG.max_ai_calls_per_day))
        return {"calls_today": _calls_today, "daily_limit": limit,
                "remaining_today": max(0, limit - _calls_today), "day_utc": _usage_day}


def record_unavailable(reason: str) -> None:
    global _blocked_until, _reason
    with _lock:
        _blocked_until = max(_blocked_until, time.time() + max(1, CFG.ai_provider_cooldown_seconds))
        _reason = reason[:240]


def record_available() -> None:
    global _blocked_until, _reason
    with _lock:
        _blocked_until, _reason = 0.0, ""


def status() -> dict:
    now = time.time()
    with _lock:
        _roll_day()
        remaining = max(0, int(_blocked_until - now))
        usage = usage_status()
        over = usage["remaining_today"] <= 0
        return {"configured": configured(),
                "entry_blocked": bool(CFG.ai_entry_block_on_unavailable and (remaining > 0 or over)),
                "blocked_seconds": remaining,
                "reason": _reason or ("daily AI API call limit reached" if over else ""),
                "calls_24h": len(_calls), **usage}


def before_cycle() -> tuple[bool, str]:
    if not configured():
        return True, "AI not configured; technical-only mode"
    st = status()
    if st["entry_blocked"]:
        return False, st["reason"] or f"AI unavailable ({st['blocked_seconds']}s cooldown)"
    return True, "AI capacity available"
