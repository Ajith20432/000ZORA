from __future__ import annotations

import json
from pathlib import Path

import ai_guard


def test_ai_usage_budget_is_hard_capped_and_persisted(monkeypatch, tmp_path: Path):
    usage_file = tmp_path / "ai_usage.json"
    monkeypatch.setattr(ai_guard, "_usage_path", usage_file)
    monkeypatch.setattr(ai_guard.CFG, "max_ai_calls_per_day", 2)
    monkeypatch.setattr(ai_guard, "_usage_day", ai_guard.datetime.now(ai_guard.timezone.utc).date().isoformat())
    monkeypatch.setattr(ai_guard, "_calls_today", 0)

    assert ai_guard.try_record_call() is True
    assert ai_guard.try_record_call() is True
    assert ai_guard.try_record_call() is False

    status = ai_guard.usage_status()
    assert status["calls_today"] == 2
    assert status["remaining_today"] == 0
    assert json.loads(usage_file.read_text()) ["calls"] == 2


def test_ai_usage_counter_restores_from_persisted_file(monkeypatch, tmp_path: Path):
    today = ai_guard.datetime.now(ai_guard.timezone.utc).date().isoformat()
    usage_file = tmp_path / "ai_usage.json"
    usage_file.write_text(json.dumps({"day": today, "calls": 7}))
    monkeypatch.setattr(ai_guard, "_usage_path", usage_file)
    monkeypatch.setattr(ai_guard, "_usage_day", "old-day")
    monkeypatch.setattr(ai_guard, "_calls_today", 0)

    ai_guard._load_usage()
    assert ai_guard.usage_status()["calls_today"] == 7
