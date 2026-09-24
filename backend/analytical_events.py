from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from backend.localization import bilingual


def analytical_event(
    stage: str,
    level: str,
    code: str,
    id_message: str,
    en_message: str,
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    scenario_id: int | None = None,
    agent_id: int | None = None,
    agent_name: str | None = None,
    round_number: int | None = None,
    metric: dict[str, Any] | None = None,
    statutory: dict[str, Any] | None = None,
    economic: dict[str, Any] | None = None,
    fallback: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": str(uuid4()),
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "level": level,
        "code": code,
        "message": en_message,
        "messages": bilingual(id_message, en_message),
        "run_id": run_id,
        "task_id": task_id,
        "scenario_id": scenario_id,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "round_number": round_number,
        "metric": metric or {"status": "not-calculated", "value": None},
        "statutory": statutory or {"status": "not-calculated", "source_tags": []},
        "economic": economic or {"status": "not-calculated", "inputs": {}, "outputs": {}},
        "fallback": fallback or {"used": False, "kind": None, "reason": None},
        "metadata": metadata or {},
    }


def legacy_event(log: dict[str, Any]) -> dict[str, Any]:
    message = str(log.get("message") or "")
    stage = str(log.get("stage") or "UNKNOWN")
    level = str(log.get("level") or "INFO")
    identity = "|".join(
        str(log.get(key) or "")
        for key in ("run_id", "task_id", "scenario_id", "agent_id", "stage", "level", "message")
    )
    return {
        "event_id": str(uuid5(NAMESPACE_URL, identity)),
        "occurred_at": log.get("occurred_at"),
        "stage": stage,
        "level": level,
        "code": "LEGACY_LOG",
        "message": message,
        "messages": bilingual(message, message),
        "run_id": log.get("run_id"),
        "task_id": log.get("task_id"),
        "scenario_id": log.get("scenario_id"),
        "agent_id": log.get("agent_id"),
        "agent_name": log.get("agent_name"),
        "round_number": log.get("round_number"),
        "metric": {"status": "not-calculated", "value": None},
        "statutory": {"status": "not-calculated", "source_tags": []},
        "economic": {"status": "not-calculated", "inputs": {}, "outputs": {}},
        "fallback": {"used": False, "kind": None, "reason": None},
        "metadata": {"legacy": True, "translation_status": "source-language-only"},
    }


def normalize_event(log: object) -> dict[str, Any]:
    if not isinstance(log, dict):
        return legacy_event({"message": str(log)})
    if isinstance(log.get("messages"), dict) and log.get("event_id"):
        return dict(log)
    return legacy_event(log)
