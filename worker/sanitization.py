import re
from copy import deepcopy
from typing import Any

_FALLBACK_METADATA: dict[str, Any] = {
    "kind": "missing_alternatives",
    "label_i18n": {
        "id": "Bukti Diperlukan",
        "en": "Evidence Required",
    },
    "reason_i18n": {
        "id": "Respons LLM tidak menyertakan alternatif kebijakan dengan angka fiskal yang dapat diverifikasi.",
        "en": "The LLM response did not provide a policy alternative with verifiable fiscal values.",
    },
    "decision_status": "evidence_required",
}


def coerce_llm_collection(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    if isinstance(value, str):
        return [
            item.strip()
            for item in re.split(r"[;\n]+", value)
            if item.strip()
        ]
    if isinstance(value, dict):
        return [value]
    return [value]


def deterministic_fallback_metadata() -> dict[str, Any]:
    return deepcopy(_FALLBACK_METADATA)
