import re
from typing import Any

_SENSITIVE_KEYS = {
    "accesskey",
    "accesskeyid",
    "accesskeys",
    "accesstoken",
    "accesstokens",
    "apikey",
    "apikeys",
    "authorization",
    "awsaccesskeyid",
    "awssecretaccesskey",
    "bearer",
    "clientsecret",
    "clientsecrets",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "hiddenreasoning",
    "hiddenreasoningtrace",
    "hiddenthoughts",
    "internalreasoning",
    "jwt",
    "llmapikey",
    "openaiapikey",
    "password",
    "passwordhash",
    "privatekey",
    "refreshtoken",
    "scratchpad",
    "secret",
    "secretkey",
    "secretkeys",
    "setcookie",
    "token",
}
_PRIVATE_KEYS = {
    "analysis",
    "chainofthought",
    "hiddenanalysis",
    "hiddenreasoning",
    "hiddenreasoningtrace",
    "hiddenthoughts",
    "internalanalysis",
    "internalreasoning",
    "reasoning",
    "reasoningtrace",
    "scratchpad",
    "thoughts",
}


_SENSITIVE_KEY_SUFFIXES = (
    "accesskey",
    "accesskeyid",
    "accesskeys",
    "apikey",
    "apikeys",
    "bearer",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "jwt",
    "password",
    "passwordhash",
    "privatekey",
    "secret",
    "secretkey",
    "secretkeys",
    "token",
    "tokens",
)
_SENSITIVE_STRUCTURED_KEY_PATTERN = re.compile(
    r"^(?:api(?:key|keys)|clientsecret|access(?:token|tokens|key|keys)|"
    r"aws(?:accesskeyid|secretaccesskey)|privatekey|secret(?:key|keys)?|"
    r"credential(?:s)?|password(?:hash)?|refreshtoken|jwt|bearer|token)"
    r"(?:value|data|json|id|ids|list|pem|der|format|encoded|encoding|base64|blob)?$"
)
_PUBLIC_REASONING_KEYS = {"reasoningsummary", "rationale"}
_PRIVATE_KEY_PREFIXES = (
    "chainofthought",
    "hiddenanalysis",
    "hiddenreasoning",
    "internalanalysis",
    "internalreasoning",
    "reasoning",
    "scratchpad",
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(?:bearer|basic|token)\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\b(?:(?:sk|rk)[-_](?:live[-_]|test[-_])?|ghp_|glpat-|xox[baprs]-|hf_|npm_)[-A-Za-z0-9_]{12,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bSG\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bSK[a-fA-F0-9]{32}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"(?is)-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----.*?-----END(?: [A-Z]+)? PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:api[-_ ]?key|authorization|secret[-_ ]?key|private[-_ ]?key)\s*[:=]\s*\S+"),
)


def _redact_string(value: str) -> str:
    redacted = value
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.casefold())


def is_private_key(key: str) -> bool:
    normalized = _normalized_key(key)
    return (
        normalized in _PRIVATE_KEYS
        or normalized not in _PUBLIC_REASONING_KEYS
        and normalized.startswith(_PRIVATE_KEY_PREFIXES)
        or normalized in _SENSITIVE_KEYS
        or _SENSITIVE_STRUCTURED_KEY_PATTERN.fullmatch(normalized) is not None
        or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)
        or normalized.startswith("authorization")
        or normalized.endswith("cookie")
    )


def _safe_error(value: object) -> str:
    code = _redact_string(str(value)).split(":", maxsplit=1)[0].strip()
    return code if re.fullmatch(r"[A-Za-z][A-Za-z0-9_. -]{0,80}", code) else "Operation failed"


def sanitize_public_error(value: object) -> str:
    return _safe_error(value)


def sanitize_public_value(value: object, parent_key: str = "") -> object:
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        normalized_parent = _normalized_key(parent_key)
        for key, item in value.items():
            key_text = str(key)
            normalized = _normalized_key(key_text)
            if is_private_key(key_text):
                continue
            if isinstance(item, str) and (
                normalized in {"error", "fallbackreason"}
                or normalized_parent == "fallback" and normalized == "reason"
            ):
                sanitized[key_text] = _safe_error(item)
            else:
                sanitized[key_text] = sanitize_public_value(item, key_text)
        return sanitized
    if isinstance(value, list):
        return [sanitize_public_value(item, parent_key) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def sanitize_public_dict(value: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_public_value(value)
    return sanitized if isinstance(sanitized, dict) else {}
