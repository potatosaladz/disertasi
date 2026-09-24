from typing import Any, Literal, cast

Language = Literal["id", "en"]
DEFAULT_LANGUAGE: Language = "id"
SUPPORTED_LANGUAGES: tuple[Language, ...] = ("id", "en")


def bilingual(id_text: str, en_text: str) -> dict[str, str]:
    return {"id": id_text, "en": en_text}


def translated(messages: object, lang: Language) -> str | None:
    if not isinstance(messages, dict):
        return None
    selected = messages.get(lang)
    if isinstance(selected, str) and selected.strip():
        return selected
    fallback = messages.get(DEFAULT_LANGUAGE) or messages.get("en")
    return fallback if isinstance(fallback, str) else None


def localize_payload(value: object, lang: Language) -> object:
    if isinstance(value, list):
        return [localize_payload(item, lang) for item in value]
    if not isinstance(value, dict):
        return value
    localized: dict[str, Any] = {
        str(key): localize_payload(item, lang) for key, item in value.items()
    }
    messages = localized.get("messages")
    selected_message = translated(messages, lang)
    if selected_message is not None:
        localized["message"] = selected_message
    for key, item in list(localized.items()):
        if not key.endswith("_i18n"):
            continue
        selected = translated(item, lang)
        if selected is not None:
            localized[key.removesuffix("_i18n")] = selected
    localized["language"] = lang if "language" in localized else localized.get("language")
    if localized.get("language") is None:
        localized.pop("language", None)
    return localized


def localized_dict(value: dict[str, Any], lang: Language) -> dict[str, Any]:
    return cast(dict[str, Any], localize_payload(value, lang))
