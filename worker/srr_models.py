import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.core_algorithms import STATUTORY_DEFICIT_CEILING_PERCENT_GDP


_NUMBER_TOKEN = r"[-+]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?"
_NUMBER_PATTERN = re.compile(_NUMBER_TOKEN)
_PERCENT_PATTERN = re.compile(rf"(?<![\d.,])({_NUMBER_TOKEN})\s*%")
_ZERO_PATTERN = re.compile(
    r"^\s*(?:zero|nol|none|no impact|tanpa dampak|tidak ada dampak)\b"
)
_INCREMENTAL_PERCENT_AMOUNT = rf"{_NUMBER_TOKEN}(?:\s*[-–—]\s*{_NUMBER_TOKEN})?\s*%"
_INCREMENTAL_DEFICIT_PATTERNS = (
    re.compile(
        rf"\b(?:deficit\s+impact|impact\s+on\s+(?:the\s+)?deficit)\s*[:=]?\s*{_INCREMENTAL_PERCENT_AMOUNT}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:increase|incremental|additional)(?:\s+deficit)?(?:\s+impact)?\s*(?:of|by|:)?\s*{_INCREMENTAL_PERCENT_AMOUNT}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:raises?|raised|increases?|increased)\s+(?:the\s+)?deficit\s+(?:by|of)\s+{_INCREMENTAL_PERCENT_AMOUNT}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\bdeficit\s+(?:rises?|rose|increases?|increased)\s+by\s+{_INCREMENTAL_PERCENT_AMOUNT}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:meningkatkan\s+defisit|defisit\s+meningkat|meningkat(?:kan)?\s+sebesar|bertambah\s+sebesar|tambahan\s+dampak\s+defisit)\D{{0,20}}{_INCREMENTAL_PERCENT_AMOUNT}",
        flags=re.IGNORECASE,
    ),
)
_TOTAL_DEFICIT_PERCENT_PATTERN = re.compile(
    rf"\b(?:total\s+(?:projected\s+)?deficit|projected\s+total\s+deficit|"
    rf"defisit\s+total|proyeksi\s+defisit\s+total)\b[^%;]{{0,80}}?({_NUMBER_TOKEN})\s*%",
    flags=re.IGNORECASE,
)
_TOTAL_DEFICIT_PATTERN = re.compile(
    r"\b(?:total\s+(?:projected\s+)?deficit|projected\s+total\s+deficit|defisit\s+total|proyeksi\s+defisit\s+total)\b",
    flags=re.IGNORECASE,
)


def _is_incremental_deficit(value: str) -> bool:
    if _TOTAL_DEFICIT_PATTERN.search(value):
        return False
    return any(pattern.search(value) for pattern in _INCREMENTAL_DEFICIT_PATTERNS)
_QUALITATIVE_SCORES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("very high", "sangat tinggi", "fully preserves", "optimal"), 0.9),
    (("high", "tinggi"), 0.75),
    (("low to medium", "rendah hingga sedang"), 0.4),
    (("very low", "sangat rendah"), 0.1),
    (("medium", "moderate", "sedang", "netral", "neutral"), 0.5),
    (("low", "rendah"), 0.25),
)


def _parse_number_token(token: str) -> float:
    value = token.strip()
    if "." in value and "," in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif value.count(",") > 1:
        value = value.replace(",", "")
    elif value.count(".") > 1:
        value = value.replace(".", "")
    elif "," in value:
        value = value.replace(",", ".")
    return float(value)


def _clean_number(
    value: object,
    *,
    prefer_percent: bool = False,
    normalise_ratio: bool = False,
    qualitative: bool = False,
) -> object:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().replace("\u00a0", " ")
        if not text:
            return None
        lowered = text.casefold()
        if _ZERO_PATTERN.search(lowered):
            return 0.0
        if prefer_percent and re.search(r"\b(?:netral|neutral|unchanged|no change|tetap)\b", lowered):
            return 0.0
        total_deficit_match = (
            _TOTAL_DEFICIT_PERCENT_PATTERN.search(text) if prefer_percent else None
        )
        percent_match = (
            total_deficit_match
            or (_PERCENT_PATTERN.search(text) if prefer_percent or normalise_ratio else None)
        )
        if qualitative:
            for labels, score in _QUALITATIVE_SCORES:
                if any(label in lowered for label in labels):
                    return score
            explicit_match = re.search(
                rf"(?:utility|score|nilai)\s*[:=]?\s*({_NUMBER_TOKEN})",
                text,
                flags=re.IGNORECASE,
            ) or re.match(rf"^\s*({_NUMBER_TOKEN})(?:\s|$)", text)
            number_match = percent_match or explicit_match
            if number_match is None:
                return value
        else:
            number_match = percent_match
            if number_match is None and prefer_percent:
                number_match = re.fullmatch(rf"\s*({_NUMBER_TOKEN})\s*", text)
            if number_match is None and not prefer_percent:
                number_match = _NUMBER_PATTERN.search(text)
        if number_match is None:
            return value
        number = _parse_number_token(number_match.group(1) if percent_match else number_match.group(0))
        if normalise_ratio and (percent_match is not None or re.search(r"/\s*100\b", text)):
            number /= 100.0
    else:
        return value
    if normalise_ratio and 1 < number <= 100:
        number /= 100.0
    return number


def _normalise_numeric_fields(
    payload: dict[str, object],
    fields: tuple[str, ...],
    *,
    prefer_percent: bool = False,
    normalise_ratio: bool = False,
    qualitative: bool = False,
) -> None:
    for field in fields:
        if field not in payload:
            continue
        original = payload[field]
        cleaned = _clean_number(
            original,
            prefer_percent=prefer_percent,
            normalise_ratio=normalise_ratio,
            qualitative=qualitative,
        )
        payload[field] = cleaned
        if isinstance(original, str) and cleaned != original:
            payload.setdefault(f"{field}_raw", original)


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in (
            "content",
            "summary",
            "decision",
            "resolution",
            "rationale",
            "arbitrated_path",
            "sandbox_run",
            "status",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return "; ".join(
            f"{key}: {item}" for key, item in value.items() if isinstance(item, (str, int, float))
        )
    if value is None:
        return ""
    return str(value)


def _coerce_text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [text for item in value if (text := _coerce_text(item))]
    text = _coerce_text(value)
    return [text] if text else []


class SRRItem(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    content: str = Field(min_length=1)
    source_tag: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalise_item(cls, value: object) -> object:
        if isinstance(value, str):
            return {"content": value.strip()}
        if isinstance(value, dict):
            payload = dict(value)
            if not payload.get("content"):
                for key in (
                    "content",
                    "text",
                    "description",
                    "claim",
                    "value",
                    "recommendation",
                    "decision",
                    "verdict",
                    "key_action",
                    "preferred_alternative",
                    "action",
                    "rationale",
                    "alternative",
                    "summary",
                ):
                    if isinstance(payload.get(key), str) and payload[key].strip():
                        payload["content"] = payload[key].strip()
                        break
            if "source_tag" not in payload:
                for key in ("source", "citation", "provenance"):
                    if isinstance(payload.get(key), str):
                        payload["source_tag"] = payload[key]
                        break
            return payload
        return value


class Evidence(SRRItem):
    pass


class Assumption(SRRItem):
    pass


class Prediction(SRRItem):
    pass


class Risk(SRRItem):
    pass


class Uncertainty(SRRItem):
    pass


class Objective(SRRItem):
    pass


class Constraint(SRRItem):
    pass


class Recommendation(SRRItem):
    pass


class Alternative(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    name: str = Field(min_length=1)
    deficit: float = Field(ge=0.0, allow_inf_nan=False)
    utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source_tag: str | None = None

    @field_validator("deficit", "utility", mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid numeric inputs")
        return value

    @model_validator(mode="before")
    @classmethod
    def normalise_alternative(cls, value: object) -> object:
        if isinstance(value, dict):
            payload = dict(value)
            if not payload.get("name"):
                for key in ("title", "alternative", "option", "content"):
                    if isinstance(payload.get(key), str) and payload[key].strip():
                        payload["name"] = payload[key].strip()
                        break
            for target, aliases in {
                "deficit": ("projected_deficit", "projected_deficit_percent_gdp"),
                "utility": ("score", "benefit", "utility_score", "utilityScore"),
            }.items():
                if target not in payload or payload[target] in (None, ""):
                    for alias in aliases:
                        if alias in payload and payload[alias] not in (None, ""):
                            payload[target] = payload[alias]
                            break
            raw_deficit = payload.get("deficit")
            if isinstance(raw_deficit, str) and _is_incremental_deficit(raw_deficit):
                raise ValueError(
                    "Incremental deficit impact cannot be treated as a total projected deficit"
                )
            _normalise_numeric_fields(payload, ("deficit",), prefer_percent=True)
            _normalise_numeric_fields(
                payload,
                ("utility",),
                normalise_ratio=True,
                qualitative=True,
            )
            return payload
        return value


class SRRResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    reasoning_summary: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    predictions: list[Prediction] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    uncertainties: list[Uncertainty] = Field(default_factory=list)
    objectives: list[Objective] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    material_information_retention_macro_f1: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="before")
    @classmethod
    def normalise_response_aliases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        for wrapper in ("analysis", "result", "data", "output", "srr"):
            nested = payload.get(wrapper)
            if isinstance(nested, dict):
                payload = {**nested, **{key: item for key, item in payload.items() if key != wrapper}}
                break
        aliases = {
            "reasoning_summary": ("agent_opinion", "opinion", "summary", "rationale"),
            "evidence": ("evidences", "required_evidence", "evidence_requirements"),
            "predictions": ("prediction", "forecast", "forecasts"),
            "risks": ("risk", "threats"),
            "uncertainties": ("uncertainty", "unknowns"),
            "alternatives": ("alternative", "options", "scenarios"),
            "recommendation": ("recommendations", "decision", "conclusion"),
            "confidence": ("confidence_score", "certainty", "confidenceScore"),
        }
        for canonical, candidates in aliases.items():
            if canonical not in payload or payload[canonical] is None:
                for candidate in candidates:
                    if candidate in payload and payload[candidate] is not None:
                        payload[canonical] = payload[candidate]
                        break
        for field in ("evidence", "predictions", "risks", "uncertainties", "alternatives"):
            if isinstance(payload.get(field), str):
                payload[field] = [item.strip() for item in re.split(r"[;\n]+", payload[field]) if item.strip()]
            elif isinstance(payload.get(field), dict):
                payload[field] = [payload[field]]
        alternatives = payload.get("alternatives")
        if isinstance(alternatives, list):
            valid_alternatives: list[dict[str, object]] = []
            discarded_alternatives: list[dict[str, object]] = []
            for alternative in alternatives:
                try:
                    valid_alternatives.append(
                        Alternative.model_validate(alternative).model_dump(mode="json")
                    )
                except (ValueError, TypeError):
                    if isinstance(alternative, dict):
                        discarded_alternatives.append(
                            {
                                "name": str(alternative.get("name") or "Unnamed alternative"),
                                "deficit_raw": alternative.get("deficit"),
                                "reason": "Deficit or utility is not numerically verifiable.",
                            }
                        )
            payload["alternatives"] = valid_alternatives
            if discarded_alternatives:
                payload["discarded_alternatives"] = discarded_alternatives
        if isinstance(payload.get("recommendation"), list):
            payload["recommendation"] = payload["recommendation"][0] if payload["recommendation"] else None
        if isinstance(payload.get("recommendation"), dict):
            recommendation = dict(payload["recommendation"])
            if "confidence" not in payload and isinstance(recommendation.get("confidence"), (int, float, str)):
                payload["confidence"] = recommendation["confidence"]
            if not recommendation.get("content"):
                for key in (
                    "text",
                    "decision",
                    "action",
                    "recommendation",
                    "verdict",
                    "key_action",
                    "preferred_alternative",
                    "rationale",
                    "summary",
                    "authorized_scope",
                ):
                    recommendation_candidate = recommendation.get(key)
                    if (
                        isinstance(recommendation_candidate, str)
                        and recommendation_candidate.strip()
                    ):
                        recommendation["content"] = recommendation_candidate.strip()
                        break
            payload["recommendation"] = recommendation if recommendation.get("content") else None
        _normalise_numeric_fields(payload, ("confidence",), normalise_ratio=True)
        _normalise_numeric_fields(
            payload,
            ("material_information_retention_macro_f1",),
            normalise_ratio=True,
        )
        return payload

    def provenance_items(self) -> list[SRRItem | Alternative]:
        items: list[SRRItem | Alternative] = [
            *self.evidence,
            *self.assumptions,
            *self.predictions,
            *self.risks,
            *self.uncertainties,
            *self.objectives,
            *self.constraints,
            *self.alternatives,
        ]
        if self.recommendation is not None:
            items.append(self.recommendation)
        return items

    def divergence_object(self) -> dict[str, object]:
        return {
            "E": [
                {"content": item.content, "source_tag": item.source_tag}
                for item in self.evidence
            ],
            "A": [item.content for item in self.assumptions],
            "P": {
                "predictions": [item.content for item in self.predictions],
                "projected_deficits": [
                    {"name": item.name, "deficit": item.deficit}
                    for item in self.alternatives
                ],
            },
            "R": [item.content for item in self.risks],
            "U": {
                "uncertainties": [item.content for item in self.uncertainties],
                "confidence": self.confidence,
            },
            "O": [item.content for item in self.objectives],
            "C": {
                "constraints": [item.content for item in self.constraints],
                "statutory_deficit_violation": any(
                    item.deficit > STATUTORY_DEFICIT_CEILING_PERCENT_GDP
                    for item in self.alternatives
                ),
            },
            "REC": self.recommendation.content
            if self.recommendation is not None
            else None,
        }


_SIMULATION_SUMMARY_FALLBACK = (
    "Simulasi makro-fiskal otomatis diselesaikan oleh arbiter native."
)

_SIMULATION_CONFLICT_FALLBACK = (
    "Tidak ada konflik tambahan yang belum diselesaikan pada ronde arbiter akhir."
)


class SimulationAlternative(Alternative):
    source_tag: str = "SIMULATION_MODELLED"
    evidence_status: Literal["modelled"] = "modelled"


class SimulationResponse(SRRResponse):
    simulation_summary: str = Field(min_length=1)
    conflict_summary: list[str] = Field(min_length=1)
    resolution: str = Field(min_length=1)
    modelled_variables: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_status: Literal["modelled"] = "modelled"
    agent_name: str = "Simulation Agent / Arbiter Simulasi Makro-Fiskal"
    simulation_version: str = "native-simulation-v1"

    @model_validator(mode="before")
    @classmethod
    def normalise_simulation_payload(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        for wrapper in ("analysis", "result", "data", "output", "srr"):
            nested = payload.get(wrapper)
            if isinstance(nested, dict):
                payload = {
                    **nested,
                    **{key: item for key, item in payload.items() if key != wrapper},
                }
                break
        simulation_summary = _coerce_text(payload.get("simulation_summary"))
        payload["simulation_summary"] = (
            simulation_summary or _SIMULATION_SUMMARY_FALLBACK
        )
        payload["resolution"] = _coerce_text(payload.get("resolution"))
        conflict_summary = _coerce_text_list(payload.get("conflict_summary"))
        payload["conflict_summary"] = conflict_summary or [
            _SIMULATION_CONFLICT_FALLBACK
        ]
        for field in ("modelled_variables", "limitations"):
            payload[field] = _coerce_text_list(payload.get(field))
        payload.setdefault("evidence_status", "modelled")
        alternatives = payload.get("alternatives")
        if isinstance(alternatives, list):
            for alternative in alternatives:
                if isinstance(alternative, dict):
                    alternative["source_tag"] = "SIMULATION_MODELLED"
                    alternative["evidence_status"] = "modelled"
        return payload

    @model_validator(mode="after")
    def tag_modelled_alternatives(self) -> "SimulationResponse":
        for alternative in self.alternatives:
            alternative.source_tag = "SIMULATION_MODELLED"
            setattr(alternative, "evidence_status", "modelled")
        return self


ConvergenceName = Literal[
    "FULL_CONSENSUS",
    "PARTIAL_CONSENSUS",
    "CONDITIONAL_CONSENSUS",
    "PARETO_SET",
    "NO_CONSENSUS",
    "INSUFFICIENT_EVIDENCE",
    "INFEASIBLE",
]
