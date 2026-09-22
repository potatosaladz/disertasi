import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_NUMBER_TOKEN = r"[-+]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?"
_NUMBER_PATTERN = re.compile(_NUMBER_TOKEN)
_PERCENT_PATTERN = re.compile(rf"(?<![\d.,])({_NUMBER_TOKEN})\s*%")
_ZERO_PATTERN = re.compile(
    r"^\s*(?:zero|nol|none|no impact|tanpa dampak|tidak ada dampak)\b"
)
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
        percent_match = _PERCENT_PATTERN.search(text) if prefer_percent or normalise_ratio else None
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
                return 0.5
        else:
            number_match = percent_match or _NUMBER_PATTERN.search(text)
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
    model_config = ConfigDict(extra="allow")

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
                for key in ("text", "description", "claim", "value", "recommendation", "action", "rationale", "alternative", "summary"):
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
    content: str = ""


class Alternative(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    deficit: float
    utility: float
    source_tag: str | None = None

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
                "deficit": ("deficit_impact", "projected_deficit", "deficitImpact"),
                "utility": ("score", "benefit", "utility_score", "utilityScore"),
            }.items():
                if target not in payload or payload[target] in (None, ""):
                    for alias in aliases:
                        if alias in payload and payload[alias] not in (None, ""):
                            payload[target] = payload[alias]
                            break
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
            "E": [item.content for item in self.evidence],
            "A": [item.content for item in self.assumptions],
            "P": [item.content for item in self.predictions],
            "R": [item.content for item in self.risks],
            "U": [item.content for item in self.uncertainties],
            "O": [item.content for item in self.objectives],
            "C": [item.content for item in self.constraints],
            "REC": self.recommendation.content if self.recommendation is not None else None,
        }


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
        for field in ("simulation_summary", "resolution"):
            payload[field] = _coerce_text(payload.get(field))
        for field in ("conflict_summary", "modelled_variables", "limitations"):
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
