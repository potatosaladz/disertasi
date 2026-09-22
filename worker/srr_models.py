import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    pass


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
                if target not in payload:
                    for alias in aliases:
                        if alias in payload:
                            payload[target] = payload[alias]
                            break
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
            recommendation = payload["recommendation"]
            if "confidence" not in payload and isinstance(recommendation.get("confidence"), (int, float, str)):
                payload["confidence"] = recommendation["confidence"]
        confidence = payload.get("confidence")
        if isinstance(confidence, str):
            confidence = confidence.strip().rstrip("%")
            try:
                payload["confidence"] = float(confidence)
            except ValueError:
                pass
        if isinstance(payload.get("confidence"), (int, float)) and 1 < payload["confidence"] <= 100:
            payload["confidence"] = payload["confidence"] / 100
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


ConvergenceName = Literal[
    "FULL_CONSENSUS",
    "PARTIAL_CONSENSUS",
    "CONDITIONAL_CONSENSUS",
    "PARETO_SET",
    "NO_CONSENSUS",
    "INSUFFICIENT_EVIDENCE",
    "INFEASIBLE",
]
