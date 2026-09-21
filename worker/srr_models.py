from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SRRItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: str = Field(min_length=1)
    source_tag: str | None = None


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
