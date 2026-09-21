from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SRRItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    deficit: float
    utility: float
    source_tag: str | None = None


class SRRResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[Evidence]
    assumptions: list[Assumption]
    predictions: list[Prediction]
    risks: list[Risk]
    uncertainties: list[Uncertainty]
    objectives: list[Objective]
    constraints: list[Constraint]
    alternatives: list[Alternative]
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    material_information_retention_macro_f1: float = Field(ge=0.0, le=1.0)

    def provenance_items(self) -> list[SRRItem | Alternative]:
        return [
            *self.evidence,
            *self.assumptions,
            *self.predictions,
            *self.risks,
            *self.uncertainties,
            *self.objectives,
            *self.constraints,
            *self.alternatives,
            self.recommendation,
        ]

    def divergence_object(self) -> dict[str, object]:
        return {
            "E": [item.content for item in self.evidence],
            "A": [item.content for item in self.assumptions],
            "P": [item.content for item in self.predictions],
            "R": [item.content for item in self.risks],
            "U": [item.content for item in self.uncertainties],
            "O": [item.content for item in self.objectives],
            "C": [item.content for item in self.constraints],
            "REC": self.recommendation.content,
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
