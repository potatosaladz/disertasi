import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .database import Base

SCENARIO_PARAMETER_FIELDS = (
    "instrument",
    "targeting",
    "program_cost",
    "duration_months",
    "evaluation_trigger",
    "program_cost_period",
    "skip_llm_formulation",
    "formulation_dry_run_only",
    "single_year_deployment",
    "proposed_reallocation",
    "reallocation_from_education",
    "proposed_additional_revenue",
    "revenue_measure_type",
    "proposed_debt_financing",
    "debt_financing_mode",
    "proposed_sal_use",
    "sal_purpose",
    "proposed_other_financing",
    "appropriation_available",
    "verified_reallocation_capacity",
    "verified_revenue_offset_capacity",
    "verified_debt_financing_headroom",
    "verified_sal_available",
    "verified_operational_cash_minimum",
    "verified_projected_cash_after_policy",
    "verified_cumulative_borrowing_pct_gdp",
    "spending_reallocation_authorized",
    "dpr_spending_adjustment_recommendation",
    "finance_minister_sal_authorized",
    "dpr_sal_approval_obtained",
    "dpr_additional_sbn_approval_obtained",
    "tax_measure_has_enacted_law",
    "pnbp_measure_has_valid_tariff_instrument",
    "output_outcome_documented",
    "domestic_product_compliance_documented",
    "growth_outlook",
    "inflation_outlook",
    "fx_outlook",
    "sbn10y_yield_outlook",
    "icp_outlook",
    "oil_lifting_outlook",
    "gas_lifting_outlook",
    "tax_revenue_forecast",
)


class ConvergenceStatus(str, enum.Enum):
    FULL_CONSENSUS = "FULL_CONSENSUS"
    PARTIAL_CONSENSUS = "PARTIAL_CONSENSUS"
    CONDITIONAL_CONSENSUS = "CONDITIONAL_CONSENSUS"
    PARETO_SET = "PARETO_SET"
    NO_CONSENSUS = "NO_CONSENSUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INFEASIBLE = "INFEASIBLE"


class Agent(Base):
    __tablename__ = "agents"
    __table_args__ = (
        CheckConstraint(
            "temperature >= 0 AND temperature <= 2",
            name="ck_agent_temperature_range",
        ),
        CheckConstraint("max_tokens > 0", name="ck_agent_max_tokens_positive"),
        CheckConstraint(
            "rar_dai_weight_mode IN ('auto', 'manual')",
            name="ck_agent_rar_dai_weight_mode",
        ),
        Index(
            "uq_agent_global_name",
            "name",
            unique=True,
            postgresql_where=text("scenario_id IS NULL"),
        ),
        Index(
            "uq_agent_scenario_name",
            "scenario_id",
            "name",
            unique=True,
            postgresql_where=text("scenario_id IS NOT NULL"),
        ),
        Index(
            "uq_agent_global_template_key",
            "template_key",
            unique=True,
            postgresql_where=text(
                "scenario_id IS NULL AND template_key IS NOT NULL"
            ),
        ),
        Index(
            "uq_agent_scenario_template_key",
            "scenario_id",
            "template_key",
            unique=True,
            postgresql_where=text(
                "scenario_id IS NOT NULL AND template_key IS NOT NULL"
            ),
        ),
        Index(
            "uq_agent_singleton_orchestrator",
            "is_orchestrator",
            unique=True,
            postgresql_where=text("is_orchestrator IS TRUE"),
        ),
        Index(
            "uq_agent_scenario_specialist_domain",
            "scenario_id",
            "specialist_domain",
            unique=True,
            postgresql_where=text(
                "scenario_id IS NOT NULL AND specialist_domain IS NOT NULL"
            ),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(255), nullable=False)
    template_key: Mapped[str | None] = mapped_column(String(100))
    scenario_id: Mapped[int | None] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=True, index=True
    )
    specialist_domain: Mapped[str | None] = mapped_column(String(100))
    is_orchestrator: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    theta_x: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    theta_q: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    theta_h: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    theta_s: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    theta_u: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    rar_dai_weight_mode: Mapped[str] = mapped_column(
        String(20), nullable=False, default="auto", server_default="auto"
    )
    llm_base_url: Mapped[str | None] = mapped_column(String(2048))
    llm_api_key: Mapped[str | None] = mapped_column(String(4096))
    llm_model: Mapped[str | None] = mapped_column(String(255))
    system_prompt: Mapped[str | None] = mapped_column(Text)
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.2, server_default="0.2")
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=4000, server_default="4000")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class GlobalLLMConfig(Base):
    __tablename__ = "global_llm_config"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_global_llm_config_singleton"),
        CheckConstraint(
            "temperature >= 0 AND temperature <= 2",
            name="ck_global_llm_temperature_range",
        ),
        CheckConstraint(
            "max_tokens > 0",
            name="ck_global_llm_max_tokens_positive",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    llm_base_url: Mapped[str | None] = mapped_column(String(2048))
    llm_api_key: Mapped[str | None] = mapped_column(String(4096))
    llm_model: Mapped[str | None] = mapped_column(String(255))
    temperature: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.2, server_default="0.2"
    )
    max_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=4000, server_default="4000"
    )
    apply_to_all: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AgentInfluenceObservation(Base):
    __tablename__ = "agent_influence_observations"
    __table_args__ = (
        CheckConstraint("gate IN (0, 1)", name="ck_influence_binary_gate"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("consensus_sessions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False, index=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    proposition: Mapped[str] = mapped_column(Text, nullable=False)
    X: Mapped[float] = mapped_column(Float, nullable=False)
    Q: Mapped[float] = mapped_column(Float, nullable=False)
    H: Mapped[float] = mapped_column(Float, nullable=False)
    S: Mapped[float] = mapped_column(Float, nullable=False)
    U: Mapped[float] = mapped_column(Float, nullable=False)
    gate: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    interaction_payload: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    calculation_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    raw_score: Mapped[float | None] = mapped_column(Float)
    normalized_weight: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SimulationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    instrument: str | None = Field(default=None, max_length=255)
    targeting: str | None = None
    program_cost: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    duration_months: int | None = Field(default=None, gt=0)
    evaluation_trigger: str | None = None
    program_cost_period: str | None = Field(default=None, max_length=100)
    skip_llm_formulation: bool | None = None
    formulation_dry_run_only: bool | None = None
    single_year_deployment: bool | None = None
    proposed_reallocation: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    reallocation_from_education: bool | None = None
    proposed_additional_revenue: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    revenue_measure_type: str | None = Field(default=None, max_length=255)
    proposed_debt_financing: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    debt_financing_mode: str | None = Field(default=None, max_length=255)
    proposed_sal_use: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    sal_purpose: str | None = None
    proposed_other_financing: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    appropriation_available: bool | None = None
    verified_reallocation_capacity: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    verified_revenue_offset_capacity: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    verified_debt_financing_headroom: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    verified_sal_available: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    verified_operational_cash_minimum: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    verified_projected_cash_after_policy: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    verified_cumulative_borrowing_pct_gdp: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    spending_reallocation_authorized: bool | None = None
    dpr_spending_adjustment_recommendation: bool | None = None
    finance_minister_sal_authorized: bool | None = None
    dpr_sal_approval_obtained: bool | None = None
    dpr_additional_sbn_approval_obtained: bool | None = None
    tax_measure_has_enacted_law: bool | None = None
    pnbp_measure_has_valid_tariff_instrument: bool | None = None
    output_outcome_documented: bool | None = None
    domestic_product_compliance_documented: bool | None = None
    growth_outlook: float | None = Field(default=None, allow_inf_nan=False)
    inflation_outlook: float | None = Field(default=None, allow_inf_nan=False)
    fx_outlook: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    sbn10y_yield_outlook: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    icp_outlook: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    oil_lifting_outlook: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    gas_lifting_outlook: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    tax_revenue_forecast: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_execution_mode(self) -> "SimulationPayload":
        enabled = sum(
            value is True
            for value in (
                self.skip_llm_formulation,
                self.formulation_dry_run_only,
            )
        )
        if enabled > 1:
            raise ValueError(
                "Direct execution and formulation dry-run are mutually exclusive"
            )
        return self

    def simulation_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class Scenario(Base):
    __tablename__ = "scenarios"
    __table_args__ = (
        CheckConstraint(
            "program_cost IS NULL OR program_cost >= 0",
            name="ck_scenario_program_cost_nonnegative",
        ),
        CheckConstraint(
            "duration_months IS NULL OR duration_months > 0",
            name="ck_scenario_duration_positive",
        ),
        CheckConstraint(
            "proposed_reallocation IS NULL OR proposed_reallocation >= 0",
            name="ck_scenario_reallocation_nonnegative",
        ),
        CheckConstraint(
            "proposed_additional_revenue IS NULL OR proposed_additional_revenue >= 0",
            name="ck_scenario_revenue_nonnegative",
        ),
        CheckConstraint(
            "proposed_debt_financing IS NULL OR proposed_debt_financing >= 0",
            name="ck_scenario_debt_nonnegative",
        ),
        CheckConstraint(
            "proposed_sal_use IS NULL OR proposed_sal_use >= 0",
            name="ck_scenario_sal_use_nonnegative",
        ),
        CheckConstraint(
            "proposed_other_financing IS NULL OR proposed_other_financing >= 0",
            name="ck_scenario_other_financing_nonnegative",
        ),
        CheckConstraint(
            "verified_reallocation_capacity IS NULL OR verified_reallocation_capacity >= 0",
            name="ck_scenario_verified_reallocation_nonnegative",
        ),
        CheckConstraint(
            "verified_revenue_offset_capacity IS NULL OR verified_revenue_offset_capacity >= 0",
            name="ck_scenario_verified_revenue_nonnegative",
        ),
        CheckConstraint(
            "verified_debt_financing_headroom IS NULL OR verified_debt_financing_headroom >= 0",
            name="ck_scenario_verified_debt_nonnegative",
        ),
        CheckConstraint(
            "verified_sal_available IS NULL OR verified_sal_available >= 0",
            name="ck_scenario_verified_sal_nonnegative",
        ),
        CheckConstraint(
            "verified_operational_cash_minimum IS NULL OR verified_operational_cash_minimum >= 0",
            name="ck_scenario_verified_cash_minimum_nonnegative",
        ),
        CheckConstraint(
            "verified_projected_cash_after_policy IS NULL OR verified_projected_cash_after_policy >= 0",
            name="ck_scenario_verified_cash_after_nonnegative",
        ),
        CheckConstraint(
            "verified_cumulative_borrowing_pct_gdp IS NULL OR verified_cumulative_borrowing_pct_gdp >= 0",
            name="ck_scenario_verified_borrowing_nonnegative",
        ),
        CheckConstraint(
            "fx_outlook IS NULL OR fx_outlook > 0",
            name="ck_scenario_fx_positive",
        ),
        CheckConstraint(
            "sbn10y_yield_outlook IS NULL OR sbn10y_yield_outlook >= 0",
            name="ck_scenario_sbn_yield_nonnegative",
        ),
        CheckConstraint(
            "icp_outlook IS NULL OR icp_outlook >= 0",
            name="ck_scenario_icp_nonnegative",
        ),
        CheckConstraint(
            "oil_lifting_outlook IS NULL OR oil_lifting_outlook >= 0",
            name="ck_scenario_oil_lifting_nonnegative",
        ),
        CheckConstraint(
            "gas_lifting_outlook IS NULL OR gas_lifting_outlook >= 0",
            name="ck_scenario_gas_lifting_nonnegative",
        ),
        CheckConstraint(
            "tax_revenue_forecast IS NULL OR tax_revenue_forecast >= 0",
            name="ck_scenario_tax_forecast_nonnegative",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    instrument: Mapped[str | None] = mapped_column(String(255))
    targeting: Mapped[str | None] = mapped_column(Text)
    program_cost: Mapped[float | None] = mapped_column(Float)
    duration_months: Mapped[int | None] = mapped_column(Integer)
    evaluation_trigger: Mapped[str | None] = mapped_column(Text)
    program_cost_period: Mapped[str | None] = mapped_column(String(100))
    skip_llm_formulation: Mapped[bool | None] = mapped_column(Boolean)
    formulation_dry_run_only: Mapped[bool | None] = mapped_column(Boolean)
    single_year_deployment: Mapped[bool | None] = mapped_column(Boolean)
    proposed_reallocation: Mapped[float | None] = mapped_column(Float)
    reallocation_from_education: Mapped[bool | None] = mapped_column(Boolean)
    proposed_additional_revenue: Mapped[float | None] = mapped_column(Float)
    revenue_measure_type: Mapped[str | None] = mapped_column(String(255))
    proposed_debt_financing: Mapped[float | None] = mapped_column(Float)
    debt_financing_mode: Mapped[str | None] = mapped_column(String(255))
    proposed_sal_use: Mapped[float | None] = mapped_column(Float)
    sal_purpose: Mapped[str | None] = mapped_column(Text)
    proposed_other_financing: Mapped[float | None] = mapped_column(Float)
    appropriation_available: Mapped[bool | None] = mapped_column(Boolean)
    verified_reallocation_capacity: Mapped[float | None] = mapped_column(Float)
    verified_revenue_offset_capacity: Mapped[float | None] = mapped_column(Float)
    verified_debt_financing_headroom: Mapped[float | None] = mapped_column(Float)
    verified_sal_available: Mapped[float | None] = mapped_column(Float)
    verified_operational_cash_minimum: Mapped[float | None] = mapped_column(Float)
    verified_projected_cash_after_policy: Mapped[float | None] = mapped_column(Float)
    verified_cumulative_borrowing_pct_gdp: Mapped[float | None] = mapped_column(Float)
    spending_reallocation_authorized: Mapped[bool | None] = mapped_column(Boolean)
    dpr_spending_adjustment_recommendation: Mapped[bool | None] = mapped_column(Boolean)
    finance_minister_sal_authorized: Mapped[bool | None] = mapped_column(Boolean)
    dpr_sal_approval_obtained: Mapped[bool | None] = mapped_column(Boolean)
    dpr_additional_sbn_approval_obtained: Mapped[bool | None] = mapped_column(Boolean)
    tax_measure_has_enacted_law: Mapped[bool | None] = mapped_column(Boolean)
    pnbp_measure_has_valid_tariff_instrument: Mapped[bool | None] = mapped_column(Boolean)
    output_outcome_documented: Mapped[bool | None] = mapped_column(Boolean)
    domestic_product_compliance_documented: Mapped[bool | None] = mapped_column(Boolean)
    growth_outlook: Mapped[float | None] = mapped_column(Float)
    inflation_outlook: Mapped[float | None] = mapped_column(Float)
    fx_outlook: Mapped[float | None] = mapped_column(Float)
    sbn10y_yield_outlook: Mapped[float | None] = mapped_column(Float)
    icp_outlook: Mapped[float | None] = mapped_column(Float)
    oil_lifting_outlook: Mapped[float | None] = mapped_column(Float)
    gas_lifting_outlook: Mapped[float | None] = mapped_column(Float)
    tax_revenue_forecast: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def simulation_payload(self) -> dict[str, Any]:
        return SimulationPayload.model_validate(self).simulation_payload()


class ScenarioMandateSnapshot(Base):
    __tablename__ = "scenario_mandate_snapshots"
    __table_args__ = (
        UniqueConstraint("scenario_id", "revision", name="uq_scenario_mandate_revision"),
        CheckConstraint("agent_count >= 0", name="ck_mandate_agent_count_nonnegative"),
        CheckConstraint("generated_count >= 0", name="ck_mandate_generated_count_nonnegative"),
        CheckConstraint("failure_count >= 0", name="ck_mandate_failure_count_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    revision: Mapped[str] = mapped_column(String(64), nullable=False)
    generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    agent_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rules: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    agent_rules: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    generated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ConsensusSession(Base):
    __tablename__ = "consensus_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scenario_id: Mapped[int] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mandate_snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("scenario_mandate_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    mandate_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    mandate_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    runtime_config_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="QUEUED", server_default="QUEUED"
    )
    error: Mapped[str | None] = mapped_column(Text)
    logs: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    progress_stage: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SimulationArtifact(Base):
    __tablename__ = "simulation_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "simulation_version",
            "round_number",
            name="uq_simulation_run_version_round",
        ),
        CheckConstraint("round_number > 0", name="ck_simulation_round_positive"),
        CheckConstraint("latency_ms >= 0", name="ck_simulation_latency_nonnegative"),
        CheckConstraint("token_usage >= 0", name="ck_simulation_tokens_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("consensus_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scenario_id: Mapped[int] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    trigger: Mapped[str] = mapped_column(String(100), nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    output_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    simulation_version: Mapped[str] = mapped_column(String(64), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    token_usage: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReasoningLog(Base):
    __tablename__ = "reasoning_logs"
    __table_args__ = (
        CheckConstraint("provenance_count >= 0", name="ck_reasoning_provenance_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False, index=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("consensus_sessions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    parsed_srr_objects: Mapped[list[dict[str, Any]] | dict[str, Any]] = mapped_column(JSONB, nullable=False)
    deliberation_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    is_schema_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    provenance_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DisagreementLog(Base):
    __tablename__ = "disagreement_logs"
    __table_args__ = (
        CheckConstraint("agent_i <> agent_j", name="ck_disagreement_distinct_agents"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("consensus_sessions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_i: Mapped[int] = mapped_column(ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False, index=True)
    agent_j: Mapped[int] = mapped_column(ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False, index=True)
    dE: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    dA: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    dP: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    dR: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    dU: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    dO: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    dC: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    dREC: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    resolution_route: Mapped[str | None] = mapped_column(String(255))
    detail_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MetricSnapshot(Base):
    __tablename__ = "metric_snapshots"
    __table_args__ = (
        CheckConstraint(
            "hard_constraint_violation_rate >= 0 AND hard_constraint_violation_rate <= 100",
            name="ck_metric_violation_rate_percent",
        ),
        CheckConstraint(
            "provenance_completeness_percent >= 0 AND provenance_completeness_percent <= 100",
            name="ck_metric_provenance_percent",
        ),
        CheckConstraint(
            "material_information_retention_macro_f1 >= 0 AND material_information_retention_macro_f1 <= 1",
            name="ck_metric_retention_f1",
        ),
        CheckConstraint("feasible_alternatives_count >= 0", name="ck_metric_feasible_count_nonnegative"),
        CheckConstraint("latency_ms >= 0", name="ck_metric_latency_nonnegative"),
        CheckConstraint("token_usage >= 0", name="ck_metric_token_usage_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("consensus_sessions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    provenance_completeness_percent: Mapped[float] = mapped_column(Float, nullable=False)
    material_information_retention_macro_f1: Mapped[float] = mapped_column(Float, nullable=False)
    hard_constraint_violation_rate: Mapped[float] = mapped_column(Float, nullable=False)
    feasible_alternatives_count: Mapped[int] = mapped_column(Integer, nullable=False)
    convergence_status: Mapped[ConvergenceStatus] = mapped_column(
        Enum(ConvergenceStatus, name="convergence_status"),
        nullable=False,
    )
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    token_usage: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
