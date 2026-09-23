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
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


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
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(255), nullable=False)
    template_key: Mapped[str | None] = mapped_column(String(100), unique=True)
    theta_x: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    theta_q: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    theta_h: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    theta_s: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    theta_u: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1.0")
    llm_base_url: Mapped[str | None] = mapped_column(String(2048))
    llm_api_key: Mapped[str | None] = mapped_column(String(4096))
    llm_model: Mapped[str | None] = mapped_column(String(255))
    system_prompt: Mapped[str | None] = mapped_column(Text)
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.2, server_default="0.2")
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=4000, server_default="4000")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


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


class Scenario(Base):
    __tablename__ = "scenarios"
    __table_args__ = (
        CheckConstraint("max_deficit_constraint >= 0", name="ck_scenario_max_deficit_nonnegative"),
        CheckConstraint(
            "program_cost IS NULL OR program_cost >= 0",
            name="ck_scenario_program_cost_nonnegative",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    program_cost: Mapped[float | None] = mapped_column(Float)
    max_deficit_constraint: Mapped[float] = mapped_column(Float, nullable=False, default=3.0, server_default="3.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


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
    celery_task_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="QUEUED", server_default="QUEUED"
    )
    error: Mapped[str | None] = mapped_column(Text)
    logs: Mapped[list[dict[str, str]]] = mapped_column(
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
