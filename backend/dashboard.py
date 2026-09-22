from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import aliased

from .agent_templates import agent_revision, resolve_agent_system_prompt
from .celery_client import celery_client
from .core_algorithms import build_agent_system_prompt, resolve_llm_runtime_config
from .database import SessionLocal
from .mandate_snapshots import refresh_mandate_snapshot
from .models import (
    Agent,
    AgentInfluenceObservation,
    ConsensusSession,
    DisagreementLog,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
    ScenarioMandateSnapshot,
    SimulationArtifact,
)

router = APIRouter(prefix="/api")


def _resolution_mechanism(log: DisagreementLog) -> str:
    if log.resolution_route:
        return log.resolution_route
    if log.dE:
        return "Provenance Retrieval Triggered"
    if log.dP:
        return "Simulation Agent Requested"
    if log.dC:
        return "Constraint Arbitration Required"
    if log.dREC or log.dO:
        return "Pareto Reconciliation"
    if any((log.dA, log.dR, log.dU)):
        return "Evidence Review Required"
    return "No Resolution Required"


def _safe_agent_rules(raw_rules: object) -> list[dict[str, Any]]:
    if not isinstance(raw_rules, list):
        return []
    normalised: list[dict[str, Any]] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        rule = dict(raw_rule)
        for key, value in {
            "agent_id": 0,
            "name": "Unknown agent",
            "role": "Unknown role",
            "template_key": None,
            "mandate": None,
            "primary_sources": [],
            "constraints": [],
            "owned_checks": [],
            "synthesis_status": "fallback",
            "scenario_mandate": None,
            "scenario_focus": [],
            "priority_questions": [],
            "required_evidence": [],
            "epistemic_logic_traceability": [],
            "structured_consensus_protocol": [],
            "regulatory_compliance_alignment": [],
            "applicable_primary_sources": [],
            "applicable_constraints": [],
            "applicable_owned_checks": [],
            "llm_model": None,
            "latency_ms": None,
            "token_usage": 0,
            "error": None,
        }.items():
            if rule.get(key) is None and value is not None:
                rule[key] = value
            else:
                rule.setdefault(key, value)
        rule["agent_id"] = rule["agent_id"] if isinstance(rule["agent_id"], int) else 0
        rule["name"] = str(rule["name"] or "Unknown agent")
        rule["role"] = str(rule["role"] or "Unknown role")
        if rule["synthesis_status"] not in {"generated", "fallback"}:
            rule["synthesis_status"] = "fallback"
        if not isinstance(rule["error"], dict):
            rule["error"] = None
        else:
            rule["error"].setdefault("code", "SCHEMA_ERROR")
            rule["error"].setdefault("message", "Stored mandate error")
            rule["error"].setdefault("retryable", False)
        for field in (
            "primary_sources",
            "constraints",
            "owned_checks",
            "scenario_focus",
            "priority_questions",
            "required_evidence",
            "epistemic_logic_traceability",
            "structured_consensus_protocol",
            "regulatory_compliance_alignment",
            "applicable_primary_sources",
            "applicable_constraints",
            "applicable_owned_checks",
        ):
            if not isinstance(rule[field], list):
                rule[field] = [str(rule[field])] if rule[field] else []
        normalised.append(rule)
    return normalised


def _simulation_payload(artifact: SimulationArtifact) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "session_id": artifact.run_id,
        "scenario_id": artifact.scenario_id,
        "trigger": artifact.trigger,
        "round_number": artifact.round_number,
        "status": artifact.status,
        "simulation_version": artifact.simulation_version,
        "input": artifact.input_payload,
        "output": artifact.output_payload,
        "latency_ms": artifact.latency_ms,
        "token_usage": artifact.token_usage,
        "created_at": artifact.created_at.isoformat(),
    }


def _metric_payload(snapshot: MetricSnapshot) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "scenario_id": snapshot.scenario_id,
        "session_id": snapshot.run_id,
        "hard_constraint_violation_rate": snapshot.hard_constraint_violation_rate,
        "provenance_completeness_percent": snapshot.provenance_completeness_percent,
        "material_information_retention_macro_f1": snapshot.material_information_retention_macro_f1,
        "feasible_alternatives_count": snapshot.feasible_alternatives_count,
        "convergence_status": snapshot.convergence_status.value,
        "latency_ms": snapshot.latency_ms,
        "token_usage": snapshot.token_usage,
        "created_at": snapshot.created_at.isoformat(),
    }


def _dashboard_payload(scenario_id: int, session_id: str | None = None) -> dict[str, Any]:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        active_session = (
            session.get(ConsensusSession, session_id)
            if session_id is not None
            else session.scalar(
                select(ConsensusSession)
                .where(ConsensusSession.scenario_id == scenario_id)
                .order_by(ConsensusSession.created_at.desc())
            )
        )
        if session_id is not None and active_session is None:
            raise HTTPException(status_code=404, detail="Consensus session not found for scenario")
        if active_session is not None and active_session.scenario_id != scenario_id:
            raise HTTPException(status_code=404, detail="Consensus session not found for scenario")
        active_session_id = active_session.id if active_session is not None else None
        artifact_session_id = active_session_id or "__NO_ACTIVE_SESSION__"

        agents = list(session.scalars(select(Agent).order_by(Agent.id)))
        mandate_snapshot = session.scalar(
            select(ScenarioMandateSnapshot)
            .where(ScenarioMandateSnapshot.scenario_id == scenario_id)
            .order_by(ScenarioMandateSnapshot.updated_at.desc(), ScenarioMandateSnapshot.id.desc())
        )
        current_revision = agent_revision(agents, scenario)
        stored_agent_rules = _safe_agent_rules(
            mandate_snapshot.agent_rules if mandate_snapshot is not None else []
        )
        stored_rules = (
            mandate_snapshot.rules
            if mandate_snapshot is not None and isinstance(mandate_snapshot.rules, dict)
            else {}
        )
        mandate_is_stale = (
            mandate_snapshot is not None
            and (
                mandate_snapshot.revision != current_revision
                or mandate_snapshot.agent_count != len(agents)
            )
        )
        domain_rules = (
            {
                "scenario_id": mandate_snapshot.scenario_id,
                "revision": mandate_snapshot.revision,
                "generated": mandate_snapshot.generated,
                "stale": mandate_is_stale,
                "agent_count": mandate_snapshot.agent_count,
                "rules": stored_rules,
                "agent_rules": stored_agent_rules,
                "status": "stale" if mandate_is_stale else mandate_snapshot.status,
                "generated_count": mandate_snapshot.generated_count,
                "failure_count": mandate_snapshot.failure_count,
                "detail": (
                    "Saved mandates are stale because the scenario or agent configuration changed."
                    if mandate_is_stale
                    else mandate_snapshot.detail
                ),
            }
            if mandate_snapshot is not None
            else {
                "scenario_id": scenario_id,
                "revision": current_revision,
                "generated": False,
                "stale": True,
                "agent_count": len(agents),
                "rules": {},
                "agent_rules": [],
                "status": "missing",
                "generated_count": 0,
                "failure_count": 0,
                "detail": "Generate scenario-specific mandates before starting deliberation.",
            }
        )

        snapshots = list(
            session.scalars(
                select(MetricSnapshot)
                .where(
                    MetricSnapshot.scenario_id == scenario_id,
                    MetricSnapshot.run_id == artifact_session_id,
                )
                .order_by(MetricSnapshot.created_at.desc(), MetricSnapshot.id.desc())
            )
        )
        reasoning_logs = list(
            session.scalars(
                select(ReasoningLog)
                .where(
                    ReasoningLog.scenario_id == scenario_id,
                    ReasoningLog.run_id == artifact_session_id,
                )
                .order_by(ReasoningLog.id)
            )
        )
        agent_i = aliased(Agent)
        agent_j = aliased(Agent)
        disagreements = session.execute(
            select(DisagreementLog, agent_i.name, agent_j.name)
            .join(agent_i, DisagreementLog.agent_i == agent_i.id)
            .join(agent_j, DisagreementLog.agent_j == agent_j.id)
            .where(
                DisagreementLog.scenario_id == scenario_id,
                DisagreementLog.run_id == artifact_session_id,
            )
            .order_by(DisagreementLog.id)
        ).all()
        simulations = list(
            session.scalars(
                select(SimulationArtifact)
                .where(
                    SimulationArtifact.scenario_id == scenario_id,
                    SimulationArtifact.run_id == artifact_session_id,
                )
                .order_by(SimulationArtifact.round_number, SimulationArtifact.id)
            )
        )
        influences = session.execute(
            select(AgentInfluenceObservation, Agent.name)
            .join(Agent, AgentInfluenceObservation.agent_id == Agent.id)
            .where(
                AgentInfluenceObservation.scenario_id == scenario_id,
                AgentInfluenceObservation.run_id == artifact_session_id,
            )
            .order_by(AgentInfluenceObservation.id)
        ).all()
        schema_validity = (
            round(
                sum(log.is_schema_valid for log in reasoning_logs)
                / len(reasoning_logs)
                * 100.0,
                2,
            )
            if reasoning_logs
            else 0.0
        )

        return {
            "session_id": active_session_id,
            "session_status": active_session.status if active_session else None,
            "scenario": {
                "id": scenario.id,
                "description": scenario.description,
                "program_cost": scenario.program_cost,
                "max_deficit_constraint": scenario.max_deficit_constraint,
            },
            "domain_rules": domain_rules,
            "latest_metric": _metric_payload(snapshots[0]) if snapshots else None,
            "metric_history": [_metric_payload(snapshot) for snapshot in snapshots],
            "schema_validity_percent": schema_validity,
            "reasoning_log_count": len(reasoning_logs),
            "disagreements": [
                {
                    "id": log.id,
                    "agent_i": left_name,
                    "agent_j": right_name,
                    "dE": log.dE,
                    "dA": log.dA,
                    "dP": log.dP,
                    "dR": log.dR,
                    "dU": log.dU,
                    "dO": log.dO,
                    "dC": log.dC,
                    "dREC": log.dREC,
                    "resolution_mechanism": _resolution_mechanism(log),
                }
                for log, left_name, right_name in disagreements
            ],
            "simulation_artifacts": [
                _simulation_payload(artifact) for artifact in simulations
            ],
            "influence_observations": [
                {
                    "agent": agent_name,
                    "proposition": observation.proposition,
                    "X": observation.X,
                    "Q": observation.Q,
                    "H": observation.H,
                    "S": observation.S,
                    "U": observation.U,
                    "gate": observation.gate,
                    "raw_score": observation.raw_score,
                    "normalized_weight": observation.normalized_weight,
                }
                for observation, agent_name in influences
            ],
        }


@router.post("/scenarios/{scenario_id}/runs", status_code=status.HTTP_202_ACCEPTED)
def start_run(scenario_id: int) -> dict[str, Any]:
    session_id = str(uuid4())
    queue_logs = [
        {
            "stage": "QUEUE",
            "level": "INFO",
            "message": "Cycle queued for worker execution.",
        }
    ]
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        agents = list(session.scalars(select(Agent).order_by(Agent.id)))
        if len(agents) < 2:
            raise HTTPException(
                status_code=409,
                detail="At least two configured agents are required for deliberation",
            )
        current_revision = agent_revision(agents, scenario)
        mandate_snapshot = session.scalar(
            select(ScenarioMandateSnapshot).where(
                ScenarioMandateSnapshot.scenario_id == scenario_id,
                ScenarioMandateSnapshot.revision == current_revision,
            )
        )
        if mandate_snapshot is None:
            stale_snapshot = session.scalar(
                select(ScenarioMandateSnapshot)
                .where(ScenarioMandateSnapshot.scenario_id == scenario_id)
                .order_by(
                    ScenarioMandateSnapshot.updated_at.desc(),
                    ScenarioMandateSnapshot.id.desc(),
                )
            )
            if stale_snapshot is not None:
                mandate_snapshot = refresh_mandate_snapshot(
                    session,
                    scenario,
                    agents,
                    stale_snapshot,
                )
        snapshot_agent_ids = {
            item.get("agent_id")
            for item in mandate_snapshot.agent_rules
            if isinstance(item, dict)
        } if mandate_snapshot is not None else set()
        if mandate_snapshot is not None and (
            not mandate_snapshot.generated
            or mandate_snapshot.agent_count != len(agents)
            or snapshot_agent_ids != {agent.id for agent in agents}
        ):
            mandate_snapshot = refresh_mandate_snapshot(
                session,
                scenario,
                agents,
                mandate_snapshot,
            )
            snapshot_agent_ids = {
                item.get("agent_id")
                for item in mandate_snapshot.agent_rules
                if isinstance(item, dict)
            }
        if (
            mandate_snapshot is None
            or not mandate_snapshot.generated
            or mandate_snapshot.agent_count != len(agents)
            or snapshot_agent_ids != {agent.id for agent in agents}
        ):
            raise HTTPException(
                status_code=409,
                detail="Generate current mandates for every agent before deliberation",
            )
        invalid_agents: list[str] = []
        for agent in agents:
            try:
                resolve_llm_runtime_config(agent)
            except ValueError:
                invalid_agents.append(agent.name)
        if invalid_agents:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Database LLM configuration is required for every agent: "
                    + ", ".join(invalid_agents)
                ),
            )
        run = ConsensusSession(
            id=session_id,
            scenario_id=scenario_id,
            mandate_snapshot_id=mandate_snapshot.id,
            mandate_revision=mandate_snapshot.revision,
            mandate_payload={
                "rules": mandate_snapshot.rules,
                "agent_rules": mandate_snapshot.agent_rules,
            },
            status="QUEUED",
            logs=queue_logs,
            progress_stage="QUEUE",
        )
        session.add(run)
        session.commit()
    try:
        task = celery_client.send_task(
            "shcr.run_full_shcr_cycle",
            args=[scenario_id, session_id],
        )
    except Exception as error:
        with SessionLocal() as session:
            run_record = session.get(ConsensusSession, session_id)
            if run_record is not None:
                run_record.status = "FAILED"
                run_record.error = f"Task dispatch failed: {type(error).__name__}: {error}"
                run_record.progress_stage = "DISPATCH_FAILED"
                run_record.completed_at = datetime.now(timezone.utc)
                session.commit()
        raise HTTPException(
            status_code=503,
            detail="Consensus cycle could not be dispatched to the worker",
        ) from error
    with SessionLocal() as session:
        run_record = session.get(ConsensusSession, session_id)
        if run_record is not None:
            run_record.celery_task_id = task.id
            session.commit()
    return {
        "task_id": task.id,
        "session_id": session_id,
        "scenario_id": scenario_id,
        "status": "QUEUED",
        "logs": queue_logs,
    }


def _run_payload(session: ConsensusSession) -> dict[str, Any]:
    with SessionLocal() as database:
        simulations = list(
            database.scalars(
                select(SimulationArtifact)
                .where(SimulationArtifact.run_id == session.id)
                .order_by(SimulationArtifact.round_number, SimulationArtifact.id)
            )
        )
    return {
        "task_id": session.celery_task_id,
        "session_id": session.id,
        "scenario_id": session.scenario_id,
        "status": session.status,
        "logs": list(session.logs or []),
        "simulation_artifacts": [
            _simulation_payload(artifact) for artifact in simulations
        ],
        "result": session.result_payload,
        "error": session.error,
        "progress_stage": session.progress_stage,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
    }


@router.get("/scenarios/{scenario_id}/runs")
def scenario_runs(scenario_id: int) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        if session.get(Scenario, scenario_id) is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        runs = list(
            session.scalars(
                select(ConsensusSession)
                .where(ConsensusSession.scenario_id == scenario_id)
                .order_by(ConsensusSession.created_at.desc(), ConsensusSession.id.desc())
            )
        )
        return [_run_payload(run) for run in runs]


@router.get("/scenarios/{scenario_id}/runs/latest")
def latest_scenario_run(scenario_id: int) -> dict[str, Any]:
    with SessionLocal() as session:
        if session.get(Scenario, scenario_id) is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        run = session.scalar(
            select(ConsensusSession)
            .where(ConsensusSession.scenario_id == scenario_id)
            .order_by(ConsensusSession.created_at.desc(), ConsensusSession.id.desc())
        )
        if run is None:
            raise HTTPException(status_code=404, detail="No consensus runs found")
        return _run_payload(run)


@router.get("/runs/{task_id}")
def run_status(task_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        run = session.scalar(
            select(ConsensusSession).where(ConsensusSession.celery_task_id == task_id)
        )
        if run is not None:
            return _run_payload(run)
    result = AsyncResult(task_id, app=celery_client)
    state_map = {
        "PENDING": "QUEUED",
        "RECEIVED": "QUEUED",
        "STARTED": "RUNNING",
        "PROGRESS": "RUNNING",
        "SUCCESS": "SUCCEEDED",
        "FAILURE": "FAILED",
        "RETRY": "RUNNING",
        "REVOKED": "FAILED",
    }
    payload: dict[str, Any] = {
        "task_id": task_id,
        "status": state_map.get(result.state, result.state),
        "logs": [],
        "result": None,
        "session_id": None,
        "scenario_id": None,
    }
    if isinstance(result.info, dict):
        payload["logs"] = result.info.get("logs", [])
    if result.successful():
        payload["result"] = result.result
        if isinstance(result.result, dict):
            payload["logs"] = result.result.get("logs", payload["logs"])
    elif result.failed():
        payload["error"] = str(result.result)
    return payload


@router.get("/scenarios/{scenario_id}/dashboard")
def scenario_dashboard(
    scenario_id: int,
    session_id: str | None = Query(default=None),
) -> dict[str, Any]:
    return _dashboard_payload(scenario_id, session_id)


@router.get("/scenarios/{scenario_id}/manifest")
def reproducibility_manifest(
    scenario_id: int,
    session_id: str | None = Query(default=None),
) -> JSONResponse:
    dashboard = _dashboard_payload(scenario_id, session_id)
    with SessionLocal() as session:
        agents = list(session.scalars(select(Agent).order_by(Agent.id)))
        reasoning = session.execute(
            select(ReasoningLog, Agent)
            .join(Agent, ReasoningLog.agent_id == Agent.id)
            .where(
                ReasoningLog.scenario_id == scenario_id,
                ReasoningLog.run_id == dashboard["session_id"],
            )
            .order_by(ReasoningLog.id)
        ).all()
        mandate_by_agent = {
            item["agent_id"]: item.get("scenario_mandate")
            for item in dashboard["domain_rules"]["agent_rules"]
            if isinstance(item, dict) and isinstance(item.get("agent_id"), int)
        }
        manifest = {
            "manifest_version": "1.0",
            "framework": "SHCR = SRR + (RAR -> DAI) + DDR + CAR",
            "session_id": dashboard["session_id"],
            "scenario": dashboard["scenario"],
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "role": agent.role,
                    "template_key": agent.template_key,
                    "theta_x": agent.theta_x,
                    "theta_q": agent.theta_q,
                    "theta_h": agent.theta_h,
                    "theta_s": agent.theta_s,
                    "theta_u": agent.theta_u,
                    "llm_base_url": agent.llm_base_url,
                    "llm_model": agent.llm_model,
                    "system_prompt": agent.system_prompt,
                    "temperature": agent.temperature,
                    "max_tokens": agent.max_tokens,
                    "has_llm_api_key": bool(agent.llm_api_key),
                }
                for agent in agents
            ],
            "prompts": [
                {
                    "agent_id": agent.id,
                     "system": build_agent_system_prompt(
                         agent.role,
                         resolve_agent_system_prompt(agent),
                         mandate_by_agent.get(agent.id),
                     ),
                    "user": (
                        f"Agent role: {agent.role}\n"
                        f"Policy goal: {dashboard['scenario']['description']}\n"
                        f"Program cost: {dashboard['scenario']['program_cost']}\n"
                        f"Automatic legal deficit ceiling: "
                        f"{dashboard['scenario']['max_deficit_constraint']}%"
                    ),
                }
                for agent in agents
            ],
            "llm_outputs": [
                {
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "raw_json": log.raw_json,
                    "parsed_srr_objects": log.parsed_srr_objects,
                    "is_schema_valid": log.is_schema_valid,
                    "provenance_count": log.provenance_count,
                }
                for log, agent in reasoning
            ],
            "influence_observations": dashboard["influence_observations"],
            "disagreements": dashboard["disagreements"],
            "simulation_artifacts": dashboard["simulation_artifacts"],
            "metrics": dashboard["metric_history"],
            "schema_validity_percent": dashboard["schema_validity_percent"],
        }
    return JSONResponse(
        manifest,
        headers={
            "Content-Disposition": f'attachment; filename="shcr-scenario-{scenario_id}-manifest.json"'
        },
    )
