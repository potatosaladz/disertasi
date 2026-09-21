from typing import Any

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import aliased

from .celery_client import celery_client
from .core_algorithms import build_agent_system_prompt
from .database import SessionLocal
from .models import (
    Agent,
    AgentInfluenceObservation,
    DisagreementLog,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
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


def _metric_payload(snapshot: MetricSnapshot) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "scenario_id": snapshot.scenario_id,
        "hard_constraint_violation_rate": snapshot.hard_constraint_violation_rate,
        "provenance_completeness_percent": snapshot.provenance_completeness_percent,
        "material_information_retention_macro_f1": snapshot.material_information_retention_macro_f1,
        "feasible_alternatives_count": snapshot.feasible_alternatives_count,
        "convergence_status": snapshot.convergence_status.value,
        "latency_ms": snapshot.latency_ms,
        "token_usage": snapshot.token_usage,
        "created_at": snapshot.created_at.isoformat(),
    }


def _dashboard_payload(scenario_id: int) -> dict[str, Any]:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")

        snapshots = list(
            session.scalars(
                select(MetricSnapshot)
                .where(MetricSnapshot.scenario_id == scenario_id)
                .order_by(MetricSnapshot.created_at.desc(), MetricSnapshot.id.desc())
            )
        )
        reasoning_logs = list(
            session.scalars(
                select(ReasoningLog)
                .where(ReasoningLog.scenario_id == scenario_id)
                .order_by(ReasoningLog.id)
            )
        )
        agent_i = aliased(Agent)
        agent_j = aliased(Agent)
        disagreements = session.execute(
            select(DisagreementLog, agent_i.name, agent_j.name)
            .join(agent_i, DisagreementLog.agent_i == agent_i.id)
            .join(agent_j, DisagreementLog.agent_j == agent_j.id)
            .where(DisagreementLog.scenario_id == scenario_id)
            .order_by(DisagreementLog.id)
        ).all()
        influences = session.execute(
            select(AgentInfluenceObservation, Agent.name)
            .join(Agent, AgentInfluenceObservation.agent_id == Agent.id)
            .where(AgentInfluenceObservation.scenario_id == scenario_id)
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
            "scenario": {
                "id": scenario.id,
                "description": scenario.description,
                "max_deficit_constraint": scenario.max_deficit_constraint,
            },
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
    with SessionLocal() as session:
        if session.get(Scenario, scenario_id) is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
    task = celery_client.send_task("shcr.run_full_shcr_cycle", args=[scenario_id])
    return {
        "task_id": task.id,
        "scenario_id": scenario_id,
        "status": "QUEUED",
        "logs": ["Cycle queued for worker execution."],
    }


@router.get("/runs/{task_id}")
def run_status(task_id: str) -> dict[str, Any]:
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
    }
    if isinstance(result.info, dict):
        payload["logs"] = result.info.get("logs", [])
    if result.successful():
        payload["result"] = result.result
        payload["logs"] = [
            "Executing SRR...",
            "Calculating DDR vector...",
            "Applying CAR filter...",
            "Persisting dissertation metrics...",
            "SHCR cycle completed.",
        ]
    elif result.failed():
        payload["error"] = str(result.result)
    return payload


@router.get("/scenarios/{scenario_id}/dashboard")
def scenario_dashboard(scenario_id: int) -> dict[str, Any]:
    return _dashboard_payload(scenario_id)


@router.get("/scenarios/{scenario_id}/manifest")
def reproducibility_manifest(scenario_id: int) -> JSONResponse:
    dashboard = _dashboard_payload(scenario_id)
    with SessionLocal() as session:
        agents = list(session.scalars(select(Agent).order_by(Agent.id)))
        reasoning = session.execute(
            select(ReasoningLog, Agent)
            .join(Agent, ReasoningLog.agent_id == Agent.id)
            .where(ReasoningLog.scenario_id == scenario_id)
            .order_by(ReasoningLog.id)
        ).all()
        manifest = {
            "manifest_version": "1.0",
            "framework": "SHCR = SRR + (RAR -> DAI) + DDR + CAR",
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
                    "system": build_agent_system_prompt(agent.role, agent.system_prompt),
                    "user": f"Agent role: {agent.role}\nScenario: {dashboard['scenario']['description']}",
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
            "metrics": dashboard["metric_history"],
            "schema_validity_percent": dashboard["schema_validity_percent"],
        }
    return JSONResponse(
        manifest,
        headers={
            "Content-Disposition": f'attachment; filename="shcr-scenario-{scenario_id}-manifest.json"'
        },
    )
