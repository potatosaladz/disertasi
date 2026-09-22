from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import aliased

from .models import (
    Agent,
    AgentInfluenceObservation,
    ConsensusSession,
    DisagreementLog,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
    SimulationArtifact,
)

GraphStatus = str

_PRIVATE_DETAIL_KEYS = {
    "analysis",
    "chain_of_thought",
    "chainofthought",
    "hidden_reasoning",
    "reasoning_trace",
    "thoughts",
}

_STAGE_ORDER = {
    "QUEUE": 0,
    "INITIALIZE": 1,
    "SRR": 2,
    "CONSENSUS": 3,
    "DDR": 4,
    "SIMULATION": 5,
    "SIMULATION_CONSENSUS": 6,
    "CAR": 7,
    "METRICS": 8,
    "COMPLETE": 9,
}


def _node(
    identifier: str,
    kind: str,
    label: str,
    status: GraphStatus,
    level: int,
    lane: int,
    *,
    subtitle: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "kind": kind,
        "label": label,
        "subtitle": subtitle,
        "status": status,
        "position": {"x": lane * 260, "y": level * 170},
        "details": _safe_detail(details or {}),
    }


def _edge(
    identifier: str,
    source: str,
    target: str,
    kind: str,
    status: GraphStatus,
    label: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "source": source,
        "target": target,
        "kind": kind,
        "status": status,
        "label": label,
        "animated": status == "RUNNING",
        "details": _safe_detail(details or {}),
    }


def _reached_stage(logs: list[dict[str, Any]], stage: str) -> bool:
    threshold = _STAGE_ORDER[stage]
    return any(_STAGE_ORDER.get(str(log.get("stage")), -1) >= threshold for log in logs)


def _stage_status(
    run: ConsensusSession,
    logs: list[dict[str, Any]],
    stage: str,
) -> GraphStatus:
    stage_logs = [log for log in logs if log.get("stage") == stage]
    if any(log.get("level") == "ERROR" for log in stage_logs):
        return "FAILED"
    if run.status == "RUNNING" and run.progress_stage == stage:
        return "RUNNING"
    if any(log.get("level") == "WARNING" for log in stage_logs):
        return "WARNING"
    if stage_logs or _reached_stage(logs, stage):
        return "SUCCEEDED"
    if run.status == "FAILED" and stage == "COMPLETE":
        return "FAILED"
    return "PENDING"


def _edge_status(source_status: GraphStatus, target_status: GraphStatus) -> GraphStatus:
    if target_status == "FAILED":
        return "FAILED"
    if target_status == "RUNNING":
        return "RUNNING"
    if target_status == "WARNING" or source_status == "WARNING":
        return "WARNING"
    if source_status in {"SUCCEEDED", "WARNING"} and target_status in {"SUCCEEDED", "WARNING"}:
        return "SUCCEEDED"
    return "PENDING"


def _safe_detail(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _safe_detail(item)
            for key, item in value.items()
            if key.casefold() not in _PRIVATE_DETAIL_KEYS
        }
    if isinstance(value, list):
        return [_safe_detail(item) for item in value]
    return value


def _safe_log(log: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "stage": log.get("stage"),
        "level": log.get("level"),
        "message": log.get("message"),
    }


def run_graph_payload(database: Any, run: ConsensusSession) -> dict[str, Any]:
    scenario = database.get(Scenario, run.scenario_id)
    if scenario is None:
        raise ValueError("Scenario for consensus run does not exist")
    mandate_agent_ids = [
        item.get("agent_id")
        for item in run.mandate_payload.get("agent_rules", [])
        if isinstance(item, dict) and isinstance(item.get("agent_id"), int)
    ]
    agents = list(
        database.scalars(
            select(Agent)
            .where(Agent.id.in_(mandate_agent_ids))
            .order_by(Agent.id)
        )
    )
    reasoning = {
        item.agent_id: item
        for item in database.scalars(
            select(ReasoningLog).where(ReasoningLog.run_id == run.id)
        )
    }
    left = aliased(Agent)
    right = aliased(Agent)
    disagreement_rows = database.execute(
        select(DisagreementLog, left.name, right.name)
        .join(left, DisagreementLog.agent_i == left.id)
        .join(right, DisagreementLog.agent_j == right.id)
        .where(DisagreementLog.run_id == run.id)
        .order_by(DisagreementLog.id)
    ).all()
    simulations = list(
        database.scalars(
            select(SimulationArtifact)
            .where(SimulationArtifact.run_id == run.id)
            .order_by(SimulationArtifact.round_number, SimulationArtifact.id)
        )
    )
    influence_observations = list(
        database.scalars(
            select(AgentInfluenceObservation)
            .where(AgentInfluenceObservation.run_id == run.id)
            .order_by(AgentInfluenceObservation.id)
        )
    )
    metric = database.scalar(
        select(MetricSnapshot)
        .where(MetricSnapshot.run_id == run.id)
        .order_by(MetricSnapshot.created_at.desc(), MetricSnapshot.id.desc())
    )
    logs = [dict(item) for item in (run.logs or [])]
    log_refs = {
        stage: [
            _safe_log(log, index)
            for index, log in enumerate(logs)
            if log.get("stage") == stage
        ]
        for stage in _STAGE_ORDER
    }

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    status_by_node: dict[str, GraphStatus] = {}

    def add_node(payload: dict[str, Any]) -> None:
        nodes.append(payload)
        status_by_node[payload["id"]] = payload["status"]

    scenario_status = "SUCCEEDED"
    add_node(
        _node(
            "scenario:start",
            "scenario",
            "Mulai Skenario & Sesi Baru",
            scenario_status,
            0,
            2,
            subtitle=f"Scenario #{scenario.id}",
            details={
                "description": scenario.description,
                "program_cost": scenario.program_cost,
                "max_deficit_constraint": scenario.max_deficit_constraint,
                "session_id": run.id,
                "task_id": run.celery_task_id,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            },
        )
    )
    mandate_status = "SUCCEEDED"
    add_node(
        _node(
            "stage:mandate",
            "stage",
            "1. Sintesis Mandat Agen Sektoral",
            mandate_status,
            1,
            2,
            subtitle=run.mandate_revision,
            details={
                "mandate_revision": run.mandate_revision,
                "mandate_snapshot_id": run.mandate_snapshot_id,
                "agent_count": len(agents),
                "logs": log_refs["INITIALIZE"],
            },
        )
    )
    agent_status = _stage_status(run, logs, "SRR")
    agent_rows = [0, 1, 2, 3, 4]
    for index, agent in enumerate(agents):
        reasoning_log = reasoning.get(agent.id)
        matching_logs = [
            item for item in log_refs["SRR"] if agent.name in str(item.get("message"))
        ]
        if reasoning_log is not None:
            status_value = "SUCCEEDED" if reasoning_log.is_schema_valid else "FAILED"
        elif any(item.get("level") == "ERROR" for item in matching_logs):
            status_value = "FAILED"
        elif any(item.get("level") == "SUCCESS" for item in matching_logs):
            status_value = "SUCCEEDED"
        elif (
            agent_status == "RUNNING"
            and matching_logs
            and matching_logs[-1].get("level") == "INFO"
        ):
            status_value = "RUNNING"
        elif matching_logs:
            status_value = "WARNING"
        else:
            status_value = "PENDING"
        add_node(
            _node(
                f"agent:{agent.id}",
                "agent",
                agent.name.split(" / ", maxsplit=1)[0],
                status_value,
                2,
                agent_rows[index] if index < len(agent_rows) else index,
                subtitle=agent.role,
                details={
                    "agent_id": agent.id,
                    "template_key": agent.template_key,
                    "role": agent.role,
                    "llm_model": agent.llm_model,
                    "mandate": next(
                        (
                            item.get("scenario_mandate")
                            for item in run.mandate_payload.get("agent_rules", [])
                            if isinstance(item, dict) and item.get("agent_id") == agent.id
                        ),
                        None,
                    ),
                    "reasoning_log_id": reasoning_log.id if reasoning_log else None,
                    "schema_valid": reasoning_log.is_schema_valid if reasoning_log else None,
                    "provenance_count": reasoning_log.provenance_count if reasoning_log else 0,
                    "decision_artifacts": reasoning_log.parsed_srr_objects if reasoning_log else {},
                    "logs": [
                        item
                        for item in log_refs["SRR"]
                        if agent.name in str(item.get("message"))
                    ],
                },
            )
        )
        edges.append(
            _edge(
                f"edge:mandate-agent:{agent.id}",
                "stage:mandate",
                f"agent:{agent.id}",
                "mandate",
                _edge_status(mandate_status, status_value),
                "Generate mandat & parameter",
            )
        )

    peer_status = _stage_status(run, logs, "CONSENSUS")
    add_node(
        _node(
            "stage:peer-review",
            "stage",
            "2. Peer Review & Exchange Outputs",
            peer_status,
            3,
            2,
            details={"logs": log_refs["CONSENSUS"], "participating_agents": len(reasoning)},
        )
    )
    for agent in agents:
        edges.append(
            _edge(
                f"edge:agent-peer:{agent.id}",
                f"agent:{agent.id}",
                "stage:peer-review",
                "peer-output",
                _edge_status(status_by_node[f"agent:{agent.id}"], peer_status),
                "Peer output",
            )
        )

    rar_status = (
        "SUCCEEDED"
        if any(item.normalized_weight is not None for item in influence_observations)
        else "RUNNING"
        if run.status == "RUNNING" and run.progress_stage == "DDR"
        else "PENDING"
    )
    add_node(
        _node(
            "stage:rar-dai",
            "stage",
            "RAR → DAI Dynamic Influence",
            rar_status,
            4,
            2,
            details={
                "observation_count": len(influence_observations),
                "weighted_observation_count": sum(
                    item.normalized_weight is not None for item in influence_observations
                ),
                "weights": [
                    {
                        "agent_id": item.agent_id,
                        "proposition": item.proposition,
                        "raw_score": item.raw_score,
                        "normalized_weight": item.normalized_weight,
                    }
                    for item in influence_observations
                ],
            },
        )
    )
    edges.append(
        _edge(
            "edge:peer-rar-dai",
            "stage:peer-review",
            "stage:rar-dai",
            "influence",
            _edge_status(peer_status, rar_status),
            "RAR scoring → DAI weights",
        )
    )

    ddr_status = _stage_status(run, logs, "DDR")
    if disagreement_rows and any(row[0].dP for row in disagreement_rows):
        ddr_status = "WARNING" if ddr_status != "RUNNING" else ddr_status
    conflict_details = [
        {
            "id": item.id,
            "agent_i": left_name,
            "agent_j": right_name,
            "components": [
                component
                for component in ("dE", "dA", "dP", "dR", "dU", "dO", "dC", "dREC")
                if getattr(item, component)
            ],
            "route": item.resolution_route,
        }
        for item, left_name, right_name in disagreement_rows
    ]
    add_node(
        _node(
            "stage:ddr",
            "decision",
            "3. DDR: Konflik / Deadlock?",
            ddr_status,
            5,
            2,
            details={
                "disagreement_count": len(disagreement_rows),
                "simulation_trigger_count": sum(bool(item.dP) for item, _, _ in disagreement_rows),
                "conflicts": conflict_details,
                "logs": log_refs["DDR"],
            },
        )
    )
    edges.append(
        _edge(
            "edge:rar-dai-ddr",
            "stage:rar-dai",
            "stage:ddr",
            "pipeline",
            _edge_status(rar_status, ddr_status),
            "Weighted divergence analysis",
        )
    )

    previous_feedback = "stage:ddr"
    for artifact in simulations:
        simulation_id = f"simulation:{artifact.round_number}"
        simulation_status = artifact.status
        if artifact.status == "SUCCEEDED" and artifact.output_payload.get("fallback_reason"):
            simulation_status = "WARNING"
        add_node(
            _node(
                simulation_id,
                "simulation",
                f"4–5. Simulation Agent / Round {artifact.round_number}",
                simulation_status,
                6,
                artifact.round_number - 1,
                subtitle="Native macro-fiscal arbiter",
                details={
                    "artifact_id": artifact.id,
                    "trigger": artifact.trigger,
                    "round_number": artifact.round_number,
                    "simulation_version": artifact.simulation_version,
                    "input": artifact.input_payload,
                    "output": artifact.output_payload,
                    "latency_ms": artifact.latency_ms,
                    "token_usage": artifact.token_usage,
                    "created_at": artifact.created_at.isoformat(),
                },
            )
        )
        edges.append(
            _edge(
                f"edge:ddr-simulation:{artifact.round_number}",
                previous_feedback,
                simulation_id,
                "escalation",
                _edge_status(status_by_node[previous_feedback], simulation_status),
                "Simulation Agent Requested",
                {"round_number": artifact.round_number},
            )
        )
        consensus_id = f"stage:simulation-consensus:{artifact.round_number}"
        follow_up = artifact.output_payload.get("follow_up_consensus_status")
        consensus_status = (
            "RUNNING"
            if run.status == "RUNNING" and run.progress_stage in {"SIMULATION_CONSENSUS", "CONSENSUS"}
            and artifact.round_number == len(simulations)
            else "SUCCEEDED" if follow_up == "completed" else "PENDING"
        )
        remaining = artifact.output_payload.get("remaining_prediction_conflicts")
        if isinstance(remaining, int) and remaining > 0 and consensus_status == "SUCCEEDED":
            consensus_status = "WARNING"
        add_node(
            _node(
                consensus_id,
                "stage",
                f"Resolusi → Peer Review / Round {artifact.round_number}",
                consensus_status,
                7,
                artifact.round_number - 1,
                details={
                    "artifact_id": artifact.id,
                    "remaining_prediction_conflicts": remaining,
                    "follow_up_consensus_status": follow_up,
                    "logs": log_refs["SIMULATION_CONSENSUS"],
                },
            )
        )
        edges.extend(
            [
                _edge(
                    f"edge:simulation-feedback:{artifact.round_number}",
                    simulation_id,
                    consensus_id,
                    "feedback",
                    _edge_status(simulation_status, consensus_status),
                    "Umpan balik hasil resolusi",
                ),
                _edge(
                    f"edge:consensus-loop:{artifact.round_number}",
                    consensus_id,
                    "stage:ddr",
                    "loop",
                    "WARNING" if remaining else _edge_status(consensus_status, ddr_status),
                    f"{remaining if isinstance(remaining, int) else '—'} konflik prediksi tersisa",
                ),
            ]
        )
        previous_feedback = consensus_id

    quorum_status = (
        "FAILED"
        if run.status == "FAILED" and any("quorum" in str(log.get("message", "")).lower() for log in logs)
        else "RUNNING" if run.status == "RUNNING" and run.progress_stage == "CAR"
        else "SUCCEEDED" if metric is not None else "PENDING"
    )
    add_node(
        _node(
            "stage:quorum",
            "decision",
            "6. Pengujian Quorum & Validasi Artefak",
            quorum_status,
            8,
            2,
            details={
                "schema_valid_count": sum(item.is_schema_valid for item in reasoning.values()),
                "reasoning_log_count": len(reasoning),
                "hard_constraint_violation_rate": metric.hard_constraint_violation_rate if metric else None,
                "feasible_alternatives_count": metric.feasible_alternatives_count if metric else None,
                "logs": log_refs["CAR"],
            },
        )
    )
    ddr_to_quorum_status = _edge_status(ddr_status, quorum_status)
    edges.append(
        _edge(
            "edge:ddr-quorum",
            "stage:ddr",
            "stage:quorum",
            "validation",
            ddr_to_quorum_status,
            "Tidak ada eskalasi / bounded completion",
        )
    )
    edges.append(
        _edge(
            "edge:quorum-retry",
            "stage:quorum",
            "stage:peer-review",
            "retry",
            "FAILED" if quorum_status == "FAILED" else "PENDING",
            "Gagal → review ulang",
        )
    )

    final_status = (
        "FAILED" if run.status == "FAILED" else "SUCCEEDED" if run.status == "SUCCEEDED" else "PENDING"
    )
    add_node(
        _node(
            "consensus:final",
            "consensus",
            "7. Konsensus Final & Output Kebijakan",
            final_status,
            9,
            2,
            subtitle=metric.convergence_status.value if metric else run.status,
            details={
                "run_status": run.status,
                "error": run.error,
                "result": run.result_payload,
                "metric_snapshot_id": metric.id if metric else None,
                "convergence_status": metric.convergence_status.value if metric else None,
                "provenance_completeness_percent": metric.provenance_completeness_percent if metric else None,
                "logs": log_refs["COMPLETE"],
            },
        )
    )
    edges.append(
        _edge(
            "edge:quorum-final",
            "stage:quorum",
            "consensus:final",
            "completion",
            _edge_status(quorum_status, final_status),
            "Quorum lolos",
        )
    )
    edges.insert(
        0,
        _edge(
            "edge:start-mandate",
            "scenario:start",
            "stage:mandate",
            "pipeline",
            _edge_status(scenario_status, mandate_status),
            "Inisialisasi sesi",
        ),
    )

    return {
        "schema_version": "1.0",
        "task_id": run.celery_task_id,
        "session_id": run.id,
        "scenario_id": run.scenario_id,
        "run_status": run.status,
        "progress_stage": run.progress_stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "agent_count": len(agents),
            "disagreement_count": len(disagreement_rows),
            "simulation_round_count": len(simulations),
            "simulation_triggered": bool(simulations),
            "quorum_status": quorum_status,
            "convergence_status": metric.convergence_status.value if metric else None,
        },
    }
