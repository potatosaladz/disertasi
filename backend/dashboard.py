from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import aliased

from .agent_templates import (
    agent_revision,
    agent_utility_metadata,
    ensure_phase_one_specialists,
    resolve_agent_system_prompt,
    scenario_deliberative_agents,
    semantic_agent_name,
)
from .analytical_events import analytical_event
from .celery_client import celery_client
from .core_algorithms import (
    STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
    build_agent_system_prompt,
    build_agent_user_prompt,
    describe_divergence_vector,
    resolve_disagreement_route,
    resolve_llm_runtime_config,
)
from .database import SessionLocal
from .localization import DEFAULT_LANGUAGE, Language, localize_payload
from .global_config import get_global_llm_config, global_config_snapshot
from .graph_network import run_graph_payload
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
from .sanitization import sanitize_public_error, sanitize_public_value

router = APIRouter(prefix="/api")

_AUTO_POLLING_STATES = {
    "PENDING",
    "PROCESSING",
    "TEMPORARY_HYDRATION_DELAY",
}


def _polling_contract(
    status_value: str,
    result_payload: object = None,
) -> dict[str, object]:
    normalized = status_value.upper()
    if normalized in {"QUEUED", "PENDING"}:
        polling_state = "PENDING"
    elif normalized in {"RUNNING", "PROCESSING"}:
        polling_state = "PROCESSING"
    elif normalized == "TEMPORARY_HYDRATION_DELAY" or (
        normalized == "SUCCEEDED" and result_payload is None
    ):
        polling_state = "TEMPORARY_HYDRATION_DELAY"
    elif normalized == "SUCCEEDED":
        polling_state = "SUCCEEDED"
    else:
        polling_state = "FAILED"
    should_poll = polling_state in _AUTO_POLLING_STATES
    return {
        "polling_state": polling_state,
        "should_poll": should_poll,
        "terminal": not should_poll,
    }


def _disagreement_vector(log: DisagreementLog) -> dict[str, bool]:
    return {
        component: bool(getattr(log, component))
        for component in ("dE", "dA", "dP", "dR", "dU", "dO", "dC", "dREC")
    }


def _resolution_mechanism(log: DisagreementLog) -> str:
    return log.resolution_route or resolve_disagreement_route(
        _disagreement_vector(log)
    )


def _disagreement_payload(
    log: DisagreementLog,
    left_name: str,
    right_name: str,
    simulations: list[SimulationArtifact],
    car_result: dict[str, Any] | None = None,
    left_display_name: str | None = None,
    right_display_name: str | None = None,
) -> dict[str, Any]:
    vector = _disagreement_vector(log)
    detail = (
        _safe_decision_value(log.detail_payload)
        if isinstance(log.detail_payload, dict)
        else {}
    )
    if not isinstance(detail, dict):
        detail = {}
    categories = detail.get("categories")
    if not isinstance(categories, list):
        categories = describe_divergence_vector(vector)
    vector_metadata = detail.get("vector_metadata")
    if not isinstance(vector_metadata, dict):
        vector_metadata = {
            component: {
                "component": component,
                "active": bool(active),
                "status": "detected" if active else "no-divergence",
                "calculation": {
                    "status": "not-calculated",
                    "value": None,
                    "reason": "Legacy disagreement record has no granular calculation metadata.",
                },
                "resolution_path": resolve_disagreement_route({component: active})
                if active
                else "No Resolution Required",
            }
            for component, active in vector.items()
        }
    matching_simulations = [
        artifact
        for artifact in simulations
        if any(
            isinstance(conflict, dict)
            and {conflict.get("agent_i"), conflict.get("agent_j")}
            == {left_name, right_name}
            for conflict in (
                artifact.input_payload.get("conflicts", [])
                if isinstance(artifact.input_payload, dict)
                else []
            )
        )
    ]
    if not matching_simulations and log.dP:
        matching_simulations = [
            artifact
            for artifact in simulations
            if isinstance(artifact.input_payload, dict)
            and any(
                isinstance(conflict, dict)
                and "dP" in conflict.get("components", [])
                and not conflict.get("agent_i")
                and not conflict.get("agent_j")
                for conflict in artifact.input_payload.get("conflicts", [])
            )
        ]
    latest_simulation = matching_simulations[-1] if matching_simulations else None
    resolution = detail.get("resolution")
    if not isinstance(resolution, dict):
        resolution = {
            "route": _resolution_mechanism(log),
            "status": "ESCALATED" if any(vector.values()) else "CLEAR",
        }
    fiscal_calculation = detail.get("fiscal_calculation")
    if not isinstance(fiscal_calculation, dict):
        fiscal_calculation = {}
    fiscal_calculation = {
        **fiscal_calculation,
        "compromise_formula": "A* = argmax utility(A), subject to projected_deficit(A) <= statutory_deficit_ceiling",
    }
    if latest_simulation is not None:
        output = latest_simulation.output_payload
        modelled_alternatives = output.get("alternatives", [])
        if isinstance(modelled_alternatives, list):
            fiscal_calculation["arbiter_alternatives"] = modelled_alternatives
            selected = output.get("selected_alternative")
            if not isinstance(selected, dict):
                selected = None
            if selected is not None:
                projected_deficit = selected.get("deficit")
                ceiling = fiscal_calculation.get(
                    "effective_deficit_ceiling_percent",
                    fiscal_calculation.get("statutory_deficit_ceiling_percent"),
                )
                fiscal_calculation["selected_compromise"] = {
                    **selected,
                    "headroom_percent": (
                        round(float(ceiling) - float(projected_deficit), 4)
                        if isinstance(ceiling, (int, float))
                        and isinstance(projected_deficit, (int, float))
                        else None
                    ),
                }
        resolution = {
            **resolution,
            "status": latest_simulation.status,
            "simulation_round": latest_simulation.round_number,
            "arbiter_conclusion": output.get("resolution"),
            "simulation_summary": output.get("simulation_summary"),
            "remaining_prediction_conflicts": output.get(
                "remaining_prediction_conflicts"
            ),
            "limitations": output.get("limitations", []),
        }
    if isinstance(car_result, dict):
        selected_car = car_result.get("selected_alternative")
        fiscal_calculation["car_solver_status"] = car_result.get("solver_status")
        fiscal_calculation["car_rejected_alternatives"] = car_result.get(
            "rejected_alternatives", []
        )
        if isinstance(selected_car, dict):
            projected_deficit = selected_car.get("deficit")
            ceiling = fiscal_calculation.get(
                "effective_deficit_ceiling_percent",
                fiscal_calculation.get("statutory_deficit_ceiling_percent"),
            )
            fiscal_calculation["selected_compromise"] = {
                **selected_car,
                "headroom_percent": (
                    round(float(ceiling) - float(projected_deficit), 4)
                    if isinstance(ceiling, (int, float))
                    and isinstance(projected_deficit, (int, float))
                    else None
                ),
            }
        elif car_result.get("status") == "INFEASIBLE":
            fiscal_calculation["selected_compromise"] = None
        resolution["proposed_resolution"] = resolution.get("arbiter_conclusion")
        resolution["status"] = car_result.get("status")
        resolution["arbiter_conclusion"] = (
            selected_car.get("name") if isinstance(selected_car, dict) else None
        )
        resolution["selected_alternative"] = selected_car
        resolution["rejected_alternatives"] = car_result.get(
            "rejected_alternatives", []
        )
    return {
        "id": log.id,
        "agent_i_id": log.agent_i,
        "agent_i": left_name,
        "agent_i_display_name": left_display_name or left_name,
        "agent_j_id": log.agent_j,
        "agent_j": right_name,
        "agent_j_display_name": right_display_name or right_name,
        **vector,
        "active_components": [key for key, value in vector.items() if value],
        "conflict_categories": categories,
        "vector_metadata": vector_metadata,
        "fiscal_calculation": fiscal_calculation,
        "influence_context": detail.get("influence_context", {}),
        "legal_basis": detail.get("legal_basis", []),
        "resolution_mechanism": _resolution_mechanism(log),
        "resolution_detail": resolution,
    }


def _car_result(run: ConsensusSession | None) -> dict[str, Any] | None:
    if run is None or not isinstance(run.result_payload, dict):
        return None
    value = run.result_payload.get("car")
    return value if isinstance(value, dict) else None


def _disagreements_payload(
    database: Any,
    run_id: str,
    simulations: list[SimulationArtifact],
    car_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    left = aliased(Agent)
    right = aliased(Agent)
    run = database.get(ConsensusSession, run_id)
    scoped_agent_ids = {
        agent.id
        for agent in scenario_deliberative_agents(database, run.scenario_id)
    } if run is not None else set()
    run_agent_ids = {
        item.get("agent_id")
        for item in run.mandate_payload.get("agent_rules", [])
        if isinstance(item, dict) and isinstance(item.get("agent_id"), int)
    } if run is not None else set()
    participant_ids = scoped_agent_ids & run_agent_ids
    rows = database.execute(
        select(DisagreementLog, left, right)
        .join(left, DisagreementLog.agent_i == left.id)
        .join(right, DisagreementLog.agent_j == right.id)
        .where(
            DisagreementLog.run_id == run_id,
            left.id.in_(participant_ids),
            right.id.in_(participant_ids),
        )
        .order_by(DisagreementLog.id)
    ).all()
    display_names = {
        item.get("agent_id"): item.get("display_name")
        for item in (
            run.mandate_payload.get("agent_rules", [])
            if run is not None and isinstance(run.mandate_payload, dict)
            else []
        )
        if isinstance(item, dict)
        and isinstance(item.get("agent_id"), int)
        and isinstance(item.get("display_name"), str)
    }
    return [
        _disagreement_payload(
            log,
            left_agent.name,
            right_agent.name,
            simulations,
            car_result,
            str(
                display_names.get(left_agent.id)
                or semantic_agent_name(left_agent)
            ),
            str(
                display_names.get(right_agent.id)
                or semantic_agent_name(right_agent)
            ),
        )
        for log, left_agent, right_agent in rows
    ]


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
            "display_name": "Fallback_Policy_Reviewer",
            "utility_metadata": {
                "task": "Evaluate the active policy scenario.",
                "result": "Stored legacy mandate metadata is incomplete.",
                "why": "A safe fallback preserves compatibility without fabricating identity.",
                "name_source": "fallback",
            },
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
    sanitized = sanitize_public_value(artifact.output_payload)
    output = dict(sanitized) if isinstance(sanitized, dict) else {}
    output.setdefault(
        "messages",
        {
            "id": str(output.get("message") or output.get("simulation_summary") or "Artefak simulasi tersedia."),
            "en": str(output.get("message") or output.get("simulation_summary") or "Simulation artifact is available."),
        },
    )
    return {
        "id": artifact.id,
        "session_id": artifact.run_id,
        "scenario_id": artifact.scenario_id,
        "trigger": artifact.trigger,
        "round_number": artifact.round_number,
        "status": artifact.status,
        "simulation_version": artifact.simulation_version,
        "input": sanitize_public_value(artifact.input_payload),
        "output": output,
        "latency_ms": artifact.latency_ms,
        "token_usage": artifact.token_usage,
        "created_at": artifact.created_at.isoformat(),
    }


def _safe_decision_value(value: object) -> object:
    return sanitize_public_value(value)


def _artifact_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "summary", "recommendation", "decision", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _statutory_gates(rule: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in (
        "constraints",
        "applicable_constraints",
        "owned_checks",
        "applicable_owned_checks",
        "regulatory_compliance_alignment",
    ):
        raw_items = rule.get(field)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            text = _artifact_text(item)
            if text and text not in values:
                values.append(text)
    return values


def _stage_payload(
    raw_stage: dict[str, Any],
    statutory_gates: list[str],
) -> dict[str, Any]:
    raw_artifacts = raw_stage.get("artifacts")
    artifacts = (
        _safe_decision_value(raw_artifacts)
        if isinstance(raw_artifacts, dict)
        else {}
    )
    if not isinstance(artifacts, dict):
        artifacts = {}
    recommendation = artifacts.get("recommendation")
    reasoning_summary = _artifact_text(artifacts.get("reasoning_summary"))
    if reasoning_summary is None:
        reasoning_summary = _artifact_text(recommendation)
    if reasoning_summary is None:
        predictions = artifacts.get("predictions")
        if isinstance(predictions, list) and predictions:
            reasoning_summary = _artifact_text(predictions[0])
    constraints = artifacts.get("constraints")
    constraints_considered = list(constraints) if isinstance(constraints, list) else []
    existing_constraints = {
        text.casefold()
        for item in constraints_considered
        if (text := _artifact_text(item)) is not None
    }
    constraints_considered.extend(
        {"content": gate, "source_tag": "AGENT_MANDATE"}
        for gate in statutory_gates
        if gate.casefold() not in existing_constraints
    )
    stage_name = str(raw_stage.get("stage") or "FINAL")
    round_number = raw_stage.get("round_number")
    return {
        "stage": stage_name,
        "round_number": round_number if isinstance(round_number, int) else 0,
        "agent_opinion": reasoning_summary,
        "reasoning_summary": reasoning_summary,
        "constraints_considered": constraints_considered,
        "statutory_gates": statutory_gates,
        "recommendation": recommendation,
        "confidence": artifacts.get("confidence"),
        "evidence": artifacts.get("evidence") if isinstance(artifacts.get("evidence"), list) else [],
        "assumptions": artifacts.get("assumptions") if isinstance(artifacts.get("assumptions"), list) else [],
        "predictions": artifacts.get("predictions") if isinstance(artifacts.get("predictions"), list) else [],
        "risks": artifacts.get("risks") if isinstance(artifacts.get("risks"), list) else [],
        "uncertainties": artifacts.get("uncertainties") if isinstance(artifacts.get("uncertainties"), list) else [],
        "objectives": artifacts.get("objectives") if isinstance(artifacts.get("objectives"), list) else [],
        "alternatives": artifacts.get("alternatives") if isinstance(artifacts.get("alternatives"), list) else [],
        "fallback_metadata": artifacts.get("fallback_metadata")
        if isinstance(artifacts.get("fallback_metadata"), dict)
        else None,
    }


def _agent_breakdown_payload(database: Any, run: ConsensusSession) -> list[dict[str, Any]]:
    scoped_agent_ids = {
        agent.id
        for agent in scenario_deliberative_agents(database, run.scenario_id)
    }
    mandate_rules = {
        item.get("agent_id"): item
        for item in run.mandate_payload.get("agent_rules", [])
        if isinstance(item, dict)
        and isinstance(item.get("agent_id"), int)
        and item.get("agent_id") in scoped_agent_ids
    }
    agent_ids = list(mandate_rules)
    if not agent_ids:
        return []
    agents = list(
        database.scalars(
            select(Agent).where(Agent.id.in_(agent_ids)).order_by(Agent.id)
        )
    )
    reasoning = {
        log.agent_id: log
        for log in database.scalars(
            select(ReasoningLog).where(
                ReasoningLog.run_id == run.id,
                ReasoningLog.agent_id.in_(agent_ids),
            )
        )
    }
    breakdown: list[dict[str, Any]] = []
    for agent in agents:
        log = reasoning.get(agent.id)
        rule = mandate_rules.get(agent.id, {})
        gates = _statutory_gates(rule)
        raw_history = log.deliberation_history if log is not None else []
        history = [item for item in raw_history if isinstance(item, dict)]
        if not history and log is not None and isinstance(log.parsed_srr_objects, dict):
            history = [
                {
                    "stage": "FINAL",
                    "round_number": 0,
                    "artifacts": log.parsed_srr_objects,
                }
            ]
        stages = [_stage_payload(item, gates) for item in history]
        pre_arbitration = next(
            (item for item in reversed(stages) if item["stage"] == "PRE_ARBITRATION"),
            stages[0] if stages else None,
        )
        final_position = next(
            (item for item in reversed(stages) if item["stage"] == "FINAL"),
            stages[-1] if stages else None,
        )
        current_position = pre_arbitration or final_position or _stage_payload({}, gates)
        stored_utility = rule.get("utility_metadata")
        utility_metadata = (
            dict(stored_utility) if isinstance(stored_utility, dict) else {}
        )
        utility_metadata.update(
            agent_utility_metadata(
                agent,
                source=str(utility_metadata.get("name_source") or rule.get("synthesis_status") or "generated"),
                result=(
                    f"Produced {len(current_position['alternatives'])} policy alternative(s); recommendation: "
                    f"{_artifact_text(current_position['recommendation']) or 'not available'}."
                ),
            )
        )
        breakdown.append(
            {
                "agent_id": agent.id,
                "agent_name": agent.name,
                "display_name": str(rule.get("display_name") or semantic_agent_name(agent)),
                "utility_metadata": utility_metadata,
                "agent_role": agent.role,
                "role": agent.role,
                "template_key": agent.template_key,
                "schema_valid": log.is_schema_valid if log is not None else None,
                "provenance_count": log.provenance_count if log is not None else 0,
                "position_stage": current_position["stage"],
                "agent_opinion": current_position["agent_opinion"],
                "reasoning_summary": current_position["reasoning_summary"],
                "constraints_considered": current_position["constraints_considered"],
                "statutory_gates": gates,
                "recommendation": current_position["recommendation"],
                "confidence": current_position["confidence"],
                "evidence": current_position["evidence"],
                "assumptions": current_position["assumptions"],
                "predictions": current_position["predictions"],
                "risks": current_position["risks"],
                "uncertainties": current_position["uncertainties"],
                "objectives": current_position["objectives"],
                "alternatives": current_position["alternatives"],
                "fallback_metadata": current_position["fallback_metadata"],
                "pre_arbitration": pre_arbitration,
                "final_position": final_position,
                "deliberation_stages": stages,
            }
        )
    return breakdown


def _collective_reasoning_payload(
    scenario: Scenario,
    run: ConsensusSession,
    agent_breakdown: list[dict[str, Any]],
    disagreements: list[dict[str, Any]],
    simulations: list[SimulationArtifact],
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    for agent in agent_breakdown:
        position = agent.get("pre_arbitration")
        if not isinstance(position, dict):
            position = agent.get("final_position")
        if not isinstance(position, dict):
            position = agent
        claims.append(
            {
                "agent_id": agent.get("agent_id"),
                "agent_name": agent.get("agent_name"),
                "display_name": agent.get("display_name"),
                "agent_role": agent.get("agent_role"),
                "schema_valid": agent.get("schema_valid"),
                "position_stage": position.get("stage"),
                "main_claim": position.get("reasoning_summary")
                or position.get("agent_opinion"),
                "recommendation": position.get("recommendation"),
                "confidence": position.get("confidence"),
                "constraints": position.get("constraints_considered", []),
                "evidence": position.get("evidence", []),
                "predictions": position.get("predictions", []),
            }
        )

    category_map: dict[str, dict[str, Any]] = {}
    for disagreement in disagreements:
        pair = f"{disagreement.get('agent_i')} × {disagreement.get('agent_j')}"
        categories = disagreement.get("conflict_categories", [])
        if not isinstance(categories, list):
            continue
        for category in categories:
            if not isinstance(category, dict):
                continue
            component = str(category.get("component") or "unknown")
            point = category_map.setdefault(
                component,
                {
                    "component": component,
                    "category": category.get("category"),
                    "category_i18n": category.get("category_i18n"),
                    "narrative": category.get("narrative"),
                    "narrative_i18n": category.get("narrative_i18n"),
                    "pair_count": 0,
                    "agent_pairs": [],
                },
            )
            point["pair_count"] += 1
            point["agent_pairs"].append(pair)
    divergence_points = list(category_map.values())

    effective_ceiling = STATUTORY_DEFICIT_CEILING_PERCENT_GDP
    valid_claims = [claim for claim in claims if claim.get("schema_valid")]
    feasible_agents = 0
    for agent in agent_breakdown:
        alternatives = agent.get("alternatives", [])
        if isinstance(alternatives, list) and any(
            isinstance(item, dict)
            and isinstance(item.get("deficit"), (int, float))
            and float(item["deficit"]) <= effective_ceiling
            for item in alternatives
        ):
            feasible_agents += 1
    post_arbitration_agents = sum(
        any(
            isinstance(stage, dict) and stage.get("stage") == "POST_SIMULATION"
            for stage in agent.get("deliberation_stages", [])
        )
        for agent in agent_breakdown
    )
    convergence_signals = [
        {
            "signal": "Decision-complete sectoral positions",
            "signal_i18n": {"id": "Posisi sektoral decision-complete", "en": "Decision-complete sectoral positions"},
            "coverage": f"{len(valid_claims)}/{len(claims)} agents",
            "narrative_i18n": {
                "id": "Agen yang lolos schema menyediakan klaim, rekomendasi, confidence, dan artefak keputusan terstruktur.",
                "en": "Schema-valid agents provide structured claims, recommendations, confidence, and decision artifacts.",
            },
        },
        {
            "signal": "Feasible sectoral alternatives",
            "signal_i18n": {"id": "Alternatif sektoral feasible", "en": "Feasible sectoral alternatives"},
            "coverage": f"{feasible_agents}/{len(claims)} agents",
            "narrative_i18n": {
                "id": f"Agen memiliki sedikitnya satu alternatif dengan proyeksi defisit tidak melebihi ceiling efektif {effective_ceiling}% PDB.",
                "en": f"Agents have at least one alternative with a projected deficit no higher than the effective {effective_ceiling}% of GDP ceiling.",
            },
        },
        {
            "signal": "Post-arbitration review",
            "signal_i18n": {"id": "Review pasca-arbitrase", "en": "Post-arbitration review"},
            "coverage": f"{post_arbitration_agents}/{len(claims)} agents",
            "narrative_i18n": {
                "id": "Agen memperbarui posisi setelah menerima hasil Simulation Agent.",
                "en": "Agents update their positions after receiving the Simulation Agent result.",
            },
        },
    ]

    latest_simulation = simulations[-1] if simulations else None
    simulation_output = (
        latest_simulation.output_payload
        if latest_simulation is not None
        and isinstance(latest_simulation.output_payload, dict)
        else {}
    )
    modelled_alternatives = simulation_output.get("alternatives", [])
    if not isinstance(modelled_alternatives, list):
        modelled_alternatives = []
    quantified_deficits = [
        float(item["deficit"])
        for item in modelled_alternatives
        if isinstance(item, dict)
        and isinstance(item.get("deficit"), (int, float))
    ]
    persisted_car = (
        run.result_payload.get("car")
        if isinstance(run.result_payload, dict)
        else None
    )
    if isinstance(persisted_car, dict):
        selected_car = persisted_car.get("selected_alternative")
        hard_constraints = persisted_car.get("hard_constraints", [])
        statutory_constraint = next(
            (
                item
                for item in hard_constraints
                if isinstance(item, dict) and item.get("code") == "DEFICIT_3PCT"
            ),
            None,
        )
        scenario_policy_constraint = next(
            (
                item
                for item in hard_constraints
                if isinstance(item, dict)
                and item.get("code") == "SCENARIO_DEFICIT_CEILING"
            ),
            None,
        )
        rejected = persisted_car.get("rejected_alternatives", [])
        legacy_rejected_deficits = [
            float(item["projected_deficit_percent_gdp"])
            for item in rejected
            if isinstance(item, dict)
            and isinstance(item.get("projected_deficit_percent_gdp"), (int, float))
        ]
        statutory_status = (
            "BREACH"
            if isinstance(statutory_constraint, dict)
            and statutory_constraint.get("status") == "violated"
            or any(
                value > STATUTORY_DEFICIT_CEILING_PERCENT_GDP
                for value in legacy_rejected_deficits
            )
            else "COMPLIANT"
            if isinstance(statutory_constraint, dict)
            and statutory_constraint.get("status") == "satisfied"
            else "NOT_EVALUATED"
        )
    else:
        selected_car = None
        scenario_policy_constraint = None
        statutory_status = (
            "COMPLIANT"
            if quantified_deficits
            and all(
                value <= STATUTORY_DEFICIT_CEILING_PERCENT_GDP
                for value in quantified_deficits
            )
            else "BREACH"
            if quantified_deficits
            else "NOT_EVALUATED"
        )
    legal_references = sorted(
        {
            str(basis.get("source_tag"))
            for disagreement in disagreements
            for basis in (
                disagreement.get("legal_basis", [])
                if isinstance(disagreement.get("legal_basis"), list)
                else []
            )
            if isinstance(basis, dict) and basis.get("source_tag")
        }
    )
    remaining_conflicts = simulation_output.get("remaining_prediction_conflicts")
    normative_principles = [
        {
            "principle": "Statutory deficit ceiling",
            "principle_i18n": {"id": "Batas defisit statutory", "en": "Statutory deficit ceiling"},
            "status": statutory_status,
            "evaluation_i18n": {
                "id": f"Alternatif modelled diuji terhadap batas statutory {STATUTORY_DEFICIT_CEILING_PERCENT_GDP}% PDB; kebijakan skenario yang lebih ketat dicatat terpisah.",
                "en": f"Modelled alternatives are tested against the statutory {STATUTORY_DEFICIT_CEILING_PERCENT_GDP}% of GDP ceiling; stricter scenario policy is recorded separately.",
            },
            "legal_sources": ["UU17_2003_P12", "UU17_2025_POSTURE"],
        },
        {
            "principle": "Scenario deficit policy ceiling",
            "principle_i18n": {"id": "Batas defisit kebijakan skenario", "en": "Scenario deficit policy ceiling"},
            "status": (
                "BREACH"
                if isinstance(scenario_policy_constraint, dict)
                and scenario_policy_constraint.get("status") == "violated"
                else "COMPLIANT"
                if isinstance(scenario_policy_constraint, dict)
                and scenario_policy_constraint.get("status") == "satisfied"
                else "NOT_EVALUATED"
            ),
            "evaluation_i18n": {
                "id": f"Batas kebijakan skenario efektif adalah {effective_ceiling}% PDB; nilai di bawah 3.0% merupakan pembatas kebijakan tambahan.",
                "en": f"The effective scenario policy ceiling is {effective_ceiling}% of GDP; values below 3.0% are an additional policy constraint.",
            },
            "legal_sources": [],
        },
        {
            "principle": "Legal and provenance traceability",
            "principle_i18n": {"id": "Ketertelusuran hukum dan provenance", "en": "Legal and provenance traceability"},
            "status": "DOCUMENTED" if legal_references else "NOT_DOCUMENTED",
            "evaluation_i18n": {
                "id": "Persetujuan atau penolakan harus dapat ditelusuri ke sumber hukum dan evidence tag, bukan pada hidden reasoning.",
                "en": "Approval or rejection must be traceable to legal sources and evidence tags, not hidden reasoning.",
            },
            "legal_sources": legal_references,
        },
        {
            "principle": "Unverified revenue offsets",
            "principle_i18n": {"id": "Offset penerimaan belum terverifikasi", "en": "Unverified revenue offsets"},
            "status": "SAFEGUARD_REQUIRED",
            "evaluation_i18n": {
                "id": "Proyeksi penerimaan yang belum terverifikasi tidak boleh diperlakukan sebagai kas, fiscal space, atau pengurang defisit yang telah terealisasi.",
                "en": "Unverified revenue projections must not be treated as cash, fiscal space, or realized deficit offsets.",
            },
            "legal_sources": ["UU17_2025_POSTURE", "UU9_2018_PNBP"],
        },
        {
            "principle": "Residual dissent",
            "principle_i18n": {"id": "Dissent residual", "en": "Residual dissent"},
            "status": (
                "OPEN"
                if isinstance(remaining_conflicts, int) and remaining_conflicts > 0
                else "RESOLVED"
                if remaining_conflicts == 0
                else "NOT_EVALUATED"
            ),
            "evaluation_i18n": {
                "id": "Dissent yang masih tersisa harus dipertahankan secara eksplisit dan tidak diubah menjadi konsensus semu.",
                "en": "Residual dissent must remain explicit and must not be converted into false consensus.",
            },
            "legal_sources": [],
        },
    ]

    payload = {
        "methodology": "SHCR = SRR + (RAR → DAI) + DDR + Simulation Arbitration + CAR",
        "run_status": run.status,
        "claims_and_positions": claims,
        "why_and_how": {
            "summary_i18n": {
                "id": f"DDR mencatat {len(disagreements)} pasangan agen dengan {sum(len(item.get('active_components', [])) for item in disagreements)} komponen divergensi. Perbedaan prediksi dP dieskalasikan ke Simulation Agent; constraint, evidence, risk, dan rekomendasi tetap menjadi artefak audit.",
                "en": f"DDR records {len(disagreements)} agent pairs with {sum(len(item.get('active_components', [])) for item in disagreements)} divergence components. dP prediction differences escalate to the Simulation Agent; constraints, evidence, risks, and recommendations remain audit artifacts.",
            },
            "divergence_points": divergence_points,
            "convergence_signals": convergence_signals,
        },
        "recommendation_and_follow_up": {
            "arbiter": "CAR / Z3" if isinstance(persisted_car, dict) else simulation_output.get("agent_name"),
            "status": persisted_car.get("status")
            if isinstance(persisted_car, dict)
            else latest_simulation.status
            if latest_simulation
            else "NOT_TRIGGERED",
            "round_number": latest_simulation.round_number if latest_simulation else 0,
            "simulation_summary": simulation_output.get("simulation_summary"),
            "proposed_resolution": simulation_output.get("resolution"),
            "final_resolution": (
                selected_car.get("name")
                if isinstance(selected_car, dict)
                else None
            )
            if isinstance(persisted_car, dict)
            else simulation_output.get("resolution"),
            "selected_alternative": selected_car,
            "rejected_alternatives": persisted_car.get("rejected_alternatives", [])
            if isinstance(persisted_car, dict)
            else [],
            "modelled_alternatives": modelled_alternatives,
            "limitations": simulation_output.get("limitations", []),
            "remaining_prediction_conflicts": remaining_conflicts,
            "follow_up_consensus_status": simulation_output.get(
                "follow_up_consensus_status"
            ),
        },
        "normative_evaluation": {
            "summary_i18n": {
                "id": "Kebijakan seharusnya dipilih hanya setelah lolos batas defisit, otoritas APBN, provenance evidence, dan larangan penggunaan offset penerimaan yang belum terverifikasi.",
                "en": "Policy should be selected only after satisfying the deficit ceiling, APBN authority, evidence provenance, and the prohibition on unverified revenue offsets.",
            },
            "principles": normative_principles,
        },
    }
    return cast(dict[str, Any], _safe_decision_value(payload))


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


def _localized_payload(payload: dict[str, Any], lang: Language) -> dict[str, Any]:
    safe_payload = sanitize_public_value(payload)
    localized = cast(dict[str, Any], localize_payload(safe_payload, lang))
    localized["language"] = lang
    return localized


def _dashboard_payload(
    scenario_id: int,
    session_id: str | None = None,
    lang: Language = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        if session_id is not None:
            active_session = session.get(ConsensusSession, session_id)
        else:
            active_session = session.scalar(
                select(ConsensusSession)
                .where(
                    ConsensusSession.scenario_id == scenario_id,
                    ConsensusSession.status == "SUCCEEDED",
                )
                .order_by(ConsensusSession.created_at.desc())
            ) or session.scalar(
                select(ConsensusSession)
                .where(ConsensusSession.scenario_id == scenario_id)
                .order_by(ConsensusSession.created_at.desc())
            )
        if session_id is not None and active_session is None:
            raise HTTPException(status_code=404, detail="Consensus session not found for scenario")
        if active_session is not None and active_session.scenario_id != scenario_id:
            raise HTTPException(status_code=404, detail="Consensus session not found for scenario")
        active_session_id = active_session.id if active_session is not None else None
        artifact_session_id = active_session_id or "__NO_ACTIVE_SESSION__"

        if active_session is not None:
            scoped_agent_ids = {
                agent.id
                for agent in scenario_deliberative_agents(session, scenario_id)
            }
            run_agent_ids = {
                item.get("agent_id")
                for item in active_session.mandate_payload.get("agent_rules", [])
                if isinstance(item, dict) and isinstance(item.get("agent_id"), int)
            }
            agents = list(
                session.scalars(
                    select(Agent)
                    .where(Agent.id.in_(run_agent_ids & scoped_agent_ids))
                    .order_by(Agent.id)
                )
            )
        else:
            agents = scenario_deliberative_agents(session, scenario_id)
        if active_session is not None:
            mandate_snapshot = session.get(
                ScenarioMandateSnapshot,
                active_session.mandate_snapshot_id,
            )
        else:
            mandate_snapshot = session.scalar(
                select(ScenarioMandateSnapshot)
                .where(ScenarioMandateSnapshot.scenario_id == scenario_id)
                .order_by(
                    ScenarioMandateSnapshot.updated_at.desc(),
                    ScenarioMandateSnapshot.id.desc(),
                )
            )
        current_agent_ids = {agent.id for agent in agents}
        current_revision = agent_revision(agents, scenario)
        stored_agent_rules = [
            rule
            for rule in _safe_agent_rules(
                mandate_snapshot.agent_rules if mandate_snapshot is not None else []
            )
            if rule.get("agent_id") in current_agent_ids
        ]
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
                    ReasoningLog.agent_id.in_(current_agent_ids),
                )
                .order_by(ReasoningLog.id)
            )
        )
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
        disagreements = _disagreements_payload(
            session,
            artifact_session_id,
            simulations,
            _car_result(active_session),
        )
        influences = session.execute(
            select(AgentInfluenceObservation, Agent.name)
            .join(Agent, AgentInfluenceObservation.agent_id == Agent.id)
            .where(
                AgentInfluenceObservation.scenario_id == scenario_id,
                AgentInfluenceObservation.run_id == artifact_session_id,
                AgentInfluenceObservation.agent_id.in_(current_agent_ids),
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
        agent_breakdown = (
            _agent_breakdown_payload(session, active_session)
            if active_session is not None
            else []
        )
        collective_reasoning = (
            _collective_reasoning_payload(
                scenario,
                active_session,
                agent_breakdown,
                disagreements,
                simulations,
            )
            if active_session is not None
            else None
        )

        payload = {
            "session_id": active_session_id,
            "session_status": active_session.status if active_session else None,
            "scenario": {
                "id": scenario.id,
                "description": scenario.description,
                **scenario.simulation_payload(),
            },
            "domain_rules": domain_rules,
            "latest_metric": _metric_payload(snapshots[0]) if snapshots else None,
            "metric_history": [_metric_payload(snapshot) for snapshot in snapshots],
            "schema_validity_percent": schema_validity,
            "reasoning_log_count": len(reasoning_logs),
            "agent_breakdown": agent_breakdown,
            "collective_reasoning": collective_reasoning,
            "disagreements": disagreements,
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
                    "interactions": observation.interaction_payload,
                    "calculation": observation.calculation_payload,
                }
                for observation, agent_name in influences
            ],
        }
        return _localized_payload(payload, lang)


@router.post("/scenarios/{scenario_id}/runs", status_code=status.HTTP_202_ACCEPTED)
def start_run(
    scenario_id: int,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
    session_id = str(uuid4())
    task_id = str(uuid4())
    queue_logs = [
        analytical_event(
            "QUEUE",
            "INFO",
            "RUN_QUEUED",
            "Siklus dimasukkan ke antrean worker.",
            "Cycle queued for worker execution.",
            run_id=session_id,
            task_id=task_id,
            scenario_id=scenario_id,
        )
    ]
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        agents, orchestration = ensure_phase_one_specialists(session, scenario)
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
                "orchestration": orchestration,
            },
            runtime_config_payload=global_config_snapshot(
                get_global_llm_config(session)
            ),
            status="QUEUED",
            celery_task_id=task_id,
            logs=queue_logs,
            progress_stage="QUEUE",
        )
        session.add(run)
        session.commit()
    try:
        task = celery_client.send_task(
            "shcr.run_full_shcr_cycle",
            args=[scenario_id, session_id],
            task_id=task_id,
        )
    except Exception as error:
        with SessionLocal() as session:
            run_record = session.get(ConsensusSession, session_id)
            if run_record is not None:
                failure_event = analytical_event(
                    "QUEUE",
                    "ERROR",
                    "RUN_DISPATCH_FAILED",
                    "Siklus konsensus gagal dikirim ke worker.",
                    "Consensus cycle could not be dispatched to the worker.",
                    run_id=session_id,
                    task_id=task_id,
                    scenario_id=scenario_id,
                    fallback={
                        "used": False,
                        "kind": None,
                        "reason": type(error).__name__,
                    },
                    metadata={"error_type": type(error).__name__},
                )
                run_record.status = "FAILED"
                run_record.error = f"Task dispatch failed: {type(error).__name__}"
                run_record.logs = [*list(run_record.logs or []), failure_event]
                run_record.progress_stage = "DISPATCH_FAILED"
                run_record.completed_at = datetime.now(timezone.utc)
                session.commit()
        raise HTTPException(
            status_code=503,
            detail="Consensus cycle could not be dispatched to the worker",
        ) from error
    response_logs = [dict(event) for event in queue_logs]
    response_logs[0]["task_id"] = task.id
    return _localized_payload(
        {
            "task_id": task.id,
            "session_id": session_id,
            "scenario_id": scenario_id,
            "status": "QUEUED",
            **_polling_contract("QUEUED"),
            "logs": response_logs,
        },
        lang,
    )


def _run_payload(
    session: ConsensusSession,
    lang: Language = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    with SessionLocal() as database:
        simulations = list(
            database.scalars(
                select(SimulationArtifact)
                .where(SimulationArtifact.run_id == session.id)
                .order_by(SimulationArtifact.round_number, SimulationArtifact.id)
            )
        )
        agent_breakdown = _agent_breakdown_payload(database, session)
        disagreements = _disagreements_payload(
            database,
            session.id,
            simulations,
            _car_result(session),
        )
        participant_ids = {
            item.get("agent_id")
            for item in session.mandate_payload.get("agent_rules", [])
            if isinstance(item, dict) and isinstance(item.get("agent_id"), int)
        }
        scoped_participant_ids = participant_ids & {
            agent.id
            for agent in scenario_deliberative_agents(database, session.scenario_id)
        }
        influences = database.execute(
            select(AgentInfluenceObservation, Agent.name)
            .join(Agent, AgentInfluenceObservation.agent_id == Agent.id)
            .where(
                AgentInfluenceObservation.run_id == session.id,
                AgentInfluenceObservation.agent_id.in_(scoped_participant_ids),
            )
            .order_by(AgentInfluenceObservation.id)
        ).all()
        scenario = database.get(Scenario, session.scenario_id)
        collective_reasoning = (
            _collective_reasoning_payload(
                scenario,
                session,
                agent_breakdown,
                disagreements,
                simulations,
            )
            if scenario is not None
            else None
        )
    payload = {
        "task_id": session.celery_task_id,
        "session_id": session.id,
        "scenario_id": session.scenario_id,
        "status": session.status,
        **_polling_contract(session.status, session.result_payload),
        "logs": list(session.logs or []),
        "simulation_artifacts": [
            _simulation_payload(artifact) for artifact in simulations
        ],
        "agent_breakdown": agent_breakdown,
        "disagreements": disagreements,
        "collective_reasoning": collective_reasoning,
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
                "interactions": observation.interaction_payload,
                "calculation": observation.calculation_payload,
            }
            for observation, agent_name in influences
        ],
        "result": session.result_payload,
        "runtime_configuration": session.runtime_config_payload,
        "error": session.error,
        "progress_stage": session.progress_stage,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
    }
    return _localized_payload(payload, lang)


@router.get("/scenarios/{scenario_id}/runs")
def scenario_runs(
    scenario_id: int,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> list[dict[str, Any]]:
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
        return [_run_payload(run, lang) for run in runs]


@router.get("/scenarios/{scenario_id}/runs/latest")
def latest_scenario_run(
    scenario_id: int,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
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
        return _run_payload(run, lang)


@router.get("/runs/{task_id}")
def run_status(
    task_id: str,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
    with SessionLocal() as session:
        run = session.scalar(
            select(ConsensusSession).where(ConsensusSession.celery_task_id == task_id)
        )
        if run is not None:
            return _run_payload(run, lang)
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
    public_status = state_map.get(result.state, result.state)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "status": public_status,
        **_polling_contract(
            public_status,
            result.result if result.successful() else None,
        ),
        "logs": [],
        "result": None,
        "session_id": None,
        "scenario_id": None,
    }
    if isinstance(result.info, dict):
        safe_info = sanitize_public_value(result.info)
        if isinstance(safe_info, dict):
            payload["logs"] = safe_info.get("logs", [])
            payload["session_id"] = safe_info.get("session_id")
            payload["scenario_id"] = safe_info.get("scenario_id")
    if result.successful():
        safe_result = sanitize_public_value(result.result)
        payload["result"] = safe_result
        if isinstance(safe_result, dict):
            payload["logs"] = safe_result.get("logs", payload["logs"])
            payload["session_id"] = safe_result.get("session_id", payload["session_id"])
            payload["scenario_id"] = safe_result.get("scenario_id", payload["scenario_id"])
    elif result.failed():
        payload["error"] = sanitize_public_error(result.result)
    return _localized_payload(payload, lang)


def _analytics_payload(
    database: Any,
    run: ConsensusSession,
    scenario: Scenario,
    lang: Language,
) -> dict[str, Any]:
    simulations = list(
        database.scalars(
            select(SimulationArtifact)
            .where(SimulationArtifact.run_id == run.id)
            .order_by(SimulationArtifact.round_number, SimulationArtifact.id)
        )
    )
    agent_breakdown = _agent_breakdown_payload(database, run)
    disagreements = _disagreements_payload(
        database,
        run.id,
        simulations,
        _car_result(run),
    )
    return _localized_payload(
        {
            "session_id": run.id,
            "scenario_id": run.scenario_id,
            "run_status": run.status,
            "collective_reasoning": _collective_reasoning_payload(
                scenario,
                run,
                agent_breakdown,
                disagreements,
                simulations,
            ),
        },
        lang,
    )


def _ddr_payload(
    database: Any,
    run: ConsensusSession,
    lang: Language,
) -> dict[str, Any]:
    simulations = list(
        database.scalars(
            select(SimulationArtifact)
            .where(SimulationArtifact.run_id == run.id)
            .order_by(SimulationArtifact.round_number, SimulationArtifact.id)
        )
    )
    return _localized_payload(
        {
            "session_id": run.id,
            "scenario_id": run.scenario_id,
            "run_status": run.status,
            "disagreements": _disagreements_payload(
                database,
                run.id,
                simulations,
                _car_result(run),
            ),
        },
        lang,
    )


@router.get("/runs/{task_id}/analytics")
def run_analytics(
    task_id: str,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
    with SessionLocal() as session:
        run = session.scalar(
            select(ConsensusSession).where(ConsensusSession.celery_task_id == task_id)
        )
        if run is None:
            raise HTTPException(status_code=404, detail="Consensus run not found")
        scenario = session.get(Scenario, run.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        return _analytics_payload(session, run, scenario, lang)


@router.get("/scenarios/{scenario_id}/runs/{session_id}/analytics")
def scenario_run_analytics(
    scenario_id: int,
    session_id: str,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
    with SessionLocal() as session:
        run = session.get(ConsensusSession, session_id)
        scenario = session.get(Scenario, scenario_id)
        if run is None or run.scenario_id != scenario_id or scenario is None:
            raise HTTPException(status_code=404, detail="Consensus run not found")
        return _analytics_payload(session, run, scenario, lang)


@router.get("/runs/{task_id}/ddr")
def run_ddr(
    task_id: str,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
    with SessionLocal() as session:
        run = session.scalar(
            select(ConsensusSession).where(ConsensusSession.celery_task_id == task_id)
        )
        if run is None:
            raise HTTPException(status_code=404, detail="Consensus run not found")
        return _ddr_payload(session, run, lang)


@router.get("/scenarios/{scenario_id}/runs/{session_id}/ddr")
def scenario_run_ddr(
    scenario_id: int,
    session_id: str,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
    with SessionLocal() as session:
        run = session.get(ConsensusSession, session_id)
        if run is None or run.scenario_id != scenario_id:
            raise HTTPException(status_code=404, detail="Consensus run not found")
        return _ddr_payload(session, run, lang)


@router.get("/runs/{task_id}/graph")
def run_graph(
    task_id: str,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
    with SessionLocal() as session:
        run = session.scalar(
            select(ConsensusSession).where(ConsensusSession.celery_task_id == task_id)
        )
        if run is None:
            raise HTTPException(status_code=404, detail="Consensus run not found")
        return run_graph_payload(session, run, lang)


@router.get("/scenarios/{scenario_id}/runs/{session_id}/graph")
def scenario_run_graph(
    scenario_id: int,
    session_id: str,
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
    with SessionLocal() as session:
        run = session.get(ConsensusSession, session_id)
        if run is None or run.scenario_id != scenario_id:
            raise HTTPException(status_code=404, detail="Consensus run not found")
        return run_graph_payload(session, run, lang)


@router.get("/scenarios/{scenario_id}/dashboard")
def scenario_dashboard(
    scenario_id: int,
    session_id: str | None = Query(default=None),
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> dict[str, Any]:
    return _dashboard_payload(scenario_id, session_id, lang)


@router.get("/scenarios/{scenario_id}/manifest")
def reproducibility_manifest(
    scenario_id: int,
    session_id: str | None = Query(default=None),
    lang: Language = Query(default=DEFAULT_LANGUAGE),
) -> JSONResponse:
    dashboard = _dashboard_payload(scenario_id, session_id, lang)
    with SessionLocal() as session:
        dashboard_agent_ids = {
            item.get("agent_id")
            for item in dashboard.get("agent_breakdown", [])
            if isinstance(item, dict) and isinstance(item.get("agent_id"), int)
        }
        agents = list(
            session.scalars(
                select(Agent)
                .where(Agent.id.in_(dashboard_agent_ids))
                .order_by(Agent.id)
            )
        )
        reasoning = session.execute(
            select(ReasoningLog, Agent)
            .join(Agent, ReasoningLog.agent_id == Agent.id)
            .where(
                ReasoningLog.scenario_id == scenario_id,
                ReasoningLog.run_id == dashboard["session_id"],
                ReasoningLog.agent_id.in_(dashboard_agent_ids),
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
                    "user": build_agent_user_prompt(
                        agent.role,
                        str(dashboard["scenario"]["description"]),
                         float(dashboard["scenario"]["program_cost"])
                         if isinstance(dashboard["scenario"].get("program_cost"), (int, float))
                         else None,
                         cast(dict[str, object], dashboard["scenario"]),
                    ),
                }
                for agent in agents
            ],
            "llm_outputs": [
                {
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "raw_json": _safe_decision_value(log.raw_json),
                    "parsed_srr_objects": _safe_decision_value(log.parsed_srr_objects),
                    "deliberation_history": _safe_decision_value(log.deliberation_history),
                    "is_schema_valid": log.is_schema_valid,
                    "provenance_count": log.provenance_count,
                }
                for log, agent in reasoning
            ],
            "influence_observations": dashboard["influence_observations"],
            "disagreements": dashboard["disagreements"],
            "agent_breakdown": dashboard["agent_breakdown"],
            "collective_reasoning": dashboard["collective_reasoning"],
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
