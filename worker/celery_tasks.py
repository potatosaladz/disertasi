import logging
import os
from collections.abc import Callable
from time import perf_counter
from typing import Any, cast

from openai import OpenAI
from pydantic import ValidationError
from sqlalchemy import select

from backend.agent_templates import resolve_agent_system_prompt
from backend.core_algorithms import (
    HardConstraints,
    build_agent_system_prompt,
    calculate_dynamic_influence,
    calculate_violation_rate,
    detect_divergence_vector,
    extract_json_object,
    neuro_symbolic_filter,
)
from backend.database import SessionLocal
from backend.models import (
    Agent,
    AgentInfluenceObservation,
    ConvergenceStatus,
    DisagreementLog,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
)
from worker.srr_models import SRRResponse

logger = logging.getLogger(__name__)
LLMCaller = Callable[[Agent, Scenario], tuple[str, int]]
ProgressReporter = Callable[[list[dict[str, str]]], None]


def _log(stage: str, level: str, message: str) -> dict[str, str]:
    return {"stage": stage, "level": level, "message": message}


def _extract_llm_response(response: object) -> tuple[str, int]:
    if isinstance(response, str):
        return response, 0
    if isinstance(response, dict):
        content = response.get("content")
        if content is None:
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("LLM response dictionary has no text content")
        usage = response.get("usage")
        tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
        return content, int(tokens)
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"Unsupported LLM response type: {type(response).__name__}")
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if not isinstance(content, str):
        raise ValueError("LLM returned no text content")
    usage = getattr(response, "usage", None)
    return content, int(getattr(usage, "total_tokens", 0) if usage else 0)


def _default_llm_call(agent: Agent, scenario: Scenario) -> tuple[str, int]:
    base_url = agent.llm_base_url or os.getenv(
        "OPENAI_BASE_URL", "http://host.docker.internal:11434/v1"
    )
    api_key = agent.llm_api_key or os.getenv("OPENAI_API_KEY", "local-llm")
    model = cast(str, agent.llm_model or os.getenv("OPENAI_MODEL") or "local-model")
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens,
        messages=[
            {
                "role": "system",
                "content": build_agent_system_prompt(
                    agent.role,
                    resolve_agent_system_prompt(agent),
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Agent role: {agent.role}\n"
                    f"Policy goal: {scenario.description}\n"
                    f"Program cost: {scenario.program_cost if scenario.program_cost is not None else 'UNKNOWN'}\n"
                    f"Automatic legal deficit ceiling: {scenario.max_deficit_constraint}%"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    return _extract_llm_response(response)


def _parse_response(raw_content: str) -> tuple[dict[str, Any], SRRResponse | None, str | None]:
    try:
        payload = extract_json_object(raw_content)
    except ValueError as error:
        return {"unparsed_content": raw_content, "parse_error": str(error)}, None, str(error)
    try:
        return payload, SRRResponse.model_validate(payload), None
    except ValidationError as error:
        return payload, None, "; ".join(
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            for detail in error.errors()
        )


def _provenance_counts(response: SRRResponse) -> tuple[int, int]:
    items = response.provenance_items()
    tagged = sum(bool(item.source_tag and item.source_tag.strip()) for item in items)
    return tagged, len(items)


def _determine_convergence(
    parsed_responses: list[SRRResponse],
    alternatives: list[Any],
    feasible_alternatives: list[Any],
) -> ConvergenceStatus:
    if not parsed_responses or not alternatives:
        return ConvergenceStatus.INSUFFICIENT_EVIDENCE
    if not feasible_alternatives:
        return ConvergenceStatus.INFEASIBLE

    recommendations = {
        response.recommendation.content if response.recommendation is not None else None
        for response in parsed_responses
    }
    alternatives_by_name: dict[str, list[Any]] = {}
    for alternative in feasible_alternatives:
        alternatives_by_name.setdefault(alternative.name, []).append(alternative)
    conflicting_utilities = any(
        len({alternative.utility for alternative in alternatives}) > 1
        for alternatives in alternatives_by_name.values()
    )
    multiple_utility_alternatives = (
        len(alternatives_by_name) > 1
        and len({alternative.utility for alternative in feasible_alternatives}) > 1
    )

    if conflicting_utilities or multiple_utility_alternatives or len(recommendations) > 1:
        return ConvergenceStatus.PARETO_SET
    return ConvergenceStatus.FULL_CONSENSUS


def execute_full_shcr_cycle(
    scenario_id: int,
    llm_call: LLMCaller | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    caller = llm_call or _default_llm_call
    emit = progress or (lambda _logs: None)
    logs = [_log("INITIALIZE", "INFO", f"Starting SHCR cycle for scenario {scenario_id}.")]
    emit(logs)

    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} does not exist")
        agents = list(session.scalars(select(Agent).order_by(Agent.id)))
        if not agents:
            raise ValueError("At least one agent is required")

        parsed_by_agent: list[tuple[Agent, SRRResponse]] = []
        observations = list(
            session.scalars(
                select(AgentInfluenceObservation)
                .where(AgentInfluenceObservation.scenario_id == scenario.id)
                .order_by(AgentInfluenceObservation.id)
            )
        )
        total_tokens = 0
        tagged_items = 0
        total_items = 0

        for agent in agents:
            logs.append(_log("SRR", "INFO", f"Calling {agent.name} using {agent.llm_model or 'environment default model'}."))
            emit(logs)
            try:
                raw_content, token_usage = caller(agent, scenario)
            except Exception as error:
                logger.exception("LLM call failed for agent %s", agent.id)
                logs.append(_log("SRR", "ERROR", f"{agent.name}: {type(error).__name__}: {error}"))
                emit(logs)
                session.add(
                    ReasoningLog(
                        agent_id=agent.id,
                        scenario_id=scenario.id,
                        raw_json={"error": str(error), "error_type": type(error).__name__},
                        parsed_srr_objects={},
                        is_schema_valid=False,
                        provenance_count=0,
                    )
                )
                continue
            total_tokens += token_usage
            raw_json, parsed, validation_error = _parse_response(raw_content)
            if parsed is None:
                logs.append(_log("SRR", "WARNING", f"{agent.name}: response failed JSON/schema validation: {validation_error}."))
                emit(logs)
                session.add(
                    ReasoningLog(
                        agent_id=agent.id,
                        scenario_id=scenario.id,
                        raw_json=raw_json,
                        parsed_srr_objects={},
                        is_schema_valid=False,
                        provenance_count=0,
                    )
                )
                continue

            tagged, count = _provenance_counts(parsed)
            logs.append(_log("SRR", "SUCCESS", f"{agent.name}: valid SRR with {tagged}/{count} sourced artifacts and {token_usage} tokens."))
            emit(logs)
            tagged_items += tagged
            total_items += count
            parsed_by_agent.append((agent, parsed))
            session.add(
                ReasoningLog(
                    agent_id=agent.id,
                    scenario_id=scenario.id,
                    raw_json=raw_json,
                    parsed_srr_objects=parsed.model_dump(mode="json"),
                    is_schema_valid=True,
                    provenance_count=tagged,
                )
            )

        logs.append(_log("DDR", "INFO", "Calculating disagreement vectors."))
        emit(logs)
        agents_by_id = {agent.id: agent for agent in agents}
        influence_inputs: list[dict[str, float | int]] = []
        influence_observations: list[AgentInfluenceObservation] = []
        for observation in observations:
            observation_agent = agents_by_id.get(observation.agent_id)
            if observation_agent is None:
                continue
            influence_observations.append(observation)
            influence_inputs.append(
                {
                    "theta_x": observation_agent.theta_x,
                    "theta_q": observation_agent.theta_q,
                    "theta_h": observation_agent.theta_h,
                    "theta_s": observation_agent.theta_s,
                    "theta_u": observation_agent.theta_u,
                    "X": observation.X,
                    "Q": observation.Q,
                    "H": observation.H,
                    "S": observation.S,
                    "U": observation.U,
                    "g_i": observation.gate,
                }
            )
        if influence_inputs and any(item["g_i"] == 1 for item in influence_inputs):
            influence_results = calculate_dynamic_influence(
                influence_inputs,
                [observation.proposition for observation in influence_observations],
            )
            for observation, result in zip(
                influence_observations,
                influence_results,
                strict=True,
            ):
                observation.raw_score = float(result["raw_score"])
                observation.normalized_weight = float(result["normalized_weight"])

        for index, (agent_i, response_i) in enumerate(parsed_by_agent):
            for agent_j, response_j in parsed_by_agent[index + 1 :]:
                vector = detect_divergence_vector(
                    response_i.divergence_object(),
                    response_j.divergence_object(),
                )
                route = "Simulation Agent Requested" if vector["dP"] else None
                if route:
                    logger.info(route)
                active_conflicts = ", ".join(key for key, value in vector.items() if value) or "none"
                logs.append(_log("DDR", "WARNING" if active_conflicts != "none" else "SUCCESS", f"{agent_i.name} vs {agent_j.name}: conflicts={active_conflicts}; route={route or 'none'}."))
                emit(logs)
                session.add(
                    DisagreementLog(
                        scenario_id=scenario.id,
                        agent_i=agent_i.id,
                        agent_j=agent_j.id,
                        resolution_route=route,
                        **vector,
                    )
                )

        logs.append(_log("CAR", "INFO", f"Applying hard deficit constraint <= {scenario.max_deficit_constraint}%."))
        emit(logs)
        parsed_responses = [response for _, response in parsed_by_agent]
        alternatives = [
            alternative
            for response in parsed_responses
            for alternative in response.alternatives
        ]
        feasible = neuro_symbolic_filter(
            alternatives,
            HardConstraints(max_deficit=scenario.max_deficit_constraint),
        )
        violation_rate = calculate_violation_rate(len(alternatives), len(feasible))
        provenance_completeness = (
            round((tagged_items / total_items) * 100.0, 2) if total_items else 0.0
        )
        retention_values = [
            response.material_information_retention_macro_f1
            for response in parsed_responses
            if response.material_information_retention_macro_f1 is not None
        ]
        retention_macro_f1 = (
            sum(retention_values) / len(retention_values) if retention_values else 0.0
        )
        convergence_status = _determine_convergence(
            parsed_responses, alternatives, feasible
        )
        latency_ms = (perf_counter() - started_at) * 1000.0

        logs.append(_log("CAR", "SUCCESS", f"{len(feasible)}/{len(alternatives)} alternatives feasible; violation rate={violation_rate}%."))
        logs.append(_log("METRICS", "INFO", "Persisting convergence, provenance, latency, and token metrics."))
        emit(logs)
        snapshot = MetricSnapshot(
            scenario_id=scenario.id,
            provenance_completeness_percent=provenance_completeness,
            material_information_retention_macro_f1=retention_macro_f1,
            hard_constraint_violation_rate=violation_rate,
            feasible_alternatives_count=len(feasible),
            convergence_status=convergence_status,
            latency_ms=latency_ms,
            token_usage=total_tokens,
        )
        session.add(snapshot)
        session.commit()
        session.refresh(snapshot)
        logs.append(_log("COMPLETE", "SUCCESS", f"Cycle completed with state {convergence_status.value}, {total_tokens} tokens, and {latency_ms:.2f} ms latency."))
        emit(logs)

        return {
            "metric_snapshot_id": snapshot.id,
            "scenario_id": scenario.id,
            "provenance_completeness_percent": provenance_completeness,
            "hard_constraint_violation_rate": violation_rate,
            "feasible_alternatives_count": len(feasible),
            "convergence_status": convergence_status.value,
            "latency_ms": latency_ms,
            "token_usage": total_tokens,
        }


def run_full_shcr_cycle(scenario_id: int) -> dict[str, Any]:
    return execute_full_shcr_cycle(scenario_id)
