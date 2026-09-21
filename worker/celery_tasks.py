import logging
import socket
from collections.abc import Callable
from time import perf_counter
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError
from sqlalchemy import select

from backend.agent_templates import resolve_agent_system_prompt
from backend.core_algorithms import (
    HardConstraints,
    build_agent_system_prompt,
    build_agent_user_prompt,
    build_consensus_prompt,
    calculate_dynamic_influence,
    calculate_violation_rate,
    detect_divergence_vector,
    extract_json_object,
    extract_llm_completion,
    llm_request_headers,
    log_llm_outbound,
    neuro_symbolic_filter,
    resolve_llm_runtime_config,
    validate_decision_artifacts,
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
ConsensusCaller = Callable[[Agent, Scenario, list[dict[str, Any]]], tuple[str, int]]
ProgressReporter = Callable[[list[dict[str, str]]], None]


def _log(stage: str, level: str, message: str) -> dict[str, str]:
    return {"stage": stage, "level": level, "message": message}


def _extract_llm_response(response: object) -> tuple[str, int]:
    return extract_llm_completion(response)


def _create_llm_completion(
    agent: Agent,
    messages: list[ChatCompletionMessageParam],
    operation: str,
) -> tuple[str, int]:
    config = resolve_llm_runtime_config(agent)
    log_llm_outbound(
        operation,
        agent.name,
        config.base_url,
        config.model,
        {
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
    )
    response = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=60.0,
        max_retries=2,
        default_headers=llm_request_headers(),
    ).chat.completions.create(
        model=config.model,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return _extract_llm_response(response)


def _default_llm_call(agent: Agent, scenario: Scenario) -> tuple[str, int]:
    return _create_llm_completion(
        agent,
        [
            {
                "role": "system",
                "content": build_agent_system_prompt(
                    agent.role,
                    resolve_agent_system_prompt(agent),
                ),
            },
            {
                "role": "user",
                "content": build_agent_user_prompt(
                    agent.role,
                    scenario.description,
                    scenario.program_cost,
                    scenario.max_deficit_constraint,
                ),
            },
        ],
        "agent_reasoning",
    )


def _default_consensus_call(
    agent: Agent,
    scenario: Scenario,
    peer_outputs: list[dict[str, Any]],
) -> tuple[str, int]:
    return _create_llm_completion(
        agent,
        [
            {
                "role": "system",
                "content": build_agent_system_prompt(
                    agent.role,
                    resolve_agent_system_prompt(agent),
                ),
            },
            {
                "role": "user",
                "content": build_consensus_prompt(
                    agent.name,
                    agent.role,
                    peer_outputs,
                )
                + "\n\n"
                + build_agent_user_prompt(
                    agent.role,
                    scenario.description,
                    scenario.program_cost,
                    scenario.max_deficit_constraint,
                ),
            },
        ],
        "consensus_review",
    )


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


def _validated_response(raw_content: str) -> tuple[dict[str, Any], SRRResponse | None, str | None]:
    raw_json, parsed, validation_error = _parse_response(raw_content)
    if parsed is None:
        return raw_json, None, validation_error
    missing = validate_decision_artifacts(parsed)
    if missing:
        return raw_json, None, f"missing decision artifacts: {', '.join(missing)}"
    return raw_json, parsed, None


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
    consensus_call: ConsensusCaller | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    caller = llm_call or _default_llm_call
    reviewer = consensus_call or (_default_consensus_call if llm_call is None else None)
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

        if len(agents) < 2:
            raise ValueError("At least two configured agents are required for deliberation")
        parsed_by_agent: list[tuple[Agent, SRRResponse]] = []
        reasoning_by_agent: dict[int, ReasoningLog] = {}
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
                if isinstance(
                    error,
                    (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError, OSError, socket.error),
                ):
                    logger.exception("Local LLM connection failed for agent %s", agent.id)
                else:
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
            raw_json, parsed, validation_error = _validated_response(raw_content)
            if parsed is None:
                logs.append(_log("SRR", "ERROR", f"{agent.name}: unusable LLM response: {validation_error}."))
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
            reasoning_log = ReasoningLog(
                agent_id=agent.id,
                scenario_id=scenario.id,
                raw_json=raw_json,
                parsed_srr_objects=parsed.model_dump(mode="json"),
                is_schema_valid=True,
                provenance_count=tagged,
            )
            reasoning_by_agent[agent.id] = reasoning_log
            session.add(reasoning_log)

        if len(parsed_by_agent) < 2:
            failure_message = (
                "Deliberation quorum failed: at least two agents must produce "
                "decision-complete LLM artifacts."
            )
            logs.append(_log("COMPLETE", "ERROR", failure_message))
            emit(logs)
            raise RuntimeError(failure_message)

        if reviewer is None:
            logs.append(_log("CONSENSUS", "INFO", "Consensus review callback not configured; retaining supplied test outputs."))
            emit(logs)
        else:
            logs.append(_log("CONSENSUS", "INFO", "Sending peer outputs to each agent for structured consensus review."))
            emit(logs)
            peer_outputs = [
                {
                    "agent": agent.name,
                    "role": agent.role,
                    "srr": response.model_dump(mode="json"),
                }
                for agent, response in parsed_by_agent
            ]
            consensus_results: list[tuple[Agent, SRRResponse]] = []
            for agent, _ in parsed_by_agent:
                try:
                    raw_content, token_usage = reviewer(agent, scenario, peer_outputs)
                    total_tokens += token_usage
                    raw_json, reviewed, validation_error = _validated_response(raw_content)
                    if reviewed is None:
                        logs.append(_log("CONSENSUS", "ERROR", f"{agent.name}: review rejected: {validation_error}."))
                        emit(logs)
                        continue
                    consensus_results.append((agent, reviewed))
                    logs.append(_log("CONSENSUS", "SUCCESS", f"{agent.name}: peer review completed with {token_usage} tokens."))
                    emit(logs)
                    reasoning_log = reasoning_by_agent[agent.id]
                    reasoning_log.raw_json = raw_json
                    reasoning_log.parsed_srr_objects = reviewed.model_dump(mode="json")
                    reasoning_log.provenance_count = _provenance_counts(reviewed)[0]
                except Exception as error:
                    if isinstance(
                        error,
                        (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError, OSError, socket.error),
                    ):
                        logger.exception("Local LLM connection failed during consensus review for agent %s", agent.id)
                    else:
                        logger.exception("Consensus review failed for agent %s", agent.id)
                    logs.append(_log("CONSENSUS", "ERROR", f"{agent.name}: {type(error).__name__}: {error}"))
                    emit(logs)
            if len(consensus_results) < 2:
                failure_message = "Structured consensus failed to produce a two-agent quorum."
                logs.append(_log("COMPLETE", "ERROR", failure_message))
                emit(logs)
                raise RuntimeError(failure_message)
            parsed_by_agent = consensus_results

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
            "logs": logs,
        }


def run_full_shcr_cycle(scenario_id: int) -> dict[str, Any]:
    return execute_full_shcr_cycle(scenario_id)
