import logging
import socket
from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from openai import APIConnectionError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError
from sqlalchemy import select

from backend.agent_templates import agent_revision, mandate_seed, resolve_agent_system_prompt
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
from backend.mandate_snapshots import refresh_mandate_snapshot
from backend.models import (
    Agent,
    AgentInfluenceObservation,
    ConsensusSession,
    ConvergenceStatus,
    DisagreementLog,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
    ScenarioMandateSnapshot,
    SimulationArtifact,
)
from backend.simulation_agent import (
    MAX_SIMULATION_ROUNDS,
    SIMULATION_AGENT_NAME,
    SIMULATION_AGENT_VERSION,
    SIMULATION_TRIGGER,
    build_deterministic_simulation,
    build_simulation_consensus_prompt,
    build_simulation_prompt,
    build_simulation_system_prompt,
    resolve_simulation_runtime_agent,
    sanitize_simulation_payload,
)
from worker.srr_models import SRRResponse, SimulationResponse

logger = logging.getLogger(__name__)
class LLMConfiguredAgent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def llm_base_url(self) -> str | None: ...

    @property
    def llm_api_key(self) -> str | None: ...

    @property
    def llm_model(self) -> str | None: ...

    @property
    def temperature(self) -> float: ...

    @property
    def max_tokens(self) -> int: ...


LLMCaller = Callable[[Agent, Scenario], tuple[str, int]]
ConsensusCaller = Callable[[Agent, Scenario, list[dict[str, Any]]], tuple[str, int]]
SimulationCaller = Callable[
    [Scenario, list[dict[str, Any]], list[dict[str, Any]]],
    tuple[str, int],
]
ProgressReporter = Callable[[list[dict[str, str]]], None]


def _log(stage: str, level: str, message: str) -> dict[str, str]:
    return {"stage": stage, "level": level, "message": message}


def _merge_run_logs(
    stored_logs: list[dict[str, str]] | None,
    cycle_logs: list[dict[str, str]],
) -> list[dict[str, str]]:
    if cycle_logs and cycle_logs[0].get("stage") == "QUEUE":
        return [dict(log) for log in cycle_logs]
    queue_logs = [
        dict(log) for log in stored_logs or [] if log.get("stage") == "QUEUE"
    ]
    return [*queue_logs, *(dict(log) for log in cycle_logs)]


def persist_run_progress(
    scenario_id: int,
    session_id: str,
    logs: list[dict[str, str]],
) -> None:
    with SessionLocal() as session:
        run = session.get(ConsensusSession, session_id)
        if run is None or run.scenario_id != scenario_id:
            raise ValueError("Consensus session is missing or does not belong to the scenario")
        run.logs = _merge_run_logs(run.logs, logs)
        run.progress_stage = logs[-1]["stage"] if logs else run.progress_stage
        if run.status == "QUEUED":
            run.status = "RUNNING"
            run.started_at = run.started_at or datetime.now(timezone.utc)
        session.commit()


def _extract_llm_response(response: object) -> tuple[str, int]:
    return extract_llm_completion(response)


def _create_llm_completion(
    agent: LLMConfiguredAgent,
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


def _mandate_for_agent(mandate_payload: dict[str, Any], agent_id: int) -> str | None:
    rules = mandate_payload.get("agent_rules")
    if not isinstance(rules, list):
        return None
    for rule in rules:
        if isinstance(rule, dict) and rule.get("agent_id") == agent_id:
            value = rule.get("scenario_mandate")
            return value if isinstance(value, str) else None
    return None


def _create_isolated_session(scenario_id: int) -> str:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} does not exist")
        agents = list(session.scalars(select(Agent).order_by(Agent.id)))
        revision = agent_revision(agents, scenario)
        snapshot = session.scalar(
            select(ScenarioMandateSnapshot).where(
                ScenarioMandateSnapshot.scenario_id == scenario_id,
                ScenarioMandateSnapshot.revision == revision,
            )
        )
        if snapshot is None:
            agent_rules = [
                {
                    "agent_id": agent.id,
                    "scenario_mandate": str(mandate_seed(agent)["mandate"]),
                }
                for agent in agents
            ]
            snapshot = ScenarioMandateSnapshot(
                scenario_id=scenario_id,
                revision=revision,
                generated=True,
                agent_count=len(agents),
                rules={},
                agent_rules=agent_rules,
                status="success",
                generated_count=len(agents),
                failure_count=0,
                detail="Isolated programmatic test session",
            )
            session.add(snapshot)
            session.flush()
        session_id = str(uuid4())
        session.add(
            ConsensusSession(
                id=session_id,
                scenario_id=scenario_id,
                mandate_snapshot_id=snapshot.id,
                mandate_revision=revision,
                mandate_payload={"rules": snapshot.rules, "agent_rules": snapshot.agent_rules},
                status="QUEUED",
            )
        )
        session.flush()
        for observation in session.scalars(
            select(AgentInfluenceObservation).where(
                AgentInfluenceObservation.scenario_id == scenario_id,
                AgentInfluenceObservation.run_id.is_(None),
            )
        ):
            observation.run_id = session_id
        session.commit()
        return session_id


def _load_session_context(
    session: Any,
    scenario_id: int,
    session_id: str,
) -> tuple[Scenario, list[Agent], ScenarioMandateSnapshot, ConsensusSession]:
    run = session.get(ConsensusSession, session_id)
    if run is None or run.scenario_id != scenario_id:
        raise ValueError("Consensus session is missing or does not belong to the scenario")
    scenario = session.get(Scenario, scenario_id)
    if scenario is None:
        raise ValueError(f"Scenario {scenario_id} does not exist")
    agents = list(session.scalars(select(Agent).order_by(Agent.id)))
    if not agents:
        raise ValueError("At least one agent is required")
    snapshot = session.get(ScenarioMandateSnapshot, run.mandate_snapshot_id)
    if snapshot is None or snapshot.scenario_id != scenario_id:
        raise ValueError("Mandate snapshot is missing or does not belong to the scenario")
    current_agent_ids = {agent.id for agent in agents}
    is_stale = (
        snapshot.revision != agent_revision(agents, scenario)
        or snapshot.agent_count != len(agents)
    )
    if is_stale:
        logger.warning(
            "Mandate snapshot %s for session %s is stale; refreshing automatically",
            snapshot.revision,
            session_id,
        )
        snapshot = refresh_mandate_snapshot(session, scenario, agents, snapshot)
        run.mandate_snapshot_id = snapshot.id
        run.mandate_revision = snapshot.revision
        run.mandate_payload = {
            "rules": snapshot.rules,
            "agent_rules": snapshot.agent_rules,
        }
        session.flush()
    mandate_agent_ids = {
        item.get("agent_id")
        for item in run.mandate_payload.get("agent_rules", [])
        if isinstance(item, dict)
    }
    if run.mandate_revision != snapshot.revision or mandate_agent_ids != current_agent_ids:
        logger.warning(
            "Consensus session %s mandate payload is stale; synchronizing with snapshot %s",
            session_id,
            snapshot.revision,
        )
        run.mandate_snapshot_id = snapshot.id
        run.mandate_revision = snapshot.revision
        run.mandate_payload = {
            "rules": snapshot.rules,
            "agent_rules": snapshot.agent_rules,
        }
        session.flush()
    return scenario, agents, snapshot, run
def _default_llm_call(
    agent: Agent,
    scenario: Scenario,
    scenario_mandate: str | None = None,
) -> tuple[str, int]:
    return _create_llm_completion(
        agent,
        [
            {
                "role": "system",
                "content": build_agent_system_prompt(
                    agent.role,
                    resolve_agent_system_prompt(agent),
                    scenario_mandate,
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
    scenario_mandate: str | None = None,
) -> tuple[str, int]:
    return _create_llm_completion(
        agent,
        [
            {
                "role": "system",
                "content": build_agent_system_prompt(
                    agent.role,
                    resolve_agent_system_prompt(agent),
                    scenario_mandate,
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
def _default_simulation_call(
    scenario: Scenario,
    conflicts: list[dict[str, Any]],
    peer_outputs: list[dict[str, Any]],
    agents: list[Agent],
) -> tuple[str, int]:
    runtime_source = resolve_simulation_runtime_agent(agents)
    if runtime_source is None:
        raise ValueError("No configured LLM runtime is available for the native Simulation Agent")
    return _create_llm_completion(
        runtime_source,
        [
            {"role": "system", "content": build_simulation_system_prompt()},
            {
                "role": "user",
                "content": build_simulation_prompt(
                    scenario.description,
                    scenario.program_cost,
                    scenario.max_deficit_constraint,
                    conflicts,
                    peer_outputs,
                ),
            },
        ],
        "native_simulation_arbitration",
    )


def _default_simulation_consensus_call(
    agent: Agent,
    scenario: Scenario,
    peer_outputs: list[dict[str, Any]],
    simulation_output: dict[str, Any],
    scenario_mandate: str | None = None,
) -> tuple[str, int]:
    return _create_llm_completion(
        agent,
        [
            {
                "role": "system",
                "content": build_agent_system_prompt(
                    agent.role,
                    resolve_agent_system_prompt(agent),
                    scenario_mandate,
                ),
            },
            {
                "role": "user",
                "content": build_simulation_consensus_prompt(
                    agent.name,
                    agent.role,
                    peer_outputs,
                    simulation_output,
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
        "simulation_assisted_consensus",
    )


def _parse_response(raw_content: str) -> tuple[dict[str, Any], SRRResponse | None, str | None]:
    try:
        payload = extract_json_object(raw_content)
    except ValueError as error:
        return {"parse_error": str(error)}, None, str(error)
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


def _persist_simulation_artifact(
    session: Any,
    scenario: Scenario,
    session_id: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    status: str,
    latency_ms: float,
    token_usage: int,
    round_number: int,
) -> SimulationArtifact:
    artifact = session.scalar(
        select(SimulationArtifact).where(
            SimulationArtifact.run_id == session_id,
            SimulationArtifact.simulation_version == SIMULATION_AGENT_VERSION,
            SimulationArtifact.round_number == round_number,
        )
    )
    if artifact is None:
        artifact = SimulationArtifact(
            run_id=session_id,
            scenario_id=scenario.id,
            trigger=SIMULATION_TRIGGER,
            round_number=round_number,
            input_payload=input_payload,
            output_payload=output_payload,
            status=status,
            simulation_version=SIMULATION_AGENT_VERSION,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
        session.add(artifact)
    else:
        artifact.input_payload = input_payload
        artifact.output_payload = output_payload
        artifact.status = status
        artifact.latency_ms = latency_ms
        artifact.token_usage = token_usage
    session.flush()
    return artifact


def _run_native_simulation(
    session: Any,
    scenario: Scenario,
    session_id: str,
    agents: list[Agent],
    conflicts: list[dict[str, Any]],
    peer_outputs: list[dict[str, Any]],
    simulation_call: SimulationCaller | None,
    logs: list[dict[str, str]],
    emit: ProgressReporter,
    round_number: int,
) -> tuple[dict[str, Any] | None, int]:
    started = perf_counter()
    logs.append(_log("SIMULATION", "INFO", f"{SIMULATION_AGENT_NAME} invoked automatically for {len(conflicts)} DDR conflict(s)."))
    emit(logs)
    safe_peer_outputs = [
        sanitize_simulation_payload(dict(peer)) for peer in peer_outputs
    ]
    simulation_input = {
        "conflicts": conflicts,
        "sectoral_inputs": [
            {
                "agent": peer.get("agent"),
                "role": peer.get("role"),
                "predictions": (
                    peer["srr"].get("predictions", [])
                    if isinstance(peer.get("srr"), dict)
                    else []
                ),
                "alternatives": (
                    peer["srr"].get("alternatives", [])
                    if isinstance(peer.get("srr"), dict)
                    else []
                ),
                "recommendation": (
                    peer["srr"].get("recommendation")
                    if isinstance(peer.get("srr"), dict)
                    else None
                ),
            }
            for peer in safe_peer_outputs
        ],
        "scenario": {
            "description": scenario.description,
            "program_cost": scenario.program_cost,
            "max_deficit_constraint": scenario.max_deficit_constraint,
        },
    }
    _persist_simulation_artifact(
        session,
        scenario,
        session_id,
        simulation_input,
        {
            "agent_name": SIMULATION_AGENT_NAME,
            "simulation_version": SIMULATION_AGENT_VERSION,
            "status": "RUNNING",
            "evidence_status": "modelled",
            "message": "Simulation request accepted; awaiting structured arbitration output.",
        },
        "RUNNING",
        0.0,
        0,
        round_number,
    )
    session.commit()
    try:
        fallback_reason: str | None = None
        try:
            if simulation_call is not None:
                raw_content, token_usage = simulation_call(scenario, conflicts, peer_outputs)
                raw_payload = extract_json_object(raw_content)
            else:
                runtime_source = resolve_simulation_runtime_agent(agents)
                if runtime_source is None:
                    raise ValueError("Native simulation LLM runtime is unavailable")
                raw_content, token_usage = _default_simulation_call(
                    scenario,
                    conflicts,
                    peer_outputs,
                    agents,
                )
                raw_payload = extract_json_object(raw_content)
            parsed = SimulationResponse.model_validate(raw_payload)
            missing = validate_decision_artifacts(parsed)
            if missing:
                raise ValueError(f"simulation missing decision artifacts: {', '.join(missing)}")
        except Exception as provider_error:
            fallback_reason = f"{type(provider_error).__name__}: {provider_error}"
            raw_payload = build_deterministic_simulation(
                scenario.description,
                scenario.max_deficit_constraint,
                conflicts,
                peer_outputs,
            )
            parsed = SimulationResponse.model_validate(raw_payload)
            token_usage = 0
        output_payload = sanitize_simulation_payload(parsed.model_dump(mode="json"))
        if fallback_reason is not None:
            output_payload["fallback_reason"] = fallback_reason
        latency_ms = (perf_counter() - started) * 1000.0
        _persist_simulation_artifact(
            session,
            scenario,
            session_id,
            simulation_input,
            output_payload,
            "SUCCEEDED",
            latency_ms,
            token_usage,
            round_number,
        )
        session.commit()
        logs.append(
            _log(
                "SIMULATION",
                "WARNING" if fallback_reason else "SUCCESS",
                (
                    f"{SIMULATION_AGENT_NAME} used deterministic fallback after provider failure."
                    if fallback_reason
                    else f"{SIMULATION_AGENT_NAME} produced a structured modelled resolution."
                ),
            )
        )
        emit(logs)
        return output_payload, token_usage
    except Exception as error:
        output_payload = {
            "agent_name": SIMULATION_AGENT_NAME,
            "simulation_version": SIMULATION_AGENT_VERSION,
            "evidence_status": "modelled",
            "error": f"{type(error).__name__}: {error}",
        }
        latency_ms = (perf_counter() - started) * 1000.0
        _persist_simulation_artifact(
            session,
            scenario,
            session_id,
            simulation_input,
            output_payload,
            "FAILED",
            latency_ms,
            0,
            round_number,
        )
        session.commit()
        logs.append(_log("SIMULATION", "ERROR", output_payload["error"]))
        emit(logs)
        return None, 0


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


def _collect_prediction_conflicts(
    parsed_by_agent: list[tuple[Agent, SRRResponse]],
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for index, (agent_i, response_i) in enumerate(parsed_by_agent):
        for agent_j, response_j in parsed_by_agent[index + 1 :]:
            vector = detect_divergence_vector(
                response_i.divergence_object(),
                response_j.divergence_object(),
            )
            if vector["dP"]:
                conflicts.append(
                    {
                        "agent_i": agent_i.name,
                        "agent_j": agent_j.name,
                        "components": [key for key, value in vector.items() if value],
                        "route": SIMULATION_TRIGGER,
                    }
                )
    return conflicts


def _detect_ddr_conflicts(
    session: Any,
    scenario: Scenario,
    session_id: str,
    parsed_by_agent: list[tuple[Agent, SRRResponse]],
    logs: list[dict[str, str]],
    emit: ProgressReporter,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for index, (agent_i, response_i) in enumerate(parsed_by_agent):
        for agent_j, response_j in parsed_by_agent[index + 1 :]:
            vector = detect_divergence_vector(
                response_i.divergence_object(),
                response_j.divergence_object(),
            )
            route = SIMULATION_TRIGGER if vector["dP"] else None
            components = [key for key, value in vector.items() if value]
            active_conflicts = ", ".join(components) or "none"
            logs.append(_log("DDR", "WARNING" if components else "SUCCESS", f"{agent_i.name} vs {agent_j.name}: conflicts={active_conflicts}; route={route or 'none'}."))
            emit(logs)
            session.add(
                DisagreementLog(
                    run_id=session_id,
                    scenario_id=scenario.id,
                    agent_i=agent_i.id,
                    agent_j=agent_j.id,
                    resolution_route=route,
                    **vector,
                )
            )
            if route:
                conflicts.append(
                    {
                        "agent_i": agent_i.name,
                        "agent_j": agent_j.name,
                        "components": components,
                        "route": route,
                    }
                )
    return conflicts


def _run_consensus_round(
    agents: list[Agent],
    scenario: Scenario,
    run: ConsensusSession,
    parsed_by_agent: list[tuple[Agent, SRRResponse]],
    reasoning_by_agent: dict[int, ReasoningLog],
    consensus_call: Callable[..., tuple[str, int]] | None,
    peer_outputs: list[dict[str, Any]],
    simulation_output: dict[str, Any] | None,
    logs: list[dict[str, str]],
    emit: ProgressReporter,
) -> tuple[list[tuple[Agent, SRRResponse]], int]:
    consensus_results: list[tuple[Agent, SRRResponse]] = []
    round_tokens = 0
    log_stage = "SIMULATION_CONSENSUS" if simulation_output is not None else "CONSENSUS"
    effective_peer_outputs = list(peer_outputs)
    if simulation_output is not None:
        effective_peer_outputs.append(
            {
                "agent": SIMULATION_AGENT_NAME,
                "role": "Native macro-fiscal simulation arbiter",
                "srr": simulation_output,
            }
        )
    if simulation_output is None:
        logs.append(_log("CONSENSUS", "INFO", "Sending peer outputs to each agent for structured consensus review."))
        emit(logs)
    for agent, initial_response in parsed_by_agent:
        try:
            if consensus_call is None:
                if simulation_output is None:
                    raw_content, token_usage = _default_consensus_call(
                        agent,
                        scenario,
                        peer_outputs,
                        _mandate_for_agent(run.mandate_payload, agent.id),
                    )
                else:
                    raw_content, token_usage = _default_simulation_consensus_call(
                        agent,
                        scenario,
                        peer_outputs,
                        simulation_output,
                        _mandate_for_agent(run.mandate_payload, agent.id),
                    )
            else:
                raw_content, token_usage = consensus_call(agent, scenario, effective_peer_outputs)
            round_tokens += token_usage
            raw_json, reviewed, _ = _validated_response(raw_content)
            if reviewed is None:
                consensus_results.append((agent, initial_response))
                logs.append(
                    _log(
                        log_stage,
                        "WARNING",
                        f"{agent.name}: review response did not pass schema validation; retained validated SRR artifacts.",
                    )
                )
                emit(logs)
                continue
            consensus_results.append((agent, reviewed))
            logs.append(_log(log_stage, "SUCCESS", f"{agent.name}: peer review completed with {token_usage} tokens."))
            emit(logs)
            reasoning_log = reasoning_by_agent.get(agent.id)
            if reasoning_log is not None:
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
            consensus_results.append((agent, initial_response))
            logs.append(
                _log(
                    log_stage,
                    "WARNING",
                    f"{agent.name}: {type(error).__name__}; retained validated SRR artifacts.",
                )
            )
            emit(logs)
    return consensus_results, round_tokens


def execute_full_shcr_cycle(
    scenario_id: int,
    session_id: Any,
    llm_call: Any = None,
    consensus_call: ConsensusCaller | None = None,
    progress: ProgressReporter | None = None,
    simulation_call: SimulationCaller | None = None,
    enable_simulation: bool = True,
) -> dict[str, Any]:
    if not isinstance(session_id, str):
        legacy_llm_call = session_id
        legacy_consensus_call = llm_call if callable(llm_call) else None
        session_id = _create_isolated_session(scenario_id)
        llm_call = legacy_llm_call
        consensus_call = legacy_consensus_call
    started_at = perf_counter()
    caller = llm_call or _default_llm_call
    reviewer = consensus_call or (_default_consensus_call if llm_call is None else None)
    external_emit = progress or (lambda _logs: None)

    def emit(current_logs: list[dict[str, str]]) -> None:
        persist_run_progress(scenario_id, session_id, current_logs)
        external_emit(current_logs)

    logs = [
        _log(
            "INITIALIZE",
            "INFO",
            f"Starting SHCR cycle for scenario {scenario_id}, session {session_id}.",
        )
    ]
    emit(logs)

    with SessionLocal() as session:
        scenario, agents, mandate_snapshot, run = _load_session_context(
            session,
            scenario_id,
            session_id,
        )
        run.status = "RUNNING"
        run.started_at = run.started_at or datetime.now(timezone.utc)
        session.commit()

        if len(agents) < 2:
            raise ValueError("At least two configured agents are required for deliberation")
        parsed_by_agent: list[tuple[Agent, SRRResponse]] = []
        reasoning_by_agent: dict[int, ReasoningLog] = {}
        observations = list(
            session.scalars(
                select(AgentInfluenceObservation)
                .where(
                    AgentInfluenceObservation.scenario_id == scenario.id,
                    AgentInfluenceObservation.run_id == session_id,
                )
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
                if llm_call is None:
                    raw_content, token_usage = _default_llm_call(
                        agent,
                        scenario,
                        _mandate_for_agent(run.mandate_payload, agent.id),
                    )
                else:
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
                        run_id=session_id,
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
                        run_id=session_id,
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
                run_id=session_id,
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
            run.status = "FAILED"
            run.error = failure_message
            run.logs = _merge_run_logs(run.logs, logs)
            run.progress_stage = "COMPLETE"
            run.completed_at = datetime.now(timezone.utc)
            session.commit()
            raise RuntimeError(failure_message)

        if reviewer is None:
            logs.append(_log("CONSENSUS", "INFO", "Consensus review callback not configured; retaining supplied test outputs."))
            emit(logs)
        else:
            peer_outputs = [
                {
                    "agent": agent.name,
                    "role": agent.role,
                    "srr": response.model_dump(mode="json"),
                }
                for agent, response in parsed_by_agent
            ]
            consensus_results, round_tokens = _run_consensus_round(
                agents,
                scenario,
                run,
                parsed_by_agent,
                reasoning_by_agent,
                consensus_call,
                peer_outputs,
                None,
                logs,
                emit,
            )
            total_tokens += round_tokens
            if len(consensus_results) < 2:
                failure_message = "Structured consensus failed to produce a two-agent quorum."
                logs.append(_log("COMPLETE", "ERROR", failure_message))
                emit(logs)
                run.status = "FAILED"
                run.error = failure_message
                run.logs = _merge_run_logs(run.logs, logs)
                run.progress_stage = "COMPLETE"
                run.completed_at = datetime.now(timezone.utc)
                session.commit()
                raise RuntimeError(failure_message)
            parsed_by_agent = consensus_results

        final_provenance = [_provenance_counts(response) for _, response in parsed_by_agent]
        tagged_items = sum(tagged for tagged, _ in final_provenance)
        total_items = sum(total for _, total in final_provenance)
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

        conflicts = _detect_ddr_conflicts(
            session,
            scenario,
            session_id,
            parsed_by_agent,
            logs,
            emit,
        )
        simulation_rounds = 0
        simulation_artifact_id: int | None = None
        remaining_conflicts = conflicts
        while (
            enable_simulation
            and reviewer is not None
            and remaining_conflicts
            and simulation_rounds < MAX_SIMULATION_ROUNDS
        ):
            round_number = simulation_rounds + 1
            peer_outputs = [
                {
                    "agent": agent.name,
                    "role": agent.role,
                    "srr": response.model_dump(mode="json"),
                }
                for agent, response in parsed_by_agent
            ]
            simulation_output, simulation_tokens = _run_native_simulation(
                session,
                scenario,
                session_id,
                agents if llm_call is None else [],
                remaining_conflicts,
                peer_outputs,
                simulation_call,
                logs,
                emit,
                round_number,
            )
            total_tokens += simulation_tokens
            artifact = session.scalar(
                select(SimulationArtifact).where(
                    SimulationArtifact.run_id == session_id,
                    SimulationArtifact.simulation_version == SIMULATION_AGENT_VERSION,
                    SimulationArtifact.round_number == round_number,
                )
            )
            if simulation_artifact_id is None and artifact is not None:
                simulation_artifact_id = artifact.id
            if simulation_output is None:
                break
            simulation_rounds = round_number
            logs.append(
                _log(
                    "SIMULATION_CONSENSUS",
                    "INFO",
                    f"Feeding native simulation round {round_number} back to all sectoral agents.",
                )
            )
            emit(logs)
            simulation_results, round_tokens = _run_consensus_round(
                agents,
                scenario,
                run,
                parsed_by_agent,
                reasoning_by_agent,
                consensus_call,
                peer_outputs,
                simulation_output,
                logs,
                emit,
            )
            total_tokens += round_tokens
            if len(simulation_results) < 2:
                break
            parsed_by_agent = simulation_results
            remaining_conflicts = _collect_prediction_conflicts(parsed_by_agent)
            if artifact is not None:
                artifact_payload = dict(artifact.output_payload)
                artifact_payload["follow_up_consensus_status"] = "completed"
                artifact_payload["remaining_prediction_conflicts"] = len(remaining_conflicts)
                artifact.output_payload = artifact_payload
                session.commit()
            logs.append(
                _log(
                    "SIMULATION_CONSENSUS",
                    "SUCCESS" if not remaining_conflicts else "WARNING",
                    f"Follow-up round {round_number} completed with {len(remaining_conflicts)} remaining prediction conflict(s).",
                )
            )
            emit(logs)
        if remaining_conflicts and simulation_rounds == MAX_SIMULATION_ROUNDS:
            logs.append(
                _log(
                    "SIMULATION_CONSENSUS",
                    "WARNING",
                    f"Stopped after the bounded maximum of {MAX_SIMULATION_ROUNDS} simulation rounds; valid dissent remains explicit.",
                )
            )
            emit(logs)

        final_provenance = [_provenance_counts(response) for _, response in parsed_by_agent]
        tagged_items = sum(tagged for tagged, _ in final_provenance)
        total_items = sum(total for _, total in final_provenance)
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
            run_id=session_id,
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
        session.flush()
        logs.append(_log("COMPLETE", "SUCCESS", f"Cycle completed with state {convergence_status.value}, {total_tokens} tokens, and {latency_ms:.2f} ms latency."))
        result_payload = {
            "metric_snapshot_id": snapshot.id,
            "session_id": session_id,
            "scenario_id": scenario.id,
            "provenance_completeness_percent": provenance_completeness,
            "hard_constraint_violation_rate": violation_rate,
            "feasible_alternatives_count": len(feasible),
            "convergence_status": convergence_status.value,
            "latency_ms": latency_ms,
            "token_usage": total_tokens,
            "simulation_artifact_id": simulation_artifact_id,
            "simulation_rounds": simulation_rounds,
            "simulation_triggered": bool(conflicts),
            "logs": logs,
        }
        run.result_payload = result_payload
        run.logs = _merge_run_logs(run.logs, logs)
        run.progress_stage = "COMPLETE"
        run.status = "SUCCEEDED"
        run.completed_at = datetime.now(timezone.utc)
        session.commit()
        emit(logs)

        return result_payload


def run_full_shcr_cycle(scenario_id: int, session_id: str) -> dict[str, Any]:
    return execute_full_shcr_cycle(scenario_id, session_id)
