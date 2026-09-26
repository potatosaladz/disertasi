import hashlib
import json
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

from backend.agent_templates import (
    agent_revision,
    mandate_seed,
    resolve_agent_system_prompt,
    scenario_deliberative_agents,
    semantic_agent_name,
)
from backend.analytical_events import analytical_event, normalize_event
from backend.core_algorithms import (
    build_agent_system_prompt,
    build_agent_user_prompt,
    build_consensus_prompt,
    calculate_dynamic_influence,
    calculate_violation_rate,
    DDR_COMPONENT_DETAILS,
    detect_divergence_vector,
    extract_json_object,
    extract_llm_completion,
    llm_request_headers,
    log_llm_outbound,
    resolve_disagreement_route,
    resolve_llm_runtime_config,
    STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
    validate_decision_artifacts,
)
from backend.database import SessionLocal
from backend.global_config import apply_global_llm_config, get_global_llm_config
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
from worker.car_solver import evaluate_car_constraints
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
ProgressReporter = Callable[[list[dict[str, Any]]], None]


def _event_id_message(stage: str, message: str) -> str:
    replacements = {
        "Sending peer outputs to each agent for structured consensus review.": "Mengirim keluaran rekan ke setiap agen untuk tinjauan konsensus terstruktur.",
        "Consensus review callback not configured; retaining supplied test outputs.": "Callback tinjauan konsensus tidak dikonfigurasi; keluaran uji yang diberikan dipertahankan.",
        "Calculating disagreement vectors.": "Menghitung vektor perbedaan.",
        "Persisting convergence, provenance, latency, and token metrics.": "Menyimpan metrik konvergensi, provenance, latensi, dan token.",
    }
    if message in replacements:
        return replacements[message]
    if message.startswith("Calling "):
        return message.replace("Calling ", "Memanggil ", 1).replace(" using ", " menggunakan ", 1)
    if message.startswith("Cycle completed with state "):
        return message.replace("Cycle completed with state ", "Siklus selesai dengan status ", 1).replace(" tokens, and ", " token, dan ", 1).replace(" ms latency.", " ms latensi.")
    if message.startswith("Follow-up round "):
        return message.replace("Follow-up round ", "Ronde tindak lanjut ", 1).replace(" completed with ", " selesai dengan ", 1).replace(" remaining prediction conflict(s).", " konflik prediksi tersisa.")
    if stage == "DDR" and ": conflicts=" in message:
        return message.replace(": conflicts=", ": konflik=", 1).replace("; route=", "; jalur=", 1).replace("none", "tidak ada")
    if stage == "SRR" and ": valid SRR with " in message:
        return message.replace(": valid SRR with ", ": SRR valid dengan ", 1).replace(" sourced artifacts and ", " artefak bersumber dan ", 1).replace(" tokens.", " token.")
    if stage == "SRR" and ": unusable LLM response:" in message:
        return message.replace(": unusable LLM response:", ": respons LLM tidak dapat digunakan:", 1)
    if stage in {"CONSENSUS", "SIMULATION_CONSENSUS"} and ": peer review completed with " in message:
        return message.replace(": peer review completed with ", ": tinjauan rekan selesai dengan ", 1).replace(" tokens.", " token.")
    if "retained validated SRR artifacts" in message:
        return message.replace("retained validated SRR artifacts", "artefak SRR tervalidasi dipertahankan")
    if message.startswith("Feeding native simulation round "):
        return message.replace("Feeding native simulation round ", "Mengirim hasil simulasi native ronde ", 1).replace(" back to all sectoral agents.", " kembali ke semua agen sektoral.")
    if message.startswith("Stopped after the bounded maximum of "):
        return message.replace("Stopped after the bounded maximum of ", "Dihentikan setelah batas maksimum ", 1).replace(" simulation rounds; valid dissent remains explicit.", " ronde simulasi; dissent yang valid tetap dinyatakan eksplisit.")
    if message.startswith("Deliberation quorum failed:"):
        return message.replace("Deliberation quorum failed:", "Kuorum deliberasi gagal:", 1)
    if message.startswith("Structured consensus failed"):
        return message.replace("Structured consensus failed", "Konsensus terstruktur gagal", 1)
    if stage == "SIMULATION":
        return f"Simulasi gagal: {message}"
    if stage == "SRR":
        return f"Kegagalan agen SRR: {message}"
    return f"Peristiwa {stage}: {message}"


def _log(
    stage: str,
    level: str,
    message: str,
    *,
    id_message: str | None = None,
    code: str | None = None,
    agent_id: int | None = None,
    agent_name: str | None = None,
    round_number: int | None = None,
    metric: dict[str, Any] | None = None,
    statutory: dict[str, Any] | None = None,
    economic: dict[str, Any] | None = None,
    fallback: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    why: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return analytical_event(
        stage,
        level,
        code or f"{stage}_{level}",
        id_message or _event_id_message(stage, message),
        message,
        agent_id=agent_id,
        agent_name=agent_name,
        round_number=round_number,
        metric=metric,
        statutory=statutory,
        economic=economic,
        fallback=fallback,
        task=task,
        result=result,
        why=why,
        metadata=metadata,
    )


def _merge_run_logs(
    stored_logs: list[dict[str, Any]] | None,
    cycle_logs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if cycle_logs and cycle_logs[0].get("stage") == "QUEUE":
        return [dict(log) for log in cycle_logs]
    queue_logs = [
        dict(log) for log in stored_logs or [] if log.get("stage") == "QUEUE"
    ]
    return [*queue_logs, *(dict(log) for log in cycle_logs)]


def persist_run_progress(
    scenario_id: int,
    session_id: str,
    logs: list[dict[str, Any]],
) -> None:
    with SessionLocal() as session:
        run = session.get(ConsensusSession, session_id)
        if run is None or run.scenario_id != scenario_id:
            raise ValueError("Consensus session is missing or does not belong to the scenario")
        enriched_logs: list[dict[str, Any]] = []
        for raw_log in logs:
            event = normalize_event(raw_log)
            event["run_id"] = session_id
            event["task_id"] = run.celery_task_id
            event["scenario_id"] = scenario_id
            enriched_logs.append(event)
        logs[:] = enriched_logs
        run.logs = _merge_run_logs(run.logs, enriched_logs)
        run.progress_stage = enriched_logs[-1]["stage"] if enriched_logs else run.progress_stage
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


def _display_name_for_agent(
    mandate_payload: dict[str, Any],
    agent: Agent,
) -> str:
    rules = mandate_payload.get("agent_rules")
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, dict) and rule.get("agent_id") == agent.id:
                value = rule.get("display_name")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return semantic_agent_name(agent)


def _create_isolated_session(scenario_id: int) -> str:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} does not exist")
        scoped_agent_ids = list(
            session.scalars(
                select(AgentInfluenceObservation.agent_id).where(
                    AgentInfluenceObservation.scenario_id == scenario_id,
                    AgentInfluenceObservation.run_id.is_(None),
                )
            )
        )
        scoped_agent_id_set = set(scoped_agent_ids)
        scenario_agents = scenario_deliberative_agents(session, scenario_id)
        agents = (
            [agent for agent in scenario_agents if agent.id in scoped_agent_id_set]
            if scoped_agent_id_set
            else scenario_agents
        )
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
    mandate_agent_ids = [
        item.get("agent_id")
        for item in run.mandate_payload.get("agent_rules", [])
        if isinstance(item, dict) and isinstance(item.get("agent_id"), int)
    ]
    mandate_agent_id_set = set(mandate_agent_ids)
    agents = [
        agent
        for agent in scenario_deliberative_agents(session, scenario_id)
        if agent.id in mandate_agent_id_set
    ]
    if not agents:
        raise ValueError("At least one agent is required")
    global_config = get_global_llm_config(session)
    if global_config.apply_to_all:
        for agent in agents:
            apply_global_llm_config(agent, global_config)
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
            **dict(run.mandate_payload or {}),
            "rules": snapshot.rules,
            "agent_rules": snapshot.agent_rules,
        }
        session.flush()
    payload_agent_ids = {
        item.get("agent_id")
        for item in run.mandate_payload.get("agent_rules", [])
        if isinstance(item, dict)
    }
    if run.mandate_revision != snapshot.revision or payload_agent_ids != current_agent_ids:
        logger.warning(
            "Consensus session %s mandate payload is stale; synchronizing with snapshot %s",
            session_id,
            snapshot.revision,
        )
        run.mandate_snapshot_id = snapshot.id
        run.mandate_revision = snapshot.revision
        run.mandate_payload = {
            **dict(run.mandate_payload or {}),
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
                    scenario.simulation_payload(),
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
                    scenario.simulation_payload(),
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
                    scenario.simulation_payload(),
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
                    scenario.simulation_payload(),
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


def _record_deliberation_stage(
    reasoning_by_agent: dict[int, ReasoningLog],
    parsed_by_agent: list[tuple[Agent, SRRResponse]],
    stage: str,
    round_number: int,
) -> None:
    for agent, response in parsed_by_agent:
        reasoning_log = reasoning_by_agent.get(agent.id)
        if reasoning_log is None:
            continue
        artifacts = sanitize_simulation_payload(response.model_dump(mode="json"))
        tagged, _ = _provenance_counts(response)
        reasoning_log.parsed_srr_objects = artifacts
        reasoning_log.provenance_count = tagged
        reasoning_log.deliberation_history = [
            *list(reasoning_log.deliberation_history or []),
            {
                "stage": stage,
                "round_number": round_number,
                "artifacts": artifacts,
            },
        ]


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
    logs: list[dict[str, Any]],
    emit: ProgressReporter,
    round_number: int,
) -> tuple[dict[str, Any] | None, int]:
    started = perf_counter()
    logs.append(
        _log(
            "SIMULATION",
            "INFO",
            f"{SIMULATION_AGENT_NAME} invoked automatically for {len(conflicts)} DDR conflict(s).",
            code="SIMULATION_REQUESTED",
            id_message=f"{SIMULATION_AGENT_NAME} dipanggil otomatis untuk {len(conflicts)} konflik DDR.",
            round_number=round_number,
            metric={"name": "ddr_conflict_count", "value": len(conflicts), "unit": "pairs", "status": "calculated"},
            statutory={"status": "pending-car", "constraint": "DEFICIT_3PCT", "ceiling_percent_gdp": STATUTORY_DEFICIT_CEILING_PERCENT_GDP, "source_tags": ["UU17_2003_P12"]},
            economic={"status": "modelled", "inputs": scenario.simulation_payload(), "outputs": {}, "reason": "Simulation uses structured sectoral alternatives; unsupported coefficients remain not-calculated."},
            metadata={"resolution_path": SIMULATION_TRIGGER},
        )
    )
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
            **scenario.simulation_payload(),
            "statutory_deficit_ceiling_percent": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
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
    token_usage = 0
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
            fallback_reason = type(provider_error).__name__
            raw_payload = build_deterministic_simulation(
                scenario.description,
                scenario.simulation_payload(),
                conflicts,
                peer_outputs,
            )
            parsed = SimulationResponse.model_validate(raw_payload)
        output_payload = sanitize_simulation_payload(parsed.model_dump(mode="json"))
        output_payload["fallback"] = {
            "used": fallback_reason is not None,
            "kind": "deterministic-native" if fallback_reason else None,
            "reason": fallback_reason,
            "retained_artifact": "structured sectoral alternatives pending CAR evaluation",
            "consensus_impact": "Fallback output is fed back to sectoral agents as modelled evidence only.",
        }
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
                code="SIMULATION_FALLBACK_USED" if fallback_reason else "SIMULATION_RESOLUTION_PRODUCED",
                id_message=(
                    f"{SIMULATION_AGENT_NAME} menggunakan fallback deterministik setelah kegagalan provider."
                    if fallback_reason
                    else f"{SIMULATION_AGENT_NAME} menghasilkan resolusi modelled terstruktur."
                ),
                round_number=round_number,
                metric={"name": "modelled_alternative_count", "value": len(output_payload.get("alternatives", [])), "unit": "alternatives", "status": "calculated"},
                statutory={"status": "pending-car", "constraint": "DEFICIT_3PCT", "ceiling_percent_gdp": STATUTORY_DEFICIT_CEILING_PERCENT_GDP, "source_tags": ["UU17_2003_P12"]},
                economic={"status": output_payload.get("calculation_status", "modelled"), "inputs": {"sectoral_input_count": len(peer_outputs)}, "outputs": {"resolution_status": output_payload.get("resolution_status")}, "impact": "Result remains modelled evidence and must pass CAR."},
                fallback=output_payload["fallback"],
                metadata={"resolution_path": SIMULATION_TRIGGER},
            )
        )
        emit(logs)
        return output_payload, token_usage
    except Exception as error:
        output_payload = {
            "agent_name": SIMULATION_AGENT_NAME,
            "simulation_version": SIMULATION_AGENT_VERSION,
            "evidence_status": "modelled",
            "error": f"{type(error).__name__}",
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
        logs.append(
            _log(
                "SIMULATION",
                "ERROR",
                output_payload["error"],
                code="SIMULATION_FAILED",
                round_number=round_number,
                fallback={"used": False, "kind": None, "reason": output_payload["error"]},
                metadata={"error_type": type(error).__name__},
            )
        )
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
        " ".join(
            response.recommendation.content.casefold().split()
        )
        if response.recommendation is not None
        else None
        for response in parsed_responses
    }
    feasible_ids = {id(alternative) for alternative in feasible_alternatives}
    alternative_profiles = {
        tuple(
            sorted(
                (
                    " ".join(alternative.name.casefold().split()),
                    round(alternative.deficit, 8),
                    round(alternative.utility, 8),
                )
                for alternative in response.decision_alternatives()
                if id(alternative) in feasible_ids
            )
        )
        for response in parsed_responses
    }

    if len(alternative_profiles) > 1 or len(recommendations) > 1:
        return ConvergenceStatus.PARETO_SET
    return ConvergenceStatus.FULL_CONSENSUS


def _agent_interactions(
    agent: Agent,
    response: SRRResponse,
    parsed_by_agent: list[tuple[Agent, SRRResponse]],
) -> list[dict[str, Any]]:
    interactions: list[dict[str, Any]] = []
    for peer, peer_response in parsed_by_agent:
        if peer.id == agent.id:
            continue
        vector = detect_divergence_vector(
            response.divergence_object(),
            peer_response.divergence_object(),
        )
        active_components = [key for key, value in vector.items() if value]
        interactions.append(
            {
                "peer_agent_id": peer.id,
                "peer_agent_name": peer.name,
                "active_components": active_components,
                "agreement_ratio": round(1.0 - len(active_components) / len(vector), 4),
            }
        )
    return interactions


def _ensure_influence_observations(
    session: Any,
    scenario: Scenario,
    session_id: str,
    parsed_by_agent: list[tuple[Agent, SRRResponse]],
    observations: list[AgentInfluenceObservation],
) -> list[AgentInfluenceObservation]:
    by_agent = {observation.agent_id: observation for observation in observations}
    for agent, response in parsed_by_agent:
        interactions = _agent_interactions(agent, response, parsed_by_agent)
        artifact_groups = (
            response.evidence,
            response.predictions,
            response.risks,
            response.uncertainties,
            response.decision_alternatives(),
        )
        complete_groups = sum(bool(group) for group in artifact_groups)
        completeness = complete_groups / len(artifact_groups)
        tagged, total = _provenance_counts(response)
        provenance_quality = tagged / total if total else 0.0
        peer_alignment = (
            sum(item["agreement_ratio"] for item in interactions) / len(interactions)
            if interactions
            else 1.0
        )
        uncertainty_count = len(response.uncertainties)
        inspectable_count = sum(len(group) for group in artifact_groups)
        uncertainty_burden = min(1.0, uncertainty_count / max(1, inspectable_count))
        observation = by_agent.get(agent.id)
        if observation is None:
            observation = AgentInfluenceObservation(
                run_id=session_id,
                agent_id=agent.id,
                scenario_id=scenario.id,
                proposition=scenario.description,
                X=completeness,
                Q=provenance_quality,
                H=1.0,
                S=peer_alignment,
                U=uncertainty_burden,
                gate=1,
            )
            session.add(observation)
            observations.append(observation)
            by_agent[agent.id] = observation
        observation.X = completeness
        observation.Q = provenance_quality
        observation.H = 1.0
        observation.S = peer_alignment
        observation.U = uncertainty_burden
        observation.gate = 1
        observation.interaction_payload = interactions
        observation.calculation_payload = {
            "version": "rar-dai-v2",
            "formula": "raw_score = theta_x*X + theta_q*Q + theta_h*H + theta_s*S - theta_u*U; normalized_weight = gated_softmax(raw_score)",
            "calculation_status": "calculated",
            "gate_reason": "Decision-complete SRR passed schema validation.",
            "dimensions": {
                "X": {
                    "label": "decision-artifact completeness ratio",
                    "value": completeness,
                    "source": "validated SRR collections",
                    "status": "calculated",
                },
                "Q": {
                    "label": "source-tagged provenance ratio",
                    "value": provenance_quality,
                    "source": "SRR source_tag coverage",
                    "status": "calculated",
                },
                "H": {
                    "label": "current-run recency prior",
                    "value": 1.0,
                    "source": "current consensus session",
                    "status": "prior",
                },
                "S": {
                    "label": "mean pairwise DDR agreement ratio",
                    "value": peer_alignment,
                    "source": "pairwise DDR vectors",
                    "status": "calculated",
                },
                "U": {
                    "label": "uncertainty artifact burden",
                    "value": uncertainty_burden,
                    "source": "validated SRR uncertainty items",
                    "status": "calculated",
                },
            },
            "interaction_count": len(interactions),
            "task": "Menghitung pengaruh relatif agen sebelum DDR.",
            "result": "Dimensi RAR siap dinormalisasi menjadi bobot DAI.",
            "why": "Bobot memprioritaskan artefak lengkap, bersumber, relevan, dan rendah ketidakpastian.",
            "statutory_authority": False,
            "interpretation": "Influence prioritization only; CAR remains authoritative for hard constraints.",
        }
    session.flush()
    return observations


def _has_effective_deficit_violation(response: SRRResponse) -> bool:
    return any(
        alternative.deficit > STATUTORY_DEFICIT_CEILING_PERCENT_GDP
        for alternative in response.decision_alternatives()
    )


def _classify_conflict(
    scenario: Scenario,
    agent_i: Agent,
    response_i: SRRResponse,
    agent_j: Agent,
    response_j: SRRResponse,
    display_names: dict[int, str],
) -> tuple[dict[str, bool], dict[str, Any]]:
    vector = detect_divergence_vector(
        response_i.divergence_object(),
        response_j.divergence_object(),
    )
    violation_i = _has_effective_deficit_violation(response_i)
    violation_j = _has_effective_deficit_violation(response_j)
    vector["dC"] = bool(vector["dC"] or violation_i != violation_j)
    components = [key for key, value in vector.items() if value]
    route = resolve_disagreement_route(vector) if components else None
    hard_stop = bool(vector["dC"] and (violation_i or violation_j))
    return vector, {
        "agent_i": agent_i.name,
        "agent_i_display_name": display_names.get(
            agent_i.id, semantic_agent_name(agent_i)
        ),
        "agent_j": agent_j.name,
        "agent_j_display_name": display_names.get(
            agent_j.id, semantic_agent_name(agent_j)
        ),
        "components": components,
        "route": route,
        "narrative": [],
        "hard_stop": hard_stop,
        "simulation_allowed": bool(vector["dP"] and not hard_stop),
    }


def _collect_prediction_conflicts(
    scenario: Scenario,
    parsed_by_agent: list[tuple[Agent, SRRResponse]],
    display_names: dict[int, str],
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for index, (agent_i, response_i) in enumerate(parsed_by_agent):
        for agent_j, response_j in parsed_by_agent[index + 1 :]:
            _, conflict = _classify_conflict(
                scenario,
                agent_i,
                response_i,
                agent_j,
                response_j,
                display_names,
            )
            if conflict["simulation_allowed"] or conflict["hard_stop"]:
                conflicts.append(conflict)
    return conflicts


def _artifact_content(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "name", "decision", "recommendation"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate.strip()
    return str(value)


def _normalised_text_set(value: object) -> set[str]:
    values = value if isinstance(value, list) else [value] if value is not None else []
    return {
        " ".join(_artifact_content(item).casefold().split())
        for item in values
        if _artifact_content(item)
    }


def _jaccard_distance(left: object, right: object) -> float:
    left_set = _normalised_text_set(left)
    right_set = _normalised_text_set(right)
    union = left_set | right_set
    return round(1.0 - len(left_set & right_set) / len(union), 6) if union else 0.0


def _source_tags(value: object) -> list[str]:
    values = value if isinstance(value, list) else [value] if value is not None else []
    return sorted(
        {
            str(item.get("source_tag"))
            for item in values
            if isinstance(item, dict) and item.get("source_tag")
        }
    )


def _content_source_pairs(value: object) -> list[str]:
    values = value if isinstance(value, list) else [value] if value is not None else []
    return sorted(
        f"{_artifact_content(item)}::{item.get('source_tag') or ''}"
        for item in values
        if isinstance(item, dict) and _artifact_content(item)
    )


def _alternative_deficit_profiles(artifacts: dict[str, Any]) -> list[str]:
    alternatives = artifacts.get("alternatives", [])
    return sorted(
        f"{' '.join(str(item.get('name') or '').casefold().split())}::{float(item['deficit']):.8f}"
        for item in alternatives
        if isinstance(item, dict)
        and isinstance(item.get("deficit"), (int, float))
        and not isinstance(item.get("deficit"), bool)
    )


def _artifact_hash(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _deficit_values(artifacts: dict[str, Any]) -> list[float]:
    alternatives = artifacts.get("alternatives", [])
    return [
        float(item["deficit"])
        for item in alternatives
        if isinstance(item, dict)
        and isinstance(item.get("deficit"), (int, float))
        and not isinstance(item.get("deficit"), bool)
    ]


def _deficit_set_distance(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    left_to_right = max(min(abs(item - peer) for peer in right) for item in left)
    right_to_left = max(min(abs(item - peer) for peer in left) for item in right)
    return round(max(left_to_right, right_to_left), 6)


def _vector_metadata(
    scenario: Scenario,
    component: str,
    active: bool,
    field: str,
    artifacts_i: dict[str, Any],
    artifacts_j: dict[str, Any],
    response_i: SRRResponse,
    response_j: SRRResponse,
) -> dict[str, Any]:
    detail = DDR_COMPONENT_DETAILS[component]
    left = artifacts_i.get(field)
    right = artifacts_j.get(field)
    jaccard = _jaccard_distance(left, right)
    calculation: dict[str, Any] = {
        "status": "calculated",
        "method": "normalized-jaccard-distance",
        "formula": detail["formula"],
        "value": jaccard,
        "unit": "ratio",
        "inputs": {"agent_i_count": len(left) if isinstance(left, list) else int(left is not None), "agent_j_count": len(right) if isinstance(right, list) else int(right is not None)},
    }
    if component == "dE":
        left_tags = _source_tags(left)
        right_tags = _source_tags(right)
        source_variance = _jaccard_distance(left_tags, right_tags)
        pair_distance = _jaccard_distance(
            _content_source_pairs(left),
            _content_source_pairs(right),
        )
        calculation.update(
            {
                "method": "content-source-pair-jaccard-distance",
                "value": pair_distance,
                "content_distance": jaccard,
                "source_variance": source_variance,
                "agent_i_provenance_hash": _artifact_hash(left),
                "agent_j_provenance_hash": _artifact_hash(right),
                "official_baseline_verification": "not-calculated",
                "not_calculated_reason": "No authoritative MoF/DJA retrieval ledger was supplied for hash verification.",
            }
        )
    elif component == "dP":
        deficits_i = _deficit_values(artifacts_i)
        deficits_j = _deficit_values(artifacts_j)
        prediction_text_distance = _jaccard_distance(
            [item.content for item in response_i.predictions],
            [item.content for item in response_j.predictions],
        )
        deficit_profile_distance = _jaccard_distance(
            _alternative_deficit_profiles(artifacts_i),
            _alternative_deficit_profiles(artifacts_j),
        )
        if deficits_i and deficits_j:
            deficit_distance = _deficit_set_distance(deficits_i, deficits_j)
            calculation.update(
                {
                    "method": "maximum-prediction-or-deficit-profile-distance",
                    "value": max(prediction_text_distance, deficit_profile_distance),
                    "prediction_text_distance": prediction_text_distance,
                    "deficit_profile_distance": deficit_profile_distance,
                    "deficit_range_gap_percent_gdp": deficit_distance,
                    "agent_i_projected_deficits": deficits_i,
                    "agent_j_projected_deficits": deficits_j,
                    "statutory_ceiling_percent_gdp": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
                }
            )
        else:
            calculation.update(
                {
                    "deficit_range_gap_percent_gdp": None,
                    "deficit_calculation_status": "not-calculated",
                    "not_calculated_reason": "Both agents did not provide numeric deficit projections.",
                }
            )
        calculation.update(
            {
                "inflation_coefficient": None,
                "fiscal_multiplier": None,
                "causal_model_status": "not-calculated",
                "causal_model_reason": "No verified RL-FRB/US, ABM baseline, multiplier, or elasticity input was supplied.",
            }
        )
    elif component == "dR":
        calculation.update(
            {
                "impact_probability_score": None,
                "threat_matrix_status": "not-calculated",
                "not_calculated_reason": "Risk items do not include verified Impact and Probability scores.",
            }
        )
    elif component == "dU":
        confidence_i = response_i.confidence
        confidence_j = response_j.confidence
        confidence_gap = (
            round(abs(confidence_i - confidence_j), 6)
            if confidence_i is not None and confidence_j is not None
            else None
        )
        calculation.update(
            {
                "method": "maximum-uncertainty-jaccard-or-confidence-gap",
                "value": max(jaccard, confidence_gap or 0.0),
                "uncertainty_text_distance": jaccard,
                "confidence_i": confidence_i,
                "confidence_j": confidence_j,
                "confidence_gap": confidence_gap,
                "confidence_gap_status": "calculated"
                if confidence_i is not None and confidence_j is not None
                else "not-calculated",
            }
        )
    elif component == "dC":
        deficits_i = _deficit_values(artifacts_i)
        deficits_j = _deficit_values(artifacts_j)
        all_deficits = [*deficits_i, *deficits_j]
        effective_ceiling = STATUTORY_DEFICIT_CEILING_PERCENT_GDP
        statutory_violations_i = [
            item > STATUTORY_DEFICIT_CEILING_PERCENT_GDP for item in deficits_i
        ]
        statutory_violations_j = [
            item > STATUTORY_DEFICIT_CEILING_PERCENT_GDP for item in deficits_j
        ]
        effective_violations_i = [item > effective_ceiling for item in deficits_i]
        effective_violations_j = [item > effective_ceiling for item in deficits_j]
        statutory_violation = any(
            [*statutory_violations_i, *statutory_violations_j]
        )
        effective_violation = statutory_violation
        gate_difference = any(effective_violations_i) != any(effective_violations_j)
        calculation.update(
            {
                "method": "maximum-constraint-jaccard-or-hard-gate-difference",
                "value": max(jaccard, 1.0 if gate_difference else 0.0),
                "constraint_text_distance": jaccard,
                "hard_constraint": "EFFECTIVE_DEFICIT_CEILING",
                "statutory_ceiling_percent_gdp": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
                "effective_ceiling_percent_gdp": effective_ceiling,
                "maximum_projected_deficit_percent_gdp": max(all_deficits) if all_deficits else None,
                "violation": effective_violation,
                "statutory_violation": statutory_violation,
                "agent_i_effective_violations": effective_violations_i,
                "agent_j_effective_violations": effective_violations_j,
                "agent_i_statutory_violations": statutory_violations_i,
                "agent_j_statutory_violations": statutory_violations_j,
                "evaluation_method": "pre-CAR numeric screening",
                "solver": None,
                "education_floor_status": "not-calculated",
                "education_floor_reason": "Verified post-policy education share was not supplied.",
            }
        )
    elif component == "dREC":
        calculation.update(
            {
                "alternative_rank_distance": None,
                "rank_status": "not-calculated",
                "not_calculated_reason": "Standardized ranked alternative IDs were not supplied by both agents.",
            }
        )
    return {
        "component": component,
        "active": active,
        "category_i18n": detail["category"],
        "meaning_i18n": detail["meaning"],
        "economic_impact_i18n": detail["impact"],
        "formula": detail["formula"],
        "agent_i": {
            "value": left,
            "normalized": sorted(_normalised_text_set(left)),
            "source_tags": _source_tags(left),
            "artifact_hash": _artifact_hash(left),
        },
        "agent_j": {
            "value": right,
            "normalized": sorted(_normalised_text_set(right)),
            "source_tags": _source_tags(right),
            "artifact_hash": _artifact_hash(right),
        },
        "calculation": calculation,
        "resolution_path": detail["route"] if active else "No Resolution Required",
        "status": "detected" if active else "no-divergence",
    }


def _fiscal_alternative_payload(
    response: SRRResponse,
) -> list[dict[str, Any]]:
    effective_ceiling = STATUTORY_DEFICIT_CEILING_PERCENT_GDP
    return [
        {
            "name": alternative.name,
            "projected_deficit_percent": alternative.deficit,
            "utility": alternative.utility,
            "headroom_percent": round(effective_ceiling - alternative.deficit, 4),
            "within_statutory_ceiling": (
                alternative.deficit <= STATUTORY_DEFICIT_CEILING_PERCENT_GDP
            ),
            "within_effective_ceiling": alternative.deficit <= effective_ceiling,
            "source_tag": alternative.source_tag,
        }
        for alternative in response.decision_alternatives()
    ]


def _legal_basis_payload(
    response_i: SRRResponse,
    response_j: SRRResponse,
) -> list[dict[str, Any]]:
    references = {
        "UU17_2003_P12": "UU Keuangan Negara (UU 17/2003) Pasal 12 — batas defisit terhadap PDB.",
        "UU17_2025_POSTURE": "UU APBN 2026 — postur dan baseline fiskal tahun anggaran 2026.",
        "UU17_2025_P28": "UU APBN 2026 Pasal 28 — otoritas pembiayaan dan pengelolaan fiskal.",
    }
    source_tags = {
        item.source_tag
        for response in (response_i, response_j)
        for item in response.provenance_items()
        if item.source_tag
    }
    source_tags.add("UU17_2003_P12")
    return [
        {"source_tag": source_tag, "basis": references.get(source_tag, source_tag)}
        for source_tag in sorted(source_tags)
        if source_tag.startswith(("UU", "UUD", "PP", "PMK"))
    ]


def _conflict_detail_payload(
    scenario: Scenario,
    agent_i: Agent,
    response_i: SRRResponse,
    agent_j: Agent,
    response_j: SRRResponse,
    vector: dict[str, bool],
    route: str | None,
    influence_by_agent: dict[int, AgentInfluenceObservation],
    display_names: dict[int, str],
) -> dict[str, Any]:
    artifacts_i = response_i.model_dump(mode="json")
    artifacts_j = response_j.model_dump(mode="json")
    component_fields = {
        "dE": "evidence",
        "dA": "assumptions",
        "dP": "predictions",
        "dR": "risks",
        "dU": "uncertainties",
        "dO": "objectives",
        "dC": "constraints",
        "dREC": "recommendation",
    }
    categories = []
    vector_metadata: dict[str, Any] = {}
    for component, field in component_fields.items():
        metadata = _vector_metadata(
            scenario,
            component,
            bool(vector.get(component)),
            field,
            artifacts_i,
            artifacts_j,
            response_i,
            response_j,
        )
        metadata["narrative_i18n"] = _conflict_narrative(
            component,
            display_names.get(agent_i.id, semantic_agent_name(agent_i)),
            display_names.get(agent_j.id, semantic_agent_name(agent_j)),
            artifacts_i,
            artifacts_j,
            metadata["calculation"],
            route,
        )
        metadata["utility_metadata"] = {
            "task": f"Detect {component} between {display_names.get(agent_i.id, semantic_agent_name(agent_i))} and {display_names.get(agent_j.id, semantic_agent_name(agent_j))}.",
            "result": "Conflict detected." if metadata["active"] else "No material conflict detected.",
            "why": metadata["economic_impact_i18n"]["id"],
        }
        vector_metadata[component] = metadata
        if metadata["active"]:
            categories.append(
                {
                    "component": component,
                    "category": metadata["category_i18n"]["en"],
                    "category_i18n": metadata["category_i18n"],
                    "narrative": metadata["narrative_i18n"]["en"],
                    "narrative_i18n": metadata["narrative_i18n"],
                    "impact": metadata["economic_impact_i18n"]["en"],
                    "impact_i18n": metadata["economic_impact_i18n"],
                    "formula": metadata["formula"],
                    "agent_i_artifacts": artifacts_i.get(field),
                    "agent_j_artifacts": artifacts_j.get(field),
                }
            )
    observation_i = influence_by_agent.get(agent_i.id)
    observation_j = influence_by_agent.get(agent_j.id)
    alternatives_i = _fiscal_alternative_payload(response_i)
    alternatives_j = _fiscal_alternative_payload(response_j)
    return sanitize_simulation_payload(
        {
            "active_components": [key for key, value in vector.items() if value],
            "categories": categories,
            "vector_metadata": vector_metadata,
            "fiscal_calculation": {
                "formula": "headroom_percent = effective_deficit_ceiling_percent - agent_reported_projected_deficit_percent",
                "statutory_deficit_ceiling_percent": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
                "effective_deficit_ceiling_percent": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
                **scenario.simulation_payload(),
                "program_cost_to_gdp_ratio": None,
                "calculation_note": "Rasio biaya program terhadap PDB tidak dihitung tanpa denominator PDB terverifikasi; angka defisit berasal dari alternatif terstruktur agen.",
                "agent_i_alternatives": alternatives_i,
                "agent_j_alternatives": alternatives_j,
            },
            "influence_context": {
                "agent_i_weight": observation_i.normalized_weight if observation_i else None,
                "agent_j_weight": observation_j.normalized_weight if observation_j else None,
                "combined_weight": round(
                    sum(
                        value
                        for value in (
                            observation_i.normalized_weight if observation_i else None,
                            observation_j.normalized_weight if observation_j else None,
                        )
                        if value is not None
                    ),
                    6,
                ),
                "formula": "gated softmax over evidence quality, completeness, recency, peer alignment, and uncertainty burden",
            },
            "legal_basis": _legal_basis_payload(response_i, response_j),
            "resolution": {
                "route": route or "No Resolution Required",
                "status": "ESCALATED" if route else "CLEAR",
                "conclusion": (
                    "Pelanggaran constraint terverifikasi diteruskan langsung ke CAR tanpa kompromi LLM."
                    if vector.get("dC")
                    and vector_metadata["dC"]["calculation"].get("violation")
                    else "Perbedaan proyeksi diteruskan ke Simulation Agent untuk kompromi fiskal berbatas."
                    if vector.get("dP")
                    else "Konflik dipertahankan sebagai dissent terstruktur untuk mekanisme resolusi terkait."
                    if route
                    else "Tidak ada perbedaan material pada pasangan agen ini."
                ),
            },
        }
    )


def _conflict_narrative(
    component: str,
    agent_i_name: str,
    agent_j_name: str,
    artifacts_i: dict[str, Any],
    artifacts_j: dict[str, Any],
    calculation: dict[str, Any],
    route: str | None,
) -> dict[str, str]:
    field = {
        "dE": "evidence",
        "dA": "assumptions",
        "dP": "predictions",
        "dR": "risks",
        "dU": "uncertainties",
        "dO": "objectives",
        "dC": "constraints",
        "dREC": "recommendation",
    }.get(component, component)
    left = artifacts_i.get(field)
    right = artifacts_j.get(field)
    left_text = "; ".join(_artifact_content(item) for item in (left if isinstance(left, list) else [left]) if _artifact_content(item))
    right_text = "; ".join(_artifact_content(item) for item in (right if isinstance(right, list) else [right]) if _artifact_content(item))
    if component == "dP":
        gap = calculation.get("deficit_range_gap_percent_gdp")
        suffix_en = f" Deficit-range gap: {gap} percentage points of GDP." if gap is not None else ""
        suffix_id = f" Selisih rentang defisit: {gap} poin persentase PDB." if gap is not None else ""
        en = f"{agent_i_name} projects {left_text or 'an unreported outcome'}, while {agent_j_name} projects {right_text or 'a different outcome'}.{suffix_en}"
        id_text = f"{agent_i_name} memproyeksikan {left_text or 'hasil yang tidak dilaporkan'}, sedangkan {agent_j_name} memproyeksikan {right_text or 'hasil yang berbeda'}.{suffix_id}"
    elif component == "dU":
        gap = calculation.get("confidence_gap")
        en = f"{agent_i_name} reports {left_text or 'different uncertainty bounds'}, while {agent_j_name} reports {right_text or 'different uncertainty bounds'}. Confidence gap: {gap if gap is not None else 'not calculated'}."
        id_text = f"{agent_i_name} melaporkan {left_text or 'batas ketidakpastian berbeda'}, sedangkan {agent_j_name} melaporkan {right_text or 'batas ketidakpastian berbeda'}. Selisih confidence: {gap if gap is not None else 'tidak dihitung'}."
    elif component == "dC":
        en = f"Constraint positions differ: {agent_i_name} reports {left_text or 'no explicit constraint artifact'}, while {agent_j_name} reports {right_text or 'no explicit constraint artifact'}. This is pre-CAR screening; verified hard constraints bypass LLM compromise."
        id_text = f"Posisi constraint berbeda: {agent_i_name} melaporkan {left_text or 'tidak ada artefak constraint eksplisit'}, sedangkan {agent_j_name} melaporkan {right_text or 'tidak ada artefak constraint eksplisit'}. Ini adalah screening pra-CAR; hard constraint terverifikasi melewati kompromi LLM."
    else:
        en = f"{agent_i_name} states {left_text or 'a different position'}, while {agent_j_name} states {right_text or 'a different position'}."
        id_text = f"{agent_i_name} menyatakan {left_text or 'posisi berbeda'}, sedangkan {agent_j_name} menyatakan {right_text or 'posisi berbeda'}."
    resolution_route = route or "No Resolution Required"
    return {
        "id": f"{id_text} Jalur: {resolution_route}.",
        "en": f"{en} Route: {resolution_route}.",
    }


def _detect_ddr_conflicts(
    session: Any,
    scenario: Scenario,
    session_id: str,
    parsed_by_agent: list[tuple[Agent, SRRResponse]],
    influence_observations: list[AgentInfluenceObservation],
    display_names: dict[int, str],
    logs: list[dict[str, Any]],
    emit: ProgressReporter,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    pair_count = 0
    active_pair_count = 0
    component_counts = {
        component: 0
        for component in ("dE", "dA", "dP", "dR", "dU", "dO", "dC", "dREC")
    }
    route_counts: dict[str, int] = {}
    pair_summaries: list[dict[str, Any]] = []
    influence_by_agent = {
        observation.agent_id: observation for observation in influence_observations
    }
    for index, (agent_i, response_i) in enumerate(parsed_by_agent):
        for agent_j, response_j in parsed_by_agent[index + 1 :]:
            vector, conflict = _classify_conflict(
                scenario,
                agent_i,
                response_i,
                agent_j,
                response_j,
                display_names,
            )
            components = conflict["components"]
            route = conflict["route"]
            pair_count += 1
            if components:
                active_pair_count += 1
            for component in components:
                component_counts[component] += 1
            route_name = str(route or "No Resolution Required")
            route_counts[route_name] = route_counts.get(route_name, 0) + 1
            pair_summaries.append(
                {
                    "agent_i_id": agent_i.id,
                    "agent_i_name": agent_i.name,
                    "agent_i_display_name": conflict["agent_i_display_name"],
                    "agent_j_id": agent_j.id,
                    "agent_j_name": agent_j.name,
                    "agent_j_display_name": conflict["agent_j_display_name"],
                    "vector": vector,
                    "active_components": components,
                    "resolution_path": route,
                    "hard_stop": conflict["hard_stop"],
                    "simulation_allowed": conflict["simulation_allowed"],
                }
            )
            detail_payload = _conflict_detail_payload(
                scenario,
                agent_i,
                response_i,
                agent_j,
                response_j,
                vector,
                route,
                influence_by_agent,
                display_names,
            )
            session.add(
                DisagreementLog(
                    run_id=session_id,
                    scenario_id=scenario.id,
                    agent_i=agent_i.id,
                    agent_j=agent_j.id,
                    resolution_route=route,
                    detail_payload=detail_payload,
                    **vector,
                )
            )
            if components:
                conflict["narrative"] = detail_payload["categories"]
                conflicts.append(conflict)
    hard_stop_count = sum(bool(item.get("hard_stop")) for item in conflicts)
    simulation_count = sum(bool(item.get("simulation_allowed")) for item in conflicts)
    logs.append(
        _log(
            "DDR",
            "WARNING" if active_pair_count else "SUCCESS",
            (
                f"DDR evaluated {pair_count} agent pair(s): {active_pair_count} "
                f"with divergence, {hard_stop_count} hard stop(s), and "
                f"{simulation_count} simulation route(s)."
            ),
            code="DDR_BATCH_EVALUATED",
            id_message=(
                f"DDR mengevaluasi {pair_count} pasangan agen: {active_pair_count} "
                f"dengan divergensi, {hard_stop_count} hard stop, dan "
                f"{simulation_count} jalur simulasi."
            ),
            metric={
                "name": "active_pair_count",
                "value": active_pair_count,
                "unit": "pairs",
                "status": "calculated",
            },
            task={"type": "ddr-batch", "pair_count": pair_count},
            result={
                "active_pair_count": active_pair_count,
                "hard_stop_count": hard_stop_count,
                "simulation_route_count": simulation_count,
            },
            why={
                "reason": "Pairwise audit records are persisted while progress output is batched to avoid repetitive warnings."
            },
            metadata={
                "component_counts": component_counts,
                "route_counts": route_counts,
                "pairs": pair_summaries,
            },
        )
    )
    emit(logs)
    return conflicts


def _run_consensus_round(
    agents: list[Agent],
    scenario: Scenario,
    run: ConsensusSession,
    parsed_by_agent: list[tuple[Agent, SRRResponse]],
    consensus_call: Callable[..., tuple[str, int]] | None,
    peer_outputs: list[dict[str, Any]],
    simulation_output: dict[str, Any] | None,
    round_number: int | None,
    logs: list[dict[str, Any]],
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
        logs.append(
            _log(
                "CONSENSUS",
                "INFO",
                "Sending peer outputs to each agent for structured consensus review.",
                code="CONSENSUS_REVIEW_STARTED",
                round_number=round_number,
                metric={
                    "name": "agent_count",
                    "value": len(parsed_by_agent),
                    "unit": "agents",
                    "status": "calculated",
                },
            )
        )
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
            _, reviewed, _ = _validated_response(raw_content)
            if reviewed is None:
                consensus_results.append((agent, initial_response))
                logs.append(
                    _log(
                        log_stage,
                        "WARNING",
                        f"{agent.name}: review response did not pass schema validation; retained validated SRR artifacts.",
                        code="CONSENSUS_SCHEMA_FALLBACK",
                        id_message=f"{agent.name}: respons review tidak lolos validasi schema; artefak SRR valid sebelumnya dipertahankan.",
                        agent_id=agent.id,
                        agent_name=agent.name,
                        round_number=round_number,
                        fallback={
                            "used": True,
                            "kind": "retain-last-valid-srr",
                            "reason": "Consensus response failed decision-complete schema validation.",
                            "retained_artifact": "last validated SRR response",
                            "consensus_impact": "The agent remains in the loop using its last auditable position.",
                        },
                        economic={"status": "not-calculated", "inputs": {}, "outputs": {}, "reason": "No new valid numeric artifacts were admitted."},
                        metadata={"resolution_path": "Retain Validated SRR"},
                    )
                )
                emit(logs)
                continue
            consensus_results.append((agent, reviewed))
            logs.append(
                _log(
                    log_stage,
                    "SUCCESS",
                    f"{agent.name}: peer review completed with {token_usage} tokens.",
                    code="CONSENSUS_REVIEW_COMPLETED",
                    agent_id=agent.id,
                    agent_name=agent.name,
                    round_number=round_number,
                    metric={
                        "name": "token_usage",
                        "value": token_usage,
                        "unit": "tokens",
                        "status": "calculated",
                    },
                )
            )
            emit(logs)
        except Exception as error:
            if isinstance(
                error,
                (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError, OSError, socket.error),
            ):
                logger.exception("Local LLM connection failed during consensus review for agent %s", agent.id)
            else:
                logger.exception("Consensus review failed for agent %s", agent.id)
            error_type = type(error).__name__
            consensus_results.append((agent, initial_response))
            logs.append(
                _log(
                    log_stage,
                    "WARNING",
                    f"{agent.name}: {error_type}; retained validated SRR artifacts.",
                    code="CONSENSUS_PROVIDER_FALLBACK",
                    agent_id=agent.id,
                    agent_name=agent.name,
                    round_number=round_number,
                    fallback={
                        "used": True,
                        "kind": "retain-last-valid-srr",
                        "reason": error_type,
                        "retained_artifact": "last validated SRR response",
                        "consensus_impact": "The agent remains in the loop using its last auditable position.",
                    },
                    metadata={"error_type": error_type},
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

    def emit(current_logs: list[dict[str, Any]]) -> None:
        persist_run_progress(scenario_id, session_id, current_logs)
        external_emit(current_logs)

    logs = [
        _log(
            "INITIALIZE",
            "INFO",
            f"Starting SHCR cycle for scenario {scenario_id}, session {session_id}.",
            code="SHCR_CYCLE_STARTED",
            id_message=f"Memulai siklus SHCR untuk skenario {scenario_id}, sesi {session_id}.",
            metadata={"framework": "SRR + (RAR → DAI) + DDR + CAR"},
        )
    ]
    emit(logs)

    with SessionLocal() as session:
        scenario, agents, mandate_snapshot, run = _load_session_context(
            session,
            scenario_id,
            session_id,
        )
        display_names = {
            agent.id: _display_name_for_agent(run.mandate_payload, agent)
            for agent in agents
        }
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
                error_type = type(error).__name__
                logs.append(
                    _log(
                        "SRR",
                        "ERROR",
                        f"{agent.name}: provider call failed ({error_type}).",
                        code="SRR_PROVIDER_FAILED",
                        id_message=f"{agent.name}: panggilan provider gagal ({error_type}).",
                        agent_id=agent.id,
                        agent_name=agent.name,
                        fallback={"used": False, "kind": None, "reason": error_type},
                        metadata={"error_type": error_type, "model": agent.llm_model},
                    )
                )
                emit(logs)
                session.add(
                    ReasoningLog(
                        agent_id=agent.id,
                        scenario_id=scenario.id,
                        run_id=session_id,
                        raw_json={"error_code": "SRR_PROVIDER_FAILED", "error_type": error_type},
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
                        raw_json=sanitize_simulation_payload(raw_json),
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
                raw_json=sanitize_simulation_payload(raw_json),
                parsed_srr_objects=parsed.model_dump(mode="json"),
                deliberation_history=[
                    {
                        "stage": "INITIAL",
                        "round_number": 0,
                        "artifacts": sanitize_simulation_payload(
                            parsed.model_dump(mode="json")
                        ),
                    }
                ],
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

        session.commit()
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
                consensus_call,
                peer_outputs,
                None,
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

        _record_deliberation_stage(
            reasoning_by_agent,
            parsed_by_agent,
            "PRE_ARBITRATION",
            0,
        )
        observations = _ensure_influence_observations(
            session,
            scenario,
            session_id,
            parsed_by_agent,
            observations,
        )
        session.commit()

        final_provenance = [_provenance_counts(response) for _, response in parsed_by_agent]
        tagged_items = sum(tagged for tagged, _ in final_provenance)
        total_items = sum(total for _, total in final_provenance)
        logs.append(
            _log(
                "RAR-DAI",
                "INFO",
                "Calculating gated dynamic influence weights before DDR.",
                code="RAR_DAI_CALCULATION_STARTED",
                id_message="Menghitung bobot pengaruh dinamis bergate sebelum DDR.",
                task={"type": "rar-dai", "phase": "pre-ddr"},
                result={"status": "pending"},
                why={
                    "reason": "Influence must be normalized before pairwise conflict routing.",
                    "formula": "w_i = softmax(g_i(Theta_X X + Theta_Q Q + Theta_H H + Theta_S S - Theta_U U))",
                },
            )
        )
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
                weighted_agent = agents_by_id[observation.agent_id]
                observation.raw_score = float(result["raw_score"])
                observation.normalized_weight = float(result["normalized_weight"])
                observation.calculation_payload = {
                    **dict(observation.calculation_payload or {}),
                    "inputs": {
                        "theta_x": weighted_agent.theta_x,
                        "theta_q": weighted_agent.theta_q,
                        "theta_h": weighted_agent.theta_h,
                        "theta_s": weighted_agent.theta_s,
                        "theta_u": weighted_agent.theta_u,
                        "X": observation.X,
                        "Q": observation.Q,
                        "H": observation.H,
                        "S": observation.S,
                        "U": observation.U,
                        "gate": observation.gate,
                    },
                    "raw_score": observation.raw_score,
                    "normalized_weight": observation.normalized_weight,
                }

        session.commit()
        logs.append(
            _log(
                "RAR-DAI",
                "SUCCESS",
                f"RAR-DAI weights calculated for {len(influence_observations)} agents.",
                code="RAR_DAI_CALCULATION_COMPLETED",
                task={"type": "rar-dai", "phase": "pre-ddr", "agent_count": len(influence_observations)},
                result={
                    "status": "calculated",
                    "weighted_agent_count": sum(item.normalized_weight is not None for item in influence_observations),
                },
                why={
                    "reason": "Weights prioritize complete, source-tagged, peer-aligned responses and penalize uncertainty.",
                    "rule": "rar-dai-v2",
                    "authority": "Influence prioritization only; CAR remains authoritative.",
                },
            )
        )
        emit(logs)
        logs.append(_log("DDR", "INFO", "Calculating disagreement vectors."))
        emit(logs)
        conflicts = _detect_ddr_conflicts(
            session,
            scenario,
            session_id,
            parsed_by_agent,
            influence_observations,
            display_names,
            logs,
            emit,
        )
        simulation_rounds = 0
        simulation_artifact_id: int | None = None
        hard_constraint_conflicts = [
            item for item in conflicts if item.get("hard_stop")
        ]
        remaining_conflicts = [
            item for item in conflicts if item.get("simulation_allowed")
        ]
        if hard_constraint_conflicts:
            remaining_conflicts = []
            logs.append(
                _log(
                    "CAR",
                    "WARNING",
                    "Verified dC violation triggered deterministic CAR hard-stop; simulation and LLM compromise were bypassed.",
                    code="DDR_DC_HARD_STOP",
                    id_message="Pelanggaran dC terverifikasi memicu hard-stop CAR deterministik; simulasi dan kompromi LLM dilewati.",
                    task={"type": "hard-constraint-gate", "phase": "post-ddr"},
                    result={
                        "status": "INFEASIBLE",
                        "hard_constraint_conflict_count": len(hard_constraint_conflicts),
                        "simulation_bypassed": True,
                    },
                    why={
                        "reason": "A verified hard constraint is non-overridable.",
                        "rule": "dC -> CAR/Z3",
                        "authority": "UU17_2003_P12",
                    },
                    metadata={"conflicts": hard_constraint_conflicts},
                )
            )
            emit(logs)
        while (
            enable_simulation
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
            if reviewer is None:
                if artifact is not None:
                    artifact_payload = dict(artifact.output_payload)
                    artifact_payload["follow_up_consensus_status"] = "not-configured"
                    artifact.output_payload = artifact_payload
                    session.commit()
                break
            logs.append(
                _log(
                    "SIMULATION_CONSENSUS",
                    "INFO",
                    f"Feeding native simulation round {round_number} back to all sectoral agents.",
                    round_number=round_number,
                )
            )
            emit(logs)
            simulation_results, round_tokens = _run_consensus_round(
                agents,
                scenario,
                run,
                parsed_by_agent,
                consensus_call,
                peer_outputs,
                simulation_output,
                round_number,
                logs,
                emit,
            )
            total_tokens += round_tokens
            if len(simulation_results) < 2:
                break
            parsed_by_agent = simulation_results
            _record_deliberation_stage(
                reasoning_by_agent,
                parsed_by_agent,
                "POST_SIMULATION",
                round_number,
            )
            session.commit()
            follow_up_conflicts = _collect_prediction_conflicts(
                scenario,
                parsed_by_agent,
                display_names,
            )
            follow_up_hard_stops = [
                item for item in follow_up_conflicts if item.get("hard_stop")
            ]
            if follow_up_hard_stops:
                hard_constraint_conflicts.extend(follow_up_hard_stops)
                remaining_conflicts = []
                logs.append(
                    _log(
                        "CAR",
                        "WARNING",
                        "Follow-up consensus introduced a verified dC violation; additional simulation was bypassed.",
                        code="DDR_DC_HARD_STOP",
                        id_message="Konsensus lanjutan menghasilkan pelanggaran dC terverifikasi; simulasi tambahan dilewati.",
                        round_number=round_number,
                        task={"type": "hard-constraint-gate", "phase": "post-simulation"},
                        result={
                            "status": "INFEASIBLE",
                            "hard_constraint_conflict_count": len(
                                follow_up_hard_stops
                            ),
                            "simulation_bypassed": True,
                        },
                        why={
                            "reason": "A verified hard constraint is non-overridable.",
                            "rule": "dC -> CAR/Z3",
                            "authority": "UU17_2003_P12",
                        },
                        metadata={"conflicts": follow_up_hard_stops},
                    )
                )
                emit(logs)
            else:
                remaining_conflicts = [
                    item
                    for item in follow_up_conflicts
                    if item.get("simulation_allowed")
                ]
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
                    round_number=round_number,
                )
            )
            emit(logs)
        if remaining_conflicts and simulation_rounds == MAX_SIMULATION_ROUNDS:
            logs.append(
                _log(
                    "SIMULATION_CONSENSUS",
                    "WARNING",
                    f"Stopped after the bounded maximum of {MAX_SIMULATION_ROUNDS} simulation rounds; valid dissent remains explicit.",
                    round_number=simulation_rounds,
                )
            )
            emit(logs)

        _record_deliberation_stage(
            reasoning_by_agent,
            parsed_by_agent,
            "FINAL",
            simulation_rounds,
        )
        session.commit()

        final_provenance = [_provenance_counts(response) for _, response in parsed_by_agent]
        tagged_items = sum(tagged for tagged, _ in final_provenance)
        total_items = sum(total for _, total in final_provenance)
        effective_ceiling = STATUTORY_DEFICIT_CEILING_PERCENT_GDP
        logs.append(
            _log(
                "CAR",
                "INFO",
                f"Applying Z3 hard deficit constraint <= {effective_ceiling}% GDP.",
                code="CAR_CONSTRAINT_EVALUATION_STARTED",
                id_message=f"Menerapkan hard constraint Z3: defisit <= {effective_ceiling}% PDB.",
                statutory={
                    "status": "pending",
                    "constraint": "DEFICIT_3PCT",
                    "ceiling_percent_gdp": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
                    "source_tags": ["UU17_2003_P12"],
                    "solver": "z3",
                },
                economic={
                    "status": "not-calculated",
                    "inputs": {"program_cost": scenario.program_cost, "gdp_denominator": None},
                    "outputs": {"program_cost_to_gdp_ratio": None},
                    "reason": "Verified GDP denominator was not supplied.",
                },
            )
        )
        emit(logs)
        parsed_responses = [response for _, response in parsed_by_agent]
        alternatives = [
            alternative
            for response in parsed_responses
        for alternative in response.decision_alternatives()

        ]
        car_evaluation = evaluate_car_constraints(
            alternatives,
            STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
            hard_stop_reason=(
                "Verified dC conflict violates a non-overridable hard constraint."
                if hard_constraint_conflicts
                else None
            ),
        )
        feasible = car_evaluation.feasible
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

        car_level = "WARNING" if convergence_status == ConvergenceStatus.INFEASIBLE else "SUCCESS"
        deficit_constraint = next(
            (
                item
                for item in car_evaluation.hard_constraints
                if item.get("code") == "DEFICIT_3PCT"
            ),
            {},
        )
        logs.append(
            _log(
                "CAR",
                car_level,
                f"Z3 CAR result: {len(feasible)}/{len(alternatives)} alternatives feasible; violation rate={violation_rate}%; state={convergence_status.value}.",
                code="CAR_CONSTRAINT_EVALUATED",
                id_message=f"Hasil CAR Z3: {len(feasible)}/{len(alternatives)} alternatif feasible; tingkat pelanggaran={violation_rate}%; status={convergence_status.value}.",
                metric={
                    "name": "hard_constraint_violation_rate",
                    "value": violation_rate,
                    "unit": "percent",
                    "status": "calculated",
                },
                statutory={
                    "status": deficit_constraint.get("status", "not-calculated"),
                    "calculation_status": deficit_constraint.get(
                        "calculation_status", "not-calculated"
                    ),
                    "constraint": "DEFICIT_3PCT",
                    "ceiling_percent_gdp": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
                    "scenario_policy_constraint": next(
                        (
                            item
                            for item in car_evaluation.hard_constraints
                            if item.get("code") == "SCENARIO_DEFICIT_CEILING"
                        ),
                        None,
                    ),
                    "source_tags": ["UU17_2003_P12"],
                    "solver": "z3",
                    "solver_status": car_evaluation.solver_status,
                },
                economic={
                    "status": "calculated",
                    "inputs": {"alternative_count": len(alternatives)},
                    "outputs": {
                        "feasible_count": len(feasible),
                        "rejected_count": len(car_evaluation.rejected),
                        "selected_alternative": getattr(car_evaluation.selected, "name", None),
                    },
                    "impact": "No alternative can enter final synthesis when CAR returns INFEASIBLE."
                    if convergence_status == ConvergenceStatus.INFEASIBLE
                    else "Only CAR-feasible alternatives remain eligible for final synthesis.",
                },
                metadata={"rejected_alternatives": car_evaluation.rejected},
            )
        )
        logs.append(
            _log(
                "METRICS",
                "INFO",
                "Persisting convergence, provenance, latency, and token metrics.",
                code="METRICS_PERSISTING",
                id_message="Menyimpan metrik konvergensi, provenance, latency, dan token.",
                metric={
                    "name": "convergence_status",
                    "value": convergence_status.value,
                    "status": "calculated",
                },
            )
        )
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
        final_logs = [normalize_event(log) for log in logs]
        for event in final_logs:
            event["run_id"] = session_id
            event["task_id"] = run.celery_task_id
            event["scenario_id"] = scenario.id
        logs[:] = final_logs
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
            "simulation_triggered": simulation_rounds > 0,
            "task": {
                "type": "shcr-cycle",
                "scenario_id": scenario.id,
                "session_id": session_id,
            },
            "result": {
                "status": car_evaluation.status
                if hard_constraint_conflicts
                else convergence_status.value,
                "feasible_alternatives_count": len(feasible),
                "selected_alternative": getattr(car_evaluation.selected, "name", None),
                **(
                    {"messages": car_evaluation.messages}
                    if car_evaluation.messages
                    else {}
                ),
            },
            "why": {
                "reason": (
                    "Verified dC violation forced deterministic constraint arbitration."
                    if hard_constraint_conflicts
                    else "Final state follows RAR-DAI weighted DDR routing and CAR feasibility."
                ),
                "framework": "SRR + RAR-DAI + DDR + CAR/Z3",
            },
            "car": {
                "solver": "z3",
                "solver_status": car_evaluation.solver_status,
                "hard_stop": {
                    "triggered": bool(hard_constraint_conflicts),
                    "reason": (
                        "Verified dC conflict violates a non-overridable hard constraint."
                        if hard_constraint_conflicts
                        else None
                    ),
                    "simulation_bypassed": bool(hard_constraint_conflicts),
                    "llm_compromise_bypassed": bool(hard_constraint_conflicts),
                    "conflict_count": len(hard_constraint_conflicts),
                },
                "hard_constraints": car_evaluation.hard_constraints,
                "rejected_alternatives": car_evaluation.rejected,
                "selected_alternative": (
                    car_evaluation.selected.model_dump(mode="json")
                    if car_evaluation.selected is not None
                    and hasattr(car_evaluation.selected, "model_dump")
                    else None
                ),
                "feasible_alternatives_count": len(feasible),
                "rejected_alternatives_count": len(car_evaluation.rejected),
                "status": (
                    "FEASIBLE"
                    if feasible
                    else "INFEASIBLE"
                    if alternatives
                    else "NOT_EVALUATED"
                ),
            },
        }
        run.result_payload = result_payload
        run.logs = _merge_run_logs(run.logs, logs)
        run.progress_stage = "COMPLETE"
        run.status = "SUCCEEDED"
        run.completed_at = datetime.now(timezone.utc)
        session.commit()
        external_emit(logs)

        return {**result_payload, "logs": logs}


def run_full_shcr_cycle(scenario_id: int, session_id: str) -> dict[str, Any]:
    return execute_full_shcr_cycle(scenario_id, session_id)
