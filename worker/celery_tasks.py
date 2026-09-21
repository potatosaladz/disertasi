import json
import logging
import os
from collections.abc import Callable
from time import perf_counter
from typing import Any

from openai import OpenAI
from pydantic import ValidationError
from sqlalchemy import select

from backend.core_algorithms import (
    HardConstraints,
    calculate_dynamic_influence,
    calculate_violation_rate,
    detect_divergence_vector,
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
ProgressReporter = Callable[[list[str]], None]


def _default_llm_call(agent: Agent, scenario: Scenario) -> tuple[str, int]:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "local-llm"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://host.docker.internal:11434/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "local-model"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only an SRR JSON object containing evidence, assumptions, "
                    "predictions, risks, uncertainties, objectives, constraints, alternatives, "
                    "recommendation, confidence, and material_information_retention_macro_f1. "
                    "Every typed item must contain content and an optional source_tag. Every "
                    "alternative must contain name, deficit, utility, and optional source_tag."
                ),
            },
            {
                "role": "user",
                "content": f"Agent role: {agent.role}\nScenario: {scenario.description}",
            },
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned no content")
    return content, response.usage.total_tokens if response.usage else 0


def _parse_response(raw_content: str) -> tuple[dict[str, Any], SRRResponse | None]:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return {"unparsed_content": raw_content}, None
    if not isinstance(payload, dict):
        return {"parsed_content": payload}, None
    try:
        return payload, SRRResponse.model_validate(payload)
    except ValidationError:
        return payload, None


def _provenance_counts(response: SRRResponse) -> tuple[int, int]:
    items = response.provenance_items()
    tagged = sum(bool(item.source_tag and item.source_tag.strip()) for item in items)
    return tagged, len(items)


def _determine_convergence(
    parsed_responses: list[SRRResponse],
    feasible_alternatives: list[Any],
) -> ConvergenceStatus:
    if not parsed_responses:
        return ConvergenceStatus.INSUFFICIENT_EVIDENCE
    if not feasible_alternatives:
        return ConvergenceStatus.INFEASIBLE

    recommendations = {
        response.recommendation.content for response in parsed_responses
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
    emit(["Executing SRR..."])

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
            raw_content, token_usage = caller(agent, scenario)
            total_tokens += token_usage
            raw_json, parsed = _parse_response(raw_content)
            if parsed is None:
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

        emit(["Executing SRR...", "Calculating DDR vector..."])
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
                session.add(
                    DisagreementLog(
                        scenario_id=scenario.id,
                        agent_i=agent_i.id,
                        agent_j=agent_j.id,
                        resolution_route=route,
                        **vector,
                    )
                )

        emit([
            "Executing SRR...",
            "Calculating DDR vector...",
            "Applying CAR filter...",
        ])
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
        retention_macro_f1 = (
            sum(
                response.material_information_retention_macro_f1
                for response in parsed_responses
            )
            / len(parsed_responses)
            if parsed_responses
            else 0.0
        )
        convergence_status = _determine_convergence(parsed_responses, feasible)
        latency_ms = (perf_counter() - started_at) * 1000.0

        emit([
            "Executing SRR...",
            "Calculating DDR vector...",
            "Applying CAR filter...",
            "Persisting dissertation metrics...",
        ])
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
        emit([
            "Executing SRR...",
            "Calculating DDR vector...",
            "Applying CAR filter...",
            "Persisting dissertation metrics...",
            "SHCR cycle completed.",
        ])

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
