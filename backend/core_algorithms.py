import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

DDR_COMPONENT_FIELDS = {
    "dE": "E",
    "dA": "A",
    "dP": "P",
    "dR": "R",
    "dU": "U",
    "dO": "O",
    "dC": "C",
    "dREC": "REC",
}
T = TypeVar("T")

SRR_OUTPUT_INSTRUCTIONS = (
    "Return only one valid SRR JSON object. A decision-complete response must include non-empty "
    "evidence, predictions, risks, uncertainties, and alternatives arrays, plus recommendation and "
    "confidence. Include assumptions, objectives, constraints, and "
    "material_information_retention_macro_f1 when supported. Every typed item must contain content "
    "and an optional source_tag. Every alternative must contain name, deficit, utility, and optional "
    "source_tag. Additional structured fields are allowed. Analyze impacts, risks, uncertainties, "
    "objections, conditions, and adjustments within the agent's fiscal mandate. Do not reveal hidden "
    "chain-of-thought; provide only concise, evidence-linked, inspectable artifacts."
)


def build_agent_system_prompt(role: str, mandate: str | None) -> str:
    sections = [f"Functional fiscal role: {role}."]
    if mandate and mandate.strip():
        sections.append(f"Agent-specific mandate and decision principles:\n{mandate.strip()}")
    sections.append(SRR_OUTPUT_INSTRUCTIONS)
    return "\n\n".join(sections)


def extract_json_object(raw_content: str) -> dict[str, Any]:
    cleaned = raw_content.strip().lstrip("\ufeff")
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response does not contain a JSON object")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
            ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object, received {type(payload).__name__}")
    return payload


@dataclass(frozen=True)
class LLMRuntimeConfig:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Alternative:
    deficit: float


@dataclass(frozen=True)
class HardConstraints:
    max_deficit: float


def resolve_llm_runtime_config(agent: object) -> LLMRuntimeConfig:
    fields = {
        "base_url": _value(agent, "llm_base_url"),
        "api_key": _value(agent, "llm_api_key"),
        "model": _value(agent, "llm_model"),
    }
    placeholder_values = {"local-model", "local-llm"}
    missing = [
        name
        for name, value in fields.items()
        if not isinstance(value, str)
        or not value.strip()
        or value.strip().lower() in placeholder_values
    ]
    if missing:
        agent_name = str(_value(agent, "name"))
        raise ValueError(
            f"Agent {agent_name!r} requires database LLM configuration: {', '.join(missing)}"
        )
    base_url = str(fields["base_url"]).strip()
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("LLM base_url must use http:// or https://")
    return LLMRuntimeConfig(
        base_url=base_url,
        api_key=str(fields["api_key"]).strip(),
        model=str(fields["model"]).strip(),
    )


def build_agent_user_prompt(
    role: str,
    policy_goal: str,
    program_cost: float | None,
    max_deficit: float,
) -> str:
    return (
        f"Agent role: {role}\n"
        f"Policy goal: {policy_goal}\n"
        f"Program cost: {program_cost if program_cost is not None else 'UNKNOWN'}\n"
        f"Automatic legal deficit ceiling: {max_deficit}%\n\n"
        "Produce a decision-complete SRR response. State evidence-backed impact claims in predictions, "
        "identify material risks and uncertainties, provide at least one quantified alternative, and "
        "finish with a recommendation and calibrated confidence."
    )


def build_consensus_prompt(
    agent_name: str,
    role: str,
    peer_outputs: Sequence[Mapping[str, object]],
) -> str:
    return (
        f"You are {agent_name}, acting as {role}. Review the structured outputs from every agent below. "
        "Verify competing claims against cited evidence, identify unresolved assumptions, risks, "
        "uncertainties, constraints, and recommendation conflicts, then return a revised decision-complete "
        "SRR JSON object. Preserve valid dissent instead of fabricating agreement. Do not reveal hidden "
        "chain-of-thought; return only inspectable structured artifacts.\n\n"
        f"PEER OUTPUTS:\n{json.dumps(list(peer_outputs), ensure_ascii=False, sort_keys=True)}"
    )


def validate_decision_artifacts(response: object) -> list[str]:
    required_collections = ("evidence", "predictions", "risks", "uncertainties", "alternatives")
    missing = [field for field in required_collections if not _value(response, field)]
    if _value(response, "recommendation") is None:
        missing.append("recommendation")
    if _value(response, "confidence") is None:
        missing.append("confidence")
    return missing


def _value(item: object, field: str) -> Any:
    if isinstance(item, Mapping):
        try:
            return item[field]
        except KeyError as error:
            raise ValueError(f"Missing required field: {field}") from error
    try:
        return getattr(item, field)
    except AttributeError as error:
        raise ValueError(f"Missing required field: {field}") from error


def _score(agent: object) -> float:
    theta_x = float(_value(agent, "theta_x"))
    theta_q = float(_value(agent, "theta_q"))
    theta_h = float(_value(agent, "theta_h"))
    theta_s = float(_value(agent, "theta_s"))
    theta_u = float(_value(agent, "theta_u"))
    x = float(_value(agent, "X"))
    q = float(_value(agent, "Q"))
    h = float(_value(agent, "H"))
    s = float(_value(agent, "S"))
    u = float(_value(agent, "U"))
    return theta_x * x + theta_q * q + theta_h * h + theta_s * s - theta_u * u


def calculate_dynamic_influence(
    agents_data: Sequence[object],
    propositions: Sequence[object],
) -> list[dict[str, Any]]:
    if len(agents_data) != len(propositions):
        raise ValueError("agents_data and propositions must have equal lengths")
    if not agents_data:
        return []

    scores = [_score(agent) for agent in agents_data]
    gates = [int(_value(agent, "g_i")) for agent in agents_data]
    if any(gate not in (0, 1) for gate in gates):
        raise ValueError("Each g_i must be binary")

    active_scores = [score for score, gate in zip(scores, gates, strict=True) if gate == 1]
    if not active_scores:
        raise ValueError("At least one agent must have g_i = 1")

    max_score = max(active_scores)
    numerators = [
        gate * math.exp(score - max_score)
        for score, gate in zip(scores, gates, strict=True)
    ]
    denominator = math.fsum(numerators)

    return [
        {
            "proposition": proposition,
            "raw_score": score,
            "gate": gate,
            "normalized_weight": numerator / denominator,
        }
        for proposition, score, gate, numerator in zip(
            propositions,
            scores,
            gates,
            numerators,
            strict=True,
        )
    ]


def neuro_symbolic_filter(
    alternatives: Sequence[T],
    C_H: object,
) -> list[T]:
    max_deficit = float(_value(C_H, "max_deficit"))
    return [
        alternative
        for alternative in alternatives
        if float(_value(alternative, "deficit")) <= max_deficit
    ]


def _component_value(item: object, component: str, semantic_field: str) -> Any:
    if isinstance(item, Mapping):
        if component in item:
            return item[component]
        if semantic_field in item:
            return item[semantic_field]
        raise ValueError(f"Missing required field: {component}")
    if hasattr(item, component):
        return getattr(item, component)
    try:
        return getattr(item, semantic_field)
    except AttributeError as error:
        raise ValueError(f"Missing required field: {component}") from error


def detect_divergence_vector(
    obj_i: object,
    obj_j: object,
) -> dict[str, bool]:
    return {
        component: _component_value(obj_i, component, field)
        != _component_value(obj_j, component, field)
        for component, field in DDR_COMPONENT_FIELDS.items()
    }


def calculate_violation_rate(total_alts: int, feasible_alts: int) -> float:
    if total_alts < 0 or feasible_alts < 0:
        raise ValueError("Alternative counts cannot be negative")
    if feasible_alts > total_alts:
        raise ValueError("feasible_alts cannot exceed total_alts")
    if total_alts == 0:
        return 0.0
    return round(((total_alts - feasible_alts) / total_alts) * 100.0, 2)
