import json
import logging
import math
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import urlsplit, urlunsplit

LLM_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
LLM_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": LLM_USER_AGENT,
    "Accept": "application/json",
}
logger = logging.getLogger(__name__)
STATUTORY_DEFICIT_CEILING_PERCENT_GDP = 3.0

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
DDR_COMPONENT_DETAILS: dict[str, dict[str, Any]] = {
    "dE": {
        "category": {"id": "Divergensi bukti dan provenance", "en": "Evidence and provenance divergence"},
        "formula": "1 - |sources_i ∩ sources_j| / |sources_i ∪ sources_j|",
        "meaning": {"id": "Perbedaan sumber atau hash artefak mencegah penerimaan data yang belum terverifikasi.", "en": "Source or artifact-hash differences prevent acceptance of unverified data."},
        "impact": {"id": "Memicu verifikasi provenance dan retrieval tambahan sebelum klaim dipakai.", "en": "Triggers provenance verification and additional retrieval before a claim is admitted."},
        "route": "Provenance Retrieval Triggered",
    },
    "dA": {
        "category": {"id": "Divergensi asumsi", "en": "Assumption divergence"},
        "formula": "1 - |assumptions_i ∩ assumptions_j| / |assumptions_i ∪ assumptions_j|",
        "meaning": {"id": "Premis masa depan yang tidak teramati berbeda antaragen.", "en": "Unobserved future premises differ between agents."},
        "impact": {"id": "Memerlukan analisis sensitivitas dan deklarasi reliabilitas eksplisit.", "en": "Requires sensitivity analysis and explicit reliability declarations."},
        "route": "Evidence Review Required",
    },
    "dP": {
        "category": {"id": "Divergensi prediksi", "en": "Prediction divergence"},
        "formula": "prediction_set_distance; deficit_range_gap = max(|deficit_i - deficit_j|)",
        "meaning": {"id": "Proyeksi dampak kausal atau pergeseran defisit berbeda.", "en": "Causal-impact projections or deficit shifts differ."},
        "impact": {"id": "Diteruskan ke Simulation Agent untuk stress-test kuantitatif berbatas.", "en": "Escalates to the Simulation Agent for bounded quantitative stress testing."},
        "route": "Simulation Agent Requested",
    },
    "dR": {
        "category": {"id": "Divergensi risiko", "en": "Risk divergence"},
        "formula": "risk_set_distance; impact_probability_gap when verified scores exist",
        "meaning": {"id": "Profil ancaman fiskal atau operasional dinilai berbeda.", "en": "Fiscal or operational threat profiles are assessed differently."},
        "impact": {"id": "Memicu stress-test historis; matriks Impact × Probability tidak dihitung tanpa skor terverifikasi.", "en": "Triggers historical stress testing; Impact × Probability is not calculated without verified scores."},
        "route": "Evidence Review Required",
    },
    "dU": {
        "category": {"id": "Divergensi ketidakpastian", "en": "Uncertainty divergence"},
        "formula": "uncertainty_set_distance; confidence_gap = |confidence_i - confidence_j|",
        "meaning": {"id": "Confidence atau batas epistemik antaragen berbeda.", "en": "Agent confidence or epistemic bounds differ."},
        "impact": {"id": "Memerlukan rekonsiliasi ketidakpastian dan memengaruhi penalti RAR-DAI.", "en": "Requires uncertainty reconciliation and affects the RAR-DAI penalty."},
        "route": "Evidence Review Required",
    },
    "dO": {
        "category": {"id": "Divergensi tujuan", "en": "Objective divergence"},
        "formula": "1 - |objectives_i ∩ objectives_j| / |objectives_i ∪ objectives_j|",
        "meaning": {"id": "Prioritas stabilitas fiskal dan manfaat sosial bertabrakan.", "en": "Fiscal-stability and social-benefit priorities conflict."},
        "impact": {"id": "Diadili melalui trade-off Pareto/MCDM tanpa menghapus posisi minoritas.", "en": "Adjudicated through Pareto/MCDM trade-offs without suppressing minority positions."},
        "route": "Pareto Reconciliation",
    },
    "dC": {
        "category": {"id": "Divergensi constraint", "en": "Constraint divergence"},
        "formula": "hard_gate = projected_deficit <= 3.0%; education_share >= 20% when verified",
        "meaning": {"id": "Batas hukum keras diterapkan atau dilanggar secara berbeda.", "en": "Hard statutory boundaries are applied or violated differently."},
        "impact": {"id": "CAR/Z3 memberi verdict INFEASIBLE saat hard constraint terverifikasi dilanggar; LLM tidak dapat mengesampingkannya.", "en": "CAR/Z3 returns INFEASIBLE when a verified hard constraint is violated; an LLM cannot override it."},
        "route": "Constraint Arbitration Required",
    },
    "dREC": {
        "category": {"id": "Divergensi rekomendasi", "en": "Recommendation divergence"},
        "formula": "recommendation_text_mismatch and alternative_rank_distance",
        "meaning": {"id": "Peringkat strategi akhir berbeda meskipun artefak telah distrukturkan.", "en": "Final strategic rankings differ despite structured artifacts."},
        "impact": {"id": "Memaksa resolusi ulang penyebab hulu dE–dC sebelum sintesis final.", "en": "Forces upstream re-resolution of dE–dC before final synthesis."},
        "route": "Upstream Re-resolution Required",
    },
}
T = TypeVar("T")

SRR_OUTPUT_INSTRUCTIONS = (
    "Return only one valid SRR JSON object. A decision-complete response must include non-empty "
    "evidence, predictions, risks, uncertainties, and alternatives arrays, plus recommendation and "
    "confidence. Include a concise reasoning_summary that states the agent's inspectable policy opinion, "
    "plus assumptions, objectives, constraints, and "
    "material_information_retention_macro_f1 when supported. Every typed item must contain content "
    "and an optional source_tag. Every alternative must contain name, deficit, utility, and optional "
    "source_tag. Additional structured fields are allowed. Analyze impacts, risks, uncertainties, "
    "objections, conditions, and adjustments within the agent's fiscal mandate. Do not reveal hidden "
    "chain-of-thought; provide only concise, evidence-linked, inspectable artifacts."
)


def build_agent_system_prompt(
    role: str,
    mandate: str | None,
    scenario_mandate: str | None = None,
) -> str:
    sections = [f"Functional fiscal role: {role}."]
    if mandate and mandate.strip():
        sections.append(f"Agent-specific mandate and decision principles:\n{mandate.strip()}")
    if scenario_mandate and scenario_mandate.strip():
        sections.append(f"Current scenario-specific mandate:\n{scenario_mandate.strip()}")
    sections.append(SRR_OUTPUT_INSTRUCTIONS)
    return "\n\n".join(sections)


def extract_json_object(raw_content: str) -> dict[str, Any]:
    cleaned = raw_content.strip().lstrip("\ufeff")
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", cleaned):
            try:
                candidate, _ = decoder.raw_decode(cleaned[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            raise ValueError("LLM response does not contain a valid JSON object")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object, received {type(payload).__name__}")
    choices = payload.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices:
        choice = choices[0]
        if isinstance(choice, Mapping):
            message = choice.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return extract_json_object(message["content"])
    return payload


def llm_request_headers() -> dict[str, str]:
    return dict(LLM_HEADERS)


def safe_llm_target(base_url: str) -> str:
    parsed = urlsplit(base_url)
    hostname = parsed.hostname or ""
    host = f"[{hostname}]" if ":" in hostname else hostname
    try:
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    except ValueError:
        netloc = host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def log_llm_outbound(
    operation: str,
    agent_name: str,
    base_url: str,
    model: str,
    payload: Mapping[str, object],
) -> None:
    logger.info(
        "Outbound LLM request operation=%s agent=%s target=%s model=%s payload=%s",
        operation,
        agent_name,
        safe_llm_target(base_url),
        model,
        json.dumps(dict(payload), ensure_ascii=False, default=str),
    )


def extract_llm_completion(response: object) -> tuple[str, int]:
    content: object = None
    usage_value: object = None
    if isinstance(response, str):
        content = response
    elif isinstance(response, Mapping):
        content = response.get("content")
        if content is None:
            choices = response.get("choices")
            if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices:
                choice = choices[0]
                if isinstance(choice, Mapping):
                    message = choice.get("message")
                    if isinstance(message, Mapping):
                        content = message.get("content")
        usage = response.get("usage")
        usage_value = usage.get("total_tokens") if isinstance(usage, Mapping) else None
    else:
        choices = getattr(response, "choices", None)
        if not choices:
            direct_content = getattr(response, "content", None)
            if isinstance(direct_content, str):
                content = direct_content
            else:
                raise ValueError(f"Unsupported LLM response type: {type(response).__name__}")
        else:
            content = getattr(getattr(choices[0], "message", None), "content", None)
        usage = getattr(response, "usage", None)
        usage_value = getattr(usage, "total_tokens", None) if usage else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned no text content")
    token_usage = (
        usage_value
        if isinstance(usage_value, int) and not isinstance(usage_value, bool) and usage_value >= 0
        else 0
    )
    return content, token_usage


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
    agent_name = str(_value(agent, "name"))
    field_sources = {
        "base_url": (_value(agent, "llm_base_url"), os.getenv("OPENAI_BASE_URL")),
        "api_key": (_value(agent, "llm_api_key"), os.getenv("OPENAI_API_KEY")),
        "model": (_value(agent, "llm_model"), os.getenv("OPENAI_MODEL")),
    }
    resolved: dict[str, str] = {}
    fallback_fields: list[str] = []
    placeholder_values = {"local-model", "local-llm"}
    for field, (agent_value, environment_value) in field_sources.items():
        if isinstance(agent_value, str) and agent_value.strip():
            resolved[field] = agent_value.strip()
        elif isinstance(environment_value, str) and environment_value.strip():
            resolved[field] = environment_value.strip()
            fallback_fields.append(field)
    missing = [
        field
        for field in field_sources
        if field not in resolved or resolved[field].lower() in placeholder_values
    ]
    if missing:
        raise ValueError(
            f"Agent {agent_name!r} requires database LLM configuration or valid environment fallback: "
            f"{', '.join(missing)}"
        )
    if fallback_fields:
        warnings.warn(
            f"Agent {agent_name!r} uses environment fallback for: {', '.join(fallback_fields)}",
            RuntimeWarning,
            stacklevel=2,
        )
    base_url = resolved["base_url"]
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("LLM base_url must use http:// or https://")
    return LLMRuntimeConfig(
        base_url=base_url,
        api_key=str(resolved["api_key"]),
        model=str(resolved["model"]),
    )


def build_agent_user_prompt(
    role: str,
    policy_goal: str,
    program_cost: float | None,
    simulation_payload: Mapping[str, object] | None = None,
) -> str:
    policy_inputs = dict(simulation_payload or {})
    policy_inputs.setdefault("program_cost", program_cost)
    return (
        f"Agent role: {role}\n"
        f"Policy goal: {policy_goal}\n"
        f"Program cost: {program_cost if program_cost is not None else 'UNKNOWN'}\n"
        f"Scenario inputs: {json.dumps(policy_inputs, ensure_ascii=False, sort_keys=True)}\n"
        f"Automatic legal deficit ceiling: {STATUTORY_DEFICIT_CEILING_PERCENT_GDP}%\n\n"
        "Produce a decision-complete SRR JSON response with all mandatory decision artifacts: "
        "non-empty evidence, predictions, risks, uncertainties, and alternatives arrays, plus "
        "a recommendation object and numeric confidence between 0 and 1. State evidence-backed impact "
        "claims in predictions. Use concise structured items with content and optional source_tag; "
        "alternatives require name, deficit, and utility."
    )


def build_mandate_synthesis_system_prompt(agent_name: str, role: str) -> str:
    return (
        f"You are {agent_name}, an autonomous expert acting as {role} in the dissertation framework "
        '"Collective Reasoning in Heterogeneous Multi-Agent Systems: A Structured Consensus Framework '
        'for Strategic Fiscal Decision Support." Produce a decision-ready mandate that covers the '
        "agent's operational responsibility, APBN domain focus, regulation-based priority questions, "
        "fiscally verifiable evidence, auditable epistemic traceability, structured disagreement and "
        "consensus handling, and strict regulatory compliance. Preserve uncertainty and valid dissent. "
        "Never treat an unverified LLM prediction as legal authority, realized revenue, fiscal space, "
        "or evidence. Return valid JSON only and do not reveal chain-of-thought."
    )


def build_mandate_synthesis_prompt(
    agent_name: str,
    role: str,
    scenario_description: str,
    primary_sources: Sequence[str] = (),
    constraints: Sequence[str] = (),
    owned_checks: Sequence[str] = (),
    simulation_payload: Mapping[str, object] | None = None,
) -> str:
    domain_contract = {
        "primary_sources": list(primary_sources),
        "constraints": list(constraints),
        "owned_checks": list(owned_checks),
        "automatic_deficit_ceiling_percent_gdp": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
        "scenario_inputs": dict(simulation_payload or {}),
    }
    return (
        f"Act autonomously as the expert {role} named {agent_name}. Generate a practical, "
        "scenario-specific operating mandate grounded in the authoritative domain contract below. "
        "Use Primary Sources as the legal/evidentiary basis, Constraints as non-negotiable fiscal or "
        "statutory boundaries, and Owned Checks as validations this agent must execute. Define how "
        "claims and inter-agent handoffs remain source-linked and auditable; how disagreements are "
        "classified, preserved, escalated, and reconciled through a structured consensus protocol; "
        "and how the legal deficit ceiling and other hard constraints are enforced without converting "
        "unverified model predictions into facts or fiscal capacity. Do not reveal chain-of-thought. "
        "Return only one JSON object with exactly these logical fields: scenario_mandate, "
        "scenario_focus, priority_questions, required_evidence, epistemic_logic_traceability, "
        "structured_consensus_protocol, regulatory_compliance_alignment. Each field may be a concise "
        "string or an array of concise inspectable statements.\n\n"
        f"AUTHORITATIVE DOMAIN CONTRACT:\n{json.dumps(domain_contract, ensure_ascii=False)}\n\n"
        f"ACTIVE POLICY SCENARIO: {scenario_description}"
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
        "SRR JSON object with non-empty evidence, predictions, risks, uncertainties, and alternatives, "
        "plus recommendation and confidence. Preserve valid dissent instead of fabricating agreement. "
        "Do not reveal hidden chain-of-thought; return only inspectable structured artifacts.\n\n"
        f"PEER OUTPUTS:\n{json.dumps(list(peer_outputs), ensure_ascii=False, sort_keys=True)}"
    )


def validate_decision_artifacts(response: object) -> list[str]:
    required_collections = ("evidence", "predictions", "risks", "uncertainties")
    missing = [field for field in required_collections if not _value(response, field)]
    alternatives = _value(response, "alternatives")
    fallback_metadata = (
        response.get("fallback_metadata")
        if isinstance(response, Mapping)
        else getattr(response, "fallback_metadata", None)
    )
    if not alternatives and not fallback_metadata:
        missing.append("alternatives")
    recommendation = _value(response, "recommendation")
    if recommendation is None:
        missing.append("recommendation")
    else:
        content = _value(recommendation, "content")
        if not isinstance(content, str) or not content.strip():
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


def calculate_orchestrator_rar_dai_weights(
    domain_alignment: float,
    evidence_completeness: float,
) -> dict[str, float]:
    if not math.isfinite(domain_alignment) or not 0.0 <= domain_alignment <= 1.0:
        raise ValueError("domain_alignment must be finite and within 0..1")
    if not math.isfinite(evidence_completeness) or not 0.0 <= evidence_completeness <= 1.0:
        raise ValueError("evidence_completeness must be finite and within 0..1")
    return {
        "theta_x": round(0.8 + 0.8 * domain_alignment, 6),
        "theta_q": round(0.8 + 1.2 * evidence_completeness, 6),
        "theta_h": round(0.6 + 0.4 * evidence_completeness, 6),
        "theta_s": round(0.75 + 0.75 * domain_alignment, 6),
        "theta_u": round(1.0 + 1.0 * (1.0 - evidence_completeness), 6),
    }


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
        math.exp(score - max_score) if gate == 1 else 0.0
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
    max_deficit = min(
        float(_value(C_H, "max_deficit")),
        STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
    )
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


def describe_divergence_vector(vector: Mapping[str, object]) -> list[dict[str, Any]]:
    return [
        {
            "component": component,
            "category": DDR_COMPONENT_DETAILS[component]["category"]["en"],
            "category_i18n": DDR_COMPONENT_DETAILS[component]["category"],
            "narrative": DDR_COMPONENT_DETAILS[component]["meaning"]["en"],
            "narrative_i18n": DDR_COMPONENT_DETAILS[component]["meaning"],
            "impact": DDR_COMPONENT_DETAILS[component]["impact"]["en"],
            "impact_i18n": DDR_COMPONENT_DETAILS[component]["impact"],
            "formula": DDR_COMPONENT_DETAILS[component]["formula"],
            "default_route": DDR_COMPONENT_DETAILS[component]["route"],
        }
        for component in DDR_COMPONENT_FIELDS
        if bool(vector.get(component))
    ]


def resolve_disagreement_route(vector: Mapping[str, object]) -> str:
    if bool(vector.get("dC")):
        return "Constraint Arbitration Required"
    if bool(vector.get("dP")):
        return "Simulation Agent Requested"
    if bool(vector.get("dE")):
        return "Provenance Retrieval Triggered"
    if bool(vector.get("dREC")):
        return "Upstream Re-resolution Required"
    if bool(vector.get("dO")):
        return "Pareto Reconciliation"
    if any(bool(vector.get(component)) for component in ("dA", "dR", "dU")):
        return "Evidence Review Required"
    return "No Resolution Required"


def _canonical_divergence_value(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _canonical_divergence_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [_canonical_divergence_value(item) for item in value]
        return tuple(sorted(items, key=repr))
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return value


def detect_divergence_vector(
    obj_i: object,
    obj_j: object,
) -> dict[str, bool]:
    return {
        component: _canonical_divergence_value(
            _component_value(obj_i, component, field)
        )
        != _canonical_divergence_value(_component_value(obj_j, component, field))
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
