from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any, Mapping, Sequence

from .core_algorithms import (
    SRR_OUTPUT_INSTRUCTIONS,
    STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
)
from .sanitization import sanitize_public_dict

SIMULATION_AGENT_KEY = "fiscal_simulation_rpc_tool"
SIMULATION_AGENT_NAME = "Fiscal Simulation & RPC Tool Agent / Agen Simulasi & Kalkulasi Fiskal"
SIMULATION_AGENT_ROLE = "Non-voting deterministic fiscal simulation and calculation service"
SIMULATION_AGENT_VERSION = "fiscal-simulation-rpc-v1"
SIMULATION_RPC_PROTOCOL_VERSION = "1"
SIMULATION_RPC_TOOLS = (
    "fiscal.calculate_baseline",
    "fiscal.compare_deficit_alternatives",
    "fiscal.calculate_cash_headroom",
)
SIMULATION_RPC_TOOL_MANIFEST = {
    "fiscal.calculate_baseline": {
        "purpose": "Calculate program-cost coverage, recognized financing, gap, and surplus.",
        "required_inputs": ["program_cost"],
    },
    "fiscal.compare_deficit_alternatives": {
        "purpose": "Compare supplied alternative deficit ratios with the statutory ceiling.",
        "required_inputs": ["peer_outputs[].srr.alternatives[].deficit"],
    },
    "fiscal.calculate_cash_headroom": {
        "purpose": "Calculate projected cash headroom over the verified operational minimum.",
        "required_inputs": [
            "verified_projected_cash_after_policy",
            "verified_operational_cash_minimum",
        ],
    },
}
SIMULATION_TRIGGER = "Simulation Agent Requested"
SIMULATION_SOURCE_TAG = "SIMULATION_MODELLED"
MAX_SIMULATION_ROUNDS = 3

SIMULATION_AGENT_MANDATE = """MANDATE:
- Expose a strict RPC-style interface for deterministic fiscal baseline, deficit-ceiling, financing-identity, and cash-headroom calculations.
- Act as the neutral macro-fiscal simulation arbiter when sectoral agents reach a decision deadlock.
- Accept calls only through the allowlisted fiscal.calculate_baseline, fiscal.compare_deficit_alternatives, and fiscal.calculate_cash_headroom methods.
- Return calculation status, input hash, outputs, limitations, decision_authority=false, vote_eligible=false, and car_eligible=false for every RPC call.
- Reproduce competing sectoral positions inside one explicit, inspectable accounting sandbox before proposing a resolution.
- Never convert a modelled outcome into legal authority, realized revenue, verified fiscal space, a vote, or a CAR verdict.

PRIMARY SOURCES:
- UU17_2003_P12 (statutory deficit ceiling)
- UU17_2025_POSTURE (enacted 2026 fiscal posture)
- UU17_2025_P28 (financing and treasury authority)

OWNED HARD CHECKS:
- DEFICIT_3PCT
- MODELLED_NOT_VERIFIED
- NO_FABRICATED_PARAMETERS

CONSTRAINTS:
- Every simulated alternative MUST carry source_tag SIMULATION_MODELLED and evidence_status modelled.
- Never fabricate a fiscal multiplier, elasticity, tax yield, or transmission magnitude that was not supplied by a sectoral agent.
- Keep projected deficit inside the statutory ceiling; a compromise that breaches the ceiling is not a resolution.
- Preserve valid dissent. Do not manufacture agreement for the sake of closing the debate.

RESOLUTION PROTOCOL:
- Restate each conflicting position in neutral, inspectable terms.
- Test each position against the hard deficit ceiling and statutory constraints supplied by sectoral agents.
- Produce compromise alternatives that reconcile competing positions within the legal ceiling.
- State residual uncertainty, failure conditions, and which parameters remain modelled rather than verified.
"""

SIMULATION_OUTPUT_CONTRACT = (
    "In addition to the SRR decision artifacts include simulation_summary, conflict_summary, resolution, "
    "modelled_variables, limitations, and evidence_status='modelled'. Every alternative must set "
    "source_tag='SIMULATION_MODELLED' and evidence_status='modelled'. Return only structured, inspectable "
    "artifacts; never reveal hidden chain-of-thought."
)


@dataclass(frozen=True)
class NativeSimulationAgent:
    name: str
    role: str
    system_prompt: str
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    temperature: float = 0.1
    max_tokens: int = 8000


def _valid_runtime_value(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalised = value.strip()
    return None if normalised.lower() in {"local-model", "local-llm"} else normalised


def build_native_simulation_agent(runtime_source: object | None = None) -> NativeSimulationAgent:
    environment_runtime = (
        _valid_runtime_value(os.getenv("OPENAI_BASE_URL")),
        _valid_runtime_value(os.getenv("OPENAI_API_KEY")),
        _valid_runtime_value(os.getenv("OPENAI_MODEL")),
    )
    source_runtime = (
        _valid_runtime_value(getattr(runtime_source, "llm_base_url", None)),
        _valid_runtime_value(getattr(runtime_source, "llm_api_key", None)),
        _valid_runtime_value(getattr(runtime_source, "llm_model", None)),
    )
    selected_runtime = (
        environment_runtime
        if all(environment_runtime)
        else source_runtime if all(source_runtime) else (None, None, None)
    )
    base_url, api_key, model = selected_runtime
    return NativeSimulationAgent(
        name=SIMULATION_AGENT_NAME,
        role=SIMULATION_AGENT_ROLE,
        system_prompt=build_simulation_system_prompt(),
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=model,
    )


def build_simulation_system_prompt() -> str:
    return (
        f"You are the native {SIMULATION_AGENT_ROLE}, a permanent component of the SHCR engine. "
        "DDR invokes you automatically when sectoral agents disagree on predictions. Arbitrate through "
        "explicit simulation rather than advocacy.\n\n"
        f"{SIMULATION_AGENT_MANDATE}\n\n"
        f"RPC TOOL MANIFEST:\n{json.dumps(SIMULATION_RPC_TOOL_MANIFEST, ensure_ascii=False, sort_keys=True)}\n\n"
        f"{SIMULATION_OUTPUT_CONTRACT}\n\n"
        f"{SRR_OUTPUT_INSTRUCTIONS}"
    )


def build_simulation_prompt(
    scenario_description: str,
    simulation_payload: Mapping[str, Any],
    conflicts: Sequence[Mapping[str, Any]],
    peer_outputs: Sequence[Mapping[str, Any]],
) -> str:
    sandbox_inputs = {
        "policy_goal": scenario_description,
        **dict(simulation_payload),
        "statutory_deficit_ceiling_percent": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
        "ddr_conflicts": [
            {
                "agent_i": conflict.get("agent_i"),
                "agent_j": conflict.get("agent_j"),
                "components": conflict.get("components"),
                "route": conflict.get("route"),
            }
            for conflict in conflicts
        ],
    }
    return (
        "The DDR detector routed this deadlock to the native Simulation Agent. Run the arbitration sandbox "
        "and return a decision-complete SRR JSON object that resolves the impasse within the statutory ceiling.\n\n"
        f"SIMULATION SANDBOX INPUTS:\n{json.dumps(sandbox_inputs, ensure_ascii=False, sort_keys=True)}\n\n"
        f"SECTORAL OUTPUTS UNDER ARBITRATION:\n{json.dumps(list(peer_outputs), ensure_ascii=False, sort_keys=True)}"
    )


def build_simulation_consensus_prompt(
    agent_name: str,
    role: str,
    peer_outputs: Sequence[Mapping[str, Any]],
    simulation_output: Mapping[str, Any],
) -> str:
    return (
        f"You are {agent_name}, acting as {role}. The native Simulation Agent has arbitrated the DDR "
        "deadlock. Review its structured resolution with every sectoral output below. Return a revised "
        "decision-complete SRR JSON object. Treat simulation claims as modelled evidence, not verified legal "
        "authority or fiscal space. Correct unsupported claims and preserve valid dissent rather than accepting "
        "a false compromise. Do not reveal hidden chain-of-thought; return only inspectable structured artifacts.\n\n"
        f"SIMULATION ARBITRATION RESULT:\n{json.dumps(dict(simulation_output), ensure_ascii=False, sort_keys=True)}\n\n"
        f"PEER OUTPUTS:\n{json.dumps(list(peer_outputs), ensure_ascii=False, sort_keys=True)}"
    )


def sanitize_simulation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return sanitize_public_dict(payload)


def _finite_number(value: object) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _rpc_result(
    tool_name: str,
    inputs: Mapping[str, Any],
    *,
    status: str,
    calculation: Mapping[str, Any],
    limitations: Sequence[str],
) -> dict[str, Any]:
    safe_inputs = sanitize_public_dict(dict(inputs))
    inputs_hash = hashlib.sha256(
        json.dumps(safe_inputs, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return {
        "protocol_version": SIMULATION_RPC_PROTOCOL_VERSION,
        "agent_name": SIMULATION_AGENT_NAME,
        "tool_name": tool_name,
        "status": status,
        "inputs_hash": inputs_hash,
        "calculation": dict(calculation),
        "limitations": list(limitations),
        "decision_authority": False,
        "vote_eligible": False,
        "car_eligible": False,
    }


def invoke_fiscal_simulation_rpc(
    tool_name: str,
    *,
    scenario_description: str,
    simulation_payload: Mapping[str, Any],
    peer_outputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if tool_name not in SIMULATION_RPC_TOOLS:
        raise ValueError(f"Unknown fiscal simulation RPC tool: {tool_name}")
    inputs = {
        "scenario_description": scenario_description,
        "simulation_payload": dict(simulation_payload),
        "peer_outputs": list(peer_outputs),
    }
    if tool_name == "fiscal.calculate_baseline":
        program_cost = _finite_number(simulation_payload.get("program_cost"))
        source_fields = (
            "proposed_reallocation",
            "proposed_additional_revenue",
            "proposed_debt_financing",
            "proposed_sal_use",
            "proposed_other_financing",
        )
        supplied_sources = {
            field: number
            for field in source_fields
            if (number := _finite_number(simulation_payload.get(field))) is not None
        }
        if program_cost is None:
            return _rpc_result(
                tool_name,
                inputs,
                status="not_calculated",
                calculation={},
                limitations=("Program cost is required for the financing baseline.",),
            )
        recognized_financing = sum(supplied_sources.values())
        return _rpc_result(
            tool_name,
            inputs,
            status="calculated",
            calculation={
                "method": "direct-financing-identity",
                "program_cost": program_cost,
                "recognized_financing": recognized_financing,
                "financing_gap": max(0.0, program_cost - recognized_financing),
                "financing_surplus": max(0.0, recognized_financing - program_cost),
                "source_components": supplied_sources,
            },
            limitations=(
                "Missing financing components are not inferred.",
                "The result is arithmetic only and is not a CAR determination.",
            ),
        )
    if tool_name == "fiscal.calculate_cash_headroom":
        projected_cash = _finite_number(
            simulation_payload.get("verified_projected_cash_after_policy")
        )
        minimum_cash = _finite_number(
            simulation_payload.get("verified_operational_cash_minimum")
        )
        if projected_cash is None or minimum_cash is None:
            return _rpc_result(
                tool_name,
                inputs,
                status="not_calculated",
                calculation={},
                limitations=(
                    "Verified projected cash and operational minimum are both required.",
                ),
            )
        return _rpc_result(
            tool_name,
            inputs,
            status="calculated",
            calculation={
                "method": "projected-cash-minus-operational-minimum",
                "projected_cash_after_policy": projected_cash,
                "operational_cash_minimum": minimum_cash,
                "cash_headroom": projected_cash - minimum_cash,
            },
            limitations=("Cash timing and source provenance require independent verification.",),
        )
    alternatives: list[dict[str, Any]] = []
    for peer in peer_outputs:
        srr = peer.get("srr")
        if not isinstance(srr, Mapping):
            continue
        raw_alternatives = srr.get("alternatives")
        if not isinstance(raw_alternatives, Sequence):
            continue
        for alternative in raw_alternatives:
            if not isinstance(alternative, Mapping):
                continue
            deficit = _finite_number(alternative.get("deficit"))
            if deficit is None:
                continue
            alternatives.append(
                {
                    "name": str(alternative.get("name") or "Unnamed alternative"),
                    "projected_deficit_percent_gdp": deficit,
                    "within_statutory_ceiling": (
                        deficit <= STATUTORY_DEFICIT_CEILING_PERCENT_GDP
                    ),
                }
            )
    return _rpc_result(
        tool_name,
        inputs,
        status="calculated" if alternatives else "not_calculated",
        calculation={
            "method": "direct-statutory-threshold-comparison",
            "ceiling_percent_gdp": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
            "alternatives": alternatives,
        },
        limitations=(
            "Alternative deficits originate in sectoral outputs and remain modelled until verified.",
            "Threshold comparison is not a CAR verdict or authorization.",
        ),
    )


def invoke_fiscal_simulation_rpc_suite(
    *,
    scenario_description: str,
    simulation_payload: Mapping[str, Any],
    peer_outputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        invoke_fiscal_simulation_rpc(
            tool_name,
            scenario_description=scenario_description,
            simulation_payload=simulation_payload,
            peer_outputs=peer_outputs,
        )
        for tool_name in SIMULATION_RPC_TOOLS
    ]


def build_deterministic_simulation(
    scenario_description: str,
    simulation_payload: Mapping[str, Any],
    conflicts: Sequence[Mapping[str, Any]],
    peer_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    alternatives: list[dict[str, Any]] = []
    risks: list[str] = []
    uncertainties: list[str] = []
    for peer in peer_outputs:
        srr = peer.get("srr")
        if not isinstance(srr, Mapping):
            continue
        for raw_alternative in srr.get("alternatives", []):
            if not isinstance(raw_alternative, Mapping):
                continue
            deficit = raw_alternative.get("deficit")
            utility = raw_alternative.get("utility")
            if (
                not isinstance(deficit, (int, float))
                or isinstance(deficit, bool)
                or not math.isfinite(float(deficit))
                or float(deficit) < 0
            ):
                continue
            if (
                not isinstance(utility, (int, float))
                or isinstance(utility, bool)
                or not math.isfinite(float(utility))
                or not 0 <= float(utility) <= 1
            ):
                continue
            alternatives.append(
                {
                    "name": str(raw_alternative.get("name") or "Modelled compromise"),
                    "deficit": float(deficit),
                    "utility": float(utility),
                    "source_tag": SIMULATION_SOURCE_TAG,
                    "evidence_status": "modelled",
                }
            )
        for field, target in (("risks", risks), ("uncertainties", uncertainties)):
            for item in srr.get(field, []):
                if isinstance(item, Mapping) and isinstance(item.get("content"), str):
                    content = item["content"].strip()
                    if content and content not in target:
                        target.append(content)
    effective_ceiling = STATUTORY_DEFICIT_CEILING_PERCENT_GDP
    feasible = [item for item in alternatives if item["deficit"] <= effective_ceiling]
    ranked = sorted(
        feasible,
        key=lambda item: (-item["utility"], item["deficit"], item["name"]),
    )
    selected = ranked[0] if ranked else None
    selected_alternatives = ranked[:5]
    conflict_summary = [
        f"{item.get('agent_i')} vs {item.get('agent_j')}: {', '.join(item.get('components') or [])}"
        for item in conflicts
    ] or ["DDR requested simulation without pair metadata"]
    resolution = (
        f"Use {selected['name']} as the bounded compromise candidate with modelled deficit "
        f"{selected['deficit']} and utility {selected['utility']}; sectoral agents must revalidate it."
        if selected is not None
        else "No modelled alternative satisfies the statutory deficit ceiling; CAR must return INFEASIBLE with no selected alternative."
    )
    return {
        "evidence": [
            {
                "content": "The arbitration uses only the structured sectoral outputs supplied in this run.",
                "source_tag": SIMULATION_SOURCE_TAG,
            }
        ],
        "predictions": [
            {
                "content": (
                    f"The selected modelled candidate remains at or below the {effective_ceiling}% deficit ceiling."
                    if selected is not None
                    else f"All supplied alternatives exceed the {effective_ceiling}% deficit ceiling."
                ),
                "source_tag": SIMULATION_SOURCE_TAG,
            }
        ],
        "risks": [
            {"content": item, "source_tag": SIMULATION_SOURCE_TAG}
            for item in risks[:5]
        ] or [{"content": "Residual implementation risk requires sectoral validation.", "source_tag": SIMULATION_SOURCE_TAG}],
        "uncertainties": [
            {"content": item, "source_tag": SIMULATION_SOURCE_TAG}
            for item in uncertainties[:5]
        ] or [{"content": "Unverified model parameters remain uncertain.", "source_tag": SIMULATION_SOURCE_TAG}],
        "constraints": [
            {
                "content": f"Projected deficit must remain at or below {effective_ceiling}%.",
                "source_tag": "UU17_2003_P12",
            }
        ],
        "alternatives": selected_alternatives,
        "recommendation": {"content": resolution, "source_tag": SIMULATION_SOURCE_TAG},
        "confidence": 0.75,
        "material_information_retention_macro_f1": 1.0,
        "simulation_summary": (
            f"Native arbitration evaluated {len(alternatives)} sectoral alternatives for "
            f"{scenario_description} with scenario inputs "
            f"{json.dumps(dict(simulation_payload), ensure_ascii=False, sort_keys=True)}"
        ),
        "conflict_summary": conflict_summary,
        "resolution": resolution,
        "resolution_status": "RESOLVED" if selected is not None else "INFEASIBLE",
        "selected_alternative": selected,
        "calculation_status": "modelled" if selected is not None else "infeasible",
        "modelled_variables": ["deficit", "utility"],
        "limitations": [
            "The native simulation does not establish legal authority or empirical causality.",
            "Unverified sectoral parameters remain modelled assumptions.",
        ],
        "evidence_status": "modelled",
        "agent_name": SIMULATION_AGENT_NAME,
        "simulation_version": SIMULATION_AGENT_VERSION,
    }


def resolve_simulation_runtime_agent(agents: Sequence[object]) -> NativeSimulationAgent | None:
    for runtime_source in agents:
        agent = build_native_simulation_agent(runtime_source)
        if all((agent.llm_base_url, agent.llm_api_key, agent.llm_model)):
            return agent
    agent = build_native_simulation_agent()
    if all((agent.llm_base_url, agent.llm_api_key, agent.llm_model)):
        return agent
    return None
