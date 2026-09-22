from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, Sequence

from .core_algorithms import SRR_OUTPUT_INSTRUCTIONS

SIMULATION_AGENT_KEY = "native_simulation_agent"
SIMULATION_AGENT_NAME = "Simulation Agent / Arbiter Simulasi Makro-Fiskal"
SIMULATION_AGENT_ROLE = "Arbiter Simulasi Makro-Fiskal"
SIMULATION_AGENT_VERSION = "native-simulation-v1"
SIMULATION_TRIGGER = "Simulation Agent Requested"
SIMULATION_SOURCE_TAG = "SIMULATION_MODELLED"
MAX_SIMULATION_ROUNDS = 3
_PRIVATE_REASONING_KEYS = {
    "analysis",
    "chain_of_thought",
    "chainofthought",
    "hidden_reasoning",
    "reasoning_trace",
    "thoughts",
}

SIMULATION_AGENT_MANDATE = """MANDATE:
- Act as the neutral macro-fiscal simulation arbiter when sectoral agents reach a decision deadlock.
- Reproduce competing sectoral positions inside one explicit, inspectable accounting sandbox before proposing a resolution.
- Bridge dissent by exposing where positions differ and which differences are evidential rather than interpretive.
- Never convert a modelled outcome into legal authority, realized revenue, or verified fiscal space.

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
        f"{SIMULATION_OUTPUT_CONTRACT}\n\n"
        f"{SRR_OUTPUT_INSTRUCTIONS}"
    )


def build_simulation_prompt(
    scenario_description: str,
    program_cost: float | None,
    max_deficit_constraint: float,
    conflicts: Sequence[Mapping[str, Any]],
    peer_outputs: Sequence[Mapping[str, Any]],
) -> str:
    sandbox_inputs = {
        "policy_goal": scenario_description,
        "program_cost": program_cost if program_cost is not None else "UNKNOWN",
        "statutory_deficit_ceiling_percent": max_deficit_constraint,
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
    def clean(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key.casefold() not in _PRIVATE_REASONING_KEYS
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return {key: clean(value) for key, value in payload.items() if key.casefold() not in _PRIVATE_REASONING_KEYS}


def build_deterministic_simulation(
    scenario_description: str,
    max_deficit_constraint: float,
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
            if not isinstance(deficit, (int, float)) or isinstance(deficit, bool):
                continue
            if not isinstance(utility, (int, float)) or isinstance(utility, bool):
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
    feasible = [item for item in alternatives if item["deficit"] <= max_deficit_constraint]
    candidates = feasible or sorted(alternatives, key=lambda item: item["deficit"])
    ranked = sorted(candidates, key=lambda item: (-item["utility"], item["deficit"], item["name"]))
    selected = ranked[0] if ranked else {
        "name": "No feasible modelled alternative",
        "deficit": max_deficit_constraint,
        "utility": 0.0,
        "source_tag": SIMULATION_SOURCE_TAG,
        "evidence_status": "modelled",
    }
    selected_alternatives = ranked[:5] or [selected]
    conflict_summary = [
        f"{item.get('agent_i')} vs {item.get('agent_j')}: {', '.join(item.get('components') or [])}"
        for item in conflicts
    ] or ["DDR requested simulation without pair metadata"]
    resolution = (
        f"Use {selected['name']} as the bounded compromise candidate with modelled deficit "
        f"{selected['deficit']} and utility {selected['utility']}; sectoral agents must revalidate it."
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
                "content": f"The selected modelled candidate remains at or below the {max_deficit_constraint}% deficit ceiling.",
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
                "content": f"Projected deficit must remain at or below {max_deficit_constraint}%.",
                "source_tag": "UU17_2003_P12",
            }
        ],
        "alternatives": selected_alternatives,
        "recommendation": {"content": resolution, "source_tag": SIMULATION_SOURCE_TAG},
        "confidence": 0.75,
        "material_information_retention_macro_f1": 1.0,
        "simulation_summary": f"Native arbitration evaluated {len(alternatives)} sectoral alternatives for {scenario_description}",
        "conflict_summary": conflict_summary,
        "resolution": resolution,
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
    agent = build_native_simulation_agent()
    if all((agent.llm_base_url, agent.llm_api_key, agent.llm_model)):
        return agent
    for runtime_source in agents:
        agent = build_native_simulation_agent(runtime_source)
        if all((agent.llm_base_url, agent.llm_api_key, agent.llm_model)):
            return agent
    return None
