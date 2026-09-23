import math

import pytest

from backend.core_algorithms import (
    Alternative,
    HardConstraints,
    build_agent_system_prompt,
    build_agent_user_prompt,
    build_consensus_prompt,
    build_mandate_synthesis_prompt,
    build_mandate_synthesis_system_prompt,
    calculate_dynamic_influence,
    calculate_violation_rate,
    detect_divergence_vector,
    extract_json_object,
    extract_llm_completion,
    llm_request_headers,
    neuro_symbolic_filter,
)
from backend.simulation_agent import (
    SIMULATION_AGENT_NAME,
    SIMULATION_SOURCE_TAG,
    build_deterministic_simulation,
    build_native_simulation_agent,
    build_simulation_consensus_prompt,
    build_simulation_system_prompt,
    sanitize_simulation_payload,
)


def make_agent(*, uncertainty: float = 0.0, gate: int = 1) -> dict[str, float | int]:
    return {
        "theta_x": 1.0,
        "theta_q": 1.0,
        "theta_h": 1.0,
        "theta_s": 1.0,
        "theta_u": 1.0,
        "X": 1.0,
        "Q": 1.0,
        "H": 1.0,
        "S": 1.0,
        "U": uncertainty,
        "g_i": gate,
    }


def test_extract_json_object_cleans_markdown_and_surrounding_text() -> None:
    assert extract_json_object('```json\n{"status": "ok"}\n```') == {"status": "ok"}
    assert extract_json_object('Result follows: {"status": "ok"} done') == {"status": "ok"}
    with pytest.raises(ValueError, match="does not contain"):
        extract_json_object("not-json")


def test_llm_headers_and_completion_extraction() -> None:
    headers = llm_request_headers()
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"
    assert headers["User-Agent"].startswith("Mozilla/5.0")
    assert extract_llm_completion("direct response") == ("direct response", 0)
    assert extract_llm_completion({"content": "mapping response"}) == ("mapping response", 0)
    assert extract_llm_completion({"choices": [{"message": {"content": "choice response"}}]}) == (
        "choice response",
        0,
    )
    direct = type("DirectResponse", (), {"content": "provider direct content", "usage": None})()
    assert extract_llm_completion(direct) == ("provider direct content", 0)
    with pytest.raises(ValueError, match="no text content"):
        extract_llm_completion({"choices": []})


def test_runtime_config_uses_environment_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.core_algorithms import resolve_llm_runtime_config

    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    config = resolve_llm_runtime_config(
        {"name": "fallback-agent", "llm_base_url": None, "llm_api_key": None, "llm_model": None}
    )
    assert config.base_url == "https://env.example/v1"
    assert config.api_key == "env-secret"
    assert config.model == "env-model"


def test_mandate_synthesis_prompt_contains_seed_and_scenario() -> None:
    prompt = build_mandate_synthesis_prompt(
        "Revenue Agent",
        "Revenue",
        "Assess a targeted support programme.",
        primary_sources=["UU17_2003_P12"],
        constraints=["Projected deficit must remain <= 3% GDP."],
        owned_checks=["DEFICIT_3PCT"],
        max_deficit_constraint=3.0,
    )
    assert "Act autonomously as the expert Revenue" in prompt
    assert "targeted support programme" in prompt
    assert "scenario_mandate" in prompt
    assert "AUTHORITATIVE DOMAIN CONTRACT" in prompt
    assert "UU17_2003_P12" in prompt
    assert "Projected deficit must remain <= 3% GDP." in prompt
    assert "DEFICIT_3PCT" in prompt
    assert "epistemic_logic_traceability" in prompt
    assert "structured_consensus_protocol" in prompt
    assert "regulatory_compliance_alignment" in prompt
    assert "automatic_deficit_ceiling_percent_gdp" in prompt
    system_prompt = build_mandate_synthesis_system_prompt("Revenue Agent", "Revenue")
    assert "Collective Reasoning in Heterogeneous Multi-Agent Systems" in system_prompt
    assert "auditable epistemic traceability" in system_prompt
    assert "structured disagreement and consensus handling" in system_prompt
    assert "strict regulatory compliance" in system_prompt


def test_build_agent_system_prompt_injects_mandate() -> None:
    prompt = build_agent_system_prompt(
        "State Revenue Agent",
        "Prioritize revenue resilience and evaluate objections.",
    )

    assert "State Revenue Agent" in prompt
    assert "revenue resilience" in prompt
    assert "impacts, risks, uncertainties, objections, conditions, and adjustments" in prompt
    assert "recommendation and confidence" in prompt
    assert "decision-complete response" in prompt
    assert "Additional structured fields are allowed" in prompt
    assert "Do not reveal hidden chain-of-thought" in prompt


def test_reasoning_prompts_require_decision_artifacts_and_peer_review() -> None:
    user_prompt = build_agent_user_prompt(
        "Fiscal Reviewer",
        "Evaluate policy",
        10.0,
        3.0,
    )
    consensus_prompt = build_consensus_prompt(
        "Revenue Agent",
        "Revenue",
        [{"agent": "Risk Agent", "srr": {"risks": []}}],
    )

    assert "decision-complete SRR" in user_prompt
    assert "evidence-backed impact claims" in user_prompt
    assert "Review the structured outputs from every agent" in consensus_prompt
    assert "Preserve valid dissent" in consensus_prompt


def test_native_simulation_agent_is_hardcoded_and_prompted_for_safe_arbitration() -> None:
    native_agent = build_native_simulation_agent()
    system_prompt = build_simulation_system_prompt()
    follow_up_prompt = build_simulation_consensus_prompt(
        "Revenue Agent",
        "Revenue",
        [{"agent": "Risk Agent", "srr": {"predictions": ["lower growth"]}}],
        {"resolution": "Use a phased implementation", "evidence_status": "modelled"},
    )

    assert native_agent.name == SIMULATION_AGENT_NAME
    assert native_agent.llm_base_url is None
    assert "agents" not in native_agent.__dict__
    assert "invokes you automatically" in system_prompt
    assert "Do not reveal hidden chain-of-thought" in system_prompt
    assert "modelled evidence" in follow_up_prompt
    assert "preserve valid dissent" in follow_up_prompt.lower()


def test_deterministic_simulation_filters_hard_constraint_and_tags_outputs() -> None:
    simulation = build_deterministic_simulation(
        "Evaluate a fiscal programme",
        3.0,
        [
            {
                "agent_i": "Fiscal",
                "agent_j": "Risk",
                "components": ["dP", "dREC"],
                "route": "Simulation Agent Requested",
            }
        ],
        [
            {
                "agent": "Fiscal",
                "role": "Fiscal",
                "srr": {
                    "alternatives": [
                        {"name": "Feasible", "deficit": 2.4, "utility": 0.8},
                        {"name": "Illegal", "deficit": 3.5, "utility": 0.99},
                    ],
                    "risks": [{"content": "Execution delay"}],
                    "uncertainties": [{"content": "Demand response"}],
                },
            }
        ],
    )

    assert simulation["agent_name"] == SIMULATION_AGENT_NAME
    assert simulation["evidence_status"] == "modelled"
    assert simulation["alternatives"][0]["name"] == "Feasible"
    assert simulation["alternatives"][0]["source_tag"] == SIMULATION_SOURCE_TAG
    assert all(item["deficit"] <= 3.0 for item in simulation["alternatives"])


def test_simulation_payload_removes_private_reasoning_recursively() -> None:
    payload = sanitize_simulation_payload(
        {
            "resolution": "Use phased implementation",
            "analysis": "private",
            "alternatives": [
                {
                    "name": "Phased",
                    "chain_of_thought": "private",
                    "metadata": {"hidden_reasoning": "private", "status": "modelled"},
                }
            ],
        }
    )

    assert "analysis" not in payload
    alternative = payload["alternatives"][0]
    assert "chain_of_thought" not in alternative
    assert "hidden_reasoning" not in alternative["metadata"]
    assert alternative["metadata"]["status"] == "modelled"


def test_rar_dai_zero_gate() -> None:
    results = calculate_dynamic_influence(
        [make_agent(gate=0), make_agent(gate=1)],
        ["p", "p"],
    )

    assert results[0]["normalized_weight"] == 0.0
    assert results[1]["normalized_weight"] == 1.0


def test_rar_dai_uncertainty_penalty() -> None:
    results = calculate_dynamic_influence(
        [make_agent(uncertainty=0.0), make_agent(uncertainty=2.0)],
        ["p", "p"],
    )

    assert results[1]["raw_score"] < results[0]["raw_score"]
    assert results[1]["normalized_weight"] < results[0]["normalized_weight"]


def test_rar_dai_exact_raw_score() -> None:
    agent = {
        "theta_x": 2.0,
        "theta_q": 3.0,
        "theta_h": 4.0,
        "theta_s": 5.0,
        "theta_u": 6.0,
        "X": 1.5,
        "Q": 2.0,
        "H": 0.5,
        "S": 1.0,
        "U": 0.25,
        "g_i": 1,
    }

    result = calculate_dynamic_influence([agent], ["p"])[0]

    assert result["raw_score"] == 14.5


def test_safe_softmax_handles_large_scores() -> None:
    agents = [make_agent(), make_agent()]
    for agent in agents:
        agent["X"] = 1_000_000.0

    results = calculate_dynamic_influence(agents, ["p", "p"])

    assert all(math.isfinite(result["normalized_weight"]) for result in results)
    assert results[0]["normalized_weight"] == pytest.approx(0.5)


def test_safe_softmax_skips_extreme_inactive_score() -> None:
    active = make_agent()
    inactive = make_agent(gate=0)
    inactive["X"] = 1_000_000.0

    results = calculate_dynamic_influence([active, inactive], ["p", "p"])

    assert results[0]["normalized_weight"] == 1.0
    assert results[1]["normalized_weight"] == 0.0


def test_softmax_sum() -> None:
    agents = [
        make_agent(uncertainty=0.0),
        make_agent(uncertainty=1.0),
        make_agent(uncertainty=2.0, gate=0),
    ]

    results = calculate_dynamic_influence(agents, ["p", "p", "p"])
    active_weight_sum = sum(
        result["normalized_weight"]
        for result in results
        if result["gate"] == 1
    )

    assert active_weight_sum == pytest.approx(1.0)
    assert results[2]["normalized_weight"] == 0.0


def test_car_filtering() -> None:
    alternatives = [Alternative(2.0), Alternative(3.5), Alternative(4.0)]

    feasible = neuro_symbolic_filter(alternatives, HardConstraints(max_deficit=3.0))
    violation_rate = calculate_violation_rate(len(alternatives), len(feasible))

    assert len(feasible) == 1
    assert feasible[0].deficit == 2.0
    assert violation_rate == 66.67


def test_ddr_divergence_vector() -> None:
    obj_i = {
        "E": "same",
        "A": 1,
        "P": True,
        "R": "low",
        "U": 0.1,
        "O": "growth",
        "C": ["deficit"],
        "REC": "increase_tax",
    }
    obj_j = {
        "E": "same",
        "A": 2,
        "P": True,
        "R": "high",
        "U": 0.1,
        "O": "growth",
        "C": ["deficit", "debt"],
        "REC": "reduce_spending",
    }

    assert detect_divergence_vector(obj_i, obj_j) == {
        "dE": False,
        "dA": True,
        "dP": False,
        "dR": True,
        "dU": False,
        "dO": False,
        "dC": True,
        "dREC": True,
    }


def test_violation_rate_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        calculate_violation_rate(2, 3)
