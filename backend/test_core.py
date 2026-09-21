import math

import pytest

from backend.core_algorithms import (
    Alternative,
    HardConstraints,
    build_agent_user_prompt,
    build_consensus_prompt,
    build_mandate_synthesis_prompt,
    calculate_dynamic_influence,
    build_agent_system_prompt,
    calculate_violation_rate,
    detect_divergence_vector,
    extract_json_object,
    neuro_symbolic_filter,
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
        25.0,
        3.0,
        {"constraints": ["Verified offsets only"], "primary_sources": ["LAW-1"]},
    )
    assert "LOCAL SEED CONTRACT" in prompt
    assert "targeted support programme" in prompt
    assert "Verified offsets only" in prompt
    assert "never invent" in prompt


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
