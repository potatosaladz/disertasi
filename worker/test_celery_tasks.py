import json
from collections.abc import Iterator

import pytest
from sqlalchemy import delete, select

from backend.database import SessionLocal
from backend.models import (
    Agent,
    AgentInfluenceObservation,
    DisagreementLog,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
)
from backend.agent_templates import AGENTS
from worker.celery_tasks import _default_llm_call, _extract_llm_response, execute_full_shcr_cycle
from backend.core_algorithms import resolve_llm_runtime_config
from worker.srr_models import Evidence, SRRResponse


@pytest.fixture
def scenario_id() -> Iterator[int]:
    with SessionLocal() as session:
        scenario = Scenario(
            description="Phase 3 mocked fiscal scenario",
            max_deficit_constraint=3.0,
        )
        agents = [
            Agent(name="phase3_fiscal", role="Fiscal Analyst"),
            Agent(name="phase3_risk", role="Risk Analyst"),
        ]
        session.add(scenario)
        session.add_all(agents)
        session.flush()
        session.add_all(
            [
                AgentInfluenceObservation(
                    agent_id=agents[0].id,
                    scenario_id=scenario.id,
                    proposition="Increase public investment",
                    X=1.0,
                    Q=0.9,
                    H=0.8,
                    S=0.7,
                    U=0.1,
                    gate=1,
                ),
                AgentInfluenceObservation(
                    agent_id=agents[1].id,
                    scenario_id=scenario.id,
                    proposition="Increase public investment",
                    X=0.8,
                    Q=0.9,
                    H=0.7,
                    S=0.6,
                    U=0.3,
                    gate=1,
                ),
            ]
        )
        session.commit()
        identifier = scenario.id

    yield identifier

    with SessionLocal() as session:
        agent_ids = list(
            session.scalars(
                select(Agent.id).where(Agent.name.in_(["phase3_fiscal", "phase3_risk"]))
            )
        )
        session.execute(delete(MetricSnapshot).where(MetricSnapshot.scenario_id == identifier))
        session.execute(delete(DisagreementLog).where(DisagreementLog.scenario_id == identifier))
        session.execute(delete(ReasoningLog).where(ReasoningLog.scenario_id == identifier))
        session.execute(
            delete(AgentInfluenceObservation).where(
                AgentInfluenceObservation.scenario_id == identifier
            )
        )
        session.execute(delete(Scenario).where(Scenario.id == identifier))
        if agent_ids:
            session.execute(delete(Agent).where(Agent.id.in_(agent_ids)))
        session.commit()


def response_payload(
    *,
    prediction: str,
    utility: float,
    recommendation: str,
) -> str:
    return json.dumps(
        {
            "evidence": [{"content": "Revenue baseline", "source_tag": "budget-2026"}],
            "assumptions": [{"content": "Stable inflation", "source_tag": None}],
            "predictions": [{"content": prediction, "source_tag": "model-v1"}],
            "risks": [{"content": "Debt pressure", "source_tag": None}],
            "uncertainties": [{"content": "Growth variance", "source_tag": None}],
            "objectives": [{"content": "Fiscal sustainability", "source_tag": "law-17"}],
            "constraints": [{"content": "Deficit cap", "source_tag": "law-17"}],
            "alternatives": [
                {
                    "name": "Targeted stimulus",
                    "deficit": 2.0,
                    "utility": utility,
                    "source_tag": "simulation-1",
                },
                {
                    "name": "Broad stimulus",
                    "deficit": 4.0,
                    "utility": 0.4,
                    "source_tag": None,
                },
            ],
            "recommendation": {"content": recommendation, "source_tag": "analysis-1"},
            "confidence": 0.8,
            "material_information_retention_macro_f1": 0.9,
        }
    )


def test_extract_llm_response_accepts_missing_usage_metadata() -> None:
    assert _extract_llm_response('{"evidence": []}') == ('{"evidence": []}', 0)
    content, tokens = _extract_llm_response({"choices": [{"message": {"content": "{}"}}]})
    assert content == "{}"
    assert tokens == 0


def test_extract_llm_response_rejects_missing_content() -> None:
    with pytest.raises(ValueError, match="no text content"):
        _extract_llm_response({})


def test_sparse_srr_response_uses_defaults_and_allows_extra_fields() -> None:
    first = SRRResponse.model_validate(
        {
            "provider_metadata": {"model": "fiscal-specialist"},
            "evidence": [
                {
                    "content": "Verified baseline",
                    "source_tag": "budget-2026",
                    "relevance": "high",
                }
            ],
        }
    )
    second = SRRResponse.model_validate({})

    first.evidence.append(Evidence(content="Additional evidence"))

    assert second.evidence == []
    assert first.alternatives == []
    assert first.recommendation is None
    assert first.confidence is None
    assert first.material_information_retention_macro_f1 is None
    assert first.model_extra == {"provider_metadata": {"model": "fiscal-specialist"}}
    assert first.evidence[0].content == "Verified baseline"
    assert second.divergence_object()["REC"] is None


def test_default_llm_call_uses_canonical_template_mandate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            message = type("Message", (), {"content": "{}"})()
            choice = type("Choice", (), {"message": message})()
            usage = type("Usage", (), {"total_tokens": 1})()
            return type("Response", (), {"choices": [choice], "usage": usage})()

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("worker.celery_tasks.OpenAI", FakeClient)
    agent = Agent(
        name="Revenue template",
        role="Penerimaan Negara",
        template_key="revenue",
        llm_base_url="https://revenue.example/v1",
        llm_api_key="revenue-key",
        llm_model="revenue-model",
        system_prompt="Manual text must not override the canonical template.",
    )
    scenario = Scenario(
        description="Evaluate revenue reform",
        program_cost=25.0,
        max_deficit_constraint=3.0,
    )

    _default_llm_call(agent, scenario)

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert AGENTS[0].mandate in messages[0]["content"]
    assert "VERIFIED_OFFSETS_ONLY" in messages[0]["content"]
    assert "Manual text must not override" not in messages[0]["content"]
    assert "Program cost: 25.0" in messages[1]["content"]
    assert "Automatic legal deficit ceiling: 3.0%" in messages[1]["content"]


def test_default_llm_call_uses_agent_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured["request"] = kwargs
            message = type("Message", (), {"content": "{}"})()
            choice = type("Choice", (), {"message": message})()
            usage = type("Usage", (), {"total_tokens": 42})()
            return type("Response", (), {"choices": [choice], "usage": usage})()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("worker.celery_tasks.OpenAI", FakeClient)
    agent = Agent(
        name="heterogeneous-test",
        role="Government Expenditure Agent",
        llm_base_url="https://expenditure.example/v1",
        llm_api_key="agent-secret",
        llm_model="fiscal-model-xl",
        system_prompt="Protect spending quality and evaluate fiscal adjustment options.",
        temperature=0.35,
        max_tokens=1800,
    )
    scenario = Scenario(description="Evaluate subsidy reform", max_deficit_constraint=3.0)

    content, tokens = _default_llm_call(agent, scenario)

    assert content == "{}"
    assert tokens == 42
    assert captured["client"] == {
        "api_key": "agent-secret",
        "base_url": "https://expenditure.example/v1",
        "timeout": 60.0,
        "max_retries": 2,
    }
    request = captured["request"]
    assert isinstance(request, dict)
    assert request["model"] == "fiscal-model-xl"
    assert request["temperature"] == 0.35
    assert request["max_tokens"] == 1800
    messages = request["messages"]
    assert isinstance(messages, list)
    assert "Government Expenditure Agent" in messages[0]["content"]
    assert "fiscal adjustment options" in messages[0]["content"]


def test_runtime_configuration_rejects_placeholder_defaults() -> None:
    with pytest.raises(ValueError, match="database LLM configuration"):
        resolve_llm_runtime_config(
            Agent(
                name="unconfigured",
                role="Reviewer",
                llm_base_url="http://localhost/v1",
                llm_api_key="local-llm",
                llm_model="local-model",
            )
        )


def test_consensus_round_reviews_peer_outputs(scenario_id: int) -> None:
    first_round = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 120),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 130),
        ]
    )
    reviewed = iter(
        [
            (response_payload(prediction="Growth 1.5%", utility=0.75, recommendation="Adopt C"), 140),
            (response_payload(prediction="Growth 1.5%", utility=0.75, recommendation="Adopt C"), 150),
        ]
    )
    peer_batches: list[list[dict[str, object]]] = []

    def consensus_call(
        _agent: Agent,
        _scenario: Scenario,
        peers: list[dict[str, object]],
    ) -> tuple[str, int]:
        peer_batches.append(peers)
        return next(reviewed)

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(first_round),
        consensus_call,
    )

    assert len(peer_batches) == 2
    assert {item["agent"] for item in peer_batches[0]} == {"phase3_fiscal", "phase3_risk"}
    assert result["token_usage"] == 540
    assert any(log["stage"] == "CONSENSUS" for log in result["logs"])


def test_run_full_shcr_cycle_populates_postgres(scenario_id: int) -> None:
    responses = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 120),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 130),
        ]
    )

    result = execute_full_shcr_cycle(scenario_id, lambda _agent, _scenario: next(responses))

    assert result["hard_constraint_violation_rate"] == 50.0
    assert result["feasible_alternatives_count"] == 2
    assert result["convergence_status"] == "PARETO_SET"
    assert result["token_usage"] == 250
    assert result["provenance_completeness_percent"] == 60.0

    with SessionLocal() as session:
        snapshot = session.scalar(
            select(MetricSnapshot).where(MetricSnapshot.id == result["metric_snapshot_id"])
        )
        assert snapshot is not None
        assert snapshot.hard_constraint_violation_rate == 50.0
        assert snapshot.material_information_retention_macro_f1 == pytest.approx(0.9)
        assert snapshot.latency_ms >= 0.0

        logs = list(
            session.scalars(
                select(ReasoningLog).where(ReasoningLog.scenario_id == scenario_id)
            )
        )
        assert len(logs) == 2
        assert all(log.is_schema_valid for log in logs)

        disagreement = session.scalar(
            select(DisagreementLog).where(DisagreementLog.scenario_id == scenario_id)
        )
        assert disagreement is not None
        assert disagreement.dP is True
        assert disagreement.resolution_route == "Simulation Agent Requested"

        observations = list(
            session.scalars(
                select(AgentInfluenceObservation).where(
                    AgentInfluenceObservation.scenario_id == scenario_id
                )
            )
        )
        assert sum(item.normalized_weight or 0.0 for item in observations) == pytest.approx(1.0)


def test_sparse_llm_json_fails_deliberation_quorum(scenario_id: int) -> None:
    responses = iter(
        [
            ('{"provider_metadata":{"model":"a"}}', 5),
            ('{"material_information_retention_macro_f1":0.8}', 7),
        ]
    )

    with pytest.raises(RuntimeError, match="quorum failed"):
        execute_full_shcr_cycle(
            scenario_id,
            lambda _agent, _scenario: next(responses),
        )


def test_invalid_llm_json_fails_deliberation_quorum(scenario_id: int) -> None:
    with pytest.raises(RuntimeError, match="quorum failed"):
        execute_full_shcr_cycle(
            scenario_id,
            lambda _agent, _scenario: ("not-json", 5),
        )
