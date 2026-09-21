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
from worker.celery_tasks import _default_llm_call, execute_full_shcr_cycle


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


def test_invalid_llm_json_marks_schema_invalid(scenario_id: int) -> None:
    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: ("not-json", 5),
    )

    assert result["convergence_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["token_usage"] == 10
    assert result["provenance_completeness_percent"] == 0.0

    with SessionLocal() as session:
        logs = list(
            session.scalars(
                select(ReasoningLog).where(ReasoningLog.scenario_id == scenario_id)
            )
        )
        assert len(logs) == 2
        assert all(not log.is_schema_valid for log in logs)
