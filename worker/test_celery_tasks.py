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
from worker.celery_tasks import execute_full_shcr_cycle


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
