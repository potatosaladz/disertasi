import json
from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from backend.database import SessionLocal
from backend.models import (
    Agent,
    AgentInfluenceObservation,
    ConsensusSession,
    ConvergenceStatus,
    DisagreementLog,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
    SimulationArtifact,
)
from backend.agent_templates import AGENTS, agent_revision
from worker.celery_tasks import (
    _create_isolated_session,
    _default_llm_call,
    _extract_llm_response,
    _load_session_context,
    _determine_convergence,
    _validated_response,
    execute_full_shcr_cycle,
    persist_run_progress,
)
from backend.core_algorithms import resolve_llm_runtime_config
from backend.simulation_agent import SIMULATION_AGENT_NAME, SIMULATION_AGENT_VERSION
from worker.car_solver import evaluate_car_constraints
from worker.srr_models import Evidence, SRRResponse, SimulationResponse


@pytest.fixture
def scenario_id() -> Iterator[int]:
    with SessionLocal() as session:
        scenario = Scenario(
            description="Phase 3 mocked fiscal scenario",
        )
        suffix = uuid4().hex
        agents = [
            Agent(name=f"phase3_fiscal_{suffix}", role="Fiscal Analyst"),
            Agent(name=f"phase3_risk_{suffix}", role="Risk Analyst"),
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
        agent_ids = [agent.id for agent in agents]

    yield identifier

    with SessionLocal() as session:
        session.execute(delete(MetricSnapshot).where(MetricSnapshot.scenario_id == identifier))
        session.execute(delete(SimulationArtifact).where(SimulationArtifact.scenario_id == identifier))
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


def fiscal_payload(
    *,
    name: str,
    prediction: str,
    deficit: float,
    constraint: str,
) -> str:
    return json.dumps(
        {
            "evidence": [{"content": "Budget baseline", "source_tag": "budget-2026"}],
            "predictions": [{"content": prediction, "source_tag": "forecast"}],
            "risks": [{"content": "Fiscal pressure"}],
            "uncertainties": [{"content": "Demand response"}],
            "constraints": [{"content": constraint, "source_tag": "law-17"}],
            "alternatives": [
                {
                    "name": name,
                    "deficit": deficit,
                    "utility": 0.8,
                    "source_tag": "forecast",
                }
            ],
            "recommendation": {"content": f"Assess {name}"},
            "confidence": 0.8,
        }
    )


def test_srr_response_coerces_scalar_lists_and_optional_numeric_strings() -> None:
    parsed = SRRResponse.model_validate(
        {
            "evidence": "Budget baseline",
            "assumptions": "Stable inflation; Stable exchange rate",
            "predictions": "Revenue remains stable",
            "risks": "Implementation delay",
            "uncertainties": "Demand response",
            "objectives": "Fiscal sustainability",
            "constraints": "Deficit ceiling",
            "alternatives": [
                {
                    "name": "Cohort rollout",
                    "deficit": "2.68% PDB",
                    "utility": "utility: 75",
                }
            ],
            "recommendation": "Adopt cohort rollout",
            "confidence": "80%",
            "material_information_retention_macro_f1": "0.91",
        }
    )

    assert [item.content for item in parsed.assumptions] == [
        "Stable inflation",
        "Stable exchange rate",
    ]
    assert parsed.objectives[0].content == "Fiscal sustainability"
    assert parsed.constraints[0].content == "Deficit ceiling"
    assert parsed.alternatives[0].deficit == pytest.approx(2.68)
    assert parsed.alternatives[0].utility == pytest.approx(0.75)
    assert parsed.confidence == pytest.approx(0.8)
    assert parsed.material_information_retention_macro_f1 == pytest.approx(0.91)


def test_srr_response_reports_fallback_metadata_without_fabricating_alternatives() -> None:
    raw_json, parsed, validation_error = _validated_response(
        json.dumps(
            {
                "evidence": ["Verified baseline"],
                "predictions": ["Stable activity"],
                "risks": ["Execution risk"],
                "uncertainties": ["Demand response"],
                "alternatives": [
                    {
                        "name": "Unquantified option",
                        "deficit": "No verified GDP estimate",
                        "utility": "high",
                    }
                ],
                "recommendation": "Wait for verified fiscal data",
                "confidence": "70%",
            }
        )
    )

    assert validation_error is None
    assert parsed is not None
    assert parsed.alternatives == []
    assert parsed.decision_alternatives() == []
    assert "fallback_metadata" not in raw_json
    assert raw_json["alternatives"][0]["name"] == "Unquantified option"
    assert parsed.fallback_metadata is not None
    assert parsed.fallback_metadata["kind"] == "missing_alternatives"
    assert parsed.fallback_metadata["decision_status"] == "evidence_required"
    assert "decision_eligible" not in parsed.fallback_metadata
    assert parsed.model_extra is not None
    assert parsed.model_extra["discarded_alternatives"][0]["name"] == "Unquantified option"


def test_llm_cannot_hide_alternative_from_car_with_eligibility_flag() -> None:
    parsed = SRRResponse.model_validate(
        {
            "alternatives": [
                {
                    "name": "Unsafe proposal",
                    "deficit": 4.0,
                    "utility": 0.9,
                    "decision_eligible": False,
                }
            ],
            "fallback_metadata": {"kind": "provider-controlled"},
        }
    )

    assert len(parsed.alternatives) == 1
    assert parsed.decision_alternatives()[0].name == "Unsafe proposal"
    assert "decision_eligible" not in parsed.alternatives[0].model_dump()
    assert parsed.fallback_metadata is None
    divergence = parsed.divergence_object()
    constraints = divergence["C"]
    assert isinstance(constraints, dict)
    assert constraints["statutory_deficit_violation"] is True
    car = evaluate_car_constraints(parsed.decision_alternatives(), 3.0)
    assert car.feasible == []
    assert car.selected is None
    assert car.rejected[0]["violated_constraints"] == ["DEFICIT_3PCT"]


def test_car_hard_stop_preserves_statutory_rejection_details() -> None:
    alternatives = [
        {"name": "Over ceiling", "deficit": 3.4, "utility": 0.8},
        {"name": "Within ceiling", "deficit": 2.4, "utility": 0.7},
    ]

    evaluation = evaluate_car_constraints(
        alternatives,
        3.0,
        hard_stop_reason="Verified dC conflict",
    )

    over_ceiling, within_ceiling = evaluation.rejected
    assert over_ceiling["violated_constraints"] == [
        "DEFICIT_3PCT",
        "DDR_DC_HARD_STOP",
    ]
    assert over_ceiling["projected_deficit_percent_gdp"] == pytest.approx(3.4)
    assert over_ceiling["ceiling_percent_gdp"] == pytest.approx(3.0)
    assert over_ceiling["excess_percent_gdp"] == pytest.approx(0.4)
    assert within_ceiling["violated_constraints"] == ["DDR_DC_HARD_STOP"]
    assert evaluation.feasible == []
    assert evaluation.selected is None


def test_ambiguous_deficit_text_uses_explicit_total_and_rejects_negated_utility() -> None:
    parsed = SRRResponse.model_validate(
        {
            "alternatives": [
                {
                    "name": "Mixed fiscal values",
                    "deficit": "Tax relief is 2%; projected deficit is 4% of GDP",
                    "utility": 0.8,
                },
                {
                    "name": "Negated utility",
                    "deficit": 2.4,
                    "utility": "not high",
                },
            ]
        }
    )

    assert [item.name for item in parsed.alternatives] == ["Mixed fiscal values"]
    assert parsed.alternatives[0].deficit == pytest.approx(4.0)
    assert parsed.fallback_metadata is None
    assert parsed.model_extra is not None
    discarded = parsed.model_extra["discarded_alternatives"]
    assert discarded[0]["name"] == "Negated utility"


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
    assert first.fallback_metadata is not None
    assert first.fallback_metadata["kind"] == "missing_alternatives"
    assert first.decision_alternatives() == []
    assert first.recommendation is None
    assert first.confidence is None
    assert first.material_information_retention_macro_f1 is None
    assert first.model_extra == {"provider_metadata": {"model": "fiscal-specialist"}}
    assert first.evidence[0].content == "Verified baseline"
    assert second.divergence_object()["REC"] is None


def test_flexible_srr_response_normalises_required_artifacts() -> None:
    parsed = SRRResponse.model_validate(
        {
            "analysis": {
                "required_evidence": "Budget baseline; Audit report",
                "prediction": {"text": "Revenue increases", "source": "forecast"},
                "risk": "Implementation delay",
                "unknowns": [{"description": "Demand response"}],
                "options": [
                    {
                        "title": "Targeted option",
                        "projected_deficit": 2.5,
                        "utility_score": 0.8,
                    }
                ],
                "decision": "Adopt with controls",
                "confidence_score": "75%",
            }
        }
    )

    assert [item.content for item in parsed.evidence] == ["Budget baseline", "Audit report"]
    assert parsed.predictions[0].source_tag == "forecast"
    assert parsed.risks[0].content == "Implementation delay"
    assert parsed.uncertainties[0].content == "Demand response"
    assert parsed.alternatives[0].deficit == 2.5
    assert parsed.alternatives[0].utility == 0.8
    assert parsed.recommendation is not None
    assert parsed.recommendation.content == "Adopt with controls"
    assert parsed.confidence == 0.75


def test_flexible_srr_response_sanitises_formatted_numbers_and_optional_recommendation() -> None:
    parsed = SRRResponse.model_validate(
        {
            "evidence": ["Budget baseline"],
            "predictions": ["Revenue remains stable"],
            "risks": ["Implementation delay"],
            "uncertainties": ["Demand response"],
            "alternatives": [
                {
                    "name": "Cohort rollout",
                    "deficit": "2.68% of GDP (Rp689.368T total)",
                    "utility": "Sangat tinggi; risiko nol terhadap plafon 3% PDB",
                },
                {
                    "name": "Mass rollout",
                    "deficit": "Meningkatkan defisit sekitar 0,04-0,05% PDB",
                    "utility": "Rendah hingga sedang",
                },
                {
                    "name": "Zero budget",
                    "deficit": "Nol",
                    "utility": "0,85",
                },
            ],
            "recommendation": {"decision": "Approve cohort rollout"},
            "confidence": "94%",
        }
    )

    assert [item.name for item in parsed.alternatives] == ["Cohort rollout", "Zero budget"]
    assert parsed.alternatives[0].deficit == pytest.approx(2.68)
    assert parsed.alternatives[0].utility == pytest.approx(0.9)
    assert parsed.alternatives[1].deficit == 0.0
    assert parsed.alternatives[1].utility == pytest.approx(0.85)
    assert parsed.model_extra is not None
    assert parsed.model_extra["discarded_alternatives"][0]["name"] == "Mass rollout"
    assert parsed.recommendation is not None
    assert parsed.recommendation.content == "Approve cohort rollout"
    assert parsed.confidence == pytest.approx(0.94)


def test_incremental_deficit_phrases_are_not_admitted_as_total_deficits() -> None:
    parsed = SRRResponse.model_validate(
        {
            "alternatives": [
                {"name": "Impact", "deficit": "Deficit impact: +0.04% of GDP", "utility": 0.8},
                {"name": "Increase", "deficit": "increase of 0.04% of GDP", "utility": 0.7},
                {"name": "Raised", "deficit": "raises the deficit by 0.04% of GDP", "utility": 0.6},
                {"name": "Rises", "deficit": "deficit rises by 0.04% of GDP", "utility": 0.5},
                {"name": "Total", "deficit": "total projected deficit including additional spending: 2.8% of GDP", "utility": 0.9},
                {"name": "Mixed", "deficit": "Additional deficit impact: 0.04% of GDP; total projected deficit after policy is 2.8% of GDP", "utility": 0.85},
            ]
        }
    )

    assert [item.name for item in parsed.alternatives] == ["Total", "Mixed"]
    assert parsed.alternatives[0].deficit == pytest.approx(2.8)
    assert parsed.alternatives[1].deficit == pytest.approx(2.8)
    assert parsed.model_extra is not None
    assert [item["name"] for item in parsed.model_extra["discarded_alternatives"]] == [
        "Impact",
        "Increase",
        "Raised",
        "Rises",
    ]


def test_optional_empty_recommendation_is_accepted() -> None:
    parsed = SRRResponse.model_validate({"recommendation": {"content": ""}})
    assert parsed.recommendation is None


def test_unverifiable_alternative_is_discarded_without_fabricating_deficit() -> None:
    parsed = SRRResponse.model_validate(
        {
            "evidence": ["Budget baseline"],
            "predictions": ["Stable deficit"],
            "risks": ["Execution risk"],
            "uncertainties": ["Demand response"],
            "alternatives": [
                {"name": "Verified phased", "deficit": "2.4%", "utility": 0.8},
                {
                    "name": "Unquantified option",
                    "deficit": "Material increase without verified GDP ratio",
                    "utility": 0.2,
                },
            ],
            "recommendation": {
                "key_action": "Adopt verified phased implementation",
            },
            "confidence": 0.8,
        }
    )
    assert [item.name for item in parsed.alternatives] == ["Verified phased"]
    assert parsed.recommendation is not None
    assert parsed.recommendation.content == "Adopt verified phased implementation"
    assert parsed.model_extra is not None
    assert parsed.model_extra["discarded_alternatives"][0]["deficit_raw"] == (
        "Material increase without verified GDP ratio"
    )


def test_simulation_response_uses_conflict_fallback_for_empty_llm_value() -> None:
    parsed = SimulationResponse.model_validate(
        {
            "simulation_summary": "Arbitration complete",
            "conflict_summary": [],
            "resolution": "Use a phased compromise",
            "evidence": ["Structured sectoral outputs"],
            "predictions": ["Deficit remains bounded"],
            "risks": ["Implementation delay"],
            "uncertainties": ["Demand response"],
            "alternatives": [{"name": "Phased", "deficit": 2.4, "utility": 0.8}],
            "recommendation": {"content": "Use phased compromise"},
            "confidence": 0.8,
        }
    )
    assert parsed.conflict_summary == [
        "Tidak ada konflik tambahan yang belum diselesaikan pada ronde arbiter akhir."
    ]


def test_simulation_response_uses_summary_fallback_for_empty_llm_value() -> None:
    parsed = SimulationResponse.model_validate(
        {
            "simulation_summary": "   ",
            "conflict_summary": ["Prediction divergence"],
            "resolution": "Use a phased compromise",
            "evidence": ["Structured sectoral outputs"],
            "predictions": ["Deficit remains bounded"],
            "risks": ["Implementation delay"],
            "uncertainties": ["Demand response"],
            "alternatives": [{"name": "Phased", "deficit": 2.4, "utility": 0.8}],
            "recommendation": {"content": "Use phased compromise"},
            "confidence": 0.8,
        }
    )
    assert parsed.simulation_summary == (
        "Simulasi makro-fiskal otomatis diselesaikan oleh arbiter native."
    )


def test_simulation_response_uses_current_service_identity_defaults() -> None:
    parsed = SimulationResponse.model_validate(
        {
            "simulation_summary": "Arbitration complete",
            "conflict_summary": ["Prediction divergence"],
            "resolution": "Use a phased compromise",
            "alternatives": [{"name": "Phased", "deficit": 2.4, "utility": 0.8}],
            "recommendation": {"content": "Use phased compromise"},
        }
    )

    assert parsed.agent_name == SIMULATION_AGENT_NAME
    assert parsed.simulation_version == SIMULATION_AGENT_VERSION


def test_recommendation_verdict_alias_is_normalised() -> None:
    parsed = SRRResponse.model_validate(
        {
            "evidence": ["Budget baseline"],
            "predictions": ["Stable deficit"],
            "risks": ["Yield pressure"],
            "uncertainties": ["Demand response"],
            "alternatives": [{"name": "Cohort", "deficit": 2.0, "utility": 0.8}],
            "recommendation": {
                "status": "APPROVED_WITH_CONDITIONS",
                "verdict": "Approve the cohort rollout",
                "conditions": ["Stay within the legal ceiling"],
            },
            "confidence": 0.9,
        }
    )
    assert parsed.recommendation is not None
    assert parsed.recommendation.content == "Approve the cohort rollout"



def test_validated_response_accepts_formatted_fiscal_deficits() -> None:
    raw_json, parsed, validation_error = _validated_response(
        json.dumps(
            {
                "evidence": ["Budget baseline"],
                "predictions": ["Revenue remains stable"],
                "risks": ["Implementation delay"],
                "uncertainties": ["Demand response"],
                "alternatives": [
                    {
                        "name": "Cohort rollout",
                        "deficit": "2.68% of GDP (Rp689.368T total)",
                        "utility": "Sangat tinggi; risiko nol terhadap plafon 3% PDB",
                    },
                    {
                        "name": "Mass rollout",
                        "deficit": "Rp11 triliun (~0,05% PDB)",
                        "utility": "Sangat rendah",
                    },
                ],
                "recommendation": {"action": "CONDITIONAL_APPROVAL"},
                "confidence": "92%",
            }
        )
    )

    assert validation_error is None
    assert parsed is not None
    assert raw_json["alternatives"][0]["deficit"] == "2.68% of GDP (Rp689.368T total)"
    assert [item.deficit for item in parsed.alternatives] == [
        pytest.approx(2.68),
        pytest.approx(0.05),
    ]
    assert parsed.alternatives[0].utility == pytest.approx(0.9)
    assert parsed.alternatives[1].utility == pytest.approx(0.1)
    assert parsed.recommendation is not None
    assert parsed.recommendation.content == "CONDITIONAL_APPROVAL"
    assert parsed.confidence == pytest.approx(0.92)


def test_default_llm_call_uses_canonical_template_mandate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    )

    _default_llm_call(agent, scenario, "Evaluate the current revenue reform scenario.")

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert AGENTS[0].mandate in messages[0]["content"]
    assert "VERIFIED_OFFSETS_ONLY" in messages[0]["content"]
    assert "Manual text must not override" not in messages[0]["content"]
    assert "Evaluate the current revenue reform scenario." in messages[0]["content"]
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
    scenario = Scenario(description="Evaluate subsidy reform")

    content, tokens = _default_llm_call(agent, scenario)

    assert content == "{}"
    assert tokens == 42
    assert captured["client"] == {
        "api_key": "agent-secret",
        "base_url": "https://expenditure.example/v1",
        "timeout": 60.0,
        "max_retries": 2,
        "default_headers": {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        },
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


def test_worker_session_excludes_specialists_from_other_scenarios(
    scenario_id: int,
) -> None:
    with SessionLocal() as session:
        foreign_scenario = Scenario(
            description="Foreign revenue specialist scope",
        )
        session.add(foreign_scenario)
        session.flush()
        foreign_specialist = Agent(
            name=f"foreign-specialist-{uuid4()}",
            role="Revenue Specialist",
            scenario_id=foreign_scenario.id,
            specialist_domain="revenue",
        )
        session.add(foreign_specialist)
        session.commit()
        foreign_agent_id = foreign_specialist.id

    session_id = _create_isolated_session(scenario_id)
    with SessionLocal() as session:
        run = session.get(ConsensusSession, session_id)
        assert run is not None
        original_ids = {
            item["agent_id"] for item in run.mandate_payload["agent_rules"]
        }
        assert foreign_agent_id not in original_ids
        run.mandate_payload = {
            **run.mandate_payload,
            "agent_rules": [
                *run.mandate_payload["agent_rules"],
                {"agent_id": foreign_agent_id, "scenario_mandate": "Injected"},
            ],
        }
        session.flush()
        _, scoped_agents, _, _ = _load_session_context(
            session,
            scenario_id,
            session_id,
        )
        assert foreign_agent_id not in {agent.id for agent in scoped_agents}


def test_worker_refreshes_snapshot_that_became_stale_after_queue(
    scenario_id: int,
) -> None:
    session_id = _create_isolated_session(scenario_id)
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        run = session.get(ConsensusSession, session_id)
        assert scenario is not None
        assert run is not None
        mandate_ids = [
            item["agent_id"]
            for item in run.mandate_payload["agent_rules"]
            if isinstance(item, dict) and isinstance(item.get("agent_id"), int)
        ]
        agents = list(
            session.scalars(
                select(Agent).where(Agent.id.in_(mandate_ids)).order_by(Agent.id)
            )
        )
        old_revision = agent_revision(agents, scenario)
        agents[0].role = "Updated after queue"
        session.commit()

    with SessionLocal() as session:
        scenario, agents, snapshot, run = _load_session_context(
            session,
            scenario_id,
            session_id,
        )
        assert snapshot.revision == agent_revision(agents, scenario)
        assert snapshot.revision != old_revision
        assert run.mandate_revision == snapshot.revision
        assert run.mandate_payload["agent_rules"][0]["role"] == "Updated after queue"
        session.rollback()


def test_worker_progress_is_persisted(scenario_id: int) -> None:
    session_id = _create_isolated_session(scenario_id)
    progress_logs = [
        {"stage": "INITIALIZE", "level": "INFO", "message": "Starting cycle."},
        {"stage": "SRR", "level": "INFO", "message": "Calling agents."},
    ]

    persist_run_progress(scenario_id, session_id, progress_logs)

    with SessionLocal() as session:
        run = session.get(ConsensusSession, session_id)
        assert run is not None
        assert run.status == "RUNNING"
        assert run.progress_stage == "SRR"
        assert run.started_at is not None
        assert run.logs == progress_logs


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
        enable_simulation=False,
    )

    assert len(peer_batches) == 2
    assert len({item["agent"] for item in peer_batches[0]}) == 2
    assert all(str(item["agent"]).startswith("phase3_") for item in peer_batches[0])
    assert result["token_usage"] == 540
    assert any(log["stage"] == "CONSENSUS" for log in result["logs"])


def test_run_creates_rar_dai_observations_when_no_seed_exists(
    scenario_id: int,
) -> None:
    session_id = _create_isolated_session(scenario_id)
    with SessionLocal() as session:
        session.execute(
            delete(AgentInfluenceObservation).where(
                AgentInfluenceObservation.run_id == session_id
            )
        )
        session.commit()
    first_round = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 10),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 11),
        ]
    )
    reviewed = iter(
        [
            (response_payload(prediction="Growth 1.5%", utility=0.75, recommendation="Adopt C"), 12),
            (response_payload(prediction="Growth 1.5%", utility=0.75, recommendation="Adopt C"), 13),
        ]
    )

    result = execute_full_shcr_cycle(
        scenario_id,
        session_id,
        llm_call=lambda _agent, _scenario: next(first_round),
        consensus_call=lambda _agent, _scenario, _peers: next(reviewed),
        enable_simulation=False,
    )
    codes = [log["code"] for log in result["logs"]]
    assert codes.index("RAR_DAI_CALCULATION_COMPLETED") < codes.index(
        "DDR_BATCH_EVALUATED"
    )
    assert codes.count("DDR_BATCH_EVALUATED") == 1
    assert "DDR_PAIR_EVALUATED" not in codes

    with SessionLocal() as session:
        observations = list(
            session.scalars(
                select(AgentInfluenceObservation)
                .where(AgentInfluenceObservation.run_id == session_id)
                .order_by(AgentInfluenceObservation.agent_id)
            )
        )
        assert len(observations) == 2
        assert sum(item.normalized_weight or 0.0 for item in observations) == pytest.approx(1.0)
        assert all(len(item.interaction_payload) == 1 for item in observations)
        assert all(item.calculation_payload["version"] == "rar-dai-v2" for item in observations)
        for item in observations:
            dimensions = item.calculation_payload["dimensions"]
            assert dimensions["X"]["value"] == pytest.approx(item.X)
            assert dimensions["Q"]["value"] == pytest.approx(item.Q)
            assert dimensions["H"]["value"] == pytest.approx(item.H)
            assert dimensions["S"]["value"] == pytest.approx(item.S)
            assert dimensions["U"]["value"] == pytest.approx(item.U)
        assert all(item.proposition == "Phase 3 mocked fiscal scenario" for item in observations)


def test_ddr_invokes_native_simulation_and_feeds_follow_up_round(
    scenario_id: int,
) -> None:
    result_session_id = _create_isolated_session(scenario_id)
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        assert scenario is not None
        scenario.instrument = "Targeted transfer"
        scenario.duration_months = 6
        scenario.appropriation_available = True
        scenario.verified_sal_available = 15.0
        scenario.growth_outlook = 5.2
        session.commit()
    first_round = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 10),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 11),
        ]
    )
    reviewed = iter(
        [
            (response_payload(prediction="Growth 1.8%", utility=0.75, recommendation="Adopt A"), 12),
            (response_payload(prediction="Growth 1.2%", utility=0.65, recommendation="Adopt B"), 13),
            (response_payload(prediction="Growth 1.6%", utility=0.75, recommendation="Adopt C"), 14),
            (response_payload(prediction="Growth 1.4%", utility=0.75, recommendation="Adopt C"), 15),
            (response_payload(prediction="Growth 1.5%", utility=0.75, recommendation="Adopt C"), 16),
            (response_payload(prediction="Growth 1.5%", utility=0.75, recommendation="Adopt C"), 17),
        ]
    )
    peer_batches: list[list[dict[str, object]]] = []
    simulation_calls: list[list[dict[str, object]]] = []
    live_artifact_states: list[tuple[str, dict[str, object]]] = []

    def consensus_call(
        _agent: Agent,
        _scenario: Scenario,
        peers: list[dict[str, object]],
    ) -> tuple[str, int]:
        peer_batches.append(peers)
        return next(reviewed)

    def simulation_call(
        _scenario: Scenario,
        _conflicts: list[dict[str, object]],
        _peers: list[dict[str, object]],
    ) -> tuple[str, int]:
        with SessionLocal() as live_session:
            live_artifact = live_session.scalar(
                select(SimulationArtifact).where(
                    SimulationArtifact.run_id == result_session_id
                )
            ) if result_session_id else None
            if live_artifact is not None:
                live_artifact_states.append((live_artifact.status, live_artifact.output_payload))
        simulation_calls.append(_conflicts)
        return (
            json.dumps(
                {
                    "evidence": ["Structured sectoral outputs"],
                    "predictions": ["Phased rollout remains within ceiling"],
                    "risks": ["Implementation delay"],
                    "uncertainties": ["Demand response"],
                    "alternatives": [
                        {
                            "name": "Modelled phased compromise",
                            "deficit": 2.4,
                            "utility": 0.8,
                        }
                    ],
                    "recommendation": {"content": "Adopt phased compromise"},
                    "confidence": 0.75,
                    "simulation_summary": "Dissent was modelled in a bounded fiscal sandbox.",
                    "conflict_summary": ["Prediction divergence"],
                    "resolution": "Adopt phased compromise",
                    "modelled_variables": ["deficit"],
                    "limitations": ["Not legal authority"],
                    "evidence_status": "modelled",
                    "agent_name": "obsolete provider identity",
                    "simulation_version": "obsolete-provider-version",
                }
            ),
            7,
        )

    result = execute_full_shcr_cycle(
        scenario_id,
        result_session_id,
        llm_call=lambda _agent, _scenario: next(first_round),
        consensus_call=consensus_call,
        simulation_call=simulation_call,
    )

    assert result["simulation_triggered"] is True
    assert result["simulation_rounds"] == 2
    assert len(simulation_calls) == 2
    assert live_artifact_states[0][0] == "RUNNING"
    assert str(live_artifact_states[0][1]["message"]).startswith("Simulation request accepted")
    assert result["simulation_artifact_id"] is not None
    assert len(peer_batches) == 6
    assert any(item["agent"] == SIMULATION_AGENT_NAME for item in peer_batches[2])
    assert any(log["stage"] == "SIMULATION" for log in result["logs"])
    assert any(log["stage"] == "SIMULATION_RPC" for log in result["logs"])
    assert any(log["code"] == "CAR_RPC_PRECHECK_COMPLETED" for log in result["logs"])
    assert result["car"]["rpc_precheck"]["decision_authority"] is False
    assert result["car"]["rpc_precheck"]["vote_eligible"] is False
    assert result["car"]["rpc_precheck"]["car_eligible"] is False
    assert any(log["stage"] == "SIMULATION_CONSENSUS" for log in result["logs"])
    simulation_consensus_logs = [
        log for log in result["logs"] if log["stage"] == "SIMULATION_CONSENSUS"
    ]
    assert {log["round_number"] for log in simulation_consensus_logs} == {1, 2}
    with SessionLocal() as session:
        artifacts = list(
            session.scalars(
                select(SimulationArtifact).where(
                    SimulationArtifact.run_id == result["session_id"]
                ).order_by(SimulationArtifact.round_number)
            )
        )
        assert len(artifacts) == 2
        assert all(
            artifact.simulation_version == SIMULATION_AGENT_VERSION
            for artifact in artifacts
        )
        assert all(artifact.status == "SUCCEEDED" for artifact in artifacts)
        assert artifacts[-1].output_payload["agent_name"] == SIMULATION_AGENT_NAME
        assert artifacts[-1].output_payload["simulation_version"] == SIMULATION_AGENT_VERSION
        assert artifacts[-1].output_payload["evidence_status"] == "modelled"
        assert len(artifacts[-1].output_payload["rpc_tool_calls"]) == 3
        assert artifacts[-1].output_payload["rpc_interface"] == {
            "protocol_version": "1",
            "agent_name": SIMULATION_AGENT_NAME,
            "vote_eligible": False,
            "decision_authority": False,
            "car_eligible": False,
        }
        assert len(artifacts[-1].input_payload["rpc_tool_calls"]) == 3
        assert artifacts[-1].output_payload["alternatives"][0]["source_tag"] == "SIMULATION_MODELLED"
        assert artifacts[-1].output_payload["remaining_prediction_conflicts"] == 0
        artifact_inputs = [artifact.input_payload for artifact in artifacts]
        assert all("max_deficit_constraint" not in item["scenario"] for item in artifact_inputs)
        assert all(item["scenario"]["instrument"] == "Targeted transfer" for item in artifact_inputs)
        assert all(item["scenario"]["duration_months"] == 6 for item in artifact_inputs)
        assert all(item["scenario"]["appropriation_available"] is True for item in artifact_inputs)
        assert all(item["scenario"]["verified_sal_available"] == 15.0 for item in artifact_inputs)
        assert all(item["scenario"]["growth_outlook"] == 5.2 for item in artifact_inputs)
        assert all(item["scenario"]["statutory_deficit_ceiling_percent"] == 3.0 for item in artifact_inputs)
        native_agent = session.scalar(
            select(Agent).where(Agent.name == SIMULATION_AGENT_NAME)
        )
        assert native_agent is None


def test_native_simulation_falls_back_when_provider_fails(
    scenario_id: int,
) -> None:
    first_round = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 10),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 11),
        ]
    )
    reviewed = iter(
        [
            (response_payload(prediction="Growth 1.8%", utility=0.75, recommendation="Adopt A"), 12),
            (response_payload(prediction="Growth 1.2%", utility=0.65, recommendation="Adopt B"), 13),
            (response_payload(prediction="Growth 1.5%", utility=0.75, recommendation="Adopt C"), 14),
            (response_payload(prediction="Growth 1.5%", utility=0.75, recommendation="Adopt C"), 15),
        ]
    )

    def failing_simulation(
        _scenario: Scenario,
        _conflicts: list[dict[str, object]],
        _peers: list[dict[str, object]],
    ) -> tuple[str, int]:
        raise ConnectionError("simulation provider unavailable")

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(first_round),
        lambda _agent, _scenario, _peers: next(reviewed),
        simulation_call=failing_simulation,
    )

    assert result["metric_snapshot_id"] > 0
    assert result["simulation_rounds"] == 1
    with SessionLocal() as session:
        artifact = session.scalar(
            select(SimulationArtifact).where(
                SimulationArtifact.run_id == result["session_id"]
            )
        )
        assert artifact is not None
        assert artifact.status == "SUCCEEDED"
        assert artifact.output_payload["fallback_reason"] == "ConnectionError"
        assert artifact.output_payload["evidence_status"] == "modelled"


def test_unanimous_deficit_violations_bypass_simulation_and_rpc(
    scenario_id: int,
) -> None:
    responses = iter(
        [
            (
                fiscal_payload(
                    name="Illegal A",
                    prediction="Growth rises",
                    deficit=3.4,
                    constraint="Deficit exceeds the legal ceiling",
                ),
                10,
            ),
            (
                fiscal_payload(
                    name="Illegal B",
                    prediction="Growth falls",
                    deficit=3.5,
                    constraint="Deficit exceeds the legal ceiling",
                ),
                11,
            ),
        ]
    )
    simulation_calls: list[list[dict[str, object]]] = []

    def simulation_call(
        _scenario: Scenario,
        conflicts: list[dict[str, object]],
        _peers: list[dict[str, object]],
    ) -> tuple[str, int]:
        simulation_calls.append(conflicts)
        raise AssertionError("simulation must be bypassed for unanimous deficit violations")

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(responses),
        simulation_call=simulation_call,
    )

    assert simulation_calls == []
    assert result["simulation_triggered"] is False
    assert result["car"]["hard_stop"]["triggered"] is True
    assert result["car"]["rpc_precheck"]["status"] == "bypassed"
    assert result["car"]["solver_status"] == "unsat"
    assert not any(log["stage"] == "SIMULATION_RPC" for log in result["logs"])


def test_invalid_consensus_review_retains_validated_initial_artifacts(
    scenario_id: int,
) -> None:
    responses = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 120),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 130),
        ]
    )

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(responses),
        lambda _agent, _scenario, _peers: ("{}", 10),
        enable_simulation=False,
    )

    consensus_logs = [log for log in result["logs"] if log["stage"] == "CONSENSUS"]
    assert any(log["level"] == "WARNING" for log in consensus_logs)
    assert not any(log["level"] == "ERROR" for log in consensus_logs)
    assert result["metric_snapshot_id"] > 0


def test_convergence_compares_agent_profiles_not_alternatives_within_one_profile() -> None:
    first = SRRResponse.model_validate(
        {
            "alternatives": [
                {"name": "A", "deficit": 2.0, "utility": 0.9},
                {"name": "B", "deficit": 2.5, "utility": 0.7},
            ],
            "recommendation": {"content": "Adopt A"},
        }
    )
    second = SRRResponse.model_validate(first.model_dump(mode="json"))
    alternatives = [*first.alternatives, *second.alternatives]

    assert (
        _determine_convergence([first, second], alternatives, alternatives)
        == ConvergenceStatus.FULL_CONSENSUS
    )

    second.alternatives[0].deficit = 2.2
    alternatives = [*first.alternatives, *second.alternatives]
    assert (
        _determine_convergence([first, second], alternatives, alternatives)
        == ConvergenceStatus.PARETO_SET
    )


def test_verified_dc_bypasses_simulation_and_forces_car_hard_stop(
    scenario_id: int,
) -> None:
    responses = iter(
        [
            (
                fiscal_payload(
                    name="Feasible",
                    prediction="Growth rises",
                    deficit=2.4,
                    constraint="Deficit must remain within the legal ceiling",
                ),
                10,
            ),
            (
                fiscal_payload(
                    name="Illegal",
                    prediction="Growth falls",
                    deficit=3.4,
                    constraint="Deficit may exceed the legal ceiling",
                ),
                11,
            ),
        ]
    )
    simulation_calls: list[list[dict[str, object]]] = []

    def simulation_call(
        _scenario: Scenario,
        conflicts: list[dict[str, object]],
        _peers: list[dict[str, object]],
    ) -> tuple[str, int]:
        simulation_calls.append(conflicts)
        raise AssertionError("simulation must be bypassed for verified dC")

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(responses),
        simulation_call=simulation_call,
    )

    assert simulation_calls == []
    assert result["simulation_triggered"] is False
    assert result["simulation_rounds"] == 0
    assert result["convergence_status"] == "INFEASIBLE"
    assert result["car"]["hard_stop"]["triggered"] is True
    assert result["car"]["hard_stop"]["simulation_bypassed"] is True
    assert result["car"]["rpc_precheck"]["status"] == "bypassed"
    assert not any(log["stage"] == "SIMULATION_RPC" for log in result["logs"])
    assert result["car"]["solver_status"] == "unsat"
    assert result["car"]["selected_alternative"] is None
    assert result["result"]["status"] == "INFEASIBLE"
    assert result["result"]["messages"]["id"] == (
        "Simulasi dibatalkan: Benturan batas keras terdeteksi pada defisit"
    )
    assert not any(log["code"] == "SHCR_CYCLE_FAILED" for log in result["logs"])
    assert sum(log["code"] == "DDR_BATCH_EVALUATED" for log in result["logs"]) == 1
    with SessionLocal() as session:
        disagreement = session.scalar(
            select(DisagreementLog).where(
                DisagreementLog.run_id == result["session_id"]
            )
        )
        assert disagreement is not None
        assert disagreement.dP is True
        assert disagreement.dC is True
        assert disagreement.resolution_route == "Constraint Arbitration Required"
        narrative = disagreement.detail_payload["vector_metadata"]["dC"][
            "narrative_i18n"
        ]
        assert "Dynamic_Fiscal_Analyst" in narrative["id"]
        assert "Dynamic_Risk_Analyst" in narrative["en"]
        assert "Jalur: Constraint Arbitration Required" in narrative["id"]


def test_scenario_cannot_override_statutory_deficit_ceiling(
    scenario_id: int,
) -> None:
    responses = iter(
        [
            (
                fiscal_payload(
                    name="Within statutory ceiling A",
                    prediction="Growth remains stable",
                    deficit=2.7,
                    constraint="Statutory ceiling is binding",
                ),
                10,
            ),
            (
                fiscal_payload(
                    name="Within statutory ceiling B",
                    prediction="Growth remains stable",
                    deficit=2.8,
                    constraint="Statutory ceiling is binding",
                ),
                11,
            ),
        ]
    )

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(responses),
        enable_simulation=False,
    )

    assert result["car"]["hard_stop"]["triggered"] is False
    assert result["car"]["solver_status"] == "sat"
    assert result["feasible_alternatives_count"] == 2
    with SessionLocal() as session:
        disagreement = session.scalar(
            select(DisagreementLog).where(
                DisagreementLog.run_id == result["session_id"]
            )
        )
        assert disagreement is not None
        calculation = disagreement.detail_payload["vector_metadata"]["dC"][
            "calculation"
        ]
        assert calculation["statutory_violation"] is False
        assert "scenario_policy_violation" not in calculation
        assert calculation["violation"] is False


def test_post_simulation_dc_stops_additional_rounds(
    scenario_id: int,
) -> None:
    first_round = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 10),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 11),
        ]
    )
    reviewed = iter(
        [
            (response_payload(prediction="Growth 1.8%", utility=0.75, recommendation="Adopt A"), 12),
            (response_payload(prediction="Growth 1.2%", utility=0.65, recommendation="Adopt B"), 13),
            (
                fiscal_payload(
                    name="Post-simulation feasible",
                    prediction="Growth 1.7%",
                    deficit=2.4,
                    constraint="Legal ceiling is binding",
                ),
                14,
            ),
            (
                fiscal_payload(
                    name="Post-simulation illegal",
                    prediction="Growth 1.1%",
                    deficit=3.4,
                    constraint="Legal ceiling may be exceeded",
                ),
                15,
            ),
        ]
    )
    simulation_calls: list[list[dict[str, object]]] = []

    def simulation_call(
        _scenario: Scenario,
        conflicts: list[dict[str, object]],
        _peers: list[dict[str, object]],
    ) -> tuple[str, int]:
        simulation_calls.append(conflicts)
        return (
            json.dumps(
                {
                    "evidence": ["Structured sectoral outputs"],
                    "predictions": ["Modelled compromise"],
                    "risks": ["Implementation delay"],
                    "uncertainties": ["Demand response"],
                    "alternatives": [
                        {"name": "Modelled compromise", "deficit": 2.4, "utility": 0.8}
                    ],
                    "recommendation": {"content": "Assess compromise"},
                    "confidence": 0.75,
                    "simulation_summary": "Bounded fiscal simulation.",
                    "conflict_summary": ["Prediction divergence"],
                    "resolution": "Assess compromise",
                    "evidence_status": "modelled",
                }
            ),
            7,
        )

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(first_round),
        lambda _agent, _scenario, _peers: next(reviewed),
        simulation_call=simulation_call,
    )

    assert len(simulation_calls) == 1
    assert result["simulation_rounds"] == 1
    assert result["car"]["hard_stop"]["triggered"] is True
    assert result["car"]["solver_status"] == "unsat"
    assert result["car"]["selected_alternative"] is None


def test_fallback_metadata_never_enters_ddr_or_car(
    scenario_id: int,
) -> None:
    def missing_alternatives(prediction: str) -> str:
        return json.dumps(
            {
                "evidence": ["Verified baseline"],
                "predictions": [prediction],
                "risks": ["Delivery risk"],
                "uncertainties": ["Demand response"],
                "recommendation": "Gather verified fiscal data",
                "confidence": "0.7",
            }
        )

    responses = iter(
        [
            (missing_alternatives("Growth is stable"), 10),
            (missing_alternatives("Growth may differ"), 11),
        ]
    )
    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(responses),
        enable_simulation=False,
    )

    assert result["feasible_alternatives_count"] == 0
    assert result["car"]["status"] == "NOT_EVALUATED"
    assert result["car"]["rejected_alternatives"] == []
    assert result["car"]["selected_alternative"] is None
    with SessionLocal() as session:
        logs = list(
            session.scalars(
                select(ReasoningLog).where(
                    ReasoningLog.run_id == result["session_id"]
                )
            )
        )
        assert len(logs) == 2
        assert all(log.is_schema_valid for log in logs)
        for log in logs:
            parsed = SRRResponse.model_validate(log.parsed_srr_objects)
            assert parsed.alternatives == []
            assert parsed.fallback_metadata is not None
            assert parsed.decision_alternatives() == []
        disagreements = list(
            session.scalars(
                select(DisagreementLog).where(
                    DisagreementLog.run_id == result["session_id"]
                )
            )
        )
        assert disagreements
        for disagreement in disagreements:
            deficit_values = disagreement.detail_payload["vector_metadata"]["dP"]["calculation"]
            assert deficit_values.get("agent_i_projected_deficits", []) == []
            assert deficit_values.get("agent_j_projected_deficits", []) == []


def test_three_agent_batch_preserves_every_pairwise_audit_row() -> None:
    with SessionLocal() as session:
        scenario = Scenario(
            description=f"Three-agent audit {uuid4()}",
        )
        agents = [
            Agent(name=f"audit-agent-{index}-{uuid4()}", role=f"Audit {index}")
            for index in range(3)
        ]
        session.add_all([scenario, *agents])
        session.commit()
        scenario_id = scenario.id

    payloads = iter(
        [
            (response_payload(prediction="Stable growth", utility=0.8, recommendation="Maintain"), 1),
            (response_payload(prediction="Stable growth", utility=0.8, recommendation="Maintain"), 1),
            (response_payload(prediction="Lower growth", utility=0.8, recommendation="Maintain"), 1),
        ]
    )
    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(payloads),
        enable_simulation=False,
    )

    with SessionLocal() as session:
        disagreements = list(
            session.scalars(
                select(DisagreementLog)
                .where(DisagreementLog.run_id == result["session_id"])
                .order_by(DisagreementLog.agent_i, DisagreementLog.agent_j)
            )
        )
        assert len(disagreements) == 3
        pairs = {
            (item.agent_i, item.agent_j): item
            for item in disagreements
        }
        assert len(pairs) == 3
        assert sum(
            not any(
                getattr(item, component)
                for component in ("dE", "dA", "dP", "dR", "dU", "dO", "dC", "dREC")
            )
            for item in disagreements
        ) == 1
        events = [
            item
            for item in result["logs"]
            if item.get("code") == "DDR_BATCH_EVALUATED"
        ]
        assert len(events) == 1
        pair_details = events[0]["metadata"]["pairs"]
        assert len(pair_details) == 3
        assert {
            (item["agent_i_id"], item["agent_j_id"])
            for item in pair_details
        } == set(pairs)
        assert events[0]["result"]["active_pair_count"] == 2


def test_all_eight_ddr_components_persist_granular_metadata(
    scenario_id: int,
) -> None:
    left = json.dumps(
        {
            "evidence": [{"content": "Baseline A", "source_tag": "source-a"}],
            "assumptions": [{"content": "Inflation stable"}],
            "predictions": [{"content": "Growth rises"}],
            "risks": [{"content": "Execution delay"}],
            "uncertainties": [{"content": "Demand response"}],
            "objectives": [{"content": "Growth"}],
            "constraints": [{"content": "Deficit ceiling"}],
            "alternatives": [{"name": "Option A", "deficit": 2.5, "utility": 0.8}],
            "recommendation": {"content": "Adopt A"},
            "confidence": 0.9,
        }
    )
    right = json.dumps(
        {
            "evidence": [{"content": "Baseline B", "source_tag": "source-b"}],
            "assumptions": [{"content": "Inflation rises"}],
            "predictions": [{"content": "Growth falls"}],
            "risks": [{"content": "Debt pressure"}],
            "uncertainties": [{"content": "Exchange-rate shock"}],
            "objectives": [{"content": "Stability"}],
            "constraints": [{"content": "Education floor"}],
            "alternatives": [{"name": "Option B", "deficit": 3.5, "utility": 0.6}],
            "recommendation": {"content": "Reject A"},
            "confidence": 0.4,
        }
    )
    responses = iter([(left, 10), (right, 11)])

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(responses),
        enable_simulation=False,
    )

    with SessionLocal() as session:
        disagreement = session.scalar(
            select(DisagreementLog).where(
                DisagreementLog.run_id == result["session_id"]
            )
        )
        assert disagreement is not None
        metadata = disagreement.detail_payload["vector_metadata"]
        assert set(metadata) == {"dE", "dA", "dP", "dR", "dU", "dO", "dC", "dREC"}
        assert set(disagreement.detail_payload["active_components"]) == set(metadata)
        for component, detail in metadata.items():
            assert detail["component"] == component
            assert detail["active"] is True
            assert detail["status"] == "detected"
            assert detail["formula"]
            assert detail["category_i18n"]["id"]
            assert detail["category_i18n"]["en"]
            assert detail["meaning_i18n"]["id"]
            assert detail["economic_impact_i18n"]["en"]
            assert detail["agent_i"]["artifact_hash"]
            assert detail["agent_j"]["artifact_hash"]
            assert detail["calculation"]["status"] == "calculated"
            assert detail["resolution_path"] != "No Resolution Required"
        assert metadata["dE"]["calculation"]["source_variance"] == 1.0
        assert metadata["dP"]["calculation"]["deficit_range_gap_percent_gdp"] == 1.0
        assert metadata["dU"]["calculation"]["confidence_gap"] == 0.5
        assert metadata["dC"]["calculation"]["violation"] is True
        assert metadata["dC"]["calculation"]["solver"] is None


def test_all_infeasible_cycle_cannot_resurrect_fallback_alternative(
    scenario_id: int,
) -> None:
    def payload(name: str, prediction: str, deficit: float) -> str:
        return json.dumps(
            {
                "evidence": [{"content": "Budget baseline", "source_tag": "budget-2026"}],
                "predictions": [{"content": prediction, "source_tag": "forecast"}],
                "risks": [{"content": "Fiscal breach"}],
                "uncertainties": [{"content": "Demand response"}],
                "alternatives": [{"name": name, "deficit": deficit, "utility": 0.8}],
                "recommendation": {"content": "Do not approve"},
                "confidence": 0.9,
            }
        )

    first_round = iter(
        [
            (payload("Over ceiling A", "Growth 2%", 3.2), 10),
            (payload("Over ceiling B", "Growth 1%", 4.0), 11),
        ]
    )
    reviewed = iter(
        [
            (payload("Over ceiling A", "Growth 1.8%", 3.2), 12),
            (payload("Over ceiling B", "Growth 1.2%", 4.0), 13),
            (payload("Over ceiling A", "Growth 1.5%", 3.2), 14),
            (payload("Over ceiling A", "Growth 1.5%", 3.2), 15),
        ]
    )

    def failing_simulation(
        _scenario: Scenario,
        _conflicts: list[dict[str, object]],
        _peers: list[dict[str, object]],
    ) -> tuple[str, int]:
        raise ConnectionError("simulation provider unavailable")

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(first_round),
        lambda _agent, _scenario, _peers: next(reviewed),
        simulation_call=failing_simulation,
    )

    assert result["convergence_status"] == "INFEASIBLE"
    assert result["feasible_alternatives_count"] == 0
    assert result["hard_constraint_violation_rate"] == 100.0
    assert result["car"]["solver_status"] == "unsat"
    assert result["car"]["selected_alternative"] is None
    assert len(result["car"]["rejected_alternatives"]) == 2
    with SessionLocal() as session:
        artifacts = list(
            session.scalars(
                select(SimulationArtifact).where(
                    SimulationArtifact.run_id == result["session_id"]
                )
            )
        )
        assert artifacts == []
    assert result["simulation_triggered"] is False
    assert result["car"]["rpc_precheck"]["status"] == "bypassed"


def test_formulation_dry_run_stops_before_deliberation(scenario_id: int) -> None:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        assert scenario is not None
        scenario.formulation_dry_run_only = True
        session.commit()

    responses = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 120),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 130),
        ]
    )

    def forbidden_consensus(*_args: object, **_kwargs: object) -> tuple[str, int]:
        raise AssertionError("dry-run must not invoke the LLM consensus stage")

    run_id = _create_isolated_session(scenario_id)
    result = execute_full_shcr_cycle(
        scenario_id,
        run_id,
        llm_call=lambda _agent, _scenario: next(responses),
        consensus_call=forbidden_consensus,
    )

    assert result["formulation_dry_run"] is True
    assert result["simulation_triggered"] is False
    assert result["car"]["status"] == "NOT_EVALUATED"
    assert result["convergence_status"] == ConvergenceStatus.INSUFFICIENT_EVIDENCE.value
    assert any(log["code"] == "FORMULATION_DRY_RUN_COMPLETE" for log in result["logs"])
    with SessionLocal() as session:
        run = session.get(ConsensusSession, result["session_id"])
        assert run is not None and run.status == "SUCCEEDED"
        assert list(
            session.scalars(
                select(SimulationArtifact).where(
                    SimulationArtifact.run_id == result["session_id"]
                )
            )
        ) == []


def test_direct_execution_skips_peer_review_but_keeps_car(scenario_id: int) -> None:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        assert scenario is not None
        scenario.skip_llm_formulation = True
        session.commit()

    responses = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 120),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 130),
        ]
    )
    consensus_calls = 0

    def forbidden_consensus(*_args: object, **_kwargs: object) -> tuple[str, int]:
        nonlocal consensus_calls
        consensus_calls += 1
        raise AssertionError("direct execution must skip LLM peer review")

    run_id = _create_isolated_session(scenario_id)
    result = execute_full_shcr_cycle(
        scenario_id,
        run_id,
        llm_call=lambda _agent, _scenario: next(responses),
        consensus_call=forbidden_consensus,
        enable_simulation=False,
    )

    assert consensus_calls == 0
    assert result["car"]["solver_status"] != "not_run"
    assert any(log["code"] == "LLM_FORMULATION_SKIPPED" for log in result["logs"])


def test_single_year_deployment_disables_iterative_simulation(scenario_id: int) -> None:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        assert scenario is not None
        scenario.single_year_deployment = True
        session.commit()

    responses = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 120),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 130),
        ]
    )

    def simulation_call(*_args: object, **_kwargs: object) -> tuple[str, int]:
        raise AssertionError("single-year deployment must not run simulation")

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(responses),
        simulation_call=simulation_call,
    )

    assert result["simulation_rounds"] == 0
    assert result["simulation_triggered"] is False
    assert any(log["code"] == "SINGLE_YEAR_DEPLOYMENT_NO_PHASING" for log in result["logs"])


def test_auto_weight_mode_recomputes_coefficients_per_run(scenario_id: int) -> None:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        assert scenario is not None
        scenario.description = "Tax revenue and inflation stabilisation"
        for agent in session.scalars(select(Agent).where(Agent.name.startswith("phase3_"))):
            agent.theta_x = 0.0
            agent.theta_q = 0.0
            agent.theta_h = 0.0
            agent.theta_s = 0.0
            agent.theta_u = 0.0
            agent.rar_dai_weight_mode = "auto"
        session.commit()

    responses = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 120),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 130),
        ]
    )

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(responses),
        enable_simulation=False,
    )

    with SessionLocal() as session:
        observations = list(
            session.scalars(
                select(AgentInfluenceObservation).where(
                    AgentInfluenceObservation.run_id == result["session_id"]
                )
            )
        )
        assert observations
        for observation in observations:
            assert observation.raw_score is not None
            assert observation.raw_score > 0.0
            assert observation.calculation_payload["weight_mode"] == "auto"
            assert observation.calculation_payload["inputs"]["theta_x"] > 0.0
    assert any(
        log["code"] == "RAR_DAI_AUTO_WEIGHTS_RECOMPUTED" for log in result["logs"]
    )


def test_manual_weight_mode_preserves_coefficients(scenario_id: int) -> None:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        assert scenario is not None
        for agent in session.scalars(select(Agent).where(Agent.name.startswith("phase3_"))):
            agent.theta_x = 1.25
            agent.theta_q = 1.0
            agent.theta_h = 0.5
            agent.theta_s = 1.0
            agent.theta_u = 0.5
            agent.rar_dai_weight_mode = "manual"
        session.commit()

    responses = iter(
        [
            (response_payload(prediction="Growth 2%", utility=0.8, recommendation="Adopt A"), 120),
            (response_payload(prediction="Growth 1%", utility=0.6, recommendation="Adopt B"), 130),
        ]
    )

    result = execute_full_shcr_cycle(
        scenario_id,
        lambda _agent, _scenario: next(responses),
        enable_simulation=False,
    )

    with SessionLocal() as session:
        observations = list(
            session.scalars(
                select(AgentInfluenceObservation).where(
                    AgentInfluenceObservation.run_id == result["session_id"]
                )
            )
        )
        assert observations
        for observation in observations:
            assert observation.calculation_payload["weight_mode"] == "manual"
            assert observation.calculation_payload["inputs"]["theta_x"] == pytest.approx(1.25)
            assert observation.calculation_payload["inputs"]["theta_h"] == pytest.approx(0.5)
    assert not any(
        log["code"] == "RAR_DAI_AUTO_WEIGHTS_RECOMPUTED" for log in result["logs"]
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
                select(ReasoningLog).where(
                    ReasoningLog.scenario_id == scenario_id,
                    ReasoningLog.run_id == result["session_id"],
                )
            )
        )
        assert len(logs) == 2
        assert all(log.is_schema_valid for log in logs)

        disagreement = session.scalar(
                select(DisagreementLog).where(
                    DisagreementLog.scenario_id == scenario_id,
                    DisagreementLog.run_id == result["session_id"],
                )
        )
        assert disagreement is not None
        assert disagreement.dP is True
        assert disagreement.resolution_route == "Simulation Agent Requested"
        assert disagreement.detail_payload["categories"]
        assert disagreement.detail_payload["fiscal_calculation"]["formula"].startswith("headroom_percent")
        assert disagreement.detail_payload["legal_basis"]

        observations = list(
            session.scalars(
                select(AgentInfluenceObservation).where(
                    AgentInfluenceObservation.scenario_id == scenario_id,
                    AgentInfluenceObservation.run_id == result["session_id"],
                )
            )
        )
        assert len(observations) == 2
        assert sum(item.normalized_weight or 0.0 for item in observations) == pytest.approx(1.0)
        assert all(item.interaction_payload for item in observations)
        assert all(item.calculation_payload["version"] == "rar-dai-v2" for item in observations)
        assert all("normalized_weight" in item.calculation_payload for item in observations)

        run = session.get(ConsensusSession, result["session_id"])
        assert run is not None
        assert run.status == "SUCCEEDED"
        assert run.progress_stage == "COMPLETE"
        assert run.started_at is not None
        assert run.completed_at is not None
        assert run.logs[-1]["stage"] == "COMPLETE"
        assert run.result_payload is not None
        assert run.result_payload["metric_snapshot_id"] == snapshot.id
        assert run.result_payload["scenario_id"] == scenario_id
        assert run.result_payload["session_id"] == run.id
        assert run.result_payload["token_usage"] == 250


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
