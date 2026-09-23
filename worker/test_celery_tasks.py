import json
from collections.abc import Iterator

import pytest
from sqlalchemy import delete, select

from backend.database import SessionLocal
from backend.models import (
    Agent,
    AgentInfluenceObservation,
    ConsensusSession,
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
    _validated_response,
    execute_full_shcr_cycle,
    persist_run_progress,
)
from backend.core_algorithms import resolve_llm_runtime_config
from backend.simulation_agent import SIMULATION_AGENT_NAME, SIMULATION_AGENT_VERSION
from worker.srr_models import Evidence, SRRResponse, SimulationResponse


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
                        "deficit_impact": 2.5,
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

    assert parsed.alternatives[0].deficit == pytest.approx(2.68)
    assert parsed.alternatives[0].utility == pytest.approx(0.9)
    assert parsed.alternatives[1].deficit == pytest.approx(0.05)
    assert parsed.alternatives[1].utility == pytest.approx(0.4)
    assert parsed.alternatives[2].deficit == 0.0
    assert parsed.alternatives[2].utility == pytest.approx(0.85)
    assert parsed.recommendation is not None
    assert parsed.recommendation.content == "Approve cohort rollout"
    assert parsed.confidence == pytest.approx(0.94)


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
        max_deficit_constraint=3.0,
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
    scenario = Scenario(description="Evaluate subsidy reform", max_deficit_constraint=3.0)

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


def test_worker_refreshes_snapshot_that_became_stale_after_queue(
    scenario_id: int,
) -> None:
    session_id = _create_isolated_session(scenario_id)
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        agents = list(session.scalars(select(Agent).order_by(Agent.id)))
        assert scenario is not None
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
    assert {item["agent"] for item in peer_batches[0]} == {"phase3_fiscal", "phase3_risk"}
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

    execute_full_shcr_cycle(
        scenario_id,
        session_id,
        llm_call=lambda _agent, _scenario: next(first_round),
        consensus_call=lambda _agent, _scenario, _peers: next(reviewed),
        enable_simulation=False,
    )

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
        assert all(item.calculation_payload["version"] == "rar-dai-v1" for item in observations)
        assert all(item.proposition == "Phase 3 mocked fiscal scenario" for item in observations)


def test_ddr_invokes_native_simulation_and_feeds_follow_up_round(
    scenario_id: int,
) -> None:
    result_session_id = _create_isolated_session(scenario_id)
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
    assert any(log["stage"] == "SIMULATION_CONSENSUS" for log in result["logs"])
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
        assert artifacts[-1].output_payload["evidence_status"] == "modelled"
        assert artifacts[-1].output_payload["alternatives"][0]["source_tag"] == "SIMULATION_MODELLED"
        assert artifacts[-1].output_payload["remaining_prediction_conflicts"] == 0
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
        assert artifact.output_payload["fallback_reason"].startswith("ConnectionError")
        assert artifact.output_payload["evidence_status"] == "modelled"


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
        assert all(item.calculation_payload["version"] == "rar-dai-v1" for item in observations)
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
