import json
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.celery_client import celery_client
from backend.agent_templates import STANDARD_APBN_AGENT_TEMPLATES, agent_revision
from backend.dashboard import _disagreement_payload
from backend.database import SessionLocal
from backend.main import app
from backend.models import (
    Agent,
    ConsensusSession,
    DisagreementLog,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
    ScenarioMandateSnapshot,
    SimulationArtifact,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_agent_theta_u_zero_persists(client: TestClient) -> None:
    name = f"ablation-a6-{uuid.uuid4()}"
    response = client.post("/api/agents", json={"name": name, "role": "Uncertainty Ablation Agent", "theta_x": 1.0, "theta_q": 1.0, "theta_h": 1.0, "theta_s": 1.0, "theta_u": 0.0})
    assert response.status_code == 201
    with SessionLocal() as session:
        saved = session.scalar(select(Agent).where(Agent.name == name))
        assert saved is not None and saved.theta_u == 0.0
        session.execute(delete(Agent).where(Agent.id == saved.id))
        session.commit()


def test_car_dashboard_headroom_uses_effective_scenario_ceiling() -> None:
    disagreement = DisagreementLog(
        scenario_id=1,
        agent_i=1,
        agent_j=2,
        dP=True,
        detail_payload={
            "fiscal_calculation": {
                "statutory_deficit_ceiling_percent": 3.0,
                "scenario_policy_ceiling_percent": 2.5,
                "effective_deficit_ceiling_percent": 2.5,
            }
        },
    )

    payload = _disagreement_payload(
        disagreement,
        "Fiscal",
        "Risk",
        [],
        {
            "solver_status": "sat",
            "status": "FEASIBLE",
            "selected_alternative": {
                "name": "Bounded",
                "deficit": 2.4,
                "utility": 0.9,
            },
        },
    )

    assert payload["fiscal_calculation"]["selected_compromise"][
        "headroom_percent"
    ] == pytest.approx(0.1)


def test_legacy_scenario_ceiling_remains_serializable(client: TestClient) -> None:
    with SessionLocal() as session:
        scenario = Scenario(
            description=f"Legacy ceiling {uuid.uuid4()}",
            max_deficit_constraint=4.0,
        )
        session.add(scenario)
        session.flush()
        agent = Agent(name=f"legacy-agent-{uuid.uuid4()}", role="Fiscal")
        session.add(agent)
        session.flush()
        revision = agent_revision([agent], scenario)
        snapshot = ScenarioMandateSnapshot(
            scenario_id=scenario.id,
            revision=revision,
            generated=True,
            agent_count=1,
            rules={"automatic_deficit_ceiling": 3.0},
            agent_rules=[
                {
                    "agent_id": agent.id,
                    "name": agent.name,
                    "role": agent.role,
                    "scenario_mandate": "Apply the effective legal ceiling.",
                }
            ],
            status="success",
            generated_count=1,
            failure_count=0,
        )
        session.add(snapshot)
        session.flush()
        run_id = str(uuid.uuid4())
        session.add(
            ConsensusSession(
                id=run_id,
                scenario_id=scenario.id,
                mandate_snapshot_id=snapshot.id,
                mandate_revision=revision,
                mandate_payload={
                    "rules": snapshot.rules,
                    "agent_rules": snapshot.agent_rules,
                },
                status="SUCCEEDED",
                result_payload={},
            )
        )
        session.commit()
        scenario_id = scenario.id
        agent_id = agent.id

    response = client.get("/api/scenarios")

    assert response.status_code == 200
    assert next(item for item in response.json() if item["id"] == scenario_id)[
        "max_deficit_constraint"
    ] == 4.0
    manifest = client.get(f"/api/scenarios/{scenario_id}/manifest?session_id={run_id}")
    assert manifest.status_code == 200
    prompt = next(
        item["user"]
        for item in manifest.json()["prompts"]
        if item["agent_id"] == agent_id
    )
    assert "Automatic legal deficit ceiling: 3.0%" in prompt
    assert "Automatic legal deficit ceiling: 4.0%" not in prompt
    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario_id))
        session.delete(session.get(Agent, agent_id))
        session.commit()


def test_agent_templates_are_available_and_idempotent(client: TestClient) -> None:
    response = client.get("/api/agent-templates")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    templates = response.json()
    assert len(templates) == 5
    assert [item["key"] for item in templates] == [
        "revenue", "expenditure", "financing", "treasury", "macro"
    ]
    assert {item["key"] for item in templates} == {
        template.key for template in STANDARD_APBN_AGENT_TEMPLATES
    }
    assert templates[0]["name"] == "State Revenue Agent / Penerimaan Negara"
    assert "UUD45_P23_23A_31" in templates[0]["primary_sources"]
    assert "VERIFIED_OFFSETS_ONLY" in templates[0]["owned_checks"]
    assert "DECISION PRINCIPLES" in templates[0]["system_prompt"]
    assert all(item["system_prompt"] for item in templates)

    first = client.post("/api/agents/load-templates")
    second = client.post("/api/agents/load-templates")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["total"] == 5
    assert second.json()["created"] == 0

    with SessionLocal() as session:
        template_names = [template.name for template in STANDARD_APBN_AGENT_TEMPLATES]
        template_agents = list(session.scalars(select(Agent).where(Agent.name.in_(template_names))))
        assert len(template_agents) == 5
        assert all(agent.system_prompt for agent in template_agents)
        revenue = next(agent for agent in template_agents if agent.template_key == "revenue")
        assert revenue.system_prompt is not None
        assert "TAX_LEGAL_BASIS" in revenue.system_prompt
        assert "Never invent a tax base" in revenue.system_prompt
        for agent in template_agents:
            session.execute(delete(Agent).where(Agent.id == agent.id))
        session.commit()


def test_template_selection_automates_system_prompt(client: TestClient) -> None:
    name = f"template-derived-{uuid.uuid4()}"
    response = client.post(
        "/api/agents",
        json={
            "name": name,
            "role": "Penerimaan Negara",
            "template_key": "revenue",
        },
    )
    assert response.status_code == 201
    assert "VERIFIED_OFFSETS_ONLY" in response.json()["system_prompt"]
    with SessionLocal() as session:
        saved = session.get(Agent, response.json()["id"])
        assert saved is not None
        assert saved.template_key == "revenue"
        assert saved.system_prompt is not None
        assert "Never invent a tax base" in saved.system_prompt
        session.delete(saved)
        session.commit()


def test_agent_heterogeneous_llm_config_create_and_update(client: TestClient) -> None:
    name = f"heterogeneous-{uuid.uuid4()}"
    response = client.post(
        "/agents",
        json={
            "name": name,
            "role": "State Revenue Agent",
            "llm_base_url": "https://revenue-llm.example/v1",
            "llm_api_key": "revenue-secret",
            "llm_model": "revenue-specialist-v2",
            "system_prompt": "Prioritize sustainable state revenue and tax compliance.",
            "temperature": 0.1,
            "max_tokens": 2500,
        },
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["llm_model"] == "revenue-specialist-v2"
    assert payload["has_llm_api_key"] is True
    assert "llm_api_key" not in payload

    updated = client.put(
        f"/agents/{payload['id']}",
        json={
            "llm_base_url": "https://revenue-llm.example/v2",
            "llm_api_key": "rotated-secret",
            "llm_model": "revenue-specialist-v3",
            "system_prompt": "Evaluate impacts, risks, objections, conditions, and adjustments.",
            "temperature": 0.3,
            "max_tokens": 3200,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["llm_base_url"].endswith("/v2")
    assert updated.json()["temperature"] == 0.3

    with SessionLocal() as session:
        saved = session.get(Agent, payload["id"])
        assert saved is not None
        assert saved.llm_api_key == "rotated-secret"
        assert saved.max_tokens == 3200
        session.delete(saved)
        session.commit()


def test_template_loader_applies_per_agent_llm_configuration(client: TestClient) -> None:
    response = client.post(
        "/api/agents/load-templates",
        json={
            "configs": {
                "revenue": {
                    "llm_base_url": "https://revenue.example/v1",
                    "llm_api_key": "secret",
                    "llm_model": "gpt-4o",
                    "temperature": 0.15,
                    "max_tokens": 2100,
                }
            }
        },
    )
    assert response.status_code == 200
    with SessionLocal() as session:
        agents = list(session.scalars(select(Agent).where(Agent.template_key.is_not(None))))
        revenue = next(agent for agent in agents if agent.template_key == "revenue")
        assert revenue.llm_model == "gpt-4o"
        assert revenue.temperature == 0.15
        for agent in agents:
            session.delete(agent)
        session.commit()


def test_agent_connection_endpoint_handles_success_and_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = client.post(
        "/api/agents",
        json={
            "name": f"connection-{uuid.uuid4()}",
            "role": "Test",
            "llm_base_url": "https://connection.example/v1",
            "llm_api_key": "connection-secret",
            "llm_model": "test-model",
        },
    )
    agent_id = response.json()["id"]

    class FakeCompletions:
        def create(self, **_kwargs: object) -> object:
            message = type("Message", (), {"content": "OK"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("backend.main.OpenAI", FakeClient)
    tested = client.post(f"/api/agents/{agent_id}/test-connection")
    assert tested.status_code == 200
    assert tested.json()["ok"] is True
    assert tested.json()["response_preview"] == "OK"

    with SessionLocal() as session:
        session.delete(session.get(Agent, agent_id))
        session.commit()


def test_agent_can_be_deleted_before_research_records(client: TestClient) -> None:
    response = client.post(
        "/api/agents",
        json={"name": f"delete-{uuid.uuid4()}", "role": "Temporary"},
    )
    assert response.status_code == 201
    deleted = client.delete(f"/api/agents/{response.json()['id']}")
    assert deleted.status_code == 204
    assert client.delete(f"/api/agents/{response.json()['id']}").status_code == 404


def test_domain_rules_aggregate_template_agents(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured["request"] = kwargs
            content = json.dumps(
                {
                    "scenario_mandate": "Evaluate scenario-specific revenue legality and timing.",
                    "scenario_focus": ["Revenue timing"],
                    "priority_questions": ["Is the offset verified?"],
                    "required_evidence": ["Collection schedule"],
                    "epistemic_logic_traceability": ["Tag every material claim with its source and verification state."],
                    "structured_consensus_protocol": ["Escalate unresolved fiscal conflicts through DDR and CAR."],
                    "regulatory_compliance_alignment": ["Enforce the 3% GDP deficit ceiling and reject unverified offsets."],
                }
            )
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice], "usage": None})()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("backend.main.OpenAI", FakeClient)
    template_agent = client.post(
        "/api/agents",
        json={
            "name": f"rules-revenue-{uuid.uuid4()}",
            "role": "Penerimaan Negara",
            "template_key": "revenue",
            "llm_base_url": "https://revenue.example/v1",
            "llm_api_key": "revenue-secret",
            "llm_model": "revenue-model",
        },
    )
    scenario = client.post(
        "/api/scenarios",
        json={"description": "Generate domain rules", "program_cost": 10.0},
    )
    generated = client.post(
        f"/api/scenarios/{scenario.json()['id']}/domain-rules"
    )
    assert generated.status_code == 200
    payload = generated.json()
    assert payload["generated"] is True
    assert payload["stale"] is False
    assert "VERIFIED_OFFSETS_ONLY" in payload["rules"]["owned_checks"]
    assert payload["rules"]["automatic_deficit_ceiling"] == 3.0
    revenue_rules = next(
        item
        for item in payload["agent_rules"]
        if item["agent_id"] == template_agent.json()["id"]
    )
    assert revenue_rules == {
        "agent_id": template_agent.json()["id"],
        "name": template_agent.json()["name"],
        "role": "Penerimaan Negara",
        "template_key": "revenue",
        "mandate": next(
            template.mandate
            for template in STANDARD_APBN_AGENT_TEMPLATES
            if template.key == "revenue"
        ),
        "primary_sources": list(STANDARD_APBN_AGENT_TEMPLATES[0].primary_sources),
        "constraints": list(STANDARD_APBN_AGENT_TEMPLATES[0].constraints),
        "owned_checks": list(STANDARD_APBN_AGENT_TEMPLATES[0].owned_checks),
        "synthesis_status": "generated",
        "scenario_mandate": "Evaluate scenario-specific revenue legality and timing.",
        "scenario_focus": ["Revenue timing"],
        "priority_questions": ["Is the offset verified?"],
        "required_evidence": ["Collection schedule"],
        "epistemic_logic_traceability": [
            "Tag every material claim with its source and verification state."
        ],
        "structured_consensus_protocol": [
            "Escalate unresolved fiscal conflicts through DDR and CAR."
        ],
        "regulatory_compliance_alignment": [
            "Enforce the 3% GDP deficit ceiling and reject unverified offsets."
        ],
        "applicable_primary_sources": list(STANDARD_APBN_AGENT_TEMPLATES[0].primary_sources),
        "applicable_constraints": list(STANDARD_APBN_AGENT_TEMPLATES[0].constraints),
        "applicable_owned_checks": list(STANDARD_APBN_AGENT_TEMPLATES[0].owned_checks),
        "llm_model": "revenue-model",
        "latency_ms": pytest.approx(revenue_rules["latency_ms"]),
        "token_usage": 0,
        "error": None,
    }
    assert captured["client"] == {
        "api_key": "revenue-secret",
        "base_url": "https://revenue.example/v1",
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
    messages = request["messages"]
    assert isinstance(messages, list)
    user_prompt = messages[1]["content"]
    assert isinstance(user_prompt, str)
    assert "UUD45_P23_23A_31" in user_prompt
    assert "Tax/compulsory-levy policy needs proper statutory legal basis." in user_prompt
    assert "VERIFIED_OFFSETS_ONLY" in user_prompt
    assert '"automatic_deficit_ceiling_percent_gdp": 3.0' in user_prompt
    assert "epistemic_logic_traceability" in user_prompt
    assert "structured_consensus_protocol" in user_prompt
    assert "regulatory_compliance_alignment" in user_prompt
    assert payload["status"] == "success"
    assert payload["generated_count"] == 1

    restored = client.get(f"/api/scenarios/{scenario.json()['id']}/domain-rules")
    assert restored.status_code == 200
    assert restored.json() == payload
    dashboard = client.get(f"/api/scenarios/{scenario.json()['id']}/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["domain_rules"] == payload

    with SessionLocal() as session:
        snapshot = session.scalar(
            select(ScenarioMandateSnapshot).where(
                ScenarioMandateSnapshot.scenario_id == scenario.json()["id"]
            )
        )
        assert snapshot is not None
        assert snapshot.agent_rules[0]["scenario_mandate"] == (
            "Evaluate scenario-specific revenue legality and timing."
        )

    updated_agent = client.put(
        f"/api/agents/{template_agent.json()['id']}",
        json={"role": "Updated Revenue Role"},
    )
    assert updated_agent.status_code == 200
    stale = client.get(f"/api/scenarios/{scenario.json()['id']}/domain-rules")
    assert stale.status_code == 200
    assert stale.json()["status"] == "stale"
    assert stale.json()["stale"] is True
    assert stale.json()["agent_rules"][0]["scenario_mandate"] == (
        "Evaluate scenario-specific revenue legality and timing."
    )

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario.json()["id"]))
        session.delete(session.get(Agent, template_agent.json()["id"]))
        session.commit()


def test_domain_rules_uses_environment_fallback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured["request"] = kwargs
            content = json.dumps(
                {
                    "scenario_mandate": "Environment-backed scenario mandate.",
                    "scenario_focus": ["Fallback configuration"],
                    "priority_questions": ["Is the environment provider available?"],
                    "required_evidence": ["Provider response"],
                }
            )
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice], "usage": None})()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    monkeypatch.setattr("backend.main.OpenAI", FakeClient)
    agent = client.post(
        "/api/agents",
        json={"name": f"fallback-rules-{uuid.uuid4()}", "role": "Fallback Test"},
    )
    scenario = client.post(
        "/api/scenarios",
        json={"description": "Environment fallback scenario"},
    )

    response = client.post(f"/api/scenarios/{scenario.json()['id']}/domain-rules")

    assert response.status_code == 200
    assert captured["client"] == {
        "api_key": "environment-secret",
        "base_url": "https://environment.example/v1",
        "timeout": 60.0,
        "max_retries": 2,
        "default_headers": {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        },
    }
    assert response.json()["agent_rules"][0]["llm_model"] == "environment-model"

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario.json()["id"]))
        session.delete(session.get(Agent, agent.json()["id"]))
        session.commit()


def test_domain_rules_accepts_flexible_response_keys(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FlexibleCompletions:
        def create(self, **_kwargs: object) -> object:
            mandate_content = json.dumps(
                {
                    "data": {
                        "Scenario Mandate": {"text": "Autonomous expert mandate."},
                        "Focus Areas": "Fiscal sustainability; Revenue resilience",
                        "Key Questions": [
                            {"question": "Is financing available?"},
                            {"text": "Is the policy lawful?"},
                        ],
                        "Evidence Requirements": "Current APBN baseline\nVerified implementation plan",
                        "Epistemic Traceability": {"traceability": "Source-linked claim ledger"},
                        "Consensus Protocol": "Classify disagreement; Preserve dissent",
                        "Compliance Alignment": [
                            {"alignment": "Enforce the statutory deficit ceiling"}
                        ],
                    }
                }
            )
            content = ": keepalive\n\n" + json.dumps(
                {"choices": [{"message": {"content": mandate_content}}]}
            )
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice], "usage": None})()

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = type("Chat", (), {"completions": FlexibleCompletions()})()

    monkeypatch.setattr("backend.main.OpenAI", FakeClient)
    agent = client.post(
        "/api/agents",
        json={
            "name": f"flexible-response-{uuid.uuid4()}",
            "role": "Revenue Expert",
            "llm_base_url": "https://flexible.example/v1",
            "llm_api_key": "flexible-secret",
            "llm_model": "flexible-model",
        },
    )
    scenario = client.post(
        "/api/scenarios",
        json={"description": "Review fiscal resilience"},
    )

    response = client.post(f"/api/scenarios/{scenario.json()['id']}/domain-rules")

    assert response.status_code == 200
    rule = response.json()["agent_rules"][0]
    assert rule["scenario_mandate"] == "Autonomous expert mandate."
    assert rule["scenario_focus"] == ["Fiscal sustainability", "Revenue resilience"]
    assert rule["priority_questions"] == ["Is financing available?", "Is the policy lawful?"]
    assert rule["required_evidence"] == ["Current APBN baseline", "Verified implementation plan"]
    assert rule["epistemic_logic_traceability"] == ["Source-linked claim ledger"]
    assert rule["structured_consensus_protocol"] == [
        "Classify disagreement",
        "Preserve dissent",
    ]
    assert rule["regulatory_compliance_alignment"] == [
        "Enforce the statutory deficit ceiling"
    ]

    single = client.post(
        f"/api/scenarios/{scenario.json()['id']}/agents/{agent.json()['id']}/domain-rules"
    )
    assert single.status_code == 200
    assert single.json()["scenario_mandate"] == "Autonomous expert mandate."
    restored = client.get(f"/api/scenarios/{scenario.json()['id']}/domain-rules")
    assert restored.json()["agent_rules"][0]["scenario_mandate"] == "Autonomous expert mandate."

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario.json()["id"]))
        session.delete(session.get(Agent, agent.json()["id"]))
        session.commit()


def test_domain_rules_falls_back_when_scenario_mandate_is_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyMandateCompletions:
        def create(self, **_kwargs: object) -> object:
            content = json.dumps(
                {
                    "scenario_mandate": "   ",
                    "scenario_focus": ["Fiscal implementation"],
                    "priority_questions": ["Is implementation feasible?"],
                    "required_evidence": ["Implementation plan"],
                }
            )
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice], "usage": None})()

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = type("Chat", (), {"completions": EmptyMandateCompletions()})()

    monkeypatch.setattr("backend.main.OpenAI", FakeClient)
    agent = client.post(
        "/api/agents",
        json={
            "name": f"empty-mandate-{uuid.uuid4()}",
            "role": "Fiscal Reviewer",
            "llm_base_url": "https://mandate.example/v1",
            "llm_api_key": "mandate-secret",
            "llm_model": "mandate-model",
        },
    )
    scenario = client.post(
        "/api/scenarios",
        json={"description": "Assess targeted APBN assistance"},
    )

    response = client.post(f"/api/scenarios/{scenario.json()['id']}/domain-rules")

    assert response.status_code == 200
    rule = response.json()["agent_rules"][0]
    assert rule["synthesis_status"] == "generated"
    assert rule["scenario_mandate"] == (
        "Evaluate the active APBN policy scenario from the Fiscal Reviewer mandate: "
        "Assess targeted APBN assistance"
    )

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario.json()["id"]))
        session.delete(session.get(Agent, agent.json()["id"]))
        session.commit()


def test_domain_rules_falls_back_for_all_empty_synthesis_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyFieldsCompletions:
        def create(self, **_kwargs: object) -> object:
            message = type("Message", (), {"content": json.dumps({})})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice], "usage": None})()

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = type("Chat", (), {"completions": EmptyFieldsCompletions()})()

    monkeypatch.setattr("backend.main.OpenAI", FakeClient)
    agent = client.post(
        "/api/agents",
        json={
            "name": f"empty-fields-{uuid.uuid4()}",
            "role": "Macro Reviewer",
            "llm_base_url": "https://empty-fields.example/v1",
            "llm_api_key": "empty-fields-secret",
            "llm_model": "empty-fields-model",
        },
    )
    scenario = client.post(
        "/api/scenarios",
        json={"description": "General APBN policy review"},
    )

    response = client.post(f"/api/scenarios/{scenario.json()['id']}/domain-rules")

    assert response.status_code == 200
    rule = response.json()["agent_rules"][0]
    assert rule["scenario_mandate"]
    assert rule["scenario_focus"] == [
        "Kebijakan Fiskal",
        "Optimalisasi Penerimaan Negara",
        "Makroekonomi",
    ]
    assert rule["priority_questions"]
    assert rule["required_evidence"]

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario.json()["id"]))
        session.delete(session.get(Agent, agent.json()["id"]))
        session.commit()


def test_domain_rules_returns_json_when_all_synthesis_calls_fail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingCompletions:
        def create(self, **_kwargs: object) -> object:
            raise RuntimeError("provider unavailable")

    class FailingClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = type("Chat", (), {"completions": FailingCompletions()})()

    monkeypatch.setattr("backend.main.OpenAI", FailingClient)
    agent = client.post(
        "/api/agents",
        json={
            "name": f"failed-rules-{uuid.uuid4()}",
            "role": "Failure Test",
            "llm_base_url": "https://failed.example/v1",
            "llm_api_key": "failed-secret",
            "llm_model": "failed-model",
        },
    )
    scenario = client.post(
        "/api/scenarios",
        json={"description": "Provider failure scenario"},
    )

    response = client.post(f"/api/scenarios/{scenario.json()['id']}/domain-rules")

    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["success"] is False
    assert response.json()["detail"] == "All agent mandate synthesis calls failed"

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario.json()["id"]))
        session.delete(session.get(Agent, agent.json()["id"]))
        session.commit()


def test_domain_rules_include_custom_agent_mandate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCompletions:
        def create(self, **_kwargs: object) -> object:
            content = json.dumps(
                {
                    "scenario_mandate": "Review scenario distributional impacts.",
                    "scenario_focus": ["Beneficiary distribution"],
                    "priority_questions": ["Who bears the cost?"],
                    "required_evidence": ["Distributional data"],
                }
            )
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice], "usage": None})()

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("backend.main.OpenAI", FakeClient)
    custom_agent = client.post(
        "/api/agents",
        json={
            "name": f"rules-custom-{uuid.uuid4()}",
            "role": "Custom Reviewer",
            "system_prompt": "Review distributional effects and cite supplied evidence.",
            "llm_base_url": "https://custom.example/v1",
            "llm_api_key": "custom-secret",
            "llm_model": "custom-model",
        },
    )
    scenario = client.post(
        "/api/scenarios",
        json={"description": "Generate custom mandate", "program_cost": 4.0},
    )

    generated = client.post(f"/api/scenarios/{scenario.json()['id']}/domain-rules")

    assert generated.status_code == 200
    custom_rules = next(
        item
        for item in generated.json()["agent_rules"]
        if item["agent_id"] == custom_agent.json()["id"]
    )
    assert custom_rules["mandate"] == "Review distributional effects and cite supplied evidence."
    assert custom_rules["primary_sources"] == []
    assert custom_rules["constraints"] == []
    assert custom_rules["owned_checks"] == []

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario.json()["id"]))
        session.delete(session.get(Agent, custom_agent.json()["id"]))
        session.commit()


def test_agent_rejects_invalid_generation_settings(client: TestClient) -> None:
    response = client.post(
        "/agents",
        json={
            "name": f"invalid-generation-{uuid.uuid4()}",
            "role": "Invalid",
            "temperature": 2.1,
            "max_tokens": 0,
        },
    )
    assert response.status_code == 422


def test_scenario_uses_automatic_constraint_and_program_cost(client: TestClient) -> None:
    response = client.post(
        "/api/scenarios",
        json={
            "description": "Evaluate a targeted fiscal support programme",
            "program_cost": 125.5,
        },
    )
    assert response.status_code == 201
    scenario_id = response.json()["id"]
    assert response.json()["max_deficit_constraint"] == 3.0
    assert response.json()["program_cost"] == 125.5
    with SessionLocal() as session:
        saved = session.get(Scenario, scenario_id)
        assert saved is not None
        assert saved.max_deficit_constraint == 3.0
        assert saved.program_cost == 125.5
        session.delete(saved)
        session.commit()


def test_agent_rejects_negative_theta(client: TestClient) -> None:
    response = client.post("/api/agents", json={"name": "invalid-agent", "role": "Invalid", "theta_u": -1.0})
    assert response.status_code == 422


def test_scenario_rejects_negative_program_cost(client: TestClient) -> None:
    response = client.post(
        "/api/scenarios",
        json={"description": "Invalid scenario", "program_cost": -0.1},
    )
    assert response.status_code == 422


def test_scenario_cannot_raise_statutory_deficit_ceiling(client: TestClient) -> None:
    response = client.post(
        "/api/scenarios",
        json={"description": "Invalid ceiling", "max_deficit_constraint": 3.1},
    )
    assert response.status_code == 422


def test_run_submission_and_status(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    with SessionLocal() as session:
        scenario = Scenario(description=f"Run test {uuid.uuid4()}", max_deficit_constraint=3.0)
        agents = [
            Agent(
                name=f"run-agent-{uuid.uuid4()}",
                role="Fiscal",
                llm_base_url="https://fiscal.example/v1",
                llm_api_key="fiscal-secret",
                llm_model="fiscal-model",
            ),
            Agent(
                name=f"run-agent-{uuid.uuid4()}",
                role="Risk",
                llm_base_url="https://risk.example/v1",
                llm_api_key="risk-secret",
                llm_model="risk-model",
            ),
        ]
        session.add(scenario)
        session.add_all(agents)
        session.flush()
        revision = agent_revision(agents, scenario)
        session.add(
            ScenarioMandateSnapshot(
                scenario_id=scenario.id,
                revision=revision,
                generated=True,
                agent_count=len(agents),
                rules={},
                agent_rules=[
                    {"agent_id": item.id, "scenario_mandate": f"Mandate for {item.name}"}
                    for item in agents
                ],
                status="success",
                generated_count=len(agents),
                failure_count=0,
            )
        )
        session.commit()
        scenario_id = scenario.id
        agent_ids = [agent.id for agent in agents]

    monkeypatch.setattr(
        celery_client,
        "send_task",
        lambda *_args, **kwargs: type("Task", (), {"id": kwargs["task_id"]})(),
    )
    response = client.post(f"/api/scenarios/{scenario_id}/runs")
    assert response.status_code == 202
    assert response.json()["task_id"]
    task_id = response.json()["task_id"]
    session_id = response.json()["session_id"]

    latest = client.get(f"/api/scenarios/{scenario_id}/runs/latest")
    history = client.get(f"/api/scenarios/{scenario_id}/runs")
    status_response = client.get(f"/api/runs/{task_id}")
    graph_response = client.get(f"/api/runs/{task_id}/graph")

    assert latest.status_code == 200
    assert latest.json()["session_id"] == session_id
    assert latest.json()["status"] == "QUEUED"
    assert latest.json()["progress_stage"] == "QUEUE"
    assert latest.json()["logs"][0]["stage"] == "QUEUE"
    assert history.status_code == 200
    assert history.json()[0]["session_id"] == session_id
    assert status_response.status_code == 200
    assert status_response.json()["session_id"] == session_id
    assert status_response.json()["logs"] == response.json()["logs"]
    assert graph_response.status_code == 200
    graph_payload = graph_response.json()
    assert graph_payload["session_id"] == session_id
    assert {node["id"] for node in graph_payload["nodes"]} >= {
        "scenario:start",
        "stage:mandate",
        "stage:peer-review",
        "stage:rar-dai",
        "stage:ddr",
        "stage:quorum",
        "consensus:final",
    }
    assert len([node for node in graph_payload["nodes"] if node["kind"] == "agent"]) == 2
    assert not any(node["kind"] == "simulation" for node in graph_payload["nodes"])

    with SessionLocal() as session:
        run = session.get(ConsensusSession, session_id)
        assert run is not None
        assert run.scenario_id == scenario_id
        assert run.celery_task_id == task_id
        assert run.logs[0]["event_id"] == response.json()["logs"][0]["event_id"]
        assert run.logs[0]["messages"] == response.json()["logs"][0]["messages"]
        assert run.logs[0]["message"] == run.logs[0]["messages"]["en"]
        assert response.json()["logs"][0]["message"] == response.json()["logs"][0]["messages"]["id"]
        assert run.progress_stage == "QUEUE"
        session.delete(session.get(Scenario, scenario_id))
        session.execute(delete(Agent).where(Agent.id.in_(agent_ids)))
        session.commit()


def test_run_submission_refreshes_stale_mandate_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        scenario = Scenario(
            description=f"Stale run test {uuid.uuid4()}",
            max_deficit_constraint=3.0,
        )
        agents = [
            Agent(
                name=f"stale-run-{uuid.uuid4()}",
                role="Fiscal",
                llm_base_url="https://fiscal.example/v1",
                llm_api_key="fiscal-secret",
                llm_model="fiscal-model",
            ),
            Agent(
                name=f"stale-run-{uuid.uuid4()}",
                role="Risk",
                llm_base_url="https://risk.example/v1",
                llm_api_key="risk-secret",
                llm_model="risk-model",
            ),
        ]
        session.add_all([scenario, *agents])
        session.flush()
        stale_revision = agent_revision(agents, scenario)
        stale_snapshot = ScenarioMandateSnapshot(
            scenario_id=scenario.id,
            revision=stale_revision,
            generated=True,
            agent_count=2,
            rules={},
            agent_rules=[
                {
                    "agent_id": agent.id,
                    "name": agent.name,
                    "role": agent.role,
                    "scenario_mandate": f"Mandate for {agent.name}",
                    "synthesis_status": "generated",
                }
                for agent in agents
            ],
            status="success",
            generated_count=2,
            failure_count=0,
        )
        session.add(stale_snapshot)
        session.flush()
        agents[0].role = "Updated Fiscal"
        session.commit()
        scenario_id = scenario.id
        agent_ids = [agent.id for agent in agents]

    monkeypatch.setattr(
        celery_client,
        "send_task",
        lambda *_args, **kwargs: type("Task", (), {"id": kwargs["task_id"]})(),
    )
    response = client.post(f"/api/scenarios/{scenario_id}/runs")

    assert response.status_code == 202
    with SessionLocal() as session:
        run = session.scalar(
            select(ConsensusSession).where(
                ConsensusSession.id == response.json()["session_id"]
            )
        )
        current_scenario = session.get(Scenario, scenario_id)
        current_agents = list(
            session.scalars(
                select(Agent)
                .where(Agent.id.in_(agent_ids))
                .order_by(Agent.id)
            )
        )
        assert run is not None
        assert current_scenario is not None
        assert run.mandate_revision == agent_revision(current_agents, current_scenario)
        assert run.mandate_revision != stale_revision
        assert run.mandate_payload["agent_rules"][0]["role"] == "Updated Fiscal"
        session.delete(current_scenario)
        session.execute(delete(Agent).where(Agent.id.in_(agent_ids)))
        session.commit()


def test_dashboard_matrix_and_manifest(client: TestClient) -> None:
    with SessionLocal() as session:
        scenario = Scenario(description=f"Dashboard test {uuid.uuid4()}", max_deficit_constraint=3.0)
        agent_i = Agent(name=f"dashboard-i-{uuid.uuid4()}", role="Fiscal")
        agent_j = Agent(name=f"dashboard-j-{uuid.uuid4()}", role="Risk")
        session.add_all([scenario, agent_i, agent_j])
        session.flush()
        revision = agent_revision([agent_i, agent_j], scenario)
        mandate = ScenarioMandateSnapshot(
            scenario_id=scenario.id,
            revision=revision,
            generated=True,
            agent_count=2,
            rules={},
            agent_rules=[
                {"agent_id": agent_i.id, "scenario_mandate": "Fiscal mandate"},
                {"agent_id": agent_j.id, "scenario_mandate": "Risk mandate"},
            ],
            status="success",
            generated_count=2,
            failure_count=0,
        )
        session.add(mandate)
        session.flush()
        run_id = str(uuid.uuid4())
        session.add(
            ConsensusSession(
                id=run_id,
                scenario_id=scenario.id,
                mandate_snapshot_id=mandate.id,
                mandate_revision=revision,
                mandate_payload={"rules": {}, "agent_rules": mandate.agent_rules},
                celery_task_id=f"simulation-dashboard-{run_id}",
                status="SUCCEEDED",
                logs=[
                    {
                        "event_id": "round-1-event",
                        "stage": "SIMULATION_CONSENSUS",
                        "level": "SUCCESS",
                        "message": "Round 1 complete with Authorization: Bearer abcdefghijklmnop",
                        "messages": {
                            "id": "Ronde 1 selesai dengan Authorization: Bearer abcdefghijklmnop",
                            "en": "Round 1 complete with Authorization: Bearer abcdefghijklmnop",
                        },
                        "round_number": 1,
                        "metadata": {"secret_key": "secret", "safe": "visible"},
                    },
                    {
                        "stage": "SIMULATION_CONSENSUS",
                        "level": "SUCCESS",
                        "message": "Round 2 complete",
                        "round_number": 2,
                    },
                ],
                result_payload={
                    "convergence_status": "INFEASIBLE",
                    "feasible_alternatives_count": 0,
                    "car": {
                        "solver": "z3",
                        "solver_status": "unsat",
                        "selected_alternative": None,
                        "rejected_alternatives": [
                            {
                                "name": "Over ceiling",
                                "projected_deficit_percent_gdp": 3.2,
                            }
                        ],
                        "status": "INFEASIBLE",
                    },
                },
            )
        )
        session.add(MetricSnapshot(run_id=run_id, scenario_id=scenario.id, provenance_completeness_percent=75.0, material_information_retention_macro_f1=0.8, hard_constraint_violation_rate=100.0, feasible_alternatives_count=0, convergence_status="INFEASIBLE", latency_ms=120.0, token_usage=500))
        session.add(MetricSnapshot(scenario_id=scenario.id, provenance_completeness_percent=10.0, material_information_retention_macro_f1=0.1, hard_constraint_violation_rate=90.0, feasible_alternatives_count=1, convergence_status="NO_CONSENSUS", latency_ms=50.0, token_usage=100))
        session.add_all([
            ReasoningLog(run_id=run_id, agent_id=agent_i.id, scenario_id=scenario.id, raw_json={"prompt": "i"}, parsed_srr_objects={"evidence": [{"content": "Fiscal baseline"}]}, deliberation_history=[{"stage": "INITIAL", "round_number": 0, "artifacts": {"reasoning_summary": "Prioritize fiscal space", "constraints": [{"content": "Deficit cap"}], "recommendation": {"content": "Use phased financing"}, "confidence": 0.8, "evidence": [{"content": "Fiscal baseline"}], "predictions": [{"content": "Stable deficit"}], "risks": [{"content": "Revenue shortfall"}], "uncertainties": [{"content": "Growth"}], "alternatives": [{"name": "Phased financing", "deficit": 2.5, "utility": 0.8}]}}, {"stage": "PRE_ARBITRATION", "round_number": 0, "artifacts": {"reasoning_summary": "Adopt phased financing before arbitration", "constraints": [{"content": "Deficit cap"}], "recommendation": {"content": "Use phased financing"}, "confidence": 0.82, "evidence": [{"content": "Fiscal baseline"}], "predictions": [{"content": "Stable deficit"}], "risks": [{"content": "Revenue shortfall"}], "uncertainties": [{"content": "Growth"}], "alternatives": [{"name": "Phased financing", "deficit": 2.5, "utility": 0.8}]}}], is_schema_valid=True, provenance_count=2),
            ReasoningLog(run_id=run_id, agent_id=agent_j.id, scenario_id=scenario.id, raw_json={"prompt": "j"}, parsed_srr_objects={"evidence": []}, is_schema_valid=False, provenance_count=0),
        ])
        session.add(DisagreementLog(run_id=run_id, scenario_id=scenario.id, agent_i=agent_i.id, agent_j=agent_j.id, dE=True, dP=True, dREC=False))
        session.add(
            SimulationArtifact(
                run_id=run_id,
                scenario_id=scenario.id,
                trigger="Simulation Agent Requested",
                round_number=1,
                input_payload={"conflicts": [{"components": ["dP"]}]},
                output_payload={
                    "agent_name": "Simulation Agent / Arbiter Simulasi Makro-Fiskal",
                    "resolution": "Use a phased compromise",
                    "evidence_status": "modelled",
                },
                status="SUCCEEDED",
                simulation_version="native-simulation-v1",
                latency_ms=10.0,
                token_usage=20,
            )
        )
        session.commit()
        scenario_id = scenario.id

    dashboard = client.get(
        f"/api/scenarios/{scenario_id}/dashboard?session_id={run_id}&lang=id"
    )
    dashboard_en = client.get(
        f"/api/scenarios/{scenario_id}/dashboard?session_id={run_id}&lang=en"
    )
    run_status_response = client.get(f"/api/runs/simulation-dashboard-{run_id}?lang=id")
    assert run_status_response.status_code == 200
    assert run_status_response.json()["simulation_artifacts"][0]["input"]["conflicts"][0]["components"] == ["dP"]
    assert run_status_response.json()["simulation_artifacts"][0]["output"]["evidence_status"] == "modelled"
    run_agents = run_status_response.json()["agent_breakdown"]
    assert run_agents[0]["agent_opinion"] == "Adopt phased financing before arbitration"
    assert run_agents[0]["recommendation"]["content"] == "Use phased financing"
    assert run_agents[0]["confidence"] == 0.82
    assert run_agents[0]["constraints_considered"][0]["content"] == "Deficit cap"
    assert len(run_agents[0]["deliberation_stages"]) == 2
    run_analysis = run_status_response.json()["collective_reasoning"]
    assert run_analysis["methodology"].startswith("SHCR")
    assert len(run_analysis["claims_and_positions"]) == 2
    assert run_analysis["claims_and_positions"][0]["main_claim"] == (
        "Adopt phased financing before arbitration"
    )
    assert run_analysis["why_and_how"]["divergence_points"]
    recommendation = run_analysis["recommendation_and_follow_up"]
    assert recommendation["final_resolution"] is None
    assert recommendation["proposed_resolution"] == "Use a phased compromise"
    assert recommendation["selected_alternative"] is None
    assert recommendation["status"] == "INFEASIBLE"
    assert recommendation["arbiter"] == "CAR / Z3"
    assert run_analysis["normative_evaluation"]["principles"][0]["principle"] == (
        "Batas defisit statutory"
    )

    analytics_response = client.get(
        f"/api/runs/simulation-dashboard-{run_id}/analytics?lang=en"
    )
    assert analytics_response.status_code == 200
    analytics_payload = analytics_response.json()["collective_reasoning"]
    assert analytics_payload["recommendation_and_follow_up"]["round_number"] == 1
    assert any(
        principle["principle"] == "Unverified revenue offsets"
        for principle in analytics_payload["normative_evaluation"]["principles"]
    )

    ddr_response = client.get(f"/api/runs/simulation-dashboard-{run_id}/ddr?lang=en")
    assert ddr_response.status_code == 200
    ddr_payload = ddr_response.json()["disagreements"][0]
    assert ddr_payload["active_components"] == ["dE", "dP"]
    assert ddr_payload["conflict_categories"][0]["category"] == (
        "Evidence and provenance divergence"
    )
    assert "compromise_formula" in ddr_payload["fiscal_calculation"]
    assert ddr_payload["resolution_detail"]["arbiter_conclusion"] is None
    assert ddr_payload["resolution_detail"]["proposed_resolution"] == (
        "Use a phased compromise"
    )

    graph_response = client.get(f"/api/scenarios/{scenario_id}/runs/{run_id}/graph")
    assert graph_response.status_code == 200
    graph = graph_response.json()
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = {edge["id"]: edge for edge in graph["edges"]}
    assert nodes["simulation:1"]["status"] == "SUCCEEDED"
    assert nodes["simulation:1"]["details"]["artifact_id"] > 0
    assert nodes["simulation:1"]["details"]["output"]["evidence_status"] == "modelled"
    consensus_node = nodes["stage:simulation-consensus:1"]
    assert consensus_node["label"].endswith("Ronde 1")
    assert [log["message"] for log in consensus_node["details"]["logs"]] == [
        "Ronde 1 selesai dengan [REDACTED]"
    ]
    assert consensus_node["details"]["logs"][0]["metadata"] == {"safe": "visible"}
    assert nodes["stage:ddr"]["details"]["disagreement_count"] == 1
    assert nodes["stage:ddr"]["details"]["simulation_trigger_count"] == 1
    assert edges["edge:ddr-simulation:1"]["kind"] == "escalation"
    assert edges["edge:simulation-feedback:1"]["kind"] == "feedback"
    assert edges["edge:simulation-feedback:1"]["target"] == "stage:simulation-consensus:1"
    assert graph["summary"]["simulation_triggered"] is True
    assert "chain_of_thought" not in json.dumps(graph)
    assert "raw_json" not in json.dumps(graph)
    assert dashboard.status_code == 200
    assert dashboard_en.status_code == 200
    assert dashboard.json()["language"] == "id"
    assert dashboard_en.json()["language"] == "en"
    assert dashboard.json()["latest_metric"] == dashboard_en.json()["latest_metric"]
    invalid_language = client.get(
        f"/api/runs/simulation-dashboard-{run_id}/analytics?lang=fr"
    )
    assert invalid_language.status_code == 422
    payload: dict[str, Any] = dashboard.json()
    assert payload["latest_metric"]["convergence_status"] == "INFEASIBLE"
    assert payload["collective_reasoning"]["normative_evaluation"]["principles"][0]["status"] == "BREACH"
    assert payload["schema_validity_percent"] == 50.0
    assert payload["disagreements"][0]["dE"] is True
    assert payload["disagreements"][0]["resolution_mechanism"] == "Simulation Agent Requested"
    assert payload["agent_breakdown"][0]["pre_arbitration"]["agent_opinion"] == "Adopt phased financing before arbitration"
    assert payload["agent_breakdown"][0]["alternatives"][0]["name"] == "Phased financing"
    assert payload["collective_reasoning"]["claims_and_positions"][0]["agent_name"] in {
        agent_i.name,
        agent_j.name,
    }
    assert payload["collective_reasoning"]["why_and_how"]["summary"].startswith(
        "DDR mencatat"
    )
    assert payload["simulation_artifacts"][0]["output"]["evidence_status"] == "modelled"

    manifest = client.get(f"/api/scenarios/{scenario_id}/manifest")
    assert manifest.status_code == 200
    manifest_payload = manifest.json()
    assert manifest_payload["agents"][0]["theta_u"] == 1.0
    assert "llm_api_key" not in manifest_payload["agents"][0]
    assert manifest_payload["llm_outputs"][0]["raw_json"]["prompt"] in {"i", "j"}
    assert manifest_payload["metrics"][0]["convergence_status"] == "INFEASIBLE"
    assert manifest_payload["agent_breakdown"][0]["evidence"][0]["content"] == "Fiscal baseline"
    assert manifest_payload["collective_reasoning"]["normative_evaluation"]["summary"]
    assert manifest_payload["simulation_artifacts"][0]["output"]["resolution"] == "Use a phased compromise"

    with SessionLocal() as session:
        session.execute(delete(DisagreementLog).where(DisagreementLog.scenario_id == scenario_id))
        session.execute(delete(SimulationArtifact).where(SimulationArtifact.scenario_id == scenario_id))
        session.execute(delete(ReasoningLog).where(ReasoningLog.scenario_id == scenario_id))
        session.execute(delete(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id))
        session.execute(delete(Scenario).where(Scenario.id == scenario_id))
        session.execute(delete(Agent).where(Agent.id.in_([agent_i.id, agent_j.id])))
        session.commit()


def test_run_status_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        state = "SUCCESS"
        result = {
            "metric_snapshot_id": 9,
            "token_usage": 42,
            "nested": {
                "apiKey": "secret",
                "client_secret": "secret",
                "x-api-key": "secret",
                "safe": "visible",
            },
            "logs": [
                {
                    "stage": "CONSENSUS",
                    "level": "SUCCESS",
                    "message": "Peer review completed.",
                },
                {
                    "stage": "COMPLETE",
                    "level": "SUCCESS",
                    "message": "SHCR cycle completed.",
                },
            ],
        }
        info = None

        def successful(self) -> bool:
            return True

        def failed(self) -> bool:
            return False

    monkeypatch.setattr("backend.dashboard.AsyncResult", lambda *_args, **_kwargs: FakeResult())
    response = client.get("/api/runs/success-task")
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCEEDED"
    assert response.json()["result"]["metric_snapshot_id"] == 9
    assert response.json()["result"]["token_usage"] == 42
    assert response.json()["result"]["nested"] == {"safe": "visible"}
    assert response.json()["logs"][0]["stage"] == "CONSENSUS"


def test_run_status_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        state = "FAILURE"
        result = ValueError("cycle failed: api_key=secret")
        info = None

        def successful(self) -> bool:
            return False

        def failed(self) -> bool:
            return True

    monkeypatch.setattr("backend.dashboard.AsyncResult", lambda *_args, **_kwargs: FakeResult())
    response = client.get("/api/runs/failed-task")
    assert response.json()["status"] == "FAILED"
    assert response.json()["error"] == "cycle failed"


def test_malformed_stored_mandate_is_normalised_for_api_and_dashboard(
    client: TestClient,
) -> None:
    with SessionLocal() as session:
        scenario = Scenario(
            description=f"Stored mandate compatibility {uuid.uuid4()}",
            max_deficit_constraint=3.0,
        )
        agent = Agent(name=f"stored-mandate-{uuid.uuid4()}", role="Fiscal Reviewer")
        session.add_all([scenario, agent])
        session.flush()
        revision = agent_revision([agent], scenario)
        session.add(
            ScenarioMandateSnapshot(
                scenario_id=scenario.id,
                revision=revision,
                generated=True,
                agent_count=1,
                rules={},
                agent_rules=[
                    {
                        "agent_id": agent.id,
                        "name": agent.name,
                        "role": agent.role,
                        "scenario_focus": "Fiscal sustainability",
                        "priority_questions": None,
                        "error": {"code": "LEGACY", "message": "legacy error"},
                    }
                ],
                status="success",
                generated_count=1,
                failure_count=0,
            )
        )
        session.commit()
        scenario_id = scenario.id
        agent_id = agent.id

    rules = client.get(f"/api/scenarios/{scenario_id}/domain-rules")
    dashboard = client.get(f"/api/scenarios/{scenario_id}/dashboard")

    assert rules.status_code == 200
    assert dashboard.status_code == 200
    stored = rules.json()["agent_rules"][0]
    assert stored["scenario_focus"] == ["Fiscal sustainability"]
    assert stored["priority_questions"] == []
    assert stored["required_evidence"] == []
    assert stored["primary_sources"] == []
    assert stored["epistemic_logic_traceability"] == []
    assert stored["structured_consensus_protocol"] == []
    assert stored["regulatory_compliance_alignment"] == []
    assert stored["error"]["code"] == "SCHEMA_ERROR"
    assert dashboard.json()["domain_rules"]["agent_rules"][0]["scenario_focus"] == [
        "Fiscal sustainability"
    ]

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario_id))
        session.delete(session.get(Agent, agent_id))
        session.commit()


def test_empty_dashboard(client: TestClient) -> None:
    with SessionLocal() as session:
        scenario = Scenario(description=f"Empty dashboard {uuid.uuid4()}", max_deficit_constraint=3.0)
        session.add(scenario)
        session.commit()
        scenario_id = scenario.id
    response = client.get(f"/api/scenarios/{scenario_id}/dashboard")
    assert response.status_code == 200
    assert response.json()["latest_metric"] is None
    assert response.json()["disagreements"] == []
    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario_id))
        session.commit()
