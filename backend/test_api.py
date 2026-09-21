import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.celery_client import celery_client
from backend.agent_templates import STANDARD_APBN_AGENT_TEMPLATES
from backend.database import SessionLocal
from backend.main import app
from backend.models import Agent, DisagreementLog, MetricSnapshot, ReasoningLog, Scenario


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


def test_agent_can_be_deleted_before_research_records(client: TestClient) -> None:
    response = client.post(
        "/api/agents",
        json={"name": f"delete-{uuid.uuid4()}", "role": "Temporary"},
    )
    assert response.status_code == 201
    deleted = client.delete(f"/api/agents/{response.json()['id']}")
    assert deleted.status_code == 204
    assert client.delete(f"/api/agents/{response.json()['id']}").status_code == 404


def test_domain_rules_aggregate_template_agents(client: TestClient) -> None:
    template_agent = client.post(
        "/api/agents",
        json={
            "name": f"rules-revenue-{uuid.uuid4()}",
            "role": "Penerimaan Negara",
            "template_key": "revenue",
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

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario.json()["id"]))
        session.delete(session.get(Agent, template_agent.json()["id"]))
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


def test_run_submission_and_status(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    with SessionLocal() as session:
        scenario = Scenario(description=f"Run test {uuid.uuid4()}", max_deficit_constraint=3.0)
        session.add(scenario)
        session.commit()
        scenario_id = scenario.id

    class FakeTask:
        id = "task-phase5"

    monkeypatch.setattr(celery_client, "send_task", lambda *_args, **_kwargs: FakeTask())
    response = client.post(f"/api/scenarios/{scenario_id}/runs")
    assert response.status_code == 202
    assert response.json()["task_id"] == "task-phase5"

    with SessionLocal() as session:
        session.delete(session.get(Scenario, scenario_id))
        session.commit()


def test_dashboard_matrix_and_manifest(client: TestClient) -> None:
    with SessionLocal() as session:
        scenario = Scenario(description=f"Dashboard test {uuid.uuid4()}", max_deficit_constraint=3.0)
        agent_i = Agent(name=f"dashboard-i-{uuid.uuid4()}", role="Fiscal")
        agent_j = Agent(name=f"dashboard-j-{uuid.uuid4()}", role="Risk")
        session.add_all([scenario, agent_i, agent_j])
        session.flush()
        session.add(MetricSnapshot(scenario_id=scenario.id, provenance_completeness_percent=75.0, material_information_retention_macro_f1=0.8, hard_constraint_violation_rate=20.0, feasible_alternatives_count=4, convergence_status="INFEASIBLE", latency_ms=120.0, token_usage=500))
        session.add_all([
            ReasoningLog(agent_id=agent_i.id, scenario_id=scenario.id, raw_json={"prompt": "i"}, parsed_srr_objects={"evidence": []}, is_schema_valid=True, provenance_count=2),
            ReasoningLog(agent_id=agent_j.id, scenario_id=scenario.id, raw_json={"prompt": "j"}, parsed_srr_objects={"evidence": []}, is_schema_valid=False, provenance_count=0),
        ])
        session.add(DisagreementLog(scenario_id=scenario.id, agent_i=agent_i.id, agent_j=agent_j.id, dE=True, dP=True, dREC=False))
        session.commit()
        scenario_id = scenario.id

    dashboard = client.get(f"/api/scenarios/{scenario_id}/dashboard")
    assert dashboard.status_code == 200
    payload: dict[str, Any] = dashboard.json()
    assert payload["latest_metric"]["convergence_status"] == "INFEASIBLE"
    assert payload["schema_validity_percent"] == 50.0
    assert payload["disagreements"][0]["dE"] is True
    assert payload["disagreements"][0]["resolution_mechanism"] == "Provenance Retrieval Triggered"

    manifest = client.get(f"/api/scenarios/{scenario_id}/manifest")
    assert manifest.status_code == 200
    manifest_payload = manifest.json()
    assert manifest_payload["agents"][0]["theta_u"] == 1.0
    assert "llm_api_key" not in manifest_payload["agents"][0]
    assert manifest_payload["llm_outputs"][0]["raw_json"]["prompt"] in {"i", "j"}
    assert manifest_payload["metrics"][0]["convergence_status"] == "INFEASIBLE"

    with SessionLocal() as session:
        session.execute(delete(DisagreementLog).where(DisagreementLog.scenario_id == scenario_id))
        session.execute(delete(ReasoningLog).where(ReasoningLog.scenario_id == scenario_id))
        session.execute(delete(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id))
        session.execute(delete(Scenario).where(Scenario.id == scenario_id))
        session.execute(delete(Agent).where(Agent.id.in_([agent_i.id, agent_j.id])))
        session.commit()


def test_run_status_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        state = "SUCCESS"
        result = {"metric_snapshot_id": 9}
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
    assert any(log["stage"] == "CAR" for log in response.json()["logs"])


def test_run_status_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        state = "FAILURE"
        result = ValueError("cycle failed")
        info = None

        def successful(self) -> bool:
            return False

        def failed(self) -> bool:
            return True

    monkeypatch.setattr("backend.dashboard.AsyncResult", lambda *_args, **_kwargs: FakeResult())
    response = client.get("/api/runs/failed-task")
    assert response.json()["status"] == "FAILED"
    assert response.json()["error"] == "cycle failed"


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
