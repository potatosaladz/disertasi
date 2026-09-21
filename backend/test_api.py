import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.celery_client import celery_client
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


def test_scenario_hard_constraint_persists(client: TestClient) -> None:
    response = client.post("/api/scenarios", json={"description": "Phase 4 API verification scenario", "max_deficit_constraint": 3.25})
    assert response.status_code == 201
    scenario_id = response.json()["id"]
    with SessionLocal() as session:
        saved = session.get(Scenario, scenario_id)
        assert saved is not None and saved.max_deficit_constraint == 3.25
        session.delete(saved)
        session.commit()


def test_agent_rejects_negative_theta(client: TestClient) -> None:
    response = client.post("/api/agents", json={"name": "invalid-agent", "role": "Invalid", "theta_u": -1.0})
    assert response.status_code == 422


def test_scenario_rejects_negative_constraint(client: TestClient) -> None:
    response = client.post("/api/scenarios", json={"description": "Invalid scenario", "max_deficit_constraint": -0.1})
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
    assert "Applying CAR filter..." in response.json()["logs"]


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
