import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.database import SessionLocal
from backend.main import app
from backend.models import Agent, Scenario


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_agent_theta_u_zero_persists(client: TestClient) -> None:
    name = f"ablation-a6-{uuid.uuid4()}"
    response = client.post(
        "/api/agents",
        json={
            "name": name,
            "role": "Uncertainty Ablation Agent",
            "theta_x": 1.0,
            "theta_q": 1.0,
            "theta_h": 1.0,
            "theta_s": 1.0,
            "theta_u": 0.0,
        },
    )

    assert response.status_code == 201
    assert response.json()["theta_u"] == 0.0

    with SessionLocal() as session:
        saved = session.scalar(select(Agent).where(Agent.name == name))
        assert saved is not None
        assert saved.theta_u == 0.0
        session.execute(delete(Agent).where(Agent.id == saved.id))
        session.commit()


def test_scenario_hard_constraint_persists(client: TestClient) -> None:
    response = client.post(
        "/api/scenarios",
        json={
            "description": "Phase 4 API verification scenario",
            "max_deficit_constraint": 3.25,
        },
    )

    assert response.status_code == 201
    scenario_id = response.json()["id"]
    assert response.json()["max_deficit_constraint"] == 3.25

    with SessionLocal() as session:
        saved = session.get(Scenario, scenario_id)
        assert saved is not None
        assert saved.max_deficit_constraint == 3.25
        session.delete(saved)
        session.commit()


def test_agent_rejects_negative_theta(client: TestClient) -> None:
    response = client.post(
        "/api/agents",
        json={
            "name": "invalid-agent",
            "role": "Invalid",
            "theta_u": -1.0,
        },
    )

    assert response.status_code == 422


def test_scenario_rejects_negative_constraint(client: TestClient) -> None:
    response = client.post(
        "/api/scenarios",
        json={"description": "Invalid scenario", "max_deficit_constraint": -0.1},
    )

    assert response.status_code == 422
