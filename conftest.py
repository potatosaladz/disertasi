from collections.abc import Iterator

import pytest
from sqlalchemy import delete
from sqlalchemy.engine import make_url

from backend.database import DATABASE_URL, SessionLocal
from backend.init_db import initialize_database
from backend.models import (
    Agent,
    AgentInfluenceObservation,
    ConsensusSession,
    DisagreementLog,
    GlobalLLMConfig,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
    ScenarioMandateSnapshot,
    SimulationArtifact,
)


@pytest.fixture(scope="session", autouse=True)
def initialize_test_database() -> None:
    database_name = make_url(DATABASE_URL).database or ""
    if not database_name.endswith("_test"):
        raise RuntimeError(
            "Pytest requires DATABASE_URL to target a database ending in '_test'"
        )
    initialize_database()


def _clear_database() -> None:
    with SessionLocal() as session:
        for model in (
            SimulationArtifact,
            DisagreementLog,
            ReasoningLog,
            MetricSnapshot,
            AgentInfluenceObservation,
            ConsensusSession,
            ScenarioMandateSnapshot,
            Scenario,
            Agent,
        ):
            session.execute(delete(model))
        config = session.get(GlobalLLMConfig, 1)
        if config is None:
            config = GlobalLLMConfig(id=1)
            session.add(config)
        config.llm_base_url = None
        config.llm_api_key = None
        config.llm_model = None
        config.temperature = 0.2
        config.max_tokens = 4000
        config.apply_to_all = False
        config.revision = 1
        session.commit()


@pytest.fixture(autouse=True)
def isolate_database() -> Iterator[None]:
    _clear_database()
    yield
    _clear_database()
