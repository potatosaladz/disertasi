from sqlalchemy import inspect, text

from .database import engine
from .models import Agent

_AGENT_COLUMNS: dict[str, str] = {
    "llm_base_url": "VARCHAR(2048)",
    "llm_api_key": "VARCHAR(4096)",
    "llm_model": "VARCHAR(255)",
    "system_prompt": "TEXT",
    "temperature": "DOUBLE PRECISION NOT NULL DEFAULT 0.2",
    "max_tokens": "INTEGER NOT NULL DEFAULT 4000",
}


def _upgrade_agents_table() -> None:
    inspector = inspect(engine)
    if "agents" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("agents")}
    with engine.begin() as connection:
        for name, definition in _AGENT_COLUMNS.items():
            if name not in existing:
                connection.execute(text(f'ALTER TABLE agents ADD COLUMN "{name}" {definition}'))
        connection.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'ck_agent_temperature_range'
                    ) THEN
                        ALTER TABLE agents ADD CONSTRAINT ck_agent_temperature_range
                            CHECK (temperature >= 0 AND temperature <= 2);
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'ck_agent_max_tokens_positive'
                    ) THEN
                        ALTER TABLE agents ADD CONSTRAINT ck_agent_max_tokens_positive
                            CHECK (max_tokens > 0);
                    END IF;
                END $$;
                """
            )
        )


def initialize_database() -> None:
    Agent.metadata.create_all(bind=engine)
    _upgrade_agents_table()


if __name__ == "__main__":
    initialize_database()
