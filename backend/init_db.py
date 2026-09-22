from sqlalchemy import inspect, text

from .database import engine
from .models import Agent

_SCENARIO_COLUMNS: dict[str, str] = {
    "program_cost": "DOUBLE PRECISION",
}

_AGENT_COLUMNS: dict[str, str] = {
    "llm_base_url": "VARCHAR(2048)",
    "template_key": "VARCHAR(100) UNIQUE",
    "llm_api_key": "VARCHAR(4096)",
    "llm_model": "VARCHAR(255)",
    "system_prompt": "TEXT",
    "temperature": "DOUBLE PRECISION NOT NULL DEFAULT 0.2",
    "max_tokens": "INTEGER NOT NULL DEFAULT 4000",
}

_SESSION_COLUMNS: dict[str, str] = {
    "reasoning_logs": 'VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE',
    "disagreement_logs": 'VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE',
    "metric_snapshots": 'VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE',
    "agent_influence_observations": 'VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE',
}


def _upgrade_session_columns() -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        for table, definition in _SESSION_COLUMNS.items():
            if table not in existing_tables:
                continue
            columns = {column["name"] for column in inspector.get_columns(table)}
            if "run_id" not in columns:
                connection.execute(text(f'ALTER TABLE {table} ADD COLUMN run_id {definition}'))
            connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_run_id ON {table} (run_id)"))


def _upgrade_agents_table() -> None:
    inspector = inspect(engine)
    if "agents" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("agents")}
    with engine.begin() as connection:
        for name, definition in _AGENT_COLUMNS.items():
            if name not in existing:
                connection.execute(text(f'ALTER TABLE agents ADD COLUMN "{name}" {definition}'))
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_agents_template_key "
            "ON agents (template_key) WHERE template_key IS NOT NULL"
        ))
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


def _upgrade_scenarios_table() -> None:
    inspector = inspect(engine)
    if "scenarios" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("scenarios")}
    with engine.begin() as connection:
        for name, definition in _SCENARIO_COLUMNS.items():
            if name not in existing:
                connection.execute(text(f'ALTER TABLE scenarios ADD COLUMN "{name}" {definition}'))
        connection.execute(
            text("ALTER TABLE scenarios ALTER COLUMN max_deficit_constraint SET DEFAULT 3.0")
        )
        connection.execute(
            text(
                "ALTER TABLE scenarios DROP CONSTRAINT IF EXISTS "
                "ck_scenario_program_cost_nonnegative"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE scenarios ADD CONSTRAINT ck_scenario_program_cost_nonnegative "
                "CHECK (program_cost IS NULL OR program_cost >= 0)"
            )
        )


def initialize_database() -> None:
    Agent.metadata.create_all(bind=engine)
    _upgrade_agents_table()
    _upgrade_scenarios_table()
    _upgrade_session_columns()


if __name__ == "__main__":
    initialize_database()
