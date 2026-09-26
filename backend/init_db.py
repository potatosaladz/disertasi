from sqlalchemy import Connection, inspect, text

from .database import engine
from .models import Agent

_SCENARIO_COLUMNS: dict[str, str] = {
    "instrument": "VARCHAR(255)",
    "targeting": "TEXT",
    "program_cost": "DOUBLE PRECISION",
    "duration_months": "INTEGER",
    "evaluation_trigger": "TEXT",
    "program_cost_period": "VARCHAR(100)",
    "no_phase0": "BOOLEAN",
    "phase0_only": "BOOLEAN",
    "no_phased": "BOOLEAN",
    "proposed_reallocation": "DOUBLE PRECISION",
    "reallocation_from_education": "BOOLEAN",
    "proposed_additional_revenue": "DOUBLE PRECISION",
    "revenue_measure_type": "VARCHAR(255)",
    "proposed_debt_financing": "DOUBLE PRECISION",
    "debt_financing_mode": "VARCHAR(255)",
    "proposed_sal_use": "DOUBLE PRECISION",
    "sal_purpose": "TEXT",
    "proposed_other_financing": "DOUBLE PRECISION",
    "appropriation_available": "BOOLEAN",
    "verified_reallocation_capacity": "DOUBLE PRECISION",
    "verified_revenue_offset_capacity": "DOUBLE PRECISION",
    "verified_debt_financing_headroom": "DOUBLE PRECISION",
    "verified_sal_available": "DOUBLE PRECISION",
    "verified_operational_cash_minimum": "DOUBLE PRECISION",
    "verified_projected_cash_after_policy": "DOUBLE PRECISION",
    "verified_cumulative_borrowing_pct_gdp": "DOUBLE PRECISION",
    "spending_reallocation_authorized": "BOOLEAN",
    "dpr_spending_adjustment_recommendation": "BOOLEAN",
    "finance_minister_sal_authorized": "BOOLEAN",
    "dpr_sal_approval_obtained": "BOOLEAN",
    "dpr_additional_sbn_approval_obtained": "BOOLEAN",
    "tax_measure_has_enacted_law": "BOOLEAN",
    "pnbp_measure_has_valid_tariff_instrument": "BOOLEAN",
    "output_outcome_documented": "BOOLEAN",
    "domestic_product_compliance_documented": "BOOLEAN",
    "growth_outlook": "DOUBLE PRECISION",
    "inflation_outlook": "DOUBLE PRECISION",
    "fx_outlook": "DOUBLE PRECISION",
    "sbn10y_yield_outlook": "DOUBLE PRECISION",
    "icp_outlook": "DOUBLE PRECISION",
    "oil_lifting_outlook": "DOUBLE PRECISION",
    "gas_lifting_outlook": "DOUBLE PRECISION",
    "tax_revenue_forecast": "DOUBLE PRECISION",
}

_AGENT_COLUMNS: dict[str, str] = {
    "llm_base_url": "VARCHAR(2048)",
    "template_key": "VARCHAR(100)",
    "scenario_id": "INTEGER REFERENCES scenarios(id) ON DELETE CASCADE",
    "specialist_domain": "VARCHAR(100)",
    "is_orchestrator": "BOOLEAN NOT NULL DEFAULT FALSE",
    "llm_api_key": "VARCHAR(4096)",
    "llm_model": "VARCHAR(255)",
    "system_prompt": "TEXT",
    "temperature": "DOUBLE PRECISION NOT NULL DEFAULT 0.2",
    "max_tokens": "INTEGER NOT NULL DEFAULT 4000",
}

_CONSENSUS_SESSION_COLUMNS: dict[str, str] = {
    "logs": "JSONB NOT NULL DEFAULT '[]'::jsonb",
    "result_payload": "JSONB",
    "progress_stage": "VARCHAR(50)",
    "runtime_config_payload": "JSONB NOT NULL DEFAULT '{}'::jsonb",
}

_REASONING_LOG_COLUMNS: dict[str, str] = {
    "deliberation_history": "JSONB NOT NULL DEFAULT '[]'::jsonb",
}

_INFLUENCE_OBSERVATION_COLUMNS: dict[str, str] = {
    "interaction_payload": "JSONB NOT NULL DEFAULT '[]'::jsonb",
    "calculation_payload": "JSONB NOT NULL DEFAULT '{}'::jsonb",
}

_DISAGREEMENT_LOG_COLUMNS: dict[str, str] = {
    "detail_payload": "JSONB NOT NULL DEFAULT '{}'::jsonb",
}


def _upgrade_disagreement_logs_table() -> None:
    inspector = inspect(engine)
    if "disagreement_logs" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("disagreement_logs")}
    with engine.begin() as connection:
        for name, definition in _DISAGREEMENT_LOG_COLUMNS.items():
            if name not in columns:
                connection.execute(
                    text(f'ALTER TABLE disagreement_logs ADD COLUMN "{name}" {definition}')
                )


def _upgrade_influence_observations_table() -> None:
    inspector = inspect(engine)
    if "agent_influence_observations" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("agent_influence_observations")
    }
    with engine.begin() as connection:
        for name, definition in _INFLUENCE_OBSERVATION_COLUMNS.items():
            if name not in columns:
                connection.execute(
                    text(
                        f'ALTER TABLE agent_influence_observations ADD COLUMN "{name}" {definition}'
                    )
                )


def _upgrade_reasoning_logs_table() -> None:
    inspector = inspect(engine)
    if "reasoning_logs" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("reasoning_logs")}
    with engine.begin() as connection:
        for name, definition in _REASONING_LOG_COLUMNS.items():
            if name not in columns:
                connection.execute(
                    text(f'ALTER TABLE reasoning_logs ADD COLUMN "{name}" {definition}')
                )


def _upgrade_consensus_sessions_table() -> None:
    inspector = inspect(engine)
    if "consensus_sessions" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("consensus_sessions")}
    with engine.begin() as connection:
        for name, definition in _CONSENSUS_SESSION_COLUMNS.items():
            if name not in columns:
                connection.execute(
                    text(f'ALTER TABLE consensus_sessions ADD COLUMN "{name}" {definition}')
                )


_SIMULATION_COLUMNS: dict[str, str] = {
    "round_number": "INTEGER NOT NULL DEFAULT 1",
    "trigger": "VARCHAR(100) NOT NULL",
    "input_payload": "JSONB NOT NULL",
    "output_payload": "JSONB NOT NULL",
    "status": "VARCHAR(30) NOT NULL",
    "simulation_version": "VARCHAR(64) NOT NULL",
    "latency_ms": "DOUBLE PRECISION NOT NULL DEFAULT 0",
    "token_usage": "INTEGER NOT NULL DEFAULT 0",
}


def _upgrade_simulation_artifacts_table() -> None:
    inspector = inspect(engine)
    if "simulation_artifacts" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("simulation_artifacts")}
    with engine.begin() as connection:
        for name, definition in _SIMULATION_COLUMNS.items():
            if name not in columns:
                connection.execute(
                    text(f'ALTER TABLE simulation_artifacts ADD COLUMN "{name}" {definition}')
                )
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_simulation_artifacts_run_id ON simulation_artifacts (run_id)"
        ))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_simulation_artifacts_scenario_id ON simulation_artifacts (scenario_id)"
        ))


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
        connection.execute(text("ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_name_key"))
        connection.execute(text("ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_template_key_key"))
        connection.execute(text("DROP INDEX IF EXISTS uq_agent_singleton_orchestrator"))
        connection.execute(text("DROP INDEX IF EXISTS ix_agents_template_key"))
        connection.execute(text("DROP INDEX IF EXISTS uq_agent_global_name"))
        connection.execute(text("DROP INDEX IF EXISTS uq_agent_scenario_name"))
        connection.execute(text("DROP INDEX IF EXISTS uq_agent_global_template_key"))
        connection.execute(text("DROP INDEX IF EXISTS uq_agent_scenario_template_key"))
        connection.execute(text(
            "CREATE UNIQUE INDEX uq_agent_global_name ON agents (name) "
            "WHERE scenario_id IS NULL"
        ))
        connection.execute(text(
            "CREATE UNIQUE INDEX uq_agent_scenario_name ON agents (scenario_id, name) "
            "WHERE scenario_id IS NOT NULL"
        ))
        connection.execute(text(
            "CREATE UNIQUE INDEX uq_agent_global_template_key ON agents (template_key) "
            "WHERE scenario_id IS NULL AND template_key IS NOT NULL"
        ))
        connection.execute(text(
            "CREATE UNIQUE INDEX uq_agent_scenario_template_key "
            "ON agents (scenario_id, template_key) "
            "WHERE scenario_id IS NOT NULL AND template_key IS NOT NULL"
        ))
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_singleton_orchestrator "
            "ON agents (is_orchestrator) WHERE is_orchestrator IS TRUE"
        ))
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_scenario_specialist_domain "
            "ON agents (scenario_id, specialist_domain) "
            "WHERE scenario_id IS NOT NULL AND specialist_domain IS NOT NULL"
        ))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_agents_scenario_id "
            "ON agents (scenario_id)"
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
            text(
                "ALTER TABLE scenarios DROP CONSTRAINT IF EXISTS "
                "ck_scenario_max_deficit_nonnegative"
            )
        )
        connection.execute(
            text("ALTER TABLE scenarios DROP COLUMN IF EXISTS max_deficit_constraint")
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
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_duration_positive",
            "duration_months > 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_reallocation_nonnegative",
            "proposed_reallocation >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_revenue_nonnegative",
            "proposed_additional_revenue >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_debt_nonnegative",
            "proposed_debt_financing >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_sal_use_nonnegative",
            "proposed_sal_use >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_other_financing_nonnegative",
            "proposed_other_financing >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_verified_reallocation_nonnegative",
            "verified_reallocation_capacity >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_verified_revenue_nonnegative",
            "verified_revenue_offset_capacity >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_verified_debt_nonnegative",
            "verified_debt_financing_headroom >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_verified_sal_nonnegative",
            "verified_sal_available >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_verified_cash_minimum_nonnegative",
            "verified_operational_cash_minimum >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_verified_cash_after_nonnegative",
            "verified_projected_cash_after_policy >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_verified_borrowing_nonnegative",
            "verified_cumulative_borrowing_pct_gdp >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_fx_positive",
            "fx_outlook > 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_sbn_yield_nonnegative",
            "sbn10y_yield_outlook >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_icp_nonnegative",
            "icp_outlook >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_oil_lifting_nonnegative",
            "oil_lifting_outlook >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_gas_lifting_nonnegative",
            "gas_lifting_outlook >= 0",
        )
        _add_scenario_check_constraint(
            connection,
            "ck_scenario_tax_forecast_nonnegative",
            "tax_revenue_forecast >= 0",
        )


def _add_scenario_check_constraint(
    connection: Connection,
    name: str,
    expression: str,
) -> None:
    connection.execute(
        text(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = '{name}'
                ) THEN
                    ALTER TABLE scenarios ADD CONSTRAINT {name}
                        CHECK ({expression});
                END IF;
            END $$;
            """
        )
    )


def _initialize_global_llm_config() -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO global_llm_config (
                    id, temperature, max_tokens, apply_to_all, revision
                )
                VALUES (1, 0.2, 4000, FALSE, 1)
                ON CONFLICT (id) DO NOTHING
                """
            )
        )


def initialize_database() -> None:
    Agent.metadata.create_all(bind=engine)
    _initialize_global_llm_config()
    _upgrade_agents_table()
    _upgrade_scenarios_table()
    _upgrade_consensus_sessions_table()
    _upgrade_reasoning_logs_table()
    _upgrade_influence_observations_table()
    _upgrade_disagreement_logs_table()
    _upgrade_simulation_artifacts_table()
    _upgrade_session_columns()


if __name__ == "__main__":
    initialize_database()
