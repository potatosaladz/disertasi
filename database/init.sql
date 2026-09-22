CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE IF EXISTS scenarios
    ADD COLUMN IF NOT EXISTS program_cost DOUBLE PRECISION;

ALTER TABLE IF EXISTS scenarios
    ALTER COLUMN max_deficit_constraint SET DEFAULT 3.0;

ALTER TABLE IF EXISTS agents
    ADD COLUMN IF NOT EXISTS template_key VARCHAR(100),
    ADD COLUMN IF NOT EXISTS llm_base_url VARCHAR(2048),
    ADD COLUMN IF NOT EXISTS llm_api_key VARCHAR(4096),
    ADD COLUMN IF NOT EXISTS llm_model VARCHAR(255),
    ADD COLUMN IF NOT EXISTS system_prompt TEXT,
    ADD COLUMN IF NOT EXISTS temperature DOUBLE PRECISION NOT NULL DEFAULT 0.2,
    ADD COLUMN IF NOT EXISTS max_tokens INTEGER NOT NULL DEFAULT 4000;

DO $$
BEGIN
    IF to_regclass('public.agents') IS NOT NULL THEN
        CREATE UNIQUE INDEX IF NOT EXISTS ix_agents_template_key
            ON agents (template_key) WHERE template_key IS NOT NULL;
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'ck_agent_temperature_range'
        ) THEN
            ALTER TABLE agents ADD CONSTRAINT ck_agent_temperature_range
                CHECK (temperature >= 0 AND temperature <= 2);
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'ck_agent_max_tokens_positive'
        ) THEN
            ALTER TABLE agents ADD CONSTRAINT ck_agent_max_tokens_positive
                CHECK (max_tokens > 0);
        END IF;
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('public.scenarios') IS NOT NULL THEN
        CREATE TABLE IF NOT EXISTS scenario_mandate_snapshots (
            id SERIAL PRIMARY KEY,
            scenario_id INTEGER NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
            revision VARCHAR(64) NOT NULL,
            generated BOOLEAN NOT NULL DEFAULT TRUE,
            agent_count INTEGER NOT NULL CHECK (agent_count >= 0),
            rules JSONB NOT NULL,
            agent_rules JSONB NOT NULL,
            status VARCHAR(20) NOT NULL,
            generated_count INTEGER NOT NULL DEFAULT 0 CHECK (generated_count >= 0),
            failure_count INTEGER NOT NULL DEFAULT 0 CHECK (failure_count >= 0),
            detail TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_scenario_mandate_revision UNIQUE (scenario_id, revision)
        );
        CREATE INDEX IF NOT EXISTS ix_scenario_mandate_snapshots_scenario_id
            ON scenario_mandate_snapshots (scenario_id);
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('public.scenarios') IS NOT NULL
       AND to_regclass('public.scenario_mandate_snapshots') IS NOT NULL THEN
        CREATE TABLE IF NOT EXISTS consensus_sessions (
            id VARCHAR(36) PRIMARY KEY,
            scenario_id INTEGER NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
            mandate_snapshot_id INTEGER NOT NULL REFERENCES scenario_mandate_snapshots(id) ON DELETE CASCADE,
            mandate_revision VARCHAR(64) NOT NULL,
            mandate_payload JSONB NOT NULL,
            celery_task_id VARCHAR(255) UNIQUE,
            status VARCHAR(20) NOT NULL DEFAULT 'QUEUED',
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ
        );
        CREATE INDEX IF NOT EXISTS ix_consensus_sessions_scenario_id
            ON consensus_sessions (scenario_id);
        ALTER TABLE IF EXISTS reasoning_logs ADD COLUMN IF NOT EXISTS run_id VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE;
        ALTER TABLE IF EXISTS disagreement_logs ADD COLUMN IF NOT EXISTS run_id VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE;
        ALTER TABLE IF EXISTS metric_snapshots ADD COLUMN IF NOT EXISTS run_id VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE;
        ALTER TABLE IF EXISTS agent_influence_observations ADD COLUMN IF NOT EXISTS run_id VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS ix_reasoning_logs_run_id ON reasoning_logs (run_id);
        CREATE INDEX IF NOT EXISTS ix_disagreement_logs_run_id ON disagreement_logs (run_id);
        CREATE INDEX IF NOT EXISTS ix_metric_snapshots_run_id ON metric_snapshots (run_id);
        CREATE INDEX IF NOT EXISTS ix_agent_influence_observations_run_id ON agent_influence_observations (run_id);
    END IF;
END $$;
