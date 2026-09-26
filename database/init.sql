CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE IF EXISTS scenarios
    ADD COLUMN IF NOT EXISTS instrument VARCHAR(255),
    ADD COLUMN IF NOT EXISTS targeting TEXT,
    ADD COLUMN IF NOT EXISTS program_cost DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS duration_months INTEGER,
    ADD COLUMN IF NOT EXISTS evaluation_trigger TEXT,
    ADD COLUMN IF NOT EXISTS program_cost_period VARCHAR(100),
    ADD COLUMN IF NOT EXISTS skip_llm_formulation BOOLEAN,
    ADD COLUMN IF NOT EXISTS formulation_dry_run_only BOOLEAN,
    ADD COLUMN IF NOT EXISTS single_year_deployment BOOLEAN,
    ADD COLUMN IF NOT EXISTS proposed_reallocation DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS reallocation_from_education BOOLEAN,
    ADD COLUMN IF NOT EXISTS proposed_additional_revenue DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS revenue_measure_type VARCHAR(255),
    ADD COLUMN IF NOT EXISTS proposed_debt_financing DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS debt_financing_mode VARCHAR(255),
    ADD COLUMN IF NOT EXISTS proposed_sal_use DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sal_purpose TEXT,
    ADD COLUMN IF NOT EXISTS proposed_other_financing DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS appropriation_available BOOLEAN,
    ADD COLUMN IF NOT EXISTS verified_reallocation_capacity DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS verified_revenue_offset_capacity DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS verified_debt_financing_headroom DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS verified_sal_available DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS verified_operational_cash_minimum DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS verified_projected_cash_after_policy DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS verified_cumulative_borrowing_pct_gdp DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS spending_reallocation_authorized BOOLEAN,
    ADD COLUMN IF NOT EXISTS dpr_spending_adjustment_recommendation BOOLEAN,
    ADD COLUMN IF NOT EXISTS finance_minister_sal_authorized BOOLEAN,
    ADD COLUMN IF NOT EXISTS dpr_sal_approval_obtained BOOLEAN,
    ADD COLUMN IF NOT EXISTS dpr_additional_sbn_approval_obtained BOOLEAN,
    ADD COLUMN IF NOT EXISTS tax_measure_has_enacted_law BOOLEAN,
    ADD COLUMN IF NOT EXISTS pnbp_measure_has_valid_tariff_instrument BOOLEAN,
    ADD COLUMN IF NOT EXISTS output_outcome_documented BOOLEAN,
    ADD COLUMN IF NOT EXISTS domestic_product_compliance_documented BOOLEAN,
    ADD COLUMN IF NOT EXISTS growth_outlook DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS inflation_outlook DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS fx_outlook DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sbn10y_yield_outlook DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS icp_outlook DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS oil_lifting_outlook DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS gas_lifting_outlook DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS tax_revenue_forecast DOUBLE PRECISION;

DO $$
BEGIN
    IF to_regclass('public.scenarios') IS NOT NULL THEN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'scenarios' AND column_name = 'no_phase0'
        ) THEN
            UPDATE scenarios SET skip_llm_formulation = COALESCE(skip_llm_formulation, no_phase0);
            ALTER TABLE scenarios DROP COLUMN no_phase0;
        END IF;
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'scenarios' AND column_name = 'phase0_only'
        ) THEN
            UPDATE scenarios SET formulation_dry_run_only = COALESCE(formulation_dry_run_only, phase0_only);
            ALTER TABLE scenarios DROP COLUMN phase0_only;
        END IF;
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'scenarios' AND column_name = 'no_phased'
        ) THEN
            UPDATE scenarios SET single_year_deployment = COALESCE(single_year_deployment, no_phased);
            ALTER TABLE scenarios DROP COLUMN no_phased;
        END IF;
    END IF;
END $$;

ALTER TABLE IF EXISTS scenarios
    DROP CONSTRAINT IF EXISTS ck_scenario_max_deficit_nonnegative,
    DROP COLUMN IF EXISTS max_deficit_constraint;

ALTER TABLE IF EXISTS agents
    ADD COLUMN IF NOT EXISTS agent_uuid VARCHAR(36),
    ADD COLUMN IF NOT EXISTS template_key VARCHAR(100),
    ADD COLUMN IF NOT EXISTS scenario_id INTEGER REFERENCES scenarios(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS specialist_domain VARCHAR(100),
    ADD COLUMN IF NOT EXISTS is_orchestrator BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS rar_dai_weight_mode VARCHAR(20) NOT NULL DEFAULT 'auto',
    ADD COLUMN IF NOT EXISTS llm_base_url VARCHAR(2048),
    ADD COLUMN IF NOT EXISTS llm_api_key VARCHAR(4096),
    ADD COLUMN IF NOT EXISTS llm_model VARCHAR(255),
    ADD COLUMN IF NOT EXISTS system_prompt TEXT,
    ADD COLUMN IF NOT EXISTS temperature DOUBLE PRECISION NOT NULL DEFAULT 0.2,
    ADD COLUMN IF NOT EXISTS max_tokens INTEGER NOT NULL DEFAULT 4000;

DO $$
BEGIN
    IF to_regclass('public.agents') IS NOT NULL THEN
        UPDATE agents
        SET agent_uuid = gen_random_uuid()::text
        WHERE agent_uuid IS NULL OR agent_uuid = '';
        ALTER TABLE agents ALTER COLUMN agent_uuid SET NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agents_agent_uuid ON agents (agent_uuid);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS global_llm_config (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    llm_base_url VARCHAR(2048),
    llm_api_key VARCHAR(4096),
    llm_model VARCHAR(255),
    temperature DOUBLE PRECISION NOT NULL DEFAULT 0.2 CHECK (temperature >= 0 AND temperature <= 2),
    max_tokens INTEGER NOT NULL DEFAULT 4000 CHECK (max_tokens > 0),
    apply_to_all BOOLEAN NOT NULL DEFAULT FALSE,
    revision INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO global_llm_config (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

DO $$
BEGIN
    IF to_regclass('public.agents') IS NOT NULL THEN
        ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_name_key;
        ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_template_key_key;
        DROP INDEX IF EXISTS ix_agents_template_key;
        DROP INDEX IF EXISTS uq_agent_singleton_orchestrator;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_global_name
            ON agents (name) WHERE scenario_id IS NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_scenario_name
            ON agents (scenario_id, name) WHERE scenario_id IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_global_template_key
            ON agents (template_key)
            WHERE scenario_id IS NULL AND template_key IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_scenario_template_key
            ON agents (scenario_id, template_key)
            WHERE scenario_id IS NOT NULL AND template_key IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_scenario_orchestrator
            ON agents (scenario_id)
            WHERE is_orchestrator IS TRUE AND scenario_id IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_scenario_specialist_domain
            ON agents (scenario_id, specialist_domain)
            WHERE scenario_id IS NOT NULL AND specialist_domain IS NOT NULL;
        CREATE INDEX IF NOT EXISTS ix_agents_scenario_id
            ON agents (scenario_id);
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'ck_agent_rar_dai_weight_mode'
        ) THEN
            ALTER TABLE agents ADD CONSTRAINT ck_agent_rar_dai_weight_mode
                CHECK (rar_dai_weight_mode IN ('auto', 'manual'));
        END IF;
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
            runtime_config_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            celery_task_id VARCHAR(255) UNIQUE,
            status VARCHAR(20) NOT NULL DEFAULT 'QUEUED',
            error TEXT,
            logs JSONB NOT NULL DEFAULT '[]'::jsonb,
            result_payload JSONB,
            progress_stage VARCHAR(50),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ
        );
        ALTER TABLE consensus_sessions
            ADD COLUMN IF NOT EXISTS logs JSONB NOT NULL DEFAULT '[]'::jsonb,
            ADD COLUMN IF NOT EXISTS result_payload JSONB,
            ADD COLUMN IF NOT EXISTS progress_stage VARCHAR(50),
            ADD COLUMN IF NOT EXISTS runtime_config_payload JSONB NOT NULL DEFAULT '{}'::jsonb;
        CREATE INDEX IF NOT EXISTS ix_consensus_sessions_scenario_id
            ON consensus_sessions (scenario_id);
        ALTER TABLE IF EXISTS reasoning_logs ADD COLUMN IF NOT EXISTS run_id VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE;
        ALTER TABLE IF EXISTS reasoning_logs ADD COLUMN IF NOT EXISTS deliberation_history JSONB NOT NULL DEFAULT '[]'::jsonb;
        ALTER TABLE IF EXISTS disagreement_logs ADD COLUMN IF NOT EXISTS run_id VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE;
        ALTER TABLE IF EXISTS disagreement_logs ADD COLUMN IF NOT EXISTS detail_payload JSONB NOT NULL DEFAULT '{}'::jsonb;
        ALTER TABLE IF EXISTS metric_snapshots ADD COLUMN IF NOT EXISTS run_id VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE;
        ALTER TABLE IF EXISTS agent_influence_observations ADD COLUMN IF NOT EXISTS run_id VARCHAR(36) REFERENCES consensus_sessions(id) ON DELETE CASCADE;
        ALTER TABLE IF EXISTS agent_influence_observations ADD COLUMN IF NOT EXISTS interaction_payload JSONB NOT NULL DEFAULT '[]'::jsonb;
        ALTER TABLE IF EXISTS agent_influence_observations ADD COLUMN IF NOT EXISTS calculation_payload JSONB NOT NULL DEFAULT '{}'::jsonb;
        CREATE INDEX IF NOT EXISTS ix_reasoning_logs_run_id ON reasoning_logs (run_id);
        CREATE INDEX IF NOT EXISTS ix_disagreement_logs_run_id ON disagreement_logs (run_id);
        CREATE INDEX IF NOT EXISTS ix_metric_snapshots_run_id ON metric_snapshots (run_id);
        CREATE INDEX IF NOT EXISTS ix_agent_influence_observations_run_id ON agent_influence_observations (run_id);
        CREATE TABLE IF NOT EXISTS simulation_artifacts (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(36) NOT NULL REFERENCES consensus_sessions(id) ON DELETE CASCADE,
            scenario_id INTEGER NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
            trigger VARCHAR(100) NOT NULL,
            round_number INTEGER NOT NULL DEFAULT 1 CHECK (round_number > 0),
            input_payload JSONB NOT NULL,
            output_payload JSONB NOT NULL,
            status VARCHAR(30) NOT NULL,
            simulation_version VARCHAR(64) NOT NULL,
            latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (latency_ms >= 0),
            token_usage INTEGER NOT NULL DEFAULT 0 CHECK (token_usage >= 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_simulation_run_version_round UNIQUE (run_id, simulation_version, round_number)
        );
        CREATE INDEX IF NOT EXISTS ix_simulation_artifacts_run_id
            ON simulation_artifacts (run_id);
        CREATE INDEX IF NOT EXISTS ix_simulation_artifacts_scenario_id
            ON simulation_artifacts (scenario_id);
    END IF;
END $$;
