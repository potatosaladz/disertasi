CREATE EXTENSION IF NOT EXISTS pgcrypto;

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
