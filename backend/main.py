import json
import logging
import re
import socket
import warnings
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from time import perf_counter, sleep
from typing import Any, Literal, cast

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .agent_templates import (
    ORCHESTRATED_AGENT_KEYS,
    agent_domain_key,
    agent_revision,
    agent_utility_metadata,
    find_agent_by_mandate,
    find_agent_by_name_and_scenario,
    get_agent_spec,
    global_deliberative_agents,
    get_or_create_master_orchestrator,
    is_generated_agent_name,
    is_master_orchestrator,
    load_standard_agent_templates,
    mandate_seed,
    orchestrated_scenario_agent_plan,
    orchestrate_scenario_agents,
    scenario_deliberative_agents,
    scenario_scoped_agents,
    semantic_agent_name,
    template_catalog,
)
from .core_algorithms import (
    STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
    build_mandate_synthesis_prompt,
    build_mandate_synthesis_system_prompt,
    extract_json_object,
    extract_llm_completion,
    llm_request_headers,
    log_llm_outbound,
    repair_json_trailing_commas,
    resolve_llm_runtime_config,
)
from .dashboard import router as dashboard_router
from .global_config import (
    apply_global_llm_config,
    apply_global_values,
    get_global_llm_config,
    global_config_snapshot,
)
from .database import SessionLocal
from .init_db import initialize_database
from .models import (
    Agent,
    AgentInfluenceObservation,
    ConsensusSession,
    GlobalLLMConfig,
    DisagreementLog,
    MetricSnapshot,
    ReasoningLog,
    Scenario,
    ScenarioMandateSnapshot,
    SimulationArtifact,
    SimulationPayload,
)


logger = logging.getLogger(__name__)


class TemplateLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_base_url: str | None = Field(default=None, max_length=2048)
    llm_api_key: str | None = Field(default=None, max_length=4096)
    llm_model: str | None = Field(default=None, max_length=255)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, allow_inf_nan=False)
    max_tokens: int = Field(default=4000, gt=0)


class LoadTemplatesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configs: dict[str, TemplateLLMConfig] = Field(default_factory=dict)


class OrchestratorConfigPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_base_url: str | None = Field(default=None, max_length=2048)
    api_key: str | None = Field(default=None, max_length=4096)
    model_name: str | None = Field(default=None, max_length=255)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, allow_inf_nan=False)
    max_tokens: int = Field(default=4000, gt=0)


class OrchestrateAgentsRequest(OrchestratorConfigPayload):
    configs: dict[str, TemplateLLMConfig] = Field(default_factory=dict)


class OrchestratorConfigResponse(BaseModel):
    scenario_id: int
    api_base_url: str | None
    model_name: str | None
    temperature: float
    max_tokens: int
    has_api_key: bool


RETRYABLE_LLM_STATUS_CODES = frozenset({500, 502, 503, 530})
MAX_LLM_RETRIES = 3


def _llm_status_code(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _create_completion_with_retry(
    completions: Any,
    *,
    operation: str,
    request: dict[str, Any],
) -> Any:
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            return completions.create(**request)
        except Exception as error:
            status_code = _llm_status_code(error)
            transient = status_code in RETRYABLE_LLM_STATUS_CODES or isinstance(
                error,
                (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError, OSError, socket.error),
            )
            if not transient or attempt == MAX_LLM_RETRIES:
                raise
            delay_seconds = 0.25 * (2**attempt)
            logger.warning(
                "%s transient LLM failure (status=%s, attempt=%s/%s); retrying in %.2fs",
                operation,
                status_code or type(error).__name__,
                attempt + 1,
                MAX_LLM_RETRIES + 1,
                delay_seconds,
            )
            sleep(delay_seconds)
    raise RuntimeError(f"{operation} retry loop terminated unexpectedly")


class GlobalLLMConfigResponse(BaseModel):
    llm_base_url: str | None
    llm_model: str | None
    temperature: float
    max_tokens: int
    apply_to_all: bool
    revision: int
    has_llm_api_key: bool


class GlobalLLMConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_base_url: str | None = Field(default=None, max_length=2048)
    llm_api_key: str | None = Field(default=None, max_length=4096)
    llm_model: str | None = Field(default=None, max_length=255)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, allow_inf_nan=False)
    max_tokens: int = Field(default=4000, gt=0)
    apply_to_all: bool = False


def global_llm_config_response(config: GlobalLLMConfig) -> GlobalLLMConfigResponse:
    return GlobalLLMConfigResponse.model_validate(global_config_snapshot(config))


class AgentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    role: str = Field(min_length=1, max_length=255)
    template_key: str | None = Field(default=None, max_length=100)
    scenario_id: int | None = Field(default=None, gt=0)
    theta_x: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    theta_q: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    theta_h: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    theta_s: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    theta_u: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    rar_dai_weight_mode: Literal["auto", "manual"] = "auto"
    llm_base_url: str | None = Field(default=None, max_length=2048)
    llm_api_key: str | None = Field(default=None, max_length=4096)
    llm_model: str | None = Field(default=None, max_length=255)
    system_prompt: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, allow_inf_nan=False)
    max_tokens: int = Field(default=4000, gt=0)


class AgentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=255)
    role: str | None = Field(default=None, min_length=1, max_length=255)
    template_key: str | None = Field(default=None, max_length=100)
    theta_x: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    theta_q: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    theta_h: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    theta_s: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    theta_u: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    rar_dai_weight_mode: Literal["auto", "manual"] | None = None
    llm_base_url: str | None = Field(default=None, max_length=2048)
    llm_api_key: str | None = Field(default=None, max_length=4096)
    llm_model: str | None = Field(default=None, max_length=255)
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, gt=0)


class AgentResponse(BaseModel):
    id: int
    agent_uuid: str
    name: str
    display_name: str
    reused: bool = False
    role: str
    template_key: str | None
    scenario_id: int | None
    specialist_domain: str | None
    theta_x: float
    theta_q: float
    theta_h: float
    theta_s: float
    theta_u: float
    rar_dai_weight_mode: Literal["auto", "manual"]
    llm_base_url: str | None
    llm_model: str | None
    system_prompt: str | None
    temperature: float
    max_tokens: int
    has_llm_api_key: bool


class LLMConnectionTestResponse(BaseModel):
    agent_id: int
    ok: bool
    model: str
    base_url: str
    latency_ms: float
    response_preview: str | None = None
    error: str | None = None


class MandateSynthesisError(BaseModel):
    code: Literal["CONFIGURATION_ERROR", "PROVIDER_ERROR", "EMPTY_CONTENT", "INVALID_JSON", "SCHEMA_ERROR"]
    message: str
    retryable: bool


class AgentDomainRules(BaseModel):
    agent_id: int
    name: str
    display_name: str
    utility_metadata: dict[str, str]
    role: str
    template_key: str | None
    mandate: str | None
    primary_sources: list[str]
    constraints: list[str]
    owned_checks: list[str]
    synthesis_status: Literal["generated", "fallback"] = "fallback"
    scenario_mandate: str | None = None
    scenario_focus: list[str] = Field(default_factory=list)
    priority_questions: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    epistemic_logic_traceability: list[str] = Field(default_factory=list)
    structured_consensus_protocol: list[str] = Field(default_factory=list)
    regulatory_compliance_alignment: list[str] = Field(default_factory=list)
    applicable_primary_sources: list[str] = Field(default_factory=list)
    applicable_constraints: list[str] = Field(default_factory=list)
    applicable_owned_checks: list[str] = Field(default_factory=list)
    llm_model: str | None = None
    latency_ms: float | None = None
    token_usage: int | None = None
    error: MandateSynthesisError | None = None


class DomainRulesResponse(BaseModel):
    scenario_id: int
    revision: str
    generated: bool
    stale: bool
    agent_count: int
    rules: dict[str, object]
    agent_rules: list[AgentDomainRules]
    status: Literal["success", "partial", "failed", "stale", "missing"] = "missing"
    generated_count: int = 0
    failure_count: int = 0
    detail: str | None = None


class ScenarioCreate(SimulationPayload):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    description: str = Field(min_length=1)


class ScenarioUpdate(SimulationPayload):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    description: str | None = Field(default=None, min_length=1)


class ScenarioResponse(SimulationPayload):
    id: int
    description: str


def agent_response(agent: Agent, *, reused: bool = False) -> AgentResponse:
    return AgentResponse(
        id=agent.id,
        agent_uuid=agent.agent_uuid,
        name=agent.name,
        display_name=semantic_agent_name(agent),
        reused=reused,
        role=agent.role,
        template_key=agent.template_key,
        scenario_id=agent.scenario_id,
        specialist_domain=agent.specialist_domain,
        theta_x=agent.theta_x,
        theta_q=agent.theta_q,
        theta_h=agent.theta_h,
        theta_s=agent.theta_s,
        theta_u=agent.theta_u,
        rar_dai_weight_mode=cast(Literal["auto", "manual"], agent.rar_dai_weight_mode),
        llm_base_url=agent.llm_base_url,
        llm_model=agent.llm_model,
        system_prompt=agent.system_prompt,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens,
        has_llm_api_key=bool(agent.llm_api_key),
    )


def scenario_response(scenario: Scenario) -> ScenarioResponse:
    return ScenarioResponse.model_validate(scenario)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    yield


app = FastAPI(title="SHCR API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3010"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(dashboard_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "SHCR API Running"}


@app.get("/health")
async def health() -> dict[str, str]:
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))
    return {"status": "healthy", "database": "connected"}


@app.get("/api/global-config", response_model=GlobalLLMConfigResponse)
def get_global_config() -> GlobalLLMConfigResponse:
    with SessionLocal() as session:
        return global_llm_config_response(get_global_llm_config(session))


@app.put("/api/global-config", response_model=GlobalLLMConfigResponse)
def update_global_config(payload: GlobalLLMConfigUpdate) -> GlobalLLMConfigResponse:
    with SessionLocal() as session:
        config = get_global_llm_config(session)
        values = payload.model_dump(exclude_unset=True)
        if values.get("llm_api_key") == "":
            values.pop("llm_api_key")
        for field, value in values.items():
            setattr(config, field, value)
        config.revision += 1
        if config.apply_to_all:
            for agent in session.scalars(
                select(Agent)
                .where(Agent.is_orchestrator.is_(False))
                .order_by(Agent.id)
            ):
                apply_global_llm_config(agent, config)
        session.commit()
        session.refresh(config)
        return global_llm_config_response(config)


@app.get("/api/agent-templates")
def list_agent_templates() -> list[dict[str, object]]:
    return template_catalog()


def _orchestrator_llm_domain_selection(
    orchestrator: Agent,
    scenario: Scenario,
    fallback_domains: list[str],
) -> tuple[list[str], str | None]:
    try:
        config = resolve_llm_runtime_config(orchestrator)
        prompt = {
            "scenario": {
                "goal": scenario.description,
                "parameters": scenario.simulation_payload(),
            },
            "candidate_domains": [
                {"key": key, "role": role}
                for key, role in (
                    ("revenue", "State Revenue Agent / Penerimaan Negara"),
                    ("expenditure", "Government Expenditure Agent / Belanja Pemerintah"),
                    ("budget", "BudgetAgent / Anggaran Pemerintah"),
                    ("financing", "Financing & Debt Agent / Pembiayaan Pemerintah"),
                    ("macro", "Macro-Fiscal Stabilization Agent / Strategi Ekonomi dan Fiskal"),
                    ("fiscal_risk", "Fiscal Risk & Contingency Agent / Manajemen Risiko Fiskal"),
                    ("critic", "Adversarial Fiscal Critic Agent / Penelaah & Kritik Kebijakan Fiskal"),
                )
            ],
            "required_output": {
                "agents": [{"domain": key} for key in ORCHESTRATED_AGENT_KEYS],
                "tool_agent": {
                    "name": "Fiscal Simulation & RPC Tool Agent / Agen Simulasi & Kalkulasi Fiskal",
                    "voting": False,
                    "included_in_agents": False,
                },
            },
        }
        instructions = (
            "You are the SHCR Master Orchestrator. Read the active scenario goal and every supplied "
            "parameter. Return JSON only with an agents array containing EXACTLY SEVEN unique "
            "domain keys: revenue, expenditure, budget, financing, macro, fiscal_risk, and critic. "
            "The roster roles are State Revenue Agent, Government Expenditure Agent, BudgetAgent, "
            "Financing & Debt Agent, Macro-Fiscal Stabilization Agent, Fiscal Risk & Contingency "
            "Agent, and Adversarial Fiscal Critic Agent. Do not include the Simulation/RPC Tool Agent "
            "in the voting roster; it is a separate non-voting service. Do not return prose or any "
            "other number of agents."
        )
        result = _create_completion_with_retry(
            OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=60.0,
                max_retries=0,
                default_headers=llm_request_headers(),
            ).chat.completions,
            operation="Orchestrator domain selection",
            request={
                "model": config.model,
                "temperature": orchestrator.temperature,
                "max_tokens": orchestrator.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True),
                    },
                ],
            },
        )
        content, _ = extract_llm_completion(result)
        parsed = extract_json_object(content)
        raw_agents = parsed.get("agents")
        if not isinstance(raw_agents, list) or len(raw_agents) != 7:
            raise ValueError("Orchestrator must return exactly seven agents")
        domains = [
            item.get("domain")
            for item in raw_agents
            if isinstance(item, dict) and isinstance(item.get("domain"), str)
        ]
        if len(domains) != 7 or len(set(domains)) != 7:
            raise ValueError("Orchestrator domains must be seven unique known keys")
        if set(domains) != set(ORCHESTRATED_AGENT_KEYS):
            raise ValueError("Orchestrator returned an incomplete or unknown voting roster")
        return cast(list[str], domains), None
    except Exception as error:
        logger.warning(
            "[WARNING] Orchestrator LLM unavailable or invalid; using deterministic seven-agent selection (%s)",
            type(error).__name__,
        )
        return fallback_domains, f"{type(error).__name__}: deterministic selection used"


def _orchestrator_config_response(
    scenario_id: int,
    orchestrator: Agent | None,
) -> OrchestratorConfigResponse:
    return OrchestratorConfigResponse(
        scenario_id=scenario_id,
        api_base_url=orchestrator.llm_base_url if orchestrator else None,
        model_name=orchestrator.llm_model if orchestrator else None,
        temperature=orchestrator.temperature if orchestrator else 0.2,
        max_tokens=orchestrator.max_tokens if orchestrator else 4000,
        has_api_key=bool(orchestrator and orchestrator.llm_api_key),
    )


@app.get(
    "/api/scenarios/{scenario_id}/orchestrator-config",
    response_model=OrchestratorConfigResponse,
)
def get_scenario_orchestrator_config(scenario_id: int) -> OrchestratorConfigResponse:
    with SessionLocal() as session:
        if session.get(Scenario, scenario_id) is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        orchestrator = session.scalar(
            select(Agent).where(
                Agent.scenario_id == scenario_id,
                Agent.is_orchestrator.is_(True),
            )
        )
        return _orchestrator_config_response(scenario_id, orchestrator)


@app.post("/api/agents/load-templates")
def load_agent_templates(payload: LoadTemplatesRequest | None = None) -> dict[str, object]:
    configs = {
        key: config.model_dump()
        for key, config in (payload.configs if payload else {}).items()
    }
    unknown = set(configs) - {item["key"] for item in template_catalog()}
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown templates: {sorted(unknown)}")
    with SessionLocal() as session:
        agents, created = load_standard_agent_templates(session, configs)
        return {
            "created": created,
            "total": len(agents),
            "agents": [agent_response(agent).model_dump() for agent in agents],
        }


@app.post("/api/scenarios/{scenario_id}/orchestrate-agents")
@app.post("/scenarios/{scenario_id}/orchestrate-agents")
def orchestrate_scenario_agent_set(
    scenario_id: int,
    payload: OrchestrateAgentsRequest | None = None,
) -> dict[str, object]:
    configs = {
        key: config.model_dump()
        for key, config in (payload.configs if payload else {}).items()
    }
    unknown = set(configs) - {item["key"] for item in template_catalog()}
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown domain configs: {sorted(unknown)}")
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        existing_orchestrator = session.scalar(
            select(Agent).where(
                Agent.scenario_id == scenario_id,
                Agent.is_orchestrator.is_(True),
            )
        )
        explicit = (
            payload
            if payload
            and any((payload.api_base_url, payload.api_key, payload.model_name))
            else None
        )
        orchestrator_values: dict[str, Any] | None = None
        if explicit is not None:
            supplied_key = explicit.api_key or (
                existing_orchestrator.llm_api_key if existing_orchestrator else None
            )
            if not explicit.api_base_url or not explicit.model_name or not supplied_key:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Explicit Orchestrator configuration requires API Base URL, "
                        "Model Name, and an API Key (new or already persisted)"
                    ),
                )
            orchestrator_values = {
                "llm_base_url": explicit.api_base_url,
                "llm_api_key": supplied_key,
                "llm_model": explicit.model_name,
                "temperature": explicit.temperature,
                "max_tokens": explicit.max_tokens,
            }
        _clear_scenario_execution_data(session, scenario_id)
        orchestrator, orchestrator_created = get_or_create_master_orchestrator(
            session,
            scenario_id,
            orchestrator_values,
        )
        fallback_domains, _ = orchestrated_scenario_agent_plan(scenario)
        selected_domains, orchestration_error = _orchestrator_llm_domain_selection(
            orchestrator,
            scenario,
            fallback_domains,
        )
        agents, orchestration = orchestrate_scenario_agents(
            session,
            scenario,
            configs,
            selected_domains,
        )
        orchestration["selection_source"] = (
            "deterministic_fallback" if orchestration_error else "master_orchestrator_llm"
        )
        orchestration["selection_error"] = orchestration_error
        session.commit()
        for agent in agents:
            session.refresh(agent)
        session.refresh(orchestrator)
        return {
            "created": len(agents),
            "total": len(agents),
            "scenario_id": scenario_id,
            "orchestrator_id": orchestrator.id,
            "orchestrator_uuid": orchestrator.agent_uuid,
            "orchestrator_created": orchestrator_created,
            "orchestrator_config": _orchestrator_config_response(
                scenario_id, orchestrator
            ).model_dump(),
            "orchestration": orchestration,
            "agents": [agent_response(agent).model_dump() for agent in agents],
        }


@app.post("/api/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
@app.post("/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
def create_agent(payload: AgentCreate, response: Response) -> AgentResponse:
    with SessionLocal() as session:
        values = apply_global_values(payload.model_dump(), get_global_llm_config(session))
        template = get_agent_spec(payload.template_key)
        if payload.template_key is not None and template is None:
            raise HTTPException(status_code=422, detail="Unknown agent template")
        if template is not None:
            values["system_prompt"] = template.system_prompt
        if payload.scenario_id is not None and session.get(Scenario, payload.scenario_id) is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        idempotent = find_agent_by_name_and_scenario(
            session,
            name=payload.name,
            scenario_id=payload.scenario_id,
        )
        if idempotent is not None:
            response.status_code = status.HTTP_200_OK
            return agent_response(idempotent, reused=True)
        existing_named_agent = (
            find_agent_by_name_and_scenario(
                session,
                name=payload.name,
                scenario_id=None,
            )
            if payload.scenario_id is None
            else None
        )
        if is_generated_agent_name(payload.name) and payload.scenario_id is None:
            if existing_named_agent is not None and (
                existing_named_agent.scenario_id is not None
                or existing_named_agent.is_orchestrator
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Generated agent name belongs to a scoped system agent",
                )
            existing_mandate_agent = existing_named_agent or find_agent_by_mandate(
                session,
                role=payload.role,
                template_key=payload.template_key,
                system_prompt=values.get("system_prompt"),
            )
            if existing_mandate_agent is not None:
                configuration_fields = (
                    "role",
                    "template_key",
                    "theta_x",
                    "theta_q",
                    "theta_h",
                    "theta_s",
                    "theta_u",
                    "llm_base_url",
                    "llm_api_key",
                    "llm_model",
                    "system_prompt",
                    "temperature",
                    "max_tokens",
                )
                mismatched = [
                    field
                    for field in configuration_fields
                    if getattr(existing_mandate_agent, field) != values.get(field)
                ]
                if mismatched:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Generated agent already exists with conflicting "
                            f"configuration: {', '.join(mismatched)}"
                        ),
                    )
                response.status_code = status.HTTP_200_OK
                return agent_response(existing_mandate_agent, reused=True)
            transient = Agent(name=payload.name, role=payload.role, template_key=payload.template_key)
            base_name = semantic_agent_name(transient)
            candidate = base_name
            suffix = 2
            while session.scalar(
                select(Agent.id).where(Agent.name == candidate)
            ) is not None:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            values["name"] = candidate
        elif existing_named_agent is not None:
            raise HTTPException(status_code=409, detail="Agent name already exists")
        inserted_id = session.scalar(
            insert(Agent)
            .values(**values)
            .on_conflict_do_nothing()
            .returning(Agent.id)
        )
        if inserted_id is None:
            agent = find_agent_by_name_and_scenario(
                session,
                name=str(values["name"]),
                scenario_id=payload.scenario_id,
            )
            if agent is None:
                raise HTTPException(
                    status_code=409,
                    detail="Agent name already exists in another scenario",
                )
            response.status_code = status.HTTP_200_OK
            return agent_response(agent, reused=True)
        session.commit()
        agent = session.get(Agent, inserted_id)
        if agent is None:
            raise HTTPException(status_code=500, detail="Agent creation failed")
        return agent_response(agent)


@app.put("/api/agents/{agent_id}", response_model=AgentResponse)
@app.put("/agents/{agent_id}", response_model=AgentResponse)
def update_agent(agent_id: int, payload: AgentUpdate) -> AgentResponse:
    with SessionLocal() as session:
        agent = session.get(Agent, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        existing = (
            session.scalar(
                select(Agent).where(
                    Agent.name == payload.name,
                    Agent.scenario_id == agent.scenario_id,
                    Agent.id != agent_id,
                )
            )
            if payload.name is not None
            else None
        )
        if existing is not None:
            raise HTTPException(status_code=409, detail="Agent name already exists")
        updates = payload.model_dump(exclude_unset=True)
        template = get_agent_spec(updates.get("template_key", agent.template_key))
        if updates.get("template_key") is not None and template is None:
            raise HTTPException(status_code=422, detail="Unknown agent template")
        if template is not None:
            updates["system_prompt"] = template.system_prompt
        if updates.get("llm_api_key") == "":
            updates.pop("llm_api_key")
        for field, value in updates.items():
            setattr(agent, field, value)
        if not agent.is_orchestrator:
            apply_global_llm_config(agent, get_global_llm_config(session))
        session.commit()
        session.refresh(agent)
        return agent_response(agent)


@app.get("/api/agents", response_model=list[AgentResponse])
def list_agents(scenario_id: int | None = None) -> list[AgentResponse]:
    with SessionLocal() as session:
        agents = (
            scenario_scoped_agents(session, scenario_id)
            if scenario_id is not None
            else global_deliberative_agents(session)
        )
        return [agent_response(agent) for agent in agents]


@app.get("/api/agents/{agent_id}", response_model=AgentResponse)
def get_agent(agent_id: int) -> AgentResponse:
    with SessionLocal() as session:
        agent = session.get(Agent, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent_response(agent)


@app.post("/api/agents/{agent_id}/test-connection", response_model=LLMConnectionTestResponse)
def test_agent_connection(agent_id: int) -> LLMConnectionTestResponse:
    started_at = perf_counter()
    with SessionLocal() as session:
        agent = session.get(Agent, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        model = agent.llm_model or "UNCONFIGURED"
        base_url = agent.llm_base_url or "UNCONFIGURED"
        try:
            config = resolve_llm_runtime_config(agent)
            model = config.model
            base_url = config.base_url
            messages: list[Any] = [
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "Connection test"},
            ]
            log_llm_outbound(
                "connection_test",
                agent.name,
                config.base_url,
                config.model,
                {"messages": messages, "temperature": 0, "max_tokens": 8},
            )
            response = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=60.0,
                max_retries=2,
                default_headers=llm_request_headers(),
            ).chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=0,
                max_tokens=8,
            )
            content, _ = extract_llm_completion(response)
            return LLMConnectionTestResponse(
                agent_id=agent.id,
                ok=True,
                model=model,
                base_url=base_url,
                latency_ms=(perf_counter() - started_at) * 1000,
                response_preview=str(content)[:120],
            )
        except Exception as error:
            if isinstance(
                error,
                (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError, OSError, socket.error),
            ):
                logger.exception("Local LLM connection failed for agent %s", agent.id)
            else:
                logger.exception("LLM connection test failed for agent %s", agent.id)
            return LLMConnectionTestResponse(
                agent_id=agent.id,
                ok=False,
                model=model,
                base_url=base_url,
                latency_ms=(perf_counter() - started_at) * 1000,
                error=f"{type(error).__name__}: {error}",
            )


def _combined_domain_rules(agents: list[Agent], scenario: Scenario) -> dict[str, object]:
    specs = [spec for agent in agents if (spec := get_agent_spec(agent_domain_key(agent)))]
    return {
        "hard_constraints": sorted({item for spec in specs for item in spec.constraints}),
        "owned_checks": sorted({item for spec in specs for item in spec.owned_checks}),
        "principles": sorted({item for spec in specs for item in spec.decision_principles}),
        "primary_sources": sorted({item for spec in specs for item in spec.primary_sources}),
        "automatic_deficit_ceiling": STATUTORY_DEFICIT_CEILING_PERCENT_GDP,
    }


class MandateSynthesisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_mandate: str | list[Any] | dict[str, Any] | None = None
    scenario_focus: str | list[Any] | dict[str, Any] | None = None
    priority_questions: str | list[Any] | dict[str, Any] | None = None
    required_evidence: str | list[Any] | dict[str, Any] | None = None
    epistemic_logic_traceability: str | list[Any] | dict[str, Any] | None = None
    structured_consensus_protocol: str | list[Any] | dict[str, Any] | None = None
    regulatory_compliance_alignment: str | list[Any] | dict[str, Any] | None = None

    @field_validator("*", mode="before")
    @classmethod
    def accept_supported_value_shapes(cls, value: object) -> object:
        return value if isinstance(value, (str, list, dict)) else None


def _mandate_json_repair_applied(raw_content: str) -> bool:
    cleaned = raw_content.strip().lstrip("\ufeff")
    if repair_json_trailing_commas(cleaned) != cleaned:
        return True
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return False
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    nested = message.get("content") if isinstance(message, dict) else None
    return isinstance(nested, str) and repair_json_trailing_commas(nested) != nested


def _recover_mandate_fields(raw_content: str) -> dict[str, Any]:
    aliases = {
        "scenario_mandate": ("scenario_mandate", "scenarioMandate", "mandate", "operating_mandate"),
        "scenario_focus": ("scenario_focus", "scenarioFocus", "focus", "focus_areas"),
        "priority_questions": ("priority_questions", "priorityQuestions", "questions", "key_questions"),
        "required_evidence": ("required_evidence", "requiredEvidence", "evidence", "evidence_requirements"),
        "epistemic_logic_traceability": ("epistemic_logic_traceability", "epistemicTraceability", "traceability", "auditability"),
        "structured_consensus_protocol": ("structured_consensus_protocol", "structuredConsensusProtocol", "consensus_protocol", "disagreement_handling"),
        "regulatory_compliance_alignment": ("regulatory_compliance_alignment", "regulatoryComplianceAlignment", "compliance_alignment", "regulatory_alignment"),
    }
    cleaned = repair_json_trailing_commas(raw_content.strip().lstrip("\ufeff"))
    decoder = json.JSONDecoder()
    recovered: dict[str, Any] = {}
    for canonical, field_aliases in aliases.items():
        key_pattern = "|".join(re.escape(alias) for alias in field_aliases)
        match = re.search(
            rf"(?i)[\"'](?:{key_pattern})[\"']\s*:\s*",
            cleaned,
        )
        if match is None:
            continue
        remainder = cleaned[match.end() :].lstrip()
        try:
            value, _ = decoder.raw_decode(remainder)
        except json.JSONDecodeError:
            if remainder.startswith("'"):
                scalar = re.match(r"'((?:\\.|[^'\\])*)'", remainder, flags=re.DOTALL)
                value = scalar.group(1) if scalar else None
            elif remainder.startswith("["):
                closing = remainder.find("]")
                fragment = remainder[: closing + 1] if closing >= 0 else remainder
                value = re.findall(r"[\"']([^\"']+)[\"']", fragment)
            else:
                value = None
        if isinstance(value, (str, list, dict)):
            recovered[canonical] = value
    return recovered


def _validated_mandate_payload(payload: dict[str, Any]) -> MandateSynthesisResponse:
    unwrapped = _mandate_payload(payload)
    choices = unwrapped.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            unwrapped = _mandate_payload(extract_json_object(message["content"]))
    return MandateSynthesisResponse.model_validate(
        {
            "scenario_mandate": _response_value(
                unwrapped,
                "scenario_mandate",
                "scenarioMandate",
                "mandate",
                "operating_mandate",
            ),
            "scenario_focus": _response_value(
                unwrapped,
                "scenario_focus",
                "scenarioFocus",
                "focus",
                "focus_areas",
            ),
            "priority_questions": _response_value(
                unwrapped,
                "priority_questions",
                "priorityQuestions",
                "questions",
                "key_questions",
            ),
            "required_evidence": _response_value(
                unwrapped,
                "required_evidence",
                "requiredEvidence",
                "evidence",
                "evidence_requirements",
            ),
            "epistemic_logic_traceability": _response_value(
                unwrapped,
                "epistemic_logic_traceability",
                "epistemicTraceability",
                "traceability",
                "auditability",
            ),
            "structured_consensus_protocol": _response_value(
                unwrapped,
                "structured_consensus_protocol",
                "structuredConsensusProtocol",
                "consensus_protocol",
                "disagreement_handling",
            ),
            "regulatory_compliance_alignment": _response_value(
                unwrapped,
                "regulatory_compliance_alignment",
                "regulatoryComplianceAlignment",
                "compliance_alignment",
                "regulatory_alignment",
            ),
        }
    )
class MandateSynthesisFallbacks(BaseModel):
    scenario_mandate: str
    scenario_focus: list[str]
    priority_questions: list[str]
    required_evidence: list[str]
    epistemic_logic_traceability: list[str]
    structured_consensus_protocol: list[str]
    regulatory_compliance_alignment: list[str]


MANDATE_DEFAULT_FOCUS = [
    "Kebijakan Fiskal",
    "Optimalisasi Penerimaan Negara",
    "Makroekonomi",
]


def _fallback_scenario_mandate(agent: Agent, scenario: Scenario) -> str:
    description = scenario.description.strip() or "evaluasi kebijakan APBN umum"
    return f"Evaluate the active APBN policy scenario from the {agent.role} mandate: {description}"


def _mandate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    wrapper_keys = {"result", "data", "output", "mandateresult"}
    for key, wrapped in payload.items():
        if _canonical_response_key(key) in wrapper_keys and isinstance(wrapped, dict):
            return wrapped
    return payload


def _canonical_response_key(key: str) -> str:
    return "".join(character.lower() for character in key if character.isalnum())


def _response_value(payload: dict[str, Any], *keys: str) -> object:
    canonical_payload = {
        _canonical_response_key(key): value
        for key, value in payload.items()
    }
    for key in keys:
        canonical_key = _canonical_response_key(key)
        if canonical_key in canonical_payload and canonical_payload[canonical_key] is not None:
            return canonical_payload[canonical_key]
    return None


def _normalise_mandate_text(value: object, fallback: str, field_name: str, agent_id: int) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        nested = _response_value(value, "content", "text", "value", "mandate")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    if isinstance(value, (list, tuple, set)):
        items = _normalise_mandate_list(value, [], field_name, agent_id)
        if items:
            return " ".join(items)
    logger.warning(
        "Agent %s returned an empty or invalid %s; using safe fallback",
        agent_id,
        field_name,
    )
    return fallback


def _normalise_mandate_list(
    value: object,
    fallback: list[str],
    field_name: str,
    agent_id: int,
) -> list[str]:
    if isinstance(value, str):
        valid_items = [item.strip(" -•\t") for item in value.replace(";", "\n").splitlines()]
        valid_items = [item for item in valid_items if item]
    elif isinstance(value, dict):
        nested = _response_value(
            value,
            "content",
            "text",
            "name",
            "question",
            "evidence",
            "description",
            "mechanism",
            "protocol",
            "alignment",
            "traceability",
        )
        if nested is not None:
            valid_items = _normalise_mandate_list(nested, [], field_name, agent_id)
        else:
            valid_items = []
            for candidate in value.values():
                if isinstance(candidate, str):
                    valid_items.extend(
                        item.strip(" -•\t")
                        for item in candidate.replace(";", "\n").splitlines()
                        if item.strip(" -•\t")
                    )
                elif isinstance(candidate, list):
                    for item in candidate:
                        if isinstance(item, str) and item.strip():
                            valid_items.append(item.strip())
    elif isinstance(value, (list, tuple, set)):
        valid_items = []
        for item in value:
            if isinstance(item, str) and item.strip():
                valid_items.append(item.strip())
            elif isinstance(item, dict):
                text = next(
                    (
                        candidate.strip()
                        for key in ("content", "text", "name", "question", "evidence", "description", "mechanism", "protocol", "alignment", "traceability")
                        if isinstance((candidate := item.get(key)), str) and candidate.strip()
                    ),
                    None,
                )
                if text is not None:
                    valid_items.append(text)
    else:
        valid_items = []
    if valid_items:
        return list(dict.fromkeys(valid_items))
    logger.warning(
        "Agent %s returned an empty or invalid %s; using safe fallback",
        agent_id,
        field_name,
    )
    return fallback.copy()


def _synthesis_fallbacks(agent: Agent, scenario: Scenario) -> MandateSynthesisFallbacks:
    return MandateSynthesisFallbacks(
        scenario_mandate=_fallback_scenario_mandate(agent, scenario),
        scenario_focus=MANDATE_DEFAULT_FOCUS.copy(),
        priority_questions=[
            "What legal, fiscal, and implementation conditions must be verified before adoption?"
        ],
        required_evidence=[
            "Current APBN baseline, source-linked fiscal assumptions, and implementation evidence"
        ],
        epistemic_logic_traceability=[
            "Attach source tags and verification status to material claims and preserve auditable inter-agent handoffs"
        ],
        structured_consensus_protocol=[
            "Classify disagreements, preserve valid dissent, and escalate unresolved conflicts through DDR and CAR"
        ],
        regulatory_compliance_alignment=[
            f"Enforce the {STATUTORY_DEFICIT_CEILING_PERCENT_GDP}% GDP deficit ceiling and reject unverified fiscal offsets"
        ],
    )


def _base_agent_domain_rules(agent: Agent) -> AgentDomainRules:
    seed = mandate_seed(agent)
    return AgentDomainRules(
        agent_id=agent.id,
        name=agent.name,
        display_name=semantic_agent_name(agent),
        utility_metadata=agent_utility_metadata(
            agent,
            source="generated",
            result="Menunggu keluaran argumentasi terstruktur.",
        ),
        role=agent.role,
        template_key=agent.template_key,
        mandate=str(seed["mandate"]) or None,
        primary_sources=list(seed["primary_sources"]),
        constraints=list(seed["constraints"]),
        owned_checks=list(seed["owned_checks"]),
    )


def _fallback_agent_domain_rules(agent: Agent, scenario: Scenario) -> AgentDomainRules:
    base = _base_agent_domain_rules(agent)
    fallback_values = _synthesis_fallbacks(agent, scenario)
    return base.model_copy(
        update={
            "display_name": semantic_agent_name(agent, "fallback"),
            "utility_metadata": agent_utility_metadata(
                agent,
                source="fallback",
                result="Mandat deterministik dipakai karena generasi dinamis tidak tersedia.",
            ),
            "scenario_mandate": fallback_values.scenario_mandate,
            "scenario_focus": fallback_values.scenario_focus,
            "priority_questions": fallback_values.priority_questions,
            "required_evidence": fallback_values.required_evidence,
            "epistemic_logic_traceability": fallback_values.epistemic_logic_traceability,
            "structured_consensus_protocol": fallback_values.structured_consensus_protocol,
            "regulatory_compliance_alignment": fallback_values.regulatory_compliance_alignment,
            "applicable_primary_sources": base.primary_sources,
            "applicable_constraints": base.constraints,
            "applicable_owned_checks": base.owned_checks,
        }
    )


def _synthesize_agent_domain_rules(agent: Agent, scenario: Scenario) -> AgentDomainRules:
    base = _base_agent_domain_rules(agent)
    started_at = perf_counter()
    try:
        with warnings.catch_warnings(record=True) as configuration_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            config = resolve_llm_runtime_config(agent)
        for warning in configuration_warnings:
            logger.warning("Mandate synthesis configuration: %s", warning.message)
        request_payload: dict[str, Any] = {
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": build_mandate_synthesis_system_prompt(agent.name, agent.role),
                },
                {
                    "role": "user",
                    "content": build_mandate_synthesis_prompt(
                        agent.name,
                        agent.role,
                        scenario.description,
                        primary_sources=base.primary_sources,
                        constraints=base.constraints,
                        owned_checks=base.owned_checks,
                        simulation_payload=scenario.simulation_payload(),
                    ),
                },
            ],
        }
        log_llm_outbound(
            "mandate_synthesis",
            agent.name,
            config.base_url,
            config.model,
            request_payload,
        )
        response = _create_completion_with_retry(
            OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=60.0,
                max_retries=0,
                default_headers=llm_request_headers(),
            ).chat.completions,
            operation=f"Mandate synthesis for agent {agent.id}",
            request={
                "model": config.model,
                "temperature": agent.temperature,
                "max_tokens": agent.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": request_payload["messages"],
            },
        )
        content, tokens = extract_llm_completion(response)
        repaired_trailing_commas = _mandate_json_repair_applied(content)
        parse_error: MandateSynthesisError | None = None
        try:
            parsed = _validated_mandate_payload(extract_json_object(content))
            if repaired_trailing_commas:
                parse_error = MandateSynthesisError(
                    code="INVALID_JSON",
                    message="JSON trailing commas were repaired before validation",
                    retryable=False,
                )
        except (ValueError, json.JSONDecodeError) as error:
            logger.exception(
                "Mandate synthesis returned malformed JSON for agent %s; attempting field-level repair",
                agent.id,
            )
            recovered = _recover_mandate_fields(content)
            parsed = _validated_mandate_payload(recovered)
            parse_error = MandateSynthesisError(
                code="INVALID_JSON",
                message=f"{type(error).__name__}: response repaired field by field",
                retryable=False,
            )
            logger.warning(
                "Mandate synthesis JSON repaired for agent %s; recovered fields: %s",
                agent.id,
                ", ".join(sorted(recovered)) or "none (safe field defaults applied)",
            )
        fallback_values = _synthesis_fallbacks(agent, scenario)
        scenario_mandate = _normalise_mandate_text(
            parsed.scenario_mandate,
            fallback_values.scenario_mandate,
            "scenario_mandate",
            agent.id,
        )
        scenario_focus = _normalise_mandate_list(
            parsed.scenario_focus,
            fallback_values.scenario_focus,
            "scenario_focus",
            agent.id,
        )
        priority_questions = _normalise_mandate_list(
            parsed.priority_questions,
            fallback_values.priority_questions,
            "priority_questions",
            agent.id,
        )
        required_evidence = _normalise_mandate_list(
            parsed.required_evidence,
            fallback_values.required_evidence,
            "required_evidence",
            agent.id,
        )
        epistemic_logic_traceability = _normalise_mandate_list(
            parsed.epistemic_logic_traceability,
            fallback_values.epistemic_logic_traceability,
            "epistemic_logic_traceability",
            agent.id,
        )
        structured_consensus_protocol = _normalise_mandate_list(
            parsed.structured_consensus_protocol,
            fallback_values.structured_consensus_protocol,
            "structured_consensus_protocol",
            agent.id,
        )
        regulatory_compliance_alignment = _normalise_mandate_list(
            parsed.regulatory_compliance_alignment,
            fallback_values.regulatory_compliance_alignment,
            "regulatory_compliance_alignment",
            agent.id,
        )
        return base.model_copy(
            update={
                "synthesis_status": "generated",
                "scenario_mandate": scenario_mandate,
                "scenario_focus": scenario_focus,
                "priority_questions": priority_questions,
                "required_evidence": required_evidence,
                "epistemic_logic_traceability": epistemic_logic_traceability,
                "structured_consensus_protocol": structured_consensus_protocol,
                "regulatory_compliance_alignment": regulatory_compliance_alignment,
                "applicable_primary_sources": base.primary_sources,
                "applicable_constraints": base.constraints,
                "applicable_owned_checks": base.owned_checks,
                "llm_model": config.model,
                "latency_ms": (perf_counter() - started_at) * 1000,
                "token_usage": tokens,
                "error": parse_error,
            }
        )
    except Exception as error:
        if isinstance(
            error,
            (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError, OSError, socket.error),
        ) or _llm_status_code(error) is not None:
            logger.warning(
                "[WARNING] LLM provider unreachable. Injecting deterministic fallback mandate "
                "for agent %s (%s, status=%s)",
                agent.id,
                type(error).__name__,
                _llm_status_code(error),
            )
        else:
            logger.warning(
                "[WARNING] Mandate synthesis fell back to static template due to LLM provider failure "
                "for agent %s (%s)",
                agent.id,
                type(error).__name__,
            )
        code: Literal["CONFIGURATION_ERROR", "PROVIDER_ERROR", "EMPTY_CONTENT", "INVALID_JSON", "SCHEMA_ERROR"] = (
            "CONFIGURATION_ERROR"
            if isinstance(error, ValueError) and "configuration" in str(error)
            else "PROVIDER_ERROR"
        )
        if isinstance(error, ValueError) and "JSON" in str(error):
            code = "INVALID_JSON"
        elif isinstance(error, ValueError) and ("scenario_" in str(error) or "required_evidence" in str(error)):
            code = "SCHEMA_ERROR"
        elif isinstance(error, ValueError) and "text content" in str(error):
            code = "EMPTY_CONTENT"
        fallback_values = _synthesis_fallbacks(agent, scenario)
        return base.model_copy(
            update={
                "scenario_mandate": fallback_values.scenario_mandate,
                "scenario_focus": fallback_values.scenario_focus,
                "priority_questions": fallback_values.priority_questions,
                "required_evidence": fallback_values.required_evidence,
                "epistemic_logic_traceability": fallback_values.epistemic_logic_traceability,
                "structured_consensus_protocol": fallback_values.structured_consensus_protocol,
                "regulatory_compliance_alignment": fallback_values.regulatory_compliance_alignment,
                "applicable_primary_sources": base.primary_sources,
                "applicable_constraints": base.constraints,
                "applicable_owned_checks": base.owned_checks,
                "latency_ms": (perf_counter() - started_at) * 1000,
                "error": MandateSynthesisError(
                    code=code,
                    message=f"{type(error).__name__}: mandate synthesis failed",
                    retryable=code in {"PROVIDER_ERROR", "EMPTY_CONTENT"},
                ),
            }
        )


def _save_domain_rules_snapshot(session: Session, response: DomainRulesResponse) -> None:
    snapshot = session.scalar(
        select(ScenarioMandateSnapshot).where(
            ScenarioMandateSnapshot.scenario_id == response.scenario_id,
            ScenarioMandateSnapshot.revision == response.revision,
        )
    )
    values = {
        "generated": response.generated,
        "agent_count": response.agent_count,
        "rules": response.rules,
        "agent_rules": [rule.model_dump(mode="json") for rule in response.agent_rules],
        "status": response.status,
        "generated_count": response.generated_count,
        "failure_count": response.failure_count,
        "detail": response.detail,
    }
    if snapshot is None:
        snapshot = ScenarioMandateSnapshot(
            scenario_id=response.scenario_id,
            revision=response.revision,
            **values,
        )
        session.add(snapshot)
    else:
        for field, value in values.items():
            setattr(snapshot, field, value)
    session.commit()


def _safe_agent_domain_rule(raw_rule: object) -> AgentDomainRules:
    raw = dict(raw_rule) if isinstance(raw_rule, dict) else {}
    defaults: dict[str, object] = {
        "agent_id": 0,
        "name": "Unknown agent",
        "display_name": "Fallback_Policy_Reviewer",
        "utility_metadata": {
            "task": "Evaluate the active policy scenario.",
            "result": "Stored legacy mandate metadata is incomplete.",
            "why": "A safe fallback preserves compatibility without fabricating identity.",
            "name_source": "fallback",
        },
        "role": "Unknown role",
        "template_key": None,
        "mandate": None,
        "primary_sources": [],
        "constraints": [],
        "owned_checks": [],
        "synthesis_status": "fallback",
        "scenario_mandate": None,
        "scenario_focus": [],
        "priority_questions": [],
        "required_evidence": [],
        "epistemic_logic_traceability": [],
        "structured_consensus_protocol": [],
        "regulatory_compliance_alignment": [],
        "applicable_primary_sources": [],
        "applicable_constraints": [],
        "applicable_owned_checks": [],
        "llm_model": None,
        "latency_ms": None,
        "token_usage": 0,
        "error": None,
    }
    for key, value in defaults.items():
        if raw.get(key) is None and value is not None:
            raw[key] = value
        else:
            raw.setdefault(key, value)
    raw["agent_id"] = raw["agent_id"] if isinstance(raw["agent_id"], int) else 0
    raw["name"] = str(raw["name"] or "Unknown agent")
    raw["role"] = str(raw["role"] or "Unknown role")
    if raw["synthesis_status"] not in {"generated", "fallback"}:
        raw["synthesis_status"] = "fallback"
    if not isinstance(raw["error"], dict):
        raw["error"] = None
    elif raw["error"].get("code") not in {
        "CONFIGURATION_ERROR",
        "PROVIDER_ERROR",
        "EMPTY_CONTENT",
        "INVALID_JSON",
        "SCHEMA_ERROR",
    }:
        raw["error"] = {
            "code": "SCHEMA_ERROR",
            "message": str(raw["error"].get("message") or "Stored mandate error"),
            "retryable": False,
        }
    else:
        raw["error"].setdefault("message", "Stored mandate error")
        raw["error"].setdefault("retryable", False)
    for key in (
        "primary_sources",
        "constraints",
        "owned_checks",
        "scenario_focus",
        "priority_questions",
        "required_evidence",
        "epistemic_logic_traceability",
        "structured_consensus_protocol",
        "regulatory_compliance_alignment",
        "applicable_primary_sources",
        "applicable_constraints",
        "applicable_owned_checks",
    ):
        if not isinstance(raw[key], list):
            raw[key] = [str(raw[key])] if raw[key] else []
    return AgentDomainRules.model_validate(raw)

def _domain_rules_from_snapshot(
    snapshot: ScenarioMandateSnapshot,
    current_revision: str,
    current_agent_count: int,
) -> DomainRulesResponse:
    is_stale = snapshot.revision != current_revision or snapshot.agent_count != current_agent_count
    detail = snapshot.detail
    if is_stale:
        detail = "Saved mandates are stale because the scenario or agent configuration changed."
    return DomainRulesResponse(
        scenario_id=snapshot.scenario_id,
        revision=snapshot.revision,
        generated=snapshot.generated,
        stale=is_stale,
        agent_count=snapshot.agent_count,
        rules=snapshot.rules if isinstance(snapshot.rules, dict) else {},
        agent_rules=[
            _safe_agent_domain_rule(rule)
            for rule in (snapshot.agent_rules if isinstance(snapshot.agent_rules, list) else [])
        ],
        status=(
            "stale"
            if is_stale
            else cast(
                Literal["success", "partial", "failed", "stale", "missing"],
                snapshot.status,
            )
        ),
        generated_count=snapshot.generated_count,
        failure_count=snapshot.failure_count,
        detail=detail,
    )


@app.post(
    "/api/scenarios/{scenario_id}/domain-rules",
    response_model=DomainRulesResponse,
    responses={
        404: {"content": {"application/json": {}}},
        409: {"content": {"application/json": {}}},
        500: {"content": {"application/json": {}}},
    },
)
def generate_domain_rules(scenario_id: int) -> DomainRulesResponse | JSONResponse:
    try:
        with SessionLocal() as session:
            scenario = session.get(Scenario, scenario_id)
            if scenario is None:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": "Scenario not found"},
                )
            agents = scenario_deliberative_agents(session, scenario_id)
            if not agents:
                return JSONResponse(
                    status_code=409,
                    content={"success": False, "detail": "At least one agent is required"},
                )
            with ThreadPoolExecutor(max_workers=min(len(agents), 5)) as executor:
                agent_rules = list(
                    executor.map(
                        lambda agent: _synthesize_agent_domain_rules(agent, scenario),
                        agents,
                    )
                )
            generated_count = sum(rule.synthesis_status == "generated" for rule in agent_rules)
            failure_count = len(agent_rules) - generated_count
            revision = agent_revision(agents, scenario)
            rules = _combined_domain_rules(agents, scenario)
            if generated_count == 0:
                detail = (
                    "LLM provider unavailable; deterministic fallback mandates were "
                    f"generated for all {failure_count} agents."
                )
                response_status: Literal["success", "partial", "failed"] = "failed"
            elif failure_count:
                detail = (
                    f"Generated {generated_count} scenario mandates; {failure_count} "
                    "agents use deterministic fallback."
                )
                response_status = "partial"
            else:
                detail = f"Generated {generated_count} scenario-specific mandates."
                response_status = "success"
            response = DomainRulesResponse(
                scenario_id=scenario_id,
                revision=revision,
                generated=True,
                stale=False,
                agent_count=len(agents),
                rules=rules,
                agent_rules=agent_rules,
                status=response_status,
                generated_count=generated_count,
                failure_count=failure_count,
                detail=detail,
            )
            _save_domain_rules_snapshot(session, response)
            return response
    except Exception as error:
        logger.exception("Unexpected domain mandate generation failure for scenario %s", scenario_id)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": f"Mandate generation failed: {type(error).__name__}",
            },
        )


@app.post(
    "/api/scenarios/{scenario_id}/agents/{agent_id}/domain-rules",
    response_model=AgentDomainRules,
)
def generate_agent_domain_rules(scenario_id: int, agent_id: int) -> AgentDomainRules:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        agent = session.get(Agent, agent_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        if is_master_orchestrator(agent):
            raise HTTPException(
                status_code=409,
                detail="Master_Orchestrator is a non-voting controller",
            )
        if agent.scenario_id is not None and agent.scenario_id != scenario_id:
            raise HTTPException(
                status_code=404,
                detail="Agent is not assigned to this scenario",
            )
        rule = _synthesize_agent_domain_rules(agent, scenario)
        agents = scenario_deliberative_agents(session, scenario_id)
        current_revision = agent_revision(agents, scenario)
        snapshot = session.scalar(
            select(ScenarioMandateSnapshot)
            .where(
                ScenarioMandateSnapshot.scenario_id == scenario_id,
                ScenarioMandateSnapshot.revision == current_revision,
            )
        )
        if snapshot is None:
            agent_rules = [
                rule if item.id == agent_id else _fallback_agent_domain_rules(item, scenario)
                for item in agents
            ]
            generated_count = sum(
                item.synthesis_status == "generated" for item in agent_rules
            )
            response = DomainRulesResponse(
                scenario_id=scenario_id,
                revision=current_revision,
                generated=True,
                stale=False,
                agent_count=len(agents),
                rules=_combined_domain_rules(agents, scenario),
                agent_rules=agent_rules,
                status="partial" if generated_count < len(agents) else "success",
                generated_count=generated_count,
                failure_count=len(agents) - generated_count,
                detail=f"Generated mandate for agent {agent.name}.",
            )
            _save_domain_rules_snapshot(session, response)
        else:
            stored_rules = [
                item for item in snapshot.agent_rules if item.get("agent_id") != agent_id
            ]
            stored_rules.append(rule.model_dump(mode="json"))
            stored_rules.sort(key=lambda item: int(item["agent_id"]))
            snapshot.agent_rules = stored_rules
            snapshot.generated_count = sum(
                item.get("synthesis_status") == "generated" for item in stored_rules
            )
            snapshot.failure_count = len(stored_rules) - snapshot.generated_count
            snapshot.status = "partial" if snapshot.failure_count else "success"
            snapshot.detail = (
                f"Updated mandate for agent {agent.name}; {snapshot.generated_count} mandates available."
            )
            session.commit()
        return rule


@app.delete("/api/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
@app.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent(agent_id: int) -> Response:
    with SessionLocal() as session:
        agent = session.get(Agent, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        references = session.scalar(
            select(ReasoningLog.id).where(ReasoningLog.agent_id == agent_id).limit(1)
        ) or session.scalar(
            select(DisagreementLog.id)
            .where((DisagreementLog.agent_i == agent_id) | (DisagreementLog.agent_j == agent_id))
            .limit(1)
        ) or session.scalar(
            select(AgentInfluenceObservation.id)
            .where(AgentInfluenceObservation.agent_id == agent_id)
            .limit(1)
        )
        if references is not None:
            raise HTTPException(status_code=409, detail="Agent is referenced by research records")
        session.delete(agent)
        session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/scenarios/{scenario_id}/domain-rules", response_model=DomainRulesResponse)
def get_domain_rules(scenario_id: int) -> DomainRulesResponse:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        agents = scenario_deliberative_agents(session, scenario_id)
        current_revision = agent_revision(agents, scenario)
        snapshot = session.scalar(
            select(ScenarioMandateSnapshot)
            .where(ScenarioMandateSnapshot.scenario_id == scenario_id)
            .order_by(ScenarioMandateSnapshot.updated_at.desc(), ScenarioMandateSnapshot.id.desc())
        )
        if snapshot is None:
            return DomainRulesResponse(
                scenario_id=scenario_id,
                revision=current_revision,
                generated=False,
                stale=True,
                agent_count=len(agents),
                rules={},
                agent_rules=[],
                status="missing",
                generated_count=0,
                failure_count=0,
                detail="Generate scenario-specific mandates before starting deliberation.",
            )
        return _domain_rules_from_snapshot(snapshot, current_revision, len(agents))


def _clear_scenario_execution_data(session: Session, scenario_id: int) -> None:
    run_ids = list(
        session.scalars(
            select(ConsensusSession.id).where(ConsensusSession.scenario_id == scenario_id)
        )
    )
    if run_ids:
        for model, column in (
            (ReasoningLog, ReasoningLog.run_id),
            (DisagreementLog, DisagreementLog.run_id),
            (MetricSnapshot, MetricSnapshot.run_id),
            (AgentInfluenceObservation, AgentInfluenceObservation.run_id),
        ):
            session.execute(delete(model).where(column.in_(run_ids)))
    session.execute(delete(ReasoningLog).where(ReasoningLog.scenario_id == scenario_id))
    session.execute(delete(DisagreementLog).where(DisagreementLog.scenario_id == scenario_id))
    session.execute(delete(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id))
    session.execute(
        delete(AgentInfluenceObservation).where(
            AgentInfluenceObservation.scenario_id == scenario_id
        )
    )
    session.execute(
        delete(SimulationArtifact).where(SimulationArtifact.scenario_id == scenario_id)
    )
    session.execute(
        delete(ConsensusSession).where(ConsensusSession.scenario_id == scenario_id)
    )
    session.execute(
        delete(ScenarioMandateSnapshot).where(
            ScenarioMandateSnapshot.scenario_id == scenario_id
        )
    )


@app.patch("/api/scenarios/{scenario_id}", response_model=ScenarioResponse)
@app.patch("/scenarios/{scenario_id}", response_model=ScenarioResponse)
def update_scenario(scenario_id: int, payload: ScenarioUpdate) -> ScenarioResponse:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        updates = payload.model_dump(exclude_unset=True)
        if "description" in updates and updates["description"] is None:
            raise HTTPException(status_code=422, detail="Description cannot be null")
        execution_modes = {
            "skip_llm_formulation": updates.get(
                "skip_llm_formulation", scenario.skip_llm_formulation
            ),
            "formulation_dry_run_only": updates.get(
                "formulation_dry_run_only", scenario.formulation_dry_run_only
            ),
        }
        if all(value is True for value in execution_modes.values()):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Direct execution and formulation dry-run are mutually exclusive"
                ),
            )
        for field, value in updates.items():
            setattr(scenario, field, value)
        session.commit()
        session.refresh(scenario)
        return scenario_response(scenario)


@app.delete("/api/scenarios/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT)
@app.delete("/scenarios/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scenario(scenario_id: int) -> Response:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        _clear_scenario_execution_data(session, scenario_id)
        session.execute(delete(Agent).where(Agent.scenario_id == scenario_id))
        session.delete(scenario)
        session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/scenarios", response_model=ScenarioResponse, status_code=status.HTTP_201_CREATED)
def create_scenario(payload: ScenarioCreate) -> ScenarioResponse:
    with SessionLocal() as session:
        scenario = Scenario(**payload.model_dump())
        session.add(scenario)
        session.commit()
        session.refresh(scenario)
        return scenario_response(scenario)


@app.get("/api/scenarios", response_model=list[ScenarioResponse])
def list_scenarios() -> list[ScenarioResponse]:
    with SessionLocal() as session:
        return [
            scenario_response(scenario)
            for scenario in session.scalars(select(Scenario).order_by(Scenario.id))
        ]
