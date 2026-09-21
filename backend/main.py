import hashlib
import json
import logging
import socket
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text

from .agent_templates import (
    get_agent_spec,
    load_standard_agent_templates,
    mandate_seed,
    template_catalog,
)
from .core_algorithms import (
    build_mandate_synthesis_prompt,
    extract_json_object,
    extract_llm_completion,
    llm_request_headers,
    log_llm_outbound,
    resolve_llm_runtime_config,
)
from .dashboard import router as dashboard_router
from .database import SessionLocal
from .init_db import initialize_database
from .models import (
    Agent,
    AgentInfluenceObservation,
    DisagreementLog,
    ReasoningLog,
    Scenario,
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


class AgentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    role: str = Field(min_length=1, max_length=255)
    template_key: str | None = Field(default=None, max_length=100)
    theta_x: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    theta_q: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    theta_h: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    theta_s: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    theta_u: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
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
    llm_base_url: str | None = Field(default=None, max_length=2048)
    llm_api_key: str | None = Field(default=None, max_length=4096)
    llm_model: str | None = Field(default=None, max_length=255)
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, gt=0)


class AgentResponse(BaseModel):
    id: int
    name: str
    role: str
    template_key: str | None
    theta_x: float
    theta_q: float
    theta_h: float
    theta_s: float
    theta_u: float
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


class ScenarioCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    program_cost: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    max_deficit_constraint: float = Field(default=3.0, ge=0.0, allow_inf_nan=False)


class ScenarioResponse(ScenarioCreate):
    id: int


def agent_response(agent: Agent) -> AgentResponse:
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        role=agent.role,
        template_key=agent.template_key,
        theta_x=agent.theta_x,
        theta_q=agent.theta_q,
        theta_h=agent.theta_h,
        theta_s=agent.theta_s,
        theta_u=agent.theta_u,
        llm_base_url=agent.llm_base_url,
        llm_model=agent.llm_model,
        system_prompt=agent.system_prompt,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens,
        has_llm_api_key=bool(agent.llm_api_key),
    )


def scenario_response(scenario: Scenario) -> ScenarioResponse:
    return ScenarioResponse(
        id=scenario.id,
        description=scenario.description,
        program_cost=scenario.program_cost,
        max_deficit_constraint=scenario.max_deficit_constraint,
    )


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


@app.get("/api/agent-templates")
def list_agent_templates() -> list[dict[str, object]]:
    return template_catalog()


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


@app.post("/api/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
@app.post("/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
def create_agent(payload: AgentCreate) -> AgentResponse:
    with SessionLocal() as session:
        existing = session.scalar(select(Agent).where(Agent.name == payload.name))
        if existing is not None:
            raise HTTPException(status_code=409, detail="Agent name already exists")
        values = payload.model_dump()
        template = get_agent_spec(payload.template_key)
        if payload.template_key is not None and template is None:
            raise HTTPException(status_code=422, detail="Unknown agent template")
        if template is not None:
            values["system_prompt"] = template.system_prompt
        agent = Agent(**values)
        session.add(agent)
        session.commit()
        session.refresh(agent)
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
                select(Agent).where(Agent.name == payload.name, Agent.id != agent_id)
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
        session.commit()
        session.refresh(agent)
        return agent_response(agent)


@app.get("/api/agents", response_model=list[AgentResponse])
def list_agents() -> list[AgentResponse]:
    with SessionLocal() as session:
        return [agent_response(agent) for agent in session.scalars(select(Agent).order_by(Agent.id))]


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


def _agent_revision(agents: list[Agent], scenario: Scenario | None = None) -> str:
    payload = {
        "scenario": (
            {
                "id": scenario.id,
                "description": scenario.description,
                "program_cost": scenario.program_cost,
                "max_deficit_constraint": scenario.max_deficit_constraint,
            }
            if scenario is not None
            else None
        ),
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "template_key": agent.template_key,
                "role": agent.role,
                "seed": mandate_seed(agent),
                "llm_base_url": agent.llm_base_url,
                "llm_model": agent.llm_model,
                "has_llm_api_key": bool(agent.llm_api_key),
                "temperature": agent.temperature,
                "max_tokens": agent.max_tokens,
                "theta_x": agent.theta_x,
                "theta_q": agent.theta_q,
                "theta_h": agent.theta_h,
                "theta_s": agent.theta_s,
                "theta_u": agent.theta_u,
            }
            for agent in agents
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _combined_domain_rules(agents: list[Agent], scenario: Scenario) -> dict[str, object]:
    specs = [spec for agent in agents if (spec := get_agent_spec(agent.template_key))]
    return {
        "hard_constraints": sorted({item for spec in specs for item in spec.constraints}),
        "owned_checks": sorted({item for spec in specs for item in spec.owned_checks}),
        "principles": sorted({item for spec in specs for item in spec.decision_principles}),
        "primary_sources": sorted({item for spec in specs for item in spec.primary_sources}),
        "automatic_deficit_ceiling": scenario.max_deficit_constraint,
    }


class MandateSynthesisFallbacks(BaseModel):
    scenario_mandate: str
    scenario_focus: list[str]
    priority_questions: list[str]
    required_evidence: list[str]


MANDATE_DEFAULT_FOCUS = [
    "Kebijakan Fiskal",
    "Optimalisasi Penerimaan Negara",
    "Makroekonomi",
]


def _fallback_scenario_mandate(agent: Agent, scenario: Scenario) -> str:
    description = scenario.description.strip() or "evaluasi kebijakan APBN umum"
    return f"Evaluate the active APBN policy scenario from the {agent.role} mandate: {description}"


def _mandate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for wrapper_key in ("result", "data", "output", "mandate_result"):
        wrapped = payload.get(wrapper_key)
        if isinstance(wrapped, dict):
            return wrapped
    return payload


def _response_value(payload: dict[str, Any], *keys: str) -> object:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _normalise_mandate_list(
    value: object,
    fallback: list[str],
    field_name: str,
    agent_id: int,
) -> list[str]:
    if isinstance(value, str):
        valid_items = [item.strip(" -•\t") for item in value.replace(";", "\n").splitlines()]
        valid_items = [item for item in valid_items if item]
    elif isinstance(value, (list, tuple, set)):
        valid_items = []
        for item in value:
            if isinstance(item, str) and item.strip():
                valid_items.append(item.strip())
            elif isinstance(item, dict):
                text = next(
                    (
                        candidate.strip()
                        for key in ("content", "text", "name", "question", "evidence")
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
    )


def _base_agent_domain_rules(agent: Agent) -> AgentDomainRules:
    seed = mandate_seed(agent)
    return AgentDomainRules(
        agent_id=agent.id,
        name=agent.name,
        role=agent.role,
        template_key=agent.template_key,
        mandate=str(seed["mandate"]) or None,
        primary_sources=list(seed["primary_sources"]),
        constraints=list(seed["constraints"]),
        owned_checks=list(seed["owned_checks"]),
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
                    "content": (
                        f"You are {agent.name}, an autonomous expert acting as {agent.role}. "
                        "Generate a concise, decision-ready operating mandate in valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": build_mandate_synthesis_prompt(
                        agent.name,
                        agent.role,
                        scenario.description,
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
        print("--- OUTBOUND PROMPT ---", flush=True)
        print(json.dumps(request_payload, indent=2, ensure_ascii=False, default=str), flush=True)
        response = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=60.0,
            max_retries=2,
            default_headers=llm_request_headers(),
        ).chat.completions.create(
            model=config.model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            response_format={"type": "json_object"},
            messages=request_payload["messages"],
        )
        content, tokens = extract_llm_completion(response)
        print("--- RAW LLM RESPONSE ---", flush=True)
        print(content, flush=True)
        payload = _mandate_payload(extract_json_object(content))
        fallback_values = _synthesis_fallbacks(agent, scenario)
        scenario_mandate = _response_value(
            payload,
            "scenario_mandate",
            "scenarioMandate",
            "mandate",
            "operating_mandate",
        )
        if not isinstance(scenario_mandate, str) or not scenario_mandate.strip():
            scenario_mandate = fallback_values.scenario_mandate
            logger.warning(
                "Agent %s returned an empty scenario_mandate; using scenario-aware fallback",
                agent.id,
            )
        scenario_focus = _normalise_mandate_list(
            _response_value(payload, "scenario_focus", "scenarioFocus", "focus", "focus_areas"),
            fallback_values.scenario_focus,
            "scenario_focus",
            agent.id,
        )
        priority_questions = _normalise_mandate_list(
            _response_value(
                payload,
                "priority_questions",
                "priorityQuestions",
                "questions",
                "key_questions",
            ),
            fallback_values.priority_questions,
            "priority_questions",
            agent.id,
        )
        required_evidence = _normalise_mandate_list(
            _response_value(
                payload,
                "required_evidence",
                "requiredEvidence",
                "evidence",
                "evidence_requirements",
            ),
            fallback_values.required_evidence,
            "required_evidence",
            agent.id,
        )
        return base.model_copy(
            update={
                "synthesis_status": "generated",
                "scenario_mandate": scenario_mandate.strip(),
                "scenario_focus": scenario_focus,
                "priority_questions": priority_questions,
                "required_evidence": required_evidence,
                "applicable_primary_sources": base.primary_sources,
                "applicable_constraints": base.constraints,
                "applicable_owned_checks": base.owned_checks,
                "llm_model": config.model,
                "latency_ms": (perf_counter() - started_at) * 1000,
                "token_usage": tokens,
            }
        )
    except Exception as error:
        if isinstance(
            error,
            (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError, OSError, socket.error),
        ):
            logger.exception("Local LLM connection failed during mandate synthesis for agent %s", agent.id)
        else:
            logger.exception("Mandate synthesis failed for agent %s", agent.id)
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
        return base.model_copy(
            update={
                "latency_ms": (perf_counter() - started_at) * 1000,
                "error": MandateSynthesisError(
                    code=code,
                    message=f"{type(error).__name__}: mandate synthesis failed",
                    retryable=code in {"PROVIDER_ERROR", "EMPTY_CONTENT"},
                ),
            }
        )


@app.post(
    "/api/scenarios/{scenario_id}/domain-rules",
    response_model=DomainRulesResponse,
    responses={
        404: {"content": {"application/json": {}}},
        409: {"content": {"application/json": {}}},
        502: {"content": {"application/json": {}}},
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
            agents = list(session.scalars(select(Agent).order_by(Agent.id)))
            if not agents:
                return JSONResponse(
                    status_code=409,
                    content={"success": False, "detail": "At least one agent is required"},
                )
            agent_rules = [_synthesize_agent_domain_rules(agent, scenario) for agent in agents]
            generated_count = sum(rule.synthesis_status == "generated" for rule in agent_rules)
            failure_count = len(agent_rules) - generated_count
            if generated_count == 0:
                return JSONResponse(
                    status_code=502,
                    content={
                        "success": False,
                        "detail": "All agent mandate synthesis calls failed",
                        "agent_rules": [rule.model_dump(mode="json") for rule in agent_rules],
                    },
                )
            response_status: Literal["success", "partial"] = (
                "partial" if failure_count else "success"
            )
            return DomainRulesResponse(
                scenario_id=scenario_id,
                revision=_agent_revision(agents, scenario),
                generated=True,
                stale=False,
                agent_count=len(agents),
                rules=_combined_domain_rules(agents, scenario),
                agent_rules=agent_rules,
                status=response_status,
                generated_count=generated_count,
                failure_count=failure_count,
                detail=(
                    f"Generated {generated_count} scenario mandates; {failure_count} agents use local fallback."
                    if failure_count
                    else f"Generated {generated_count} scenario-specific mandates."
                ),
            )
    except Exception as error:
        logger.exception("Unexpected domain mandate generation failure for scenario %s", scenario_id)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": f"Mandate generation failed: {type(error).__name__}",
            },
        )


@app.delete("/api/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
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
        agents = list(session.scalars(select(Agent).order_by(Agent.id)))
        return DomainRulesResponse(
            scenario_id=scenario_id,
            revision=_agent_revision(agents, scenario),
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
