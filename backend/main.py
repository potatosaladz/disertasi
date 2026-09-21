import hashlib
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Response, status
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text

from .agent_templates import (
    get_agent_spec,
    load_standard_agent_templates,
    template_catalog,
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


class AgentDomainRules(BaseModel):
    agent_id: int
    name: str
    role: str
    template_key: str | None
    mandate: str | None
    primary_sources: list[str]
    constraints: list[str]
    owned_checks: list[str]


class DomainRulesResponse(BaseModel):
    scenario_id: int
    revision: str
    generated: bool
    stale: bool
    agent_count: int
    rules: dict[str, object]
    agent_rules: list[AgentDomainRules]


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
        base_url = (
            agent.llm_base_url
            or os.getenv("OPENAI_BASE_URL")
            or "http://host.docker.internal:11434/v1"
        )
        api_key = agent.llm_api_key or os.getenv("OPENAI_API_KEY") or "local-llm"
        model = agent.llm_model or os.getenv("OPENAI_MODEL") or "local-model"
        try:
            response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Reply with exactly: OK"},
                    {"role": "user", "content": "Connection test"},
                ],
                temperature=0,
                max_tokens=8,
            )
            content = response if isinstance(response, str) else response.choices[0].message.content
            return LLMConnectionTestResponse(
                agent_id=agent.id,
                ok=True,
                model=model,
                base_url=base_url,
                latency_ms=(perf_counter() - started_at) * 1000,
                response_preview=str(content)[:120],
            )
        except Exception as error:
            return LLMConnectionTestResponse(
                agent_id=agent.id,
                ok=False,
                model=model,
                base_url=base_url,
                latency_ms=(perf_counter() - started_at) * 1000,
                error=f"{type(error).__name__}: {error}",
            )


def _agent_revision(agents: list[Agent]) -> str:
    payload = [
        {
            "id": agent.id,
            "template_key": agent.template_key,
            "role": agent.role,
            "llm_model": agent.llm_model,
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
            "theta_x": agent.theta_x,
            "theta_q": agent.theta_q,
            "theta_h": agent.theta_h,
            "theta_s": agent.theta_s,
            "theta_u": agent.theta_u,
        }
        for agent in agents
    ]
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


def _agent_domain_rules(agents: list[Agent]) -> list[AgentDomainRules]:
    rules: list[AgentDomainRules] = []
    for agent in agents:
        spec = get_agent_spec(agent.template_key)
        rules.append(
            AgentDomainRules(
                agent_id=agent.id,
                name=agent.name,
                role=agent.role,
                template_key=agent.template_key,
                mandate=spec.mandate if spec is not None else agent.system_prompt,
                primary_sources=list(spec.primary_sources) if spec is not None else [],
                constraints=list(spec.constraints) if spec is not None else [],
                owned_checks=list(spec.owned_checks) if spec is not None else [],
            )
        )
    return rules


@app.post("/api/scenarios/{scenario_id}/domain-rules", response_model=DomainRulesResponse)
def generate_domain_rules(scenario_id: int) -> DomainRulesResponse:
    with SessionLocal() as session:
        scenario = session.get(Scenario, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        agents = list(session.scalars(select(Agent).order_by(Agent.id)))
        if not agents:
            raise HTTPException(status_code=409, detail="At least one agent is required")
        return DomainRulesResponse(
            scenario_id=scenario_id,
            revision=_agent_revision(agents),
            generated=True,
            stale=False,
            agent_count=len(agents),
            rules=_combined_domain_rules(agents, scenario),
            agent_rules=_agent_domain_rules(agents),
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
            revision=_agent_revision(agents),
            generated=False,
            stale=True,
            agent_count=len(agents),
            rules={},
            agent_rules=[],
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
