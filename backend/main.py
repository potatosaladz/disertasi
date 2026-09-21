from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text

from .agent_templates import load_standard_agent_templates, template_catalog
from .dashboard import router as dashboard_router
from .database import SessionLocal
from .init_db import initialize_database
from .models import Agent, Scenario


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


class ScenarioCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    max_deficit_constraint: float = Field(ge=0.0, allow_inf_nan=False)


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
def load_agent_templates() -> dict[str, object]:
    with SessionLocal() as session:
        agents, created = load_standard_agent_templates(session)
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
        agent = Agent(**payload.model_dump())
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
