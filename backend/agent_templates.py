import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import delete, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .core_algorithms import calculate_orchestrator_rar_dai_weights
from .models import Agent, Scenario
from .global_config import apply_global_values, get_global_llm_config


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    mandate: str
    primary_sources: tuple[str, ...]
    owned_checks: tuple[str, ...]
    parameters: tuple[str, ...]
    constraints: tuple[str, ...]
    impact_dimensions: tuple[str, ...]
    risk_dimensions: tuple[str, ...]
    uncertainty_dimensions: tuple[str, ...]
    decision_principles: tuple[str, ...]

    @property
    def role(self) -> str:
        return self.name.split(" / ", maxsplit=1)[-1]

    @property
    def system_prompt(self) -> str:
        sections = (
            ("MANDATE", (self.mandate,)),
            ("PRIMARY SOURCES", self.primary_sources),
            ("OWNED HARD CHECKS", self.owned_checks),
            ("VERIFIED PARAMETERS", self.parameters),
            ("CONSTRAINTS", self.constraints),
            ("IMPACT DIMENSIONS", self.impact_dimensions),
            ("RISK DIMENSIONS", self.risk_dimensions),
            ("UNCERTAINTY DIMENSIONS", self.uncertainty_dimensions),
            ("DECISION PRINCIPLES", self.decision_principles),
        )
        return "\n\n".join(
            f"{title}:\n" + "\n".join(f"- {item}" for item in items)
            for title, items in sections
        )

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "role": self.role,
                "description": self.mandate,
                "system_prompt": self.system_prompt,
                "temperature": 0.2,
                "max_tokens": 4000,
                "theta_x": 1.0,
                "theta_q": 1.0,
                "theta_h": 1.0,
                "theta_s": 1.0,
                "theta_u": 1.0,
                "rar_dai_weight_mode": "auto",
            }
        )
        return payload

    def as_agent_values(self) -> dict[str, Any]:
        return {
            "template_key": self.key,
            "name": self.name,
            "role": self.role,
            "system_prompt": self.system_prompt,
            "temperature": 0.2,
            "max_tokens": 4000,
            "theta_x": 1.0,
            "theta_q": 1.0,
            "theta_h": 1.0,
            "theta_s": 1.0,
            "theta_u": 1.0,
            "rar_dai_weight_mode": "auto",
        }


AGENTS: list[AgentSpec] = [
    AgentSpec(
        key="revenue",
        name="State Revenue Agent / Penerimaan Negara",
        mandate=(
            "Evaluate tax, customs/excise-relevant revenue implications, PNBP, grants, revenue realism, "
            "revenue timing, and the legality/verification of revenue offsets. Never invent a tax base, "
            "tariff, elasticity, compliance yield, or additional collection capacity."
        ),
        primary_sources=("UUD45_P23_23A_31", "UU17_2025_POSTURE", "UU17_2025_P42", "UU9_2018_PNBP", "PP44_2025_PNBP"),
        owned_checks=("TAX_LEGAL_BASIS", "PNBP_LEGAL_BASIS", "VERIFIED_OFFSETS_ONLY"),
        parameters=(
            "2026 state revenue target Rp3,153.580466863T",
            "tax revenue target Rp2,693.71425T",
            "PNBP target Rp459.199942626T",
            "grants target Rp0.666274237T",
            "Article 42 tax-revenue decline trigger: at least 10% from target",
            "verified revenue-offset capacity supplied externally",
            "optional tax-revenue forecast supplied externally",
        ),
        constraints=(
            "Tax/compulsory-levy policy needs proper statutory legal basis.",
            "PNBP tariff changes must follow UU 9/2018 and current PP 44/2025 framework.",
            "Only verified revenue yield may reduce the simulated deficit.",
            "A proposed future revenue yield is not treated as cash or fiscal space merely because an LLM predicts it.",
        ),
        impact_dimensions=(
            "revenue adequacy relative to the APBN target",
            "timing of cash inflows versus the proposed spending schedule",
            "recognized revenue offset and remaining unverified offset",
            "possible tax-base/compliance or PNBP-behaviour effects, only as qualitative inference unless evidence is supplied",
        ),
        risk_dimensions=(
            "revenue shortfall",
            "collection-timing mismatch",
            "unverified policy/administrative yield",
            "legal or administrative implementation risk of a revenue measure",
        ),
        uncertainty_dimensions=(
            "tax/PNBP realization forecast",
            "tax buoyancy or behavioural response",
            "timing and durability of additional collections",
            "evidence supporting the estimated revenue yield",
        ),
        decision_principles=(
            "Reject when an owned hard legal/evidence gate fails.",
            "Condition support when a material revenue claim is still unverified or a forecast relevant to the decision is unknown.",
            "Do not object to spending merely because it is spending; object only where the revenue mandate is materially affected.",
        ),
    ),
    AgentSpec(
        key="expenditure",
        name="Government Expenditure Agent / Belanja Pemerintah",
        mandate=(
            "Evaluate legal budget availability, expenditure composition, education protection, output/outcome orientation, "
            "domestic-product prioritization, implementation feasibility, targeting quality, and verified reallocation feasibility."
        ),
        primary_sources=("UU1_2004_P3", "UU17_2025_POSTURE", "UU17_2025_P8", "UU17_2025_P22", "UU17_2025_P28", "UUD45_P23_23A_31"),
        owned_checks=("APPROPRIATION_AVAILABLE", "EDUCATION_20PCT", "OUTPUT_OUTCOME", "DOMESTIC_PRODUCT_PRIORITY", "SPENDING_ADJUSTMENT_AUTHORITY"),
        parameters=(
            "2026 total State Expenditure Rp3,842.728369471T",
            "central-government spending Rp3,149.733390760T",
            "TKD Rp692.994978711T",
            "education budget Rp769.086869324T = 20.0% of State Expenditure",
            "verified appropriation and reallocation capacity supplied externally",
        ),
        constraints=(
            "No APBN expenditure without available/sufficient budget.",
            "Education share may not fall below the constitutional/APBN minimum.",
            "Central spending should be output/outcome oriented.",
            "Domestic production/TKDN prioritization must follow applicable rules.",
            "LLM claims are not execution evidence for budget authorization.",
        ),
        impact_dimensions=(
            "appropriation sufficiency and executable budget",
            "composition of spending and effect of reallocation on other programmes",
            "education-budget share",
            "expected outputs/outcomes and implementation readiness",
            "targeting/beneficiary quality where relevant to the proposed instrument",
        ),
        risk_dimensions=(
            "appropriation gap",
            "crowding out or disruption of priority programmes through reallocation",
            "implementation/execution delay",
            "weak output/outcome design",
            "targeting inclusion/exclusion or delivery risk when relevant",
        ),
        uncertainty_dimensions=(
            "line-item reallocation capacity",
            "programme implementation readiness",
            "beneficiary/targeting data quality",
            "real-world effectiveness of the spending instrument",
        ),
        decision_principles=(
            "Reject when expenditure lacks sufficient authorized budget or breaches a protected hard allocation.",
            "Condition support when implementation, outcome, or reallocation evidence is incomplete but the proposal can be made feasible.",
            "Never hard-code a 40-50% spending-review target unless it is supplied as scenario evidence or an applicable rule.",
        ),
    ),
    AgentSpec(
        key="financing",
        name="Budget Financing & Debt Agent / Pembiayaan Anggaran",
        mandate=(
            "Evaluate the APBN deficit, additional financing identity, financing gap, debt-financing authority, and borrowing "
            "sustainability without substituting unrelated debt statistics for a legal definition."
        ),
        primary_sources=("UU17_2003_P12", "UU17_2025_POSTURE", "UU17_2025_P28", "DJPPR_DEBT_20260630"),
        owned_checks=("DEFICIT_3PCT", "ADDITIONAL_FINANCING_IDENTITY", "BORROWING_60PCT", "DEBT_FINANCING_AUTHORITY"),
        parameters=(
            "enacted 2026 deficit Rp689.147902608T = 2.68% GDP",
            "general legal deficit ceiling 3% GDP",
            "2026 financing target Rp689.147902608T",
            "debt financing plan Rp832.208898829T",
            "official debt observation 30 Jun 2026: Rp10,293.69T / 41.26% GDP (context only)",
            "verified debt-financing headroom supplied externally",
        ),
        constraints=(
            "Projected deficit must remain <= 3% GDP.",
            "Incremental deficit requires recognized financing in the sandbox accounting identity.",
            "Additional SBN/cash-loan actions follow Article 28 authority/approval rules.",
            "60% borrowing test is NOT evaluated without a verified ratio matching the legal definition.",
        ),
        impact_dimensions=(
            "projected deficit and remaining statutory headroom",
            "additional financing need and financing gap",
            "recognized versus unverified debt financing",
            "borrowing ratio only when a definition-compatible verified baseline is supplied",
            "potential market/refinancing consequences stated qualitatively unless market evidence is supplied",
        ),
        risk_dimensions=(
            "deficit-limit breach",
            "unfunded financing gap",
            "unverified debt-financing headroom or authority",
            "interest-rate/refinancing/market absorption risk",
        ),
        uncertainty_dimensions=(
            "verified debt-financing headroom",
            "definition-compatible cumulative borrowing ratio",
            "market yield and investor-demand conditions",
            "refinancing profile affected by the proposal",
        ),
        decision_principles=(
            "Reject a hard deficit breach or unresolved funding gap that cannot be financed with verified resources.",
            "Condition support when debt/market information needed for prudent financing judgment is unknown.",
            "Do not claim that debt issuance will increase yields unless evidence supports the claim; report it as a potential risk otherwise.",
        ),
    ),
    AgentSpec(
        key="treasury",
        name="Treasury & Fiscal Liquidity Agent / Perbendaharaan dan Kas Negara",
        mandate=(
            "Protect the Government's ability to meet payment obligations, evaluate SAL use, cash availability, purpose-specific "
            "approval, timing of inflows/outflows, and operational cash requirements. Never invent a SAL floor."
        ),
        primary_sources=("UU17_2025_P27", "UU17_2025_P28", "UU17_2025_P42", "PMK44_2024_CASH"),
        owned_checks=("SAL_AVAILABILITY", "SAL_AUTHORITY", "OPERATIONAL_CASH_MINIMUM"),
        parameters=(
            "verified available SAL supplied externally; otherwise UNKNOWN",
            "verified operational cash minimum/projected cash supplied externally; otherwise UNKNOWN",
            "SAL purpose: cash management / cover deficit / SBN market stabilization / other",
            "Minister of Finance authorization and DPR approval evidence supplied externally",
        ),
        constraints=(
            "No hard-coded 1.5%-of-revenue SAL floor.",
            "SAL use cannot exceed verified availability.",
            "SBN-market-stabilization SAL use requires DPR approval under Article 27.",
            "Other purpose-specific SAL approval rules follow Article 28.",
            "Cash management must preserve sufficient access to cash and consider operational minimum/risk.",
        ),
        impact_dimensions=(
            "verified SAL remaining after the proposal",
            "projected cash position relative to a verified operational minimum",
            "timing/concentration of payment outflows",
            "continuity of government payment obligations",
        ),
        risk_dimensions=(
            "cash shortfall or timing mismatch",
            "SAL overuse or use without required authority",
            "concentration of large outflows before inflows",
            "payment-continuity risk",
        ),
        uncertainty_dimensions=(
            "actual available SAL",
            "cash-flow forecast after the policy",
            "operational minimum cash requirement",
            "payment calendar and timing of revenue inflows",
        ),
        decision_principles=(
            "Reject SAL use that exceeds verified availability or lacks a required hard authorization.",
            "Condition support when cash sufficiency cannot be evaluated because core cash evidence is missing.",
            "Phasing is a possible risk mitigation, not an invented statutory requirement or fixed quarterly rule.",
        ),
    ),
    AgentSpec(
        key="macro",
        name="Macro-Fiscal Stabilization Agent / Strategi Ekonomi dan Fiskal",
        mandate=(
            "Assess the policy against enacted 2026 macro assumptions, Article 42 APBN-adjustment triggers, fiscal stance, and "
            "plausible macro transmission. Distinguish deterministic facts from qualitative model inference."
        ),
        primary_sources=("UU17_2025_POSTURE", "UU17_2025_P42", "UU17_2003_P12"),
        owned_checks=("DEFICIT_3PCT",),
        parameters=(
            "growth 5.4%",
            "inflation 2.5%",
            "IDR/USD 16,500",
            "10-year SBN yield 6.9%",
            "ICP USD70/bbl",
            "oil lifting 610 kbpd",
            "gas lifting 984 kboepd",
            "Article 42 10% trigger rules",
            "optional current/prognosis indicators supplied externally",
        ),
        constraints=(
            "Do not invent a current/prognosis macro value when it was not supplied.",
            "Do not invent an MPC, fiscal multiplier, growth effect, inflation effect, or primary-balance effect.",
            "Clearly distinguish APBN adjustment triggers from hard illegality.",
            "Respect the 3% deficit ceiling when judging fiscal feasibility.",
        ),
        impact_dimensions=(
            "fiscal stance and deficit headroom",
            "growth/consumption transmission direction, without fabricated magnitude",
            "inflation/stability implications, without fabricated magnitude",
            "interaction with Article 42 macro-monitoring triggers",
            "policy reversibility/adaptability under changing macro conditions",
        ),
        risk_dimensions=(
            "pro-cyclicality or insufficient counter-cyclical response",
            "inflation or demand pressure",
            "macro-assumption deterioration",
            "loss of fiscal buffer under a further shock",
        ),
        uncertainty_dimensions=(
            "actual fiscal multiplier and transmission lag",
            "future growth/inflation/exchange-rate path",
            "behavioural response of households/firms",
            "feedback between fiscal action, financing conditions, and revenues",
        ),
        decision_principles=(
            "Reject a hard fiscal constraint breach.",
            "Use macro uncertainty explicitly; do not convert an uncertain transmission channel into a fabricated number.",
            "A triggered Article 42 indicator raises review/adjustment concern but is not itself a legal violation.",
        ),
    ),
]

STANDARD_APBN_AGENT_TEMPLATES = tuple(AGENTS)
MASTER_ORCHESTRATOR_TEMPLATE_KEY = "master_orchestrator"
MASTER_ORCHESTRATOR_NAME = "Master_Orchestrator"
MASTER_ORCHESTRATOR_ROLE = "Non-voting SHCR orchestration controller"
MASTER_ORCHESTRATOR_MANDATE = (
    "Coordinate phase ordering, detect uncovered scenario domains, reuse existing "
    "specialists, and create a missing specialist only when deterministic Phase 1 "
    "coverage analysis identifies a domain gap. The orchestrator never votes, "
    "receives RAR-DAI weight, or participates in DDR."
)
def _domain_patterns(*terms: str) -> tuple[re.Pattern[str], ...]:
    return tuple(
        re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])")
        for term in terms
    )


_DOMAIN_GAP_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "revenue": _domain_patterns("revenue", "tax", "pnbp", "penerimaan", "pajak", "cukai"),
    "expenditure": _domain_patterns("expenditure", "spending", "belanja", "education", "pendidikan"),
    "financing": _domain_patterns("financing", "debt", "deficit", "pembiayaan", "utang", "defisit"),
    "treasury": _domain_patterns("treasury", "liquidity", "cash", "sal", "kas", "likuiditas"),
    "macro": _domain_patterns("macro", "inflation", "growth", "exchange rate", "makro", "inflasi", "pertumbuhan", "nilai tukar", "stabilization"),
}
_AGENT_SPECS_BY_KEY: dict[str, AgentSpec] = {agent.key: agent for agent in AGENTS}
_DOMAIN_SCENARIO_FIELDS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "proposed_additional_revenue",
        "revenue_measure_type",
        "verified_revenue_offset_capacity",
        "tax_measure_has_enacted_law",
        "pnbp_measure_has_valid_tariff_instrument",
        "tax_revenue_forecast",
    ),
    "expenditure": (
        "instrument",
        "targeting",
        "program_cost",
        "duration_months",
        "proposed_reallocation",
        "reallocation_from_education",
        "appropriation_available",
        "verified_reallocation_capacity",
        "spending_reallocation_authorized",
        "dpr_spending_adjustment_recommendation",
        "output_outcome_documented",
        "domestic_product_compliance_documented",
    ),
    "financing": (
        "proposed_debt_financing",
        "debt_financing_mode",
        "proposed_other_financing",
        "verified_debt_financing_headroom",
        "verified_cumulative_borrowing_pct_gdp",
        "dpr_additional_sbn_approval_obtained",
    ),
    "treasury": (
        "proposed_sal_use",
        "sal_purpose",
        "verified_sal_available",
        "verified_operational_cash_minimum",
        "verified_projected_cash_after_policy",
        "finance_minister_sal_authorized",
        "dpr_sal_approval_obtained",
    ),
    "macro": (
        "growth_outlook",
        "inflation_outlook",
        "fx_outlook",
        "sbn10y_yield_outlook",
        "icp_outlook",
        "oil_lifting_outlook",
        "gas_lifting_outlook",
    ),
}

_DOMAIN_EVIDENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "verified_revenue_offset_capacity",
        "tax_measure_has_enacted_law",
        "pnbp_measure_has_valid_tariff_instrument",
        "tax_revenue_forecast",
    ),
    "expenditure": (
        "appropriation_available",
        "verified_reallocation_capacity",
        "spending_reallocation_authorized",
        "dpr_spending_adjustment_recommendation",
        "output_outcome_documented",
        "domestic_product_compliance_documented",
    ),
    "financing": (
        "verified_debt_financing_headroom",
        "verified_cumulative_borrowing_pct_gdp",
        "dpr_additional_sbn_approval_obtained",
    ),
    "treasury": (
        "verified_sal_available",
        "verified_operational_cash_minimum",
        "verified_projected_cash_after_policy",
        "finance_minister_sal_authorized",
        "dpr_sal_approval_obtained",
    ),
    "macro": (
        "growth_outlook",
        "inflation_outlook",
        "fx_outlook",
        "sbn10y_yield_outlook",
        "icp_outlook",
        "oil_lifting_outlook",
        "gas_lifting_outlook",
    ),
}

_SEMANTIC_AGENT_NAMES = {
    "revenue": "Dynamic_Revenue_Validator",
    "expenditure": "Dynamic_Expenditure_Reviewer",
    "financing": "Dynamic_Fiscal_Reviewer",
    "treasury": "Dynamic_Treasury_Guardian",
    "macro": "Dynamic_Macro_Fiscal_Analyst",
}


def is_generated_agent_name(name: str) -> bool:
    candidate = name.strip()
    return bool(
        re.fullmatch(
            r"(?:run-agent|fallback-rules|dynamic-agent|auto-agent)[-_][0-9a-f-]{8,}",
            candidate,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            r"(?:Dynamic|Fallback)_[A-Za-z0-9_]+",
            candidate,
        )
    )


def agent_domain_key(agent: Agent) -> str | None:
    return agent.template_key or agent.specialist_domain


def semantic_agent_name(agent: Agent, source: str = "generated") -> str:
    if agent.is_orchestrator:
        return MASTER_ORCHESTRATOR_NAME
    template_name = _SEMANTIC_AGENT_NAMES.get(agent_domain_key(agent) or "")
    if template_name is not None:
        return template_name if source == "generated" else template_name.replace("Dynamic_", "Fallback_", 1)
    role = "_".join(part.capitalize() for part in re.sub(r"[^A-Za-z0-9]+", " ", agent.role).split())
    role_name = role or "Policy_Reviewer"
    prefix = "Fallback" if source == "fallback" else "Dynamic"
    return f"{prefix}_{role_name}"[:255]


def agent_utility_metadata(agent: Agent, *, source: str, result: str) -> dict[str, Any]:
    spec = get_agent_spec(agent_domain_key(agent))
    checks = list(spec.owned_checks) if spec is not None else []
    task = (
        f"Validate {', '.join(checks)} for the active scenario."
        if checks
        else f"Evaluate the active scenario from the {agent.role} mandate."
    )
    why_by_template = {
        "financing": "Mencegah bias LLM tunggal pada evaluasi defisit dan pembiayaan.",
        "revenue": "Memisahkan penerimaan terverifikasi dari proyeksi LLM yang belum terealisasi.",
        "expenditure": "Menjaga legalitas, kualitas, dan prioritas belanja dalam proses kolektif.",
        "treasury": "Menjaga likuiditas kas dan otoritas pembiayaan agar tidak diasumsikan oleh LLM.",
        "macro": "Mengisolasi asumsi makro dan ketidakpastian transmisi dari fakta fiskal terverifikasi.",
    }
    return {
        "task": task,
        "result": result,
        "why": why_by_template.get(
            agent_domain_key(agent) or "",
            "Menambahkan perspektif fungsional independen untuk mengurangi bias agen tunggal.",
        ),
        "name_source": source,
    }


def mandate_seed(agent: Agent) -> dict[str, Any]:
    domain_key = agent_domain_key(agent)
    template = _AGENT_SPECS_BY_KEY.get(domain_key) if domain_key else None
    if template is not None:
        return {
            "mandate": template.mandate,
            "primary_sources": list(template.primary_sources),
            "owned_checks": list(template.owned_checks),
            "parameters": list(template.parameters),
            "constraints": list(template.constraints),
            "impact_dimensions": list(template.impact_dimensions),
            "risk_dimensions": list(template.risk_dimensions),
            "uncertainty_dimensions": list(template.uncertainty_dimensions),
            "decision_principles": list(template.decision_principles),
        }
    return {
        "mandate": agent.system_prompt or "",
        "primary_sources": [],
        "owned_checks": [],
        "parameters": [],
        "constraints": [],
        "impact_dimensions": [],
        "risk_dimensions": [],
        "uncertainty_dimensions": [],
        "decision_principles": [],
    }


def agent_revision(agents: list[Agent], scenario: object | None = None) -> str:
    payload = {
        "mandate_contract_version": "dissertation-consensus-v2",
        "scenario": (
            {
                "id": getattr(scenario, "id"),
                "description": getattr(scenario, "description"),
                "program_cost": getattr(scenario, "program_cost"),
                "simulation_payload": (
                    scenario.simulation_payload()
                    if isinstance(scenario, Scenario)
                    else {}
                ),
            }
            if scenario is not None
            else None
        ),
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "template_key": agent.template_key,
                "specialist_domain": agent.specialist_domain,
                "scenario_id": agent.scenario_id,
                "is_orchestrator": agent.is_orchestrator,
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
                "rar_dai_weight_mode": agent.rar_dai_weight_mode,
            }
            for agent in agents
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def get_agent_spec(template_key: str | None) -> AgentSpec | None:
    if template_key is None:
        return None
    return _AGENT_SPECS_BY_KEY.get(template_key)


def resolve_agent_system_prompt(agent: Agent) -> str | None:
    template = get_agent_spec(agent_domain_key(agent))
    return template.system_prompt if template is not None else agent.system_prompt


def template_catalog() -> list[dict[str, Any]]:
    return [template.as_payload() for template in AGENTS]


def find_agent_by_name_and_scenario(
    session: Session,
    *,
    name: str,
    scenario_id: int | None,
) -> Agent | None:
    scenario_filter = (
        Agent.scenario_id.is_(None)
        if scenario_id is None
        else Agent.scenario_id == scenario_id
    )
    return session.scalar(
        select(Agent).where(
            Agent.name == name,
            scenario_filter,
        )
    )


def find_agent_by_mandate(
    session: Session,
    *,
    role: str,
    template_key: str | None,
    system_prompt: str | None,
) -> Agent | None:
    candidates = list(
        session.scalars(
            select(Agent).where(
                Agent.role == role,
                Agent.template_key == template_key,
                Agent.scenario_id.is_(None),
                Agent.is_orchestrator.is_(False),
            ).order_by(Agent.id)
        )
    )
    normalized_mandate = " ".join((system_prompt or "").casefold().split())
    if normalized_mandate:
        return next(
            (
                agent
                for agent in candidates
                if " ".join((agent.system_prompt or "").casefold().split())
                == normalized_mandate
            ),
            None,
        )
    return next(
        (
            agent
            for agent in candidates
            if agent.name.startswith(("Dynamic_", "Fallback_"))
        ),
        None,
    )


def is_master_orchestrator(agent: Agent) -> bool:
    return agent.is_orchestrator


def deliberative_agents(agents: list[Agent]) -> list[Agent]:
    return [agent for agent in agents if not agent.is_orchestrator]


def global_deliberative_agents(session: Session) -> list[Agent]:
    return list(
        session.scalars(
            select(Agent)
            .where(
                Agent.is_orchestrator.is_(False),
                Agent.scenario_id.is_(None),
            )
            .order_by(Agent.id)
        )
    )


def scenario_scoped_agents(
    session: Session,
    scenario_id: int,
) -> list[Agent]:
    return list(
        session.scalars(
            select(Agent)
            .where(
                Agent.is_orchestrator.is_(False),
                Agent.scenario_id == scenario_id,
            )
            .order_by(Agent.id)
        )
    )


def scenario_deliberative_agents(
    session: Session,
    scenario_id: int,
) -> list[Agent]:
    scoped_agents = scenario_scoped_agents(session, scenario_id)
    has_scoped_template_set = any(
        agent.template_key is not None for agent in scoped_agents
    )
    if has_scoped_template_set:
        return scoped_agents
    global_agents = global_deliberative_agents(session)
    return [*global_agents, *scoped_agents]


def get_or_create_master_orchestrator(session: Session) -> tuple[Agent, bool]:
    orchestrator = session.scalar(
        select(Agent).where(
            or_(
                Agent.is_orchestrator.is_(True),
                Agent.template_key == MASTER_ORCHESTRATOR_TEMPLATE_KEY,
            )
        )
    )
    if orchestrator is not None:
        orchestrator.is_orchestrator = True
        if orchestrator.template_key is None:
            orchestrator.template_key = MASTER_ORCHESTRATOR_TEMPLATE_KEY
        session.flush()
        return orchestrator, False
    global_config = get_global_llm_config(session)
    values = apply_global_values(
        {
            "template_key": MASTER_ORCHESTRATOR_TEMPLATE_KEY,
            "name": MASTER_ORCHESTRATOR_NAME,
            "role": MASTER_ORCHESTRATOR_ROLE,
            "system_prompt": MASTER_ORCHESTRATOR_MANDATE,
            "temperature": 0.0,
            "max_tokens": 1000,
            "theta_x": 0.0,
            "theta_q": 0.0,
            "theta_h": 0.0,
            "theta_s": 0.0,
            "theta_u": 0.0,
            "is_orchestrator": True,
        },
        global_config,
    )
    inserted_id = session.scalar(
        insert(Agent)
        .values(**values)
        .on_conflict_do_nothing()
        .returning(Agent.id)
    )
    session.flush()
    orchestrator = session.scalar(
        select(Agent).where(Agent.is_orchestrator.is_(True))
    )
    if orchestrator is None:
        raise ValueError("Master_Orchestrator identity conflicts with an existing agent")
    return orchestrator, inserted_id is not None


def detected_scenario_domains(scenario: Scenario) -> list[str]:
    description = " ".join(scenario.description.casefold().split())
    return [
        key
        for key, patterns in _DOMAIN_GAP_PATTERNS.items()
        if any(pattern.search(description) for pattern in patterns)
    ]


def scenario_agent_weights(
    agent: Agent,
    scenario: Scenario,
    evidence_completeness: float,
) -> dict[str, float]:
    domain = agent_domain_key(agent) or agent.specialist_domain
    if domain not in _DOMAIN_GAP_PATTERNS:
        return calculate_orchestrator_rar_dai_weights(0.0, evidence_completeness)
    _, plan = orchestrated_scenario_agent_plan(scenario)
    alignment = plan["metrics"][domain]["domain_alignment"]
    return calculate_orchestrator_rar_dai_weights(alignment, evidence_completeness)


def orchestrated_scenario_agent_plan(
    scenario: Scenario,
) -> tuple[list[str], dict[str, Any]]:
    description = " ".join(scenario.description.casefold().split())
    simulation_payload = scenario.simulation_payload()
    domain_scores: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    for domain, patterns in _DOMAIN_GAP_PATTERNS.items():
        token_score = float(sum(bool(pattern.search(description)) for pattern in patterns))
        relevant_fields = _DOMAIN_SCENARIO_FIELDS[domain]
        supplied_fields = sum(
            simulation_payload.get(field) is not None
            and simulation_payload.get(field) != ""
            for field in relevant_fields
        )
        field_alignment = supplied_fields / len(relevant_fields)
        evidence_fields = _DOMAIN_EVIDENCE_FIELDS[domain]
        verified_fields = sum(
            simulation_payload.get(field) is not None
            and simulation_payload.get(field) != ""
            for field in evidence_fields
        )
        completeness = verified_fields / len(evidence_fields)
        token_alignment = min(1.0, token_score / 2.0)
        alignment = max(token_alignment, field_alignment)
        score = 0.65 * alignment + 0.35 * completeness
        domain_scores[domain] = score
        metrics[domain] = {
            "domain_alignment": alignment,
            "verified_evidence_completeness": completeness,
            "orchestration_score": score,
        }
    ordered = sorted(domain_scores, key=lambda key: (-domain_scores[key], key))
    selected_count = min(5, max(3, sum(score > 0.0 for score in domain_scores.values())))
    selected_domains = ordered[:selected_count]
    weights = {
        domain: calculate_orchestrator_rar_dai_weights(
            metrics[domain]["domain_alignment"],
            metrics[domain]["verified_evidence_completeness"],
        )
        for domain in selected_domains
    }
    return selected_domains, {"metrics": metrics, "weights": weights}


def ensure_phase_one_specialists(
    session: Session,
    scenario: Scenario,
) -> tuple[list[Agent], dict[str, Any]]:
    orchestrator, orchestrator_created = get_or_create_master_orchestrator(session)
    scoped_agents = scenario_scoped_agents(session, scenario.id)
    global_agents = global_deliberative_agents(session)
    has_scenario_templates = any(agent.template_key is not None for agent in scoped_agents)
    coverage_agents = scoped_agents if has_scenario_templates else global_agents
    existing_by_key = {
        agent_domain_key(agent): agent
        for agent in coverage_agents
        if agent_domain_key(agent) is not None
    }
    selected_domains = detected_scenario_domains(scenario)
    _, orchestration_plan = orchestrated_scenario_agent_plan(scenario)
    created_specialists: list[Agent] = []
    reused_specialists: list[Agent] = []
    global_config = get_global_llm_config(session)
    for key in selected_domains:
        spec = _AGENT_SPECS_BY_KEY[key]
        specialist_name = f"Dynamic_{key.title()}_Reviewer_S{scenario.id}"
        specialist = existing_by_key.get(key)
        if specialist is None:
            specialist = find_agent_by_name_and_scenario(
                session,
                name=specialist_name,
                scenario_id=scenario.id,
            )
        if specialist is None:
            specialist = session.scalar(
                select(Agent).where(
                    Agent.scenario_id == scenario.id,
                    Agent.specialist_domain == key,
                    Agent.is_orchestrator.is_(False),
                )
            )
        if specialist is None:
            weights = orchestration_plan["weights"][key]
            values = {
                **spec.as_agent_values(),
                **weights,
                "rar_dai_weight_mode": "auto",
                "template_key": None,
                "name": specialist_name,
                "scenario_id": scenario.id,
                "specialist_domain": key,
            }
            if global_config.llm_base_url:
                values["llm_base_url"] = global_config.llm_base_url
            if global_config.llm_api_key:
                values["llm_api_key"] = global_config.llm_api_key
            if global_config.llm_model:
                values["llm_model"] = global_config.llm_model
            values["temperature"] = global_config.temperature
            values["max_tokens"] = global_config.max_tokens
            inserted_id = session.scalar(
                insert(Agent)
                .values(**values)
                .on_conflict_do_nothing()
                .returning(Agent.id)
            )
            specialist = session.scalar(
                select(Agent).where(
                    Agent.scenario_id == scenario.id,
                    Agent.specialist_domain == key,
                    Agent.is_orchestrator.is_(False),
                )
            )
            if specialist is None:
                raise ValueError(f"Could not provision scenario specialist for domain '{key}'")
            if inserted_id is not None:
                created_specialists.append(specialist)
            else:
                reused_specialists.append(specialist)
        else:
            reused_specialists.append(specialist)
    session.flush()
    participants = scenario_deliberative_agents(session, scenario.id)
    return participants, {
        "orchestrator_id": orchestrator.id,
        "orchestrator_name": orchestrator.name,
        "orchestrator_created": orchestrator_created,
        "detected_domains": selected_domains,
        "selected_domains": selected_domains,
        "created_specialist_ids": [agent.id for agent in created_specialists],
        "reused_specialist_ids": [agent.id for agent in reused_specialists],
        "selection_metrics": orchestration_plan["metrics"],
        "rar_dai_weights": orchestration_plan["weights"],
        "weight_mode": "auto",
        "non_voting": True,
    }


def orchestrate_scenario_agents(
    session: Session,
    scenario: Scenario,
    configs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[Agent], dict[str, Any]]:
    session.execute(delete(Agent).where(Agent.scenario_id == scenario.id))
    session.flush()
    selected_domains, orchestration_plan = orchestrated_scenario_agent_plan(scenario)
    configurations = configs or {}
    global_config = get_global_llm_config(session)
    agents: list[Agent] = []
    for domain in selected_domains:
        template = _AGENT_SPECS_BY_KEY[domain]
        values = {
            **apply_global_values(template.as_agent_values(), global_config),
            **orchestration_plan["weights"][domain],
            "rar_dai_weight_mode": "auto",
            "scenario_id": scenario.id,
            "specialist_domain": domain,
        }
        for field, value in configurations.get(domain, {}).items():
            if value not in (None, "") and not global_config.apply_to_all:
                values[field] = value
        agent = Agent(**values)
        session.add(agent)
        agents.append(agent)
    session.flush()
    return agents, {
        "selected_domains": selected_domains,
        "selection_metrics": orchestration_plan["metrics"],
        "rar_dai_weights": orchestration_plan["weights"],
        "weight_mode": "auto",
        "agent_count": len(agents),
    }


def load_standard_agent_templates(
    session: Session,
    configs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[Agent], int]:
    existing_by_key = {
        agent.template_key: agent
        for agent in session.scalars(
            select(Agent).where(
                Agent.scenario_id.is_(None),
                Agent.template_key.in_([template.key for template in AGENTS]),
            )
        )
    }
    existing_by_name = {
        agent.name: agent
        for agent in session.scalars(
            select(Agent).where(
                Agent.scenario_id.is_(None),
                Agent.name.in_([template.name for template in AGENTS]),
            )
        )
    }
    created = 0
    agents: list[Agent] = []
    configurations = configs or {}
    global_config = get_global_llm_config(session)
    for template in AGENTS:
        agent = existing_by_key.get(template.key) or existing_by_name.get(template.name)
        if agent is None:
            values = apply_global_values(template.as_agent_values(), global_config)
            agent = Agent(**values)
            session.add(agent)
            created += 1
        elif agent.template_key is None:
            agent.template_key = template.key
        for field, value in configurations.get(template.key, {}).items():
            if value not in (None, "") and not global_config.apply_to_all:
                setattr(agent, field, value)
        if global_config.apply_to_all:
            for field in (
                "llm_base_url",
                "llm_api_key",
                "llm_model",
                "temperature",
                "max_tokens",
            ):
                setattr(agent, field, getattr(global_config, field))
        agents.append(agent)
    session.commit()
    for agent in agents:
        session.refresh(agent)
    return agents, created
