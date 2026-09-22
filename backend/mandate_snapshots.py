from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .agent_templates import agent_revision, get_agent_spec, mandate_seed
from .models import Agent, Scenario, ScenarioMandateSnapshot


def _default_dynamic_fields(agent: Agent, scenario: Scenario) -> dict[str, object]:
    return {
        "synthesis_status": "fallback",
        "scenario_mandate": (
            f"Evaluate the active APBN policy scenario from the {agent.role} mandate: "
            f"{scenario.description.strip()}"
        ),
        "scenario_focus": [
            "Kebijakan Fiskal",
            "Optimalisasi Penerimaan Negara",
            "Makroekonomi",
        ],
        "priority_questions": [
            "What legal, fiscal, and implementation conditions must be verified before adoption?"
        ],
        "required_evidence": [
            "Current APBN baseline, source-linked fiscal assumptions, and implementation evidence"
        ],
        "epistemic_logic_traceability": [
            "Attach source tags and verification status to material claims and preserve auditable inter-agent handoffs"
        ],
        "structured_consensus_protocol": [
            "Classify disagreements, preserve valid dissent, and escalate unresolved conflicts through DDR and CAR"
        ],
        "regulatory_compliance_alignment": [
            f"Enforce the {scenario.max_deficit_constraint}% GDP deficit ceiling and reject unverified fiscal offsets"
        ],
        "llm_model": agent.llm_model,
        "latency_ms": None,
        "token_usage": 0,
        "error": None,
    }


def _rebased_agent_rule(
    agent: Agent,
    scenario: Scenario,
    source_rule: dict[str, Any] | None,
) -> dict[str, Any]:
    seed = mandate_seed(agent)
    dynamic = _default_dynamic_fields(agent, scenario)
    if source_rule is not None:
        for field in dynamic:
            if source_rule.get(field) not in (None, [], ""):
                dynamic[field] = source_rule[field]
    primary_sources = list(seed["primary_sources"])
    constraints = list(seed["constraints"])
    owned_checks = list(seed["owned_checks"])
    return {
        "agent_id": agent.id,
        "name": agent.name,
        "role": agent.role,
        "template_key": agent.template_key,
        "mandate": str(seed["mandate"]) or None,
        "primary_sources": primary_sources,
        "constraints": constraints,
        "owned_checks": owned_checks,
        **dynamic,
        "applicable_primary_sources": primary_sources,
        "applicable_constraints": constraints,
        "applicable_owned_checks": owned_checks,
    }


def _combined_rules(agents: list[Agent], scenario: Scenario) -> dict[str, object]:
    specs = [spec for agent in agents if (spec := get_agent_spec(agent.template_key))]
    return {
        "hard_constraints": sorted({item for spec in specs for item in spec.constraints}),
        "owned_checks": sorted({item for spec in specs for item in spec.owned_checks}),
        "principles": sorted({item for spec in specs for item in spec.decision_principles}),
        "primary_sources": sorted({item for spec in specs for item in spec.primary_sources}),
        "automatic_deficit_ceiling": scenario.max_deficit_constraint,
    }


def refresh_mandate_snapshot(
    session: Session,
    scenario: Scenario,
    agents: list[Agent],
    source_snapshot: ScenarioMandateSnapshot,
) -> ScenarioMandateSnapshot:
    revision = agent_revision(agents, scenario)
    target = session.scalar(
        select(ScenarioMandateSnapshot).where(
            ScenarioMandateSnapshot.scenario_id == scenario.id,
            ScenarioMandateSnapshot.revision == revision,
        )
    )
    source_rules = {
        rule.get("agent_id"): rule
        for rule in source_snapshot.agent_rules
        if isinstance(rule, dict) and isinstance(rule.get("agent_id"), int)
    }
    agent_rules = [
        _rebased_agent_rule(agent, scenario, source_rules.get(agent.id))
        for agent in agents
    ]
    generated_count = sum(rule["synthesis_status"] == "generated" for rule in agent_rules)
    failure_count = len(agent_rules) - generated_count
    values = {
        "generated": True,
        "agent_count": len(agents),
        "rules": _combined_rules(agents, scenario),
        "agent_rules": agent_rules,
        "status": "partial" if failure_count else "success",
        "generated_count": generated_count,
        "failure_count": failure_count,
        "detail": (
            f"Automatically refreshed stale mandate revision {source_snapshot.revision} "
            f"to {revision}."
        ),
    }
    if target is None:
        target = ScenarioMandateSnapshot(
            scenario_id=scenario.id,
            revision=revision,
            **values,
        )
        session.add(target)
    else:
        for field, value in values.items():
            setattr(target, field, value)
    session.flush()
    return target
