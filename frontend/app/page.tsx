"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import RunGraph, { type RunGraphPayload } from "./RunGraph";
import { useI18n } from "./i18n";

type Agent = {
  id: number;
  name: string;
  role: string;
  template_key: string | null;
  theta_x: number;
  theta_q: number;
  theta_h: number;
  theta_s: number;
  theta_u: number;
  llm_base_url: string | null;
  llm_model: string | null;
  system_prompt: string | null;
  temperature: number;
  max_tokens: number;
  has_llm_api_key: boolean;
};
type AgentTemplate = {
  key: string;
  name: string;
  role: string;
  description: string;
  system_prompt: string;
  primary_sources: string[];
  owned_checks: string[];
  parameters: string[];
  constraints: string[];
  impact_dimensions: string[];
  risk_dimensions: string[];
  uncertainty_dimensions: string[];
  decision_principles: string[];
  temperature: number;
  max_tokens: number;
  theta_x: number;
  theta_q: number;
  theta_h: number;
  theta_s: number;
  theta_u: number;
};
type Scenario = { id: number; description: string; program_cost: number | null; max_deficit_constraint: number };
type Metric = {
  id: number;
  hard_constraint_violation_rate: number;
  provenance_completeness_percent: number;
  material_information_retention_macro_f1: number;
  feasible_alternatives_count: number;
  convergence_status: string;
  latency_ms: number;
  token_usage: number;
  created_at: string;
};
type VectorMetadata = { component: string; active: boolean; category?: string; meaning?: string; economic_impact?: string; formula?: string; agent_i?: { value?: unknown; normalized?: string[]; source_tags?: string[]; artifact_hash?: string }; agent_j?: { value?: unknown; normalized?: string[]; source_tags?: string[]; artifact_hash?: string }; calculation?: Record<string, unknown> & { status?: string; value?: number | null; not_calculated_reason?: string; confidence_gap?: number | null; deficit_range_gap_percent_gdp?: number | null; violation?: boolean }; resolution_path?: string; status?: string };
type Disagreement = {
  id: number;
  agent_i: string;
  agent_j: string;
  dE: boolean;
  dA: boolean;
  dP: boolean;
  dR: boolean;
  dU: boolean;
  dO: boolean;
  dC: boolean;
  dREC: boolean;
  active_components: string[];
  conflict_categories: { component: string; category: string; narrative: string; impact?: string; formula?: string; agent_i_artifacts?: unknown; agent_j_artifacts?: unknown }[];
  vector_metadata: Record<string, VectorMetadata>;
  fiscal_calculation: { formula?: string; compromise_formula?: string; statutory_deficit_ceiling_percent?: number; program_cost?: number | null; calculation_note?: string; agent_i_alternatives?: SimulationAlternative[]; agent_j_alternatives?: SimulationAlternative[]; arbiter_alternatives?: SimulationAlternative[]; selected_compromise?: SimulationAlternative & { headroom_percent?: number | null } };
  influence_context: { agent_i_weight?: number | null; agent_j_weight?: number | null; combined_weight?: number; formula?: string };
  legal_basis: { source_tag: string; basis: string }[];
  resolution_mechanism: string;
  resolution_detail: { route?: string; status?: string; conclusion?: string; simulation_round?: number; arbiter_conclusion?: string; simulation_summary?: string; remaining_prediction_conflicts?: number; limitations?: string[] };
};
type Influence = { agent: string; proposition: string; normalized_weight: number | null; raw_score: number | null; gate: number; interactions?: { peer_agent_name: string; active_components: string[]; agreement_ratio: number }[]; calculation?: { dimensions?: Record<string, string>; interaction_count?: number; inputs?: Record<string, number> } };
type SimulationConflict = { agent_i?: string; agent_j?: string; components?: string[]; route?: string };
type SimulationAlternative = { name?: string; deficit?: number; utility?: number; source_tag?: string; evidence_status?: string };
type SimulationOutput = { agent_name?: string; resolution?: string; simulation_summary?: string; evidence_status?: string; follow_up_consensus_status?: string; remaining_prediction_conflicts?: number; fallback_reason?: string; limitations?: string[]; conflict_summary?: string[]; modelled_variables?: string[]; alternatives?: SimulationAlternative[]; risks?: { content?: string }[]; uncertainties?: { content?: string }[]; message?: string };
type SimulationArtifact = { id: number; session_id: string; scenario_id: number; trigger: string; round_number: number; status: string; simulation_version: string; input?: { conflicts?: SimulationConflict[]; sectoral_inputs?: { agent?: string; role?: string; predictions?: { content?: string }[]; alternatives?: SimulationAlternative[]; recommendation?: unknown }[]; scenario?: { description?: string; program_cost?: number | null; max_deficit_constraint?: number } }; output: SimulationOutput; latency_ms: number; token_usage: number; created_at: string };
type DecisionItem = { content?: string; source_tag?: string; name?: string; deficit?: number; utility?: number };
type AgentPosition = { stage: string; round_number: number; agent_opinion: string | null; reasoning_summary: string | null; constraints_considered: DecisionItem[]; statutory_gates: string[]; recommendation: DecisionItem | string | null; confidence: number | null; evidence: DecisionItem[]; assumptions: DecisionItem[]; predictions: DecisionItem[]; risks: DecisionItem[]; uncertainties: DecisionItem[]; objectives: DecisionItem[]; alternatives: DecisionItem[] };
type AgentBreakdown = { agent_id: number; agent_name: string; agent_role: string; role: string; template_key: string | null; schema_valid: boolean | null; provenance_count: number; position_stage: string; agent_opinion: string | null; reasoning_summary: string | null; constraints_considered: DecisionItem[]; statutory_gates: string[]; recommendation: DecisionItem | string | null; confidence: number | null; evidence: DecisionItem[]; assumptions: DecisionItem[]; predictions: DecisionItem[]; risks: DecisionItem[]; uncertainties: DecisionItem[]; objectives: DecisionItem[]; alternatives: DecisionItem[]; pre_arbitration: AgentPosition | null; final_position: AgentPosition | null; deliberation_stages: AgentPosition[] };
type CollectiveClaim = { agent_id?: number; agent_name?: string; agent_role?: string; schema_valid?: boolean; position_stage?: string; main_claim?: string | null; recommendation?: DecisionItem | string | null; confidence?: number | null; constraints?: DecisionItem[]; evidence?: DecisionItem[]; predictions?: DecisionItem[] };
type CollectiveReasoning = { methodology: string; run_status: string; claims_and_positions: CollectiveClaim[]; why_and_how: { summary: string; summary_i18n?: { id?: string; en?: string }; divergence_points: { component: string; category?: string; narrative?: string; pair_count: number; agent_pairs: string[] }[]; convergence_signals: { signal: string; coverage: string; narrative: string }[] }; recommendation_and_follow_up: { arbiter?: string | null; status: string; round_number: number; simulation_summary?: string | null; final_resolution?: string | null; modelled_alternatives: SimulationAlternative[]; limitations: string[]; remaining_prediction_conflicts?: number | null; follow_up_consensus_status?: string | null }; normative_evaluation: { summary: string; principles: { principle: string; status: string; evaluation: string; legal_sources: string[] }[] } };
type Dashboard = {
  session_id: string | null;
  session_status: string | null;
  scenario: Scenario;
  domain_rules: DomainRules;
  latest_metric: Metric | null;
  metric_history: Metric[];
  schema_validity_percent: number;
  reasoning_log_count: number;
  agent_breakdown: AgentBreakdown[];
  collective_reasoning: CollectiveReasoning | null;
  disagreements: Disagreement[];
  simulation_artifacts: SimulationArtifact[];
  influence_observations: Influence[];
};
type RunLog = { stage: string; level: string; message: string; code?: string; messages?: { id?: string; en?: string }; metric?: Record<string, unknown>; statutory?: Record<string, unknown>; economic?: Record<string, unknown>; fallback?: { used?: boolean; kind?: string | null; reason?: string | null; retained_artifact?: string; consensus_impact?: string } };
type RunState = { task_id: string | null; session_id: string; scenario_id: number; status: "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED"; logs: RunLog[]; simulation_artifacts?: SimulationArtifact[]; agent_breakdown?: AgentBreakdown[]; collective_reasoning?: CollectiveReasoning | null; disagreements?: Disagreement[]; progress_stage?: string | null; result?: Record<string, unknown> | null; error?: string | null; created_at?: string | null; started_at?: string | null; completed_at?: string | null };
type AgentDomainRules = { agent_id: number; name: string; role: string; template_key: string | null; mandate: string | null; primary_sources: string[]; constraints: string[]; owned_checks: string[]; synthesis_status: "generated" | "fallback"; scenario_mandate: string | null; scenario_focus: string[]; priority_questions: string[]; required_evidence: string[]; epistemic_logic_traceability: string[]; structured_consensus_protocol: string[]; regulatory_compliance_alignment: string[]; llm_model: string | null; token_usage: number | null; error: { code: string; message: string } | null };
type DomainRules = { scenario_id: number; revision: string; generated: boolean; stale: boolean; agent_count: number; rules: { hard_constraints?: string[]; owned_checks?: string[]; principles?: string[]; primary_sources?: string[]; automatic_deficit_ceiling?: number }; agent_rules: AgentDomainRules[]; status: "success" | "partial" | "failed" | "stale" | "missing"; generated_count: number; failure_count: number; detail: string | null };
type AgentForm = Omit<Agent, "id" | "has_llm_api_key" | "template_key" | "system_prompt"> & { llm_api_key: string };
type ScenarioForm = Omit<Scenario, "id" | "max_deficit_constraint">;

const apiUrl = "";
const components = ["dE", "dA", "dP", "dR", "dU", "dO", "dC", "dREC"] as const;
const componentLabels: Record<(typeof components)[number], { id: string; en: string }> = { dE: { id: "Bukti", en: "Evidence" }, dA: { id: "Asumsi", en: "Assumption" }, dP: { id: "Prediksi", en: "Prediction" }, dR: { id: "Risiko", en: "Risk" }, dU: { id: "Ketidakpastian", en: "Uncertainty" }, dO: { id: "Tujuan", en: "Objective" }, dC: { id: "Constraint", en: "Constraint" }, dREC: { id: "Rekomendasi", en: "Recommendation" } };
const initialAgent: AgentForm = {
  name: "",
  role: "",
  theta_x: 1,
  theta_q: 1,
  theta_h: 1,
  theta_s: 1,
  theta_u: 1,
  llm_base_url: "",
  llm_api_key: "",
  llm_model: "",
  temperature: 0.2,
  max_tokens: 4000,
};
const initialScenario: ScenarioForm = { description: "", program_cost: null };

function NumericField({ label, value, onChange, hint }: { label: string; value: number; onChange: (value: number) => void; hint: string }) {
  return <label className="form-field"><strong>{label}</strong><input type="number" min="0" step="0.01" value={value} onChange={(event) => onChange(Number(event.target.value))} required /><small>{hint}</small></label>;
}

function MetricCard({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: string }) {
  return <div className={`metric-card ${tone ?? ""}`}><span>{label}</span><strong>{value}<small>{unit}</small></strong></div>;
}

function TemplateContractPreview({ template }: { template: AgentTemplate }) {
  const { t } = useI18n();
  return <div className="template-contract"><div><span>{t("template.mandate")}</span><strong>{template.description}</strong></div><div><span>{t("template.sources")}</span><strong>{(template.primary_sources ?? []).join(" · ")}</strong></div><div><span>{t("template.constraints")}</span><strong>{(template.constraints ?? []).join(" · ")}</strong></div><div><span>{t("template.checks")}</span><strong>{(template.owned_checks ?? []).join(" · ")}</strong></div></div>;
}

function RuleList({ title, items, empty }: { title: string; items?: string[]; empty: string }) {
  const safeItems = items ?? [];
  return <section className="rule-list"><span>{title}</span>{safeItems.length ? <ul>{safeItems.map((item) => <li key={item}>{item}</li>)}</ul> : <p>{empty}</p>}</section>;
}

function AgentMandateCard({ rules, index, onGenerate, busy }: { rules: AgentDomainRules; index: number; onGenerate: (agentId: number) => void; busy: boolean }) {
  const { t, formatNumber } = useI18n();
  return <details className="agent-mandate-card" open={index === 0}><summary><div><span>{String(index + 1).padStart(2, "0")} / {rules.template_key?.toUpperCase() ?? "CUSTOM"} / {rules.synthesis_status.toUpperCase()}</span><strong>{rules.name}</strong><small>{rules.role}</small></div><b>{formatNumber((rules.primary_sources ?? []).length)} {t("common.sources")} · {formatNumber((rules.constraints ?? []).length)} {t("common.constraints")}</b></summary><div className="agent-mandate-body"><div className="mandate-card-actions"><button type="button" className="template-load-button" onClick={() => onGenerate(rules.agent_id)} disabled={busy}>{busy ? t("common.generating") : t("common.generateMandate")}</button></div>{rules.scenario_mandate && <section className="mandate-copy"><span>{t("mandate.scenario")}</span><p>{rules.scenario_mandate}</p></section>}<section className="mandate-copy"><span>{t("mandate.seed")}</span><p>{rules.mandate ?? t("mandate.unavailable")}</p></section>{rules.error && <section className="mandate-copy"><span>{rules.error.code}</span><p>{rules.error.message}</p></section>}<div className="agent-rule-columns"><RuleList title={t("mandate.focus")} items={rules.scenario_focus} empty={t("mandate.empty.focus")} /><RuleList title={t("mandate.questions")} items={rules.priority_questions} empty={t("mandate.empty.questions")} /><RuleList title={t("mandate.evidence")} items={rules.required_evidence} empty={t("mandate.empty.evidence")} /><RuleList title={t("mandate.traceability")} items={rules.epistemic_logic_traceability} empty={t("mandate.empty.traceability")} /><RuleList title={t("mandate.consensus")} items={rules.structured_consensus_protocol} empty={t("mandate.empty.consensus")} /><RuleList title={t("mandate.regulatory")} items={rules.regulatory_compliance_alignment} empty={t("mandate.empty.regulatory")} /><RuleList title={t("template.sources")} items={rules.primary_sources} empty={t("mandate.empty.sources")} /><RuleList title={t("template.constraints")} items={rules.constraints} empty={t("mandate.empty.constraints")} /><RuleList title="OWNED CHECKS" items={rules.owned_checks} empty="Tidak ada owned checks terstruktur." /></div></div></details>;
}

function decisionText(value: unknown, fallback = "Belum tersedia") {
  if (typeof value === "string" && value.trim()) return value;
  if (value && typeof value === "object") {
    const item = value as DecisionItem;
    return item.content ?? item.name ?? fallback;
  }
  return fallback;
}

function DecisionArtifactList({ items, empty }: { items?: DecisionItem[]; empty: string }) {
  const { t, formatNumber } = useI18n();
  const safeItems = items ?? [];
  return safeItems.length ? <ul>{safeItems.map((item, index) => <li key={`${decisionText(item, t("common.unknown"))}-${index}`}><span>{decisionText(item, t("common.unknown"))}</span>{item.source_tag && <small>{item.source_tag}</small>}{item.deficit !== undefined && <small>{t("common.deficit")} {formatNumber(item.deficit)}% · {t("common.utility")} {item.utility === undefined ? "—" : formatNumber(item.utility)}</small>}</li>)}</ul> : <p>{empty}</p>;
}

function AgentBreakdownCard({ agent, index }: { agent: AgentBreakdown; index: number }) {
  const { t, formatNumber } = useI18n();
  const stages = agent.deliberation_stages ?? [];
  const initialStage = agent.pre_arbitration?.stage ?? stages[0]?.stage ?? "FINAL";
  const initialRound = agent.pre_arbitration?.round_number ?? stages[0]?.round_number ?? 0;
  const [selectedKey, setSelectedKey] = useState(`${initialStage}:${initialRound}`);
  const selected = stages.find((stage) => `${stage.stage}:${stage.round_number}` === selectedKey) ?? agent.pre_arbitration ?? agent.final_position;
  const position: AgentPosition = selected ?? agent.pre_arbitration ?? agent.final_position ?? { stage: agent.position_stage, round_number: 0, agent_opinion: agent.agent_opinion, reasoning_summary: agent.reasoning_summary, constraints_considered: agent.constraints_considered, statutory_gates: agent.statutory_gates, recommendation: agent.recommendation, confidence: agent.confidence, evidence: agent.evidence, assumptions: agent.assumptions, predictions: agent.predictions, risks: agent.risks, uncertainties: agent.uncertainties, objectives: agent.objectives, alternatives: agent.alternatives };
  const confidence = typeof position.confidence === "number" ? `${formatNumber(position.confidence * 100, { maximumFractionDigits: 0 })}%` : "—";
  return <details className="agent-breakdown-card" open={index === 0}><summary><div><span>{String(index + 1).padStart(2, "0")} / {agent.template_key?.toUpperCase() ?? "SECTORAL"}</span><strong>{agent.agent_name}</strong><small>{agent.agent_role}</small></div><div className="agent-confidence"><b>{confidence}</b><small>{t("common.confidence")}</small></div></summary><div className="agent-breakdown-body"><div className="position-tabs" role="group" aria-label={t("agent.positionStages", { name: agent.agent_name })}>{stages.map((stage, stageIndex) => { const key = `${stage.stage}:${stage.round_number}`; const label = stage.stage === "INITIAL" ? t("agent.initial") : stage.stage === "PRE_ARBITRATION" ? t("agent.pre") : stage.stage === "POST_SIMULATION" ? `${t("agent.post")} · ${t("common.round")} ${formatNumber(stage.round_number)}` : t("agent.final"); return <button type="button" aria-pressed={selectedKey === key} className={selectedKey === key ? "active" : ""} onClick={() => setSelectedKey(key)} key={`${key}-${stageIndex}`}>{label}</button>; })}</div><div className="agent-opinion"><span>{t("agent.opinion")}</span><p>{position.reasoning_summary ?? position.agent_opinion ?? t("agent.noSummary")}</p></div><div className="agent-decision-grid"><section className="statutory-gate"><span>{t("agent.constraints")}</span><DecisionArtifactList items={position.constraints_considered} empty={t("agent.noConstraints")} /></section><section className="partial-recommendation"><span>{t("agent.recommendation")}</span><strong>{decisionText(position.recommendation, t("common.unknown"))}</strong><small>{t("agent.confidenceLine", { confidence, position: position.stage.replaceAll("_", " ").toLowerCase() })}</small></section></div><div className="agent-artifact-grid"><section><span>{t("agent.preAlternatives")}</span><DecisionArtifactList items={position.alternatives} empty={t("agent.noAlternatives")} /></section><section><span>{t("agent.evidencePredictions")}</span><DecisionArtifactList items={[...(position.evidence ?? []), ...(position.predictions ?? [])]} empty={t("agent.noEvidencePredictions")} /></section><section><span>{t("agent.riskUncertainty")}</span><DecisionArtifactList items={[...(position.risks ?? []), ...(position.uncertainties ?? [])]} empty={t("agent.noRiskUncertainty")} /></section></div></div></details>;
}

function artifactText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    const candidate = record.content ?? record.name ?? record.recommendation ?? record.decision;
    if (typeof candidate === "string" && candidate.trim()) return candidate;
  }
  return JSON.stringify(value);
}

function VectorDetail({ metadata }: { metadata: VectorMetadata }) {
  const { lang, t, formatNumber } = useI18n();
  const calculation = metadata.calculation ?? {};
  const calculationValue = calculation.value ?? calculation.confidence_gap ?? calculation.deficit_range_gap_percent_gdp;
  return <details className={`vector-detail ${metadata.active ? "active" : "clear"}`}><summary><span>{metadata.component}</span><strong>{metadata.category ?? componentLabels[metadata.component as keyof typeof componentLabels]?.[lang] ?? metadata.component}</strong><b>{metadata.active ? "1" : "0"}</b></summary><div><section><span>{t("ddr.formula")}</span><code>{metadata.formula ?? "—"}</code><p>{calculationValue !== undefined && calculationValue !== null ? formatNumber(calculationValue) : calculation.not_calculated_reason ?? t("ddr.notCalculated")}</p></section><section><span>{t("ddr.meaning")}</span><p>{metadata.meaning ?? "—"}</p></section><section><span>{t("ddr.impact")}</span><p>{metadata.economic_impact ?? "—"}</p></section><section><span>{t("ddr.routePath")}</span><p>{metadata.resolution_path ?? "—"}</p></section></div></details>;
}

function DisagreementInspector({ item, breakdown }: { item: Disagreement; breakdown: AgentBreakdown[] }) {
  const { t, formatNumber } = useI18n();
  const agentIPosition = breakdown.find((agent) => agent.agent_name === item.agent_i)?.pre_arbitration ?? null;
  const agentJPosition = breakdown.find((agent) => agent.agent_name === item.agent_j)?.pre_arbitration ?? null;
  const calculation = item.fiscal_calculation ?? {};
  const resolution = item.resolution_detail ?? {};
  const active = item.active_components ?? components.filter((component) => item[component]);
  return <div className="ddr-inspector"><div className="ddr-inspector-head"><span>{t("ddr.inspector")} / #{item.id}</span><h3>{item.agent_i}<em> × </em>{item.agent_j}</h3><p>{resolution.conclusion ?? item.resolution_mechanism}</p><b className={`ddr-status ${active.length ? "conflict" : "clear"}`}>{active.length ? `${t("ddr.activeConflict")} / ${formatNumber(active.length)}` : `${t("ddr.clean")} / 0`}</b></div><div className="ddr-vector-details">{components.map((component) => <VectorDetail metadata={item.vector_metadata?.[component] ?? { component, active: item[component], status: item[component] ? "detected" : "no-divergence" }} key={component} />)}</div>{item.conflict_categories?.length > 0 && <div className="ddr-category-narrative">{item.conflict_categories.map((category) => <section key={category.component}><span>{category.component} / {category.category}</span><p>{category.narrative}</p>{category.impact && <p><b>{t("ddr.impact")}:</b> {category.impact}</p>}<div className="ddr-category-compare"><div><b>{item.agent_i}</b>{Array.isArray(category.agent_i_artifacts) && category.agent_i_artifacts.length ? category.agent_i_artifacts.slice(0, 3).map((artifact, index) => <small key={index}>{artifactText(artifact)}</small>) : <small>—</small>}</div><div><b>{item.agent_j}</b>{Array.isArray(category.agent_j_artifacts) && category.agent_j_artifacts.length ? category.agent_j_artifacts.slice(0, 3).map((artifact, index) => <small key={index}>{artifactText(artifact)}</small>) : <small>—</small>}</div></div></section>)}</div>}<div className="ddr-analysis-grid"><section><span>{t("ddr.deficitCalculation")}</span><p>{calculation.calculation_note ?? t("ddr.notCalculated")}</p><div className="ddr-formula"><b>{t("ddr.ceiling")}</b><small>{calculation.statutory_deficit_ceiling_percent === undefined ? "—" : formatNumber(calculation.statutory_deficit_ceiling_percent)}% {t("common.gdp")}</small><b>{t("ddr.headroomFormula")}</b><small>{calculation.formula ?? "ceiling − projected deficit"}</small><b>{t("ddr.compromiseFormula")}</b><small>{calculation.compromise_formula ?? "argmax utility subject to deficit ≤ ceiling"}</small></div>{calculation.selected_compromise && <div className="ddr-selected-compromise"><span>{t("ddr.selectedCompromise")} · {t("common.round")} {resolution.simulation_round === undefined ? "—" : formatNumber(resolution.simulation_round)}</span><strong>{calculation.selected_compromise.name ?? t("analytics.unnamedAlternative")}</strong><small>{calculation.selected_compromise.deficit === undefined ? "—" : formatNumber(calculation.selected_compromise.deficit)}% {t("common.gdp")} · {t("common.utility")} {calculation.selected_compromise.utility === undefined ? "—" : formatNumber(calculation.selected_compromise.utility)} · {t("ddr.headroom")} {calculation.selected_compromise.headroom_percent === null || calculation.selected_compromise.headroom_percent === undefined ? "—" : formatNumber(calculation.selected_compromise.headroom_percent)} pp</small></div>}</section><section><span>{t("ddr.legal")}</span>{item.legal_basis?.length ? <ul className="ddr-legal-list">{item.legal_basis.map((basis) => <li key={basis.source_tag}><b>{basis.source_tag}</b><small>{basis.basis}</small></li>)}</ul> : <p>{t("ddr.noLegal")}</p>}</section></div><div className="ddr-agent-compare"><section><span>{t("ddr.prePosition")} / {item.agent_i}</span><p>{agentIPosition?.reasoning_summary ?? t("common.unknown")}</p><b>{t("agent.recommendation")}</b><small>{decisionText(agentIPosition?.recommendation ?? null, t("common.unknown"))}</small></section><section><span>{t("ddr.prePosition")} / {item.agent_j}</span><p>{agentJPosition?.reasoning_summary ?? t("common.unknown")}</p><b>{t("agent.recommendation")}</b><small>{decisionText(agentJPosition?.recommendation ?? null, t("common.unknown"))}</small></section></div><div className="ddr-resolution-block"><div><span>{t("ddr.arbiterResolution")}</span><strong>{resolution.arbiter_conclusion ?? resolution.route ?? item.resolution_mechanism}</strong><small>{resolution.simulation_summary ?? t("common.unknown")}</small></div><div><span>{t("ddr.residual")}</span><strong>{resolution.status ?? "—"}</strong><small>{resolution.remaining_prediction_conflicts === undefined ? "—" : formatNumber(resolution.remaining_prediction_conflicts)} · RAR-DAI {item.influence_context?.combined_weight === undefined ? "—" : formatNumber(item.influence_context.combined_weight)}</small></div></div></div>;
}

function DdrNetworkPanel({ disagreements, breakdown }: { disagreements: Disagreement[]; breakdown: AgentBreakdown[] }) {
  const { lang, t } = useI18n();
  const [selectedId, setSelectedId] = useState<number | null>(disagreements[0]?.id ?? null);
  const selected = disagreements.find((item) => item.id === selectedId) ?? disagreements[0] ?? null;
  if (!disagreements.length) return <div className="empty-state"><strong>{t("ddr.none")}</strong><span>{t("ddr.noneDetail")}</span></div>;
  return <div className="ddr-layout"><div className="ddr-matrix-column"><div className="matrix-wrap"><table className="ddr-table"><caption>{t("ddr.caption")}</caption><thead><tr><th scope="col">{t("ddr.pair")}</th>{components.map((component) => <th scope="col" title={componentLabels[component][lang]} key={component}>{component}</th>)}<th scope="col">{t("ddr.route")}</th></tr></thead><tbody>{disagreements.map((item) => <tr className={selected?.id === item.id ? "selected" : ""} key={item.id}><th scope="row"><button type="button" className="ddr-pair-button" aria-expanded={selected?.id === item.id} aria-controls="ddr-inspector" onClick={() => setSelectedId(item.id)}><strong>{item.agent_i}</strong><small>× {item.agent_j}</small></button></th>{components.map((component) => <td key={component}><span className={`bool ${item[component] ? "conflict" : "clear"}`} title={item.vector_metadata?.[component]?.meaning} aria-label={`${componentLabels[component][lang]}: ${item[component] ? t("ddr.active") : t("ddr.clear")}`}>{item[component] ? "1" : "0"}</span></td>)}<td className="resolution">{item.resolution_mechanism}</td></tr>)}</tbody></table></div><p className="ddr-matrix-hint">{t("ddr.hint")}</p></div><aside id="ddr-inspector" className="ddr-inspector-column">{selected && <DisagreementInspector item={selected} breakdown={breakdown} />}</aside></div>;
}

function AgentBreakdownPanel({ agents, disagreements, simulations }: { agents: AgentBreakdown[]; disagreements: Disagreement[]; simulations: SimulationArtifact[] }) {
  const { t, formatNumber } = useI18n();
  const safeAgents = agents ?? [];
  const safeDisagreements = disagreements ?? [];
  const safeSimulations = simulations ?? [];
  const conflictCount = safeDisagreements.filter((item) => components.some((component) => item[component])).length || safeSimulations.reduce((total, artifact) => total + (artifact.input?.conflicts?.length ?? 0), 0);
  return <div id="agent-breakdown" className="agent-breakdown"><div className="deliberation-flow"><div><span>{t("agent.flow.arguments")}</span><strong>{t("agent.flow.argumentCount", { count: formatNumber(safeAgents.length) })}</strong><small>{t("agent.flow.argumentDetail")}</small></div><i>→</i><div className={conflictCount ? "conflicted" : ""}><span>{t("agent.flow.conflicts")}</span><strong>{t("agent.flow.conflictCount", { count: conflictCount ? formatNumber(conflictCount) : "—" })}</strong><small>{t("agent.flow.conflictDetail")}</small></div><i>→</i><div className={safeSimulations.length ? "resolved" : ""}><span>{t("agent.flow.simulation")}</span><strong>{t("agent.flow.simulationCount", { count: safeSimulations.length ? formatNumber(safeSimulations.length) : "—" })}</strong><small>{t("agent.flow.simulationDetail")}</small></div></div><div className="agent-breakdown-heading"><div><span>{t("agent.transparency")}</span><h3>{t("agent.breakdownTitle")}</h3></div><p>{t("agent.breakdownDescription")}</p></div>{safeAgents.length ? <div className="agent-breakdown-grid">{safeAgents.map((agent, index) => <AgentBreakdownCard agent={agent} index={index} key={agent.agent_id} />)}</div> : <div className="empty-state"><strong>{t("agent.empty")}</strong><span>{t("agent.emptyDetail")}</span></div>}</div>;
}

function CollectiveReasoningPanel({ analysis }: { analysis: CollectiveReasoning | null }) {
  const { lang, t, formatNumber } = useI18n();
  if (!analysis) return <div className="collective-empty"><strong>{t("analytics.noNarrative")}</strong><span>{t("analytics.noNarrativeDetail")}</span></div>;
  const recommendation = analysis.recommendation_and_follow_up ?? { status: "—", round_number: 0, modelled_alternatives: [], limitations: [] };
  const why = analysis.why_and_how ?? { summary: "", divergence_points: [], convergence_signals: [] };
  const normative = analysis.normative_evaluation ?? { summary: "", principles: [] };
  return <section className="collective-reasoning"><div className="collective-method"><span>{t("analytics.dossier")}</span><strong>{analysis.methodology}</strong><small>{t("analytics.runStatus")}: {analysis.run_status}</small></div><section className="collective-section claims"><header><span>{t("analytics.claims")}</span><h3>{t("analytics.claimsTitle")}</h3><p>{t("analytics.claimsDescription")}</p></header><div className="collective-claim-grid">{(analysis.claims_and_positions ?? []).map((claim, index) => <article className={claim.schema_valid ? "valid" : "invalid"} key={`${claim.agent_id}-${index}`}><div><span>{String(index + 1).padStart(2, "0")} / {claim.position_stage?.replaceAll("_", " ") ?? t("analytics.noPosition")}</span><b>{typeof claim.confidence === "number" ? `${formatNumber(claim.confidence * 100, { maximumFractionDigits: 0 })}%` : "—"}</b></div><h4>{claim.agent_name ?? t("analytics.unknownAgent")}</h4><small>{claim.agent_role}</small><p>{claim.main_claim ?? t("analytics.invalidClaim")}</p><section><span>{t("analytics.recommendationLabel")}</span><strong>{decisionText(claim.recommendation, t("common.unknown"))}</strong></section></article>)}</div></section><section className="collective-section rationale"><header><span>{t("analytics.why")}</span><h3>{t("analytics.whyTitle")}</h3><p>{why.summary_i18n?.[lang] ?? why.summary}</p></header><div className="rationale-layout"><div className="divergence-ledger">{(why.divergence_points ?? []).map((point) => <article key={point.component}><span>{point.component}</span><div><strong>{point.category ?? t("analytics.divergence")}</strong><p>{point.narrative}</p><small>{formatNumber(point.pair_count)} {t("analytics.pairs")} · {(point.agent_pairs ?? []).slice(0, 3).join(" · ")}</small></div></article>)}</div><div className="convergence-ledger"><span>{t("analytics.measuredAgreement")}</span>{(why.convergence_signals ?? []).map((signal) => <article key={signal.signal}><b>{signal.coverage}</b><div><strong>{signal.signal}</strong><p>{signal.narrative}</p></div></article>)}</div></div></section><section className="collective-section resolution"><header><span>{t("analytics.recommendation")}</span><h3>{t("analytics.recommendationTitle")}</h3><p>{recommendation.simulation_summary ?? t("analytics.noSimulation")}</p></header><div className="resolution-verdict"><div><span>{t("analytics.finalResolution")}</span><strong>{recommendation.final_resolution ?? t("analytics.noResolution")}</strong><small>{recommendation.arbiter ?? t("simulation.native")} · {t("common.round")} {formatNumber(recommendation.round_number)} · {recommendation.status}</small></div><div><span>{t("analytics.followUp")}</span><strong>{recommendation.follow_up_consensus_status ?? t("analytics.notRequired")}</strong><small>{recommendation.remaining_prediction_conflicts === null || recommendation.remaining_prediction_conflicts === undefined ? "—" : formatNumber(recommendation.remaining_prediction_conflicts)} {t("analytics.remainingConflicts")}</small></div></div><div className="modelled-option-grid">{(recommendation.modelled_alternatives ?? []).length ? (recommendation.modelled_alternatives ?? []).map((alternative, index) => <article key={`${alternative.name}-${index}`}><span>{t("analytics.modelledOption")} {String(index + 1).padStart(2, "0")}</span><strong>{alternative.name ?? t("analytics.unnamedAlternative")}</strong><div><b>{alternative.deficit === undefined ? "—" : formatNumber(alternative.deficit)}%</b><small>{t("analytics.projectedDeficit")}</small><b>{alternative.utility === undefined ? "—" : formatNumber(alternative.utility)}</b><small>{t("analytics.utility")}</small></div><small>{alternative.source_tag ?? "SIMULATION_MODELLED"}</small></article>) : <p className="empty">{t("analytics.noAlternatives")}</p>}</div>{(recommendation.limitations ?? []).length > 0 && <div className="analytics-limitations"><span>{t("analytics.limitations")}</span>{(recommendation.limitations ?? []).map((limitation) => <p key={limitation}>{limitation}</p>)}</div>}</section><section className="collective-section normative"><header><span>{t("analytics.normative")}</span><h3>{t("analytics.normativeTitle")}</h3><p>{normative.summary}</p></header><div className="normative-grid">{(normative.principles ?? []).map((principle) => <article key={principle.principle}><div><span>{principle.principle}</span><b className={principle.status.toLowerCase()}>{principle.status.replaceAll("_", " ")}</b></div><p>{principle.evaluation}</p><small>{(principle.legal_sources ?? []).length ? (principle.legal_sources ?? []).join(" · ") : t("analytics.defaultMethod")}</small></article>)}</div></section></section>;
}

function SimulationResolutionPanel({ artifact }: { artifact: SimulationArtifact }) {
  const { t, formatNumber } = useI18n();
  const conflicts = artifact.input?.conflicts ?? [];
  const output = artifact.output ?? {};
  const alternatives = output.alternatives ?? [];
  const limitations = output.limitations ?? [];
  return <section className="simulation-resolution"><div className="simulation-resolution-head"><div><span className="section-number">SIM / {t("simulation.round")} {formatNumber(artifact.round_number)}</span><h3>{t("simulation.title")}</h3></div><span className={`simulation-status ${artifact.status.toLowerCase()}`}>{artifact.status}</span></div><div className="simulation-meta"><div><span>{t("simulation.request")}</span><strong>{artifact.trigger}</strong><small>{formatNumber(conflicts.length)} {t("simulation.conflictPair")} · {conflicts.flatMap((item) => item.components ?? []).join(", ") || t("simulation.awaitingDetail")}</small></div><div><span>{t("simulation.arbiter")}</span><strong>{output.agent_name ?? t("simulation.native")}</strong><small>{artifact.simulation_version} · {output.evidence_status ?? t("simulation.modelledEvidence")}</small></div><div><span>{t("simulation.followUp")}</span><strong>{output.follow_up_consensus_status ?? (artifact.status === "RUNNING" ? t("simulation.inProgress") : t("simulation.notStarted"))}</strong><small>{output.remaining_prediction_conflicts === undefined ? "—" : formatNumber(output.remaining_prediction_conflicts)} {t("simulation.predictionRemaining")}</small></div></div><div className="simulation-resolution-body"><div className="simulation-resolution-copy"><span>{t("simulation.what")}</span><p>{output.simulation_summary ?? (artifact.status === "RUNNING" ? output.message : t("simulation.defaultSummary"))}</p><span>{t("simulation.resolution")}</span><p>{output.resolution ?? output.message ?? t("simulation.accepted")}</p>{output.fallback_reason && <small className="simulation-fallback">{t("simulation.fallback")}: {output.fallback_reason}</small>}</div>{alternatives.length ? <div className="simulation-alternatives"><span>{t("simulation.alternatives")}</span>{alternatives.slice(0, 4).map((alternative, index) => <div className="simulation-alternative" key={`${alternative.name}-${index}`}><strong>{alternative.name ?? t("simulation.alternative", { index: formatNumber(index + 1) })}</strong><small>{t("common.deficit")} {alternative.deficit === undefined ? "—" : formatNumber(alternative.deficit)}% · {t("common.utility")} {alternative.utility === undefined ? "—" : formatNumber(alternative.utility)}</small><em>{alternative.source_tag ?? "SIMULATION_MODELLED"}</em></div>)}</div> : null}</div><div className="simulation-impact-grid"><section><span>{t("simulation.riskUncertainty")}</span>{[...(output.risks ?? []), ...(output.uncertainties ?? [])].slice(0, 6).map((item, index) => <p key={index}>{item.content}</p>)}</section><section><span>{t("analytics.limitations")}</span>{limitations.length ? limitations.map((item, index) => <p key={index}>{item}</p>) : <p>{output.evidence_status ?? t("simulation.noLimitations")}</p>}</section></div></section>;
}

export default function Home() {
  const { lang, setLang, t, query, formatNumber, formatDateTime } = useI18n();
  const [agent, setAgent] = useState<AgentForm>(initialAgent);
  const [scenario, setScenario] = useState<ScenarioForm>(initialScenario);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [templates, setTemplates] = useState<AgentTemplate[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState("");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<number | null>(null);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [graph, setGraph] = useState<RunGraphPayload | null>(null);
  const [run, setRun] = useState<RunState | null>(null);
  const [notice, setNotice] = useState("");
  const [domainRules, setDomainRules] = useState<DomainRules | null>(null);
  const [rulesBusy, setRulesBusy] = useState(false);
  const [busyAgentId, setBusyAgentId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [editingAgentId, setEditingAgentId] = useState<number | null>(null);
  const [scenarioExpanded, setScenarioExpanded] = useState(false);
  const [templateModalOpen, setTemplateModalOpen] = useState(false);
  const [templateConfigs, setTemplateConfigs] = useState<Record<string, { llm_base_url: string; llm_api_key: string; llm_model: string; temperature: number; max_tokens: number }>>({});
  const [connectionTests, setConnectionTests] = useState<Record<number, { ok: boolean; message: string }>>({});
  const [mandateLogs, setMandateLogs] = useState<RunLog[]>([]);
  const [trackerHistory, setTrackerHistory] = useState<RunLog[]>([]);
  const trackerLogCounts = useRef<Record<string, number>>({});
  const trackerConsoleRef = useRef<HTMLDivElement>(null);
  const templateButtonRef = useRef<HTMLButtonElement>(null);
  const templateModalRef = useRef<HTMLDivElement>(null);
  const apiFetch = (url: string, init?: RequestInit) => fetch(query(url), init);

  function appendTrackerLogs(taskId: string, logs: RunLog[]) {
    const previousCount = trackerLogCounts.current[taskId] ?? 0;
    const additions = logs.slice(previousCount);
    trackerLogCounts.current[taskId] = Math.max(previousCount, logs.length);
    if (additions.length) setTrackerHistory((current) => [...current, ...additions]);
  }

  function appendQueueLog(taskId: string, logs: RunLog[]) {
    setTrackerHistory(logs);
    trackerLogCounts.current = { [taskId]: logs.length };
  }

  async function loadTemplates() {
    const response = await apiFetch("/api/agent-templates", { cache: "no-store" });
    if (!response.ok) throw new Error(t("error.templateStatus", { status: response.status }));
    const payload: AgentTemplate[] = await response.json();
    if (!Array.isArray(payload) || payload.length !== 5) throw new Error(t("error.templateInvalid"));
    setTemplates(payload);
  }

  async function loadSetup() {
    const [agentResponse, scenarioResponse] = await Promise.all([
      apiFetch(`${apiUrl}/api/agents`),
      apiFetch(`${apiUrl}/api/scenarios`),
    ]);
    if (!agentResponse.ok || !scenarioResponse.ok) throw new Error(t("error.apiUnavailable"));
    const nextAgents: Agent[] = await agentResponse.json();
    const nextScenarios: Scenario[] = await scenarioResponse.json();
    setAgents(nextAgents);
    setScenarios(nextScenarios);
    if (selectedScenario === null && nextScenarios.length > 0) setSelectedScenario(nextScenarios[nextScenarios.length - 1].id);
  }

  async function loadDashboard(id: number, sessionId?: string) {
    const endpoint = sessionId
      ? `${apiUrl}/api/scenarios/${id}/dashboard?session_id=${encodeURIComponent(sessionId)}`
      : `${apiUrl}/api/scenarios/${id}/dashboard`;
    const response = await apiFetch(endpoint, { cache: "no-store" });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || t("error.dashboard", { status: response.status }));
    }
    const payload: Dashboard = await response.json();
    setDashboard(payload);
    setDomainRules(payload.domain_rules);
  }

  async function loadRunGraph(taskId: string | null, scenarioId: number, sessionId: string) {
    const endpoint = taskId
      ? `${apiUrl}/api/runs/${taskId}/graph`
      : `${apiUrl}/api/scenarios/${scenarioId}/runs/${sessionId}/graph`;
    const response = await apiFetch(endpoint, { cache: "no-store" });
    if (!response.ok) throw new Error(t("error.graph", { status: response.status }));
    const payload: RunGraphPayload = await response.json();
    setGraph(payload);
  }

  async function loadLatestRun(id: number) {
    const response = await apiFetch(`${apiUrl}/api/scenarios/${id}/runs/latest`, { cache: "no-store" });
    if (response.status === 404) {
      setRun(null);
      setGraph(null);
      setTrackerHistory([]);
      return;
    }
    if (!response.ok) throw new Error(t("error.history", { status: response.status }));
    const payload: RunState = await response.json();
    setRun(payload);
    setTrackerHistory(payload.logs ?? []);
    if (payload.task_id) trackerLogCounts.current[payload.task_id] = payload.logs?.length ?? 0;
    await loadRunGraph(payload.task_id, payload.scenario_id, payload.session_id);
    if (payload.status === "SUCCEEDED" && payload.scenario_id === id) await loadDashboard(id, payload.session_id);
  }
  async function loadDomainRules(id: number) {
    const response = await apiFetch(`${apiUrl}/api/scenarios/${id}/domain-rules`, { cache: "no-store" });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || t("error.mandateData", { status: response.status }));
    }
    const payload: DomainRules = await response.json();
    setDomainRules(payload);
    return payload;
  }

  async function refreshMandateData(id: number) {
    const results = await Promise.allSettled([loadDashboard(id), loadDomainRules(id)]);
    if (results.every((result) => result.status === "rejected")) {
      throw new Error(t("error.mandateRefresh"));
    }
  }
  useEffect(() => {
    loadTemplates().catch((error) => setNotice(t("notice.templatesFailed", { message: error.message })));
    loadSetup().catch(() => setNotice(t("notice.apiPending")));
  }, []);
  useEffect(() => {
    if (selectedScenario !== null) {
      Promise.all([loadDashboard(selectedScenario), loadLatestRun(selectedScenario)]).catch(() => setNotice(t("notice.noDashboard")));
    }
  }, [selectedScenario, lang]);
  useEffect(() => {
    const consoleElement = trackerConsoleRef.current;
    if (consoleElement) consoleElement.scrollTop = consoleElement.scrollHeight;
  }, [trackerHistory.length]);

  useEffect(() => {
    if (!run || ["SUCCEEDED", "FAILED"].includes(run.status)) return;
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        if (!run.task_id) {
          await loadLatestRun(run.scenario_id);
        } else {
          const response = await apiFetch(`${apiUrl}/api/runs/${run.task_id}`, { cache: "no-store" });
          if (!response.ok) {
            if (!cancelled) setNotice(t("notice.trackerError", { status: response.status }));
          } else {
            const nextRun: RunState = await response.json();
            if (cancelled) return;
            setRun(nextRun);
            if (nextRun.task_id) appendTrackerLogs(nextRun.task_id, nextRun.logs);
            await loadRunGraph(nextRun.task_id, nextRun.scenario_id, nextRun.session_id);
            if (cancelled) return;
            if (nextRun.status === "SUCCEEDED" && selectedScenario === nextRun.scenario_id) {
              await loadDashboard(nextRun.scenario_id, nextRun.session_id);
              if (!cancelled) setNotice(t("notice.complete"));
            } else if (nextRun.status === "FAILED") {
              setNotice(nextRun.error ?? t("notice.failed"));
            }
          }
        }
      } catch {
        if (!cancelled) setNotice(t("notice.apiPending"));
      } finally {
        if (!cancelled) timer = window.setTimeout(poll, 1200);
      }
    };
    timer = window.setTimeout(poll, 1200);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [run?.task_id, run?.status, run?.scenario_id, selectedScenario, lang]);

  useEffect(() => {
    if (!run || !["SUCCEEDED", "FAILED"].includes(run.status)) return;
    const graphPending = graph?.session_id !== run.session_id || graph.run_status !== run.status;
    const dashboardPending = run.status === "SUCCEEDED" && selectedScenario === run.scenario_id && dashboard?.session_id !== run.session_id;
    if (!graphPending && !dashboardPending) {
      setNotice(run.status === "SUCCEEDED" ? t("notice.complete") : run.error ?? t("notice.failed"));
      return;
    }
    let cancelled = false;
    let timer: number | undefined;
    let attempts = 0;
    const maximumAttempts = 5;
    const hydrate = async () => {
      attempts += 1;
      const requests: Promise<void>[] = [];
      if (graphPending) requests.push(loadRunGraph(run.task_id, run.scenario_id, run.session_id));
      if (dashboardPending) requests.push(loadDashboard(run.scenario_id, run.session_id));
      const results = await Promise.allSettled(requests);
      if (cancelled) return;
      if (results.every((result) => result.status === "fulfilled")) {
        setNotice(run.status === "SUCCEEDED" ? t("notice.complete") : run.error ?? t("notice.failed"));
      } else if (attempts < maximumAttempts) {
        setNotice(t("notice.hydrationRetry", { attempt: attempts, maximum: maximumAttempts }));
        timer = window.setTimeout(hydrate, 1200);
      } else {
        setNotice(t("notice.hydrationFailed"));
      }
    };
    void hydrate();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [run?.status, run?.session_id, run?.scenario_id, run?.task_id, run?.error, graph?.session_id, graph?.run_status, dashboard?.session_id, selectedScenario, lang]);

  useEffect(() => {
    if (!templateModalOpen) return;
    const modal = templateModalRef.current;
    const focusable = modal?.querySelector<HTMLElement>("button, input, select, textarea, [tabindex]:not([tabindex='-1'])");
    focusable?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setTemplateModalOpen(false);
        templateButtonRef.current?.focus();
      }
      if (event.key !== "Tab" || !modal) return;
      const items = Array.from(modal.querySelectorAll<HTMLElement>("button, input, select, textarea, [tabindex]:not([tabindex='-1'])")).filter((item) => !item.hasAttribute("disabled"));
      if (!items.length) return;
      const first = items[0];
      const last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [templateModalOpen]);

  function closeTemplateModal() {
    setTemplateModalOpen(false);
    window.requestAnimationFrame(() => templateButtonRef.current?.focus());
  }

  function applyTemplate(key: string) {
    setSelectedTemplate(key);
    const template = templates.find((item) => item.key === key);
    if (!template) return;
    setAgent({
      ...agent,
      name: template.name,
      role: template.role,
      temperature: template.temperature,
      max_tokens: template.max_tokens,
      theta_x: template.theta_x,
      theta_q: template.theta_q,
      theta_h: template.theta_h,
      theta_s: template.theta_s,
      theta_u: template.theta_u,
    });
  }

  function openTemplateModal() {
    setTemplateConfigs(Object.fromEntries(templates.map((item) => [item.key, { llm_base_url: "", llm_api_key: "", llm_model: "", temperature: item.temperature, max_tokens: item.max_tokens }])));
    setTemplateModalOpen(true);
  }

  async function loadAllTemplates() {
    setBusy(true);
    try {
      const configs = Object.fromEntries(Object.entries(templateConfigs).map(([key, value]) => [key, { ...value, llm_base_url: value.llm_base_url || null, llm_api_key: value.llm_api_key || null, llm_model: value.llm_model || null }]));
      const response = await apiFetch(`${apiUrl}/api/agents/load-templates`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ configs }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? t("error.templatesLoad"));
      await Promise.all([loadSetup(), loadTemplates()]); setDomainRules(null); closeTemplateModal();
      setNotice(t("notice.templatesLoaded", { created: formatNumber(payload.created), total: formatNumber(payload.total) }));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : t("error.templatesLoad"));
    } finally {
      setBusy(false);
    }
  }

  async function submitAgent(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true);
    try {
      const agentPayload = {
        ...agent,
        template_key: selectedTemplate || null,
        llm_base_url: agent.llm_base_url || null,
        llm_api_key: agent.llm_api_key || undefined,
        llm_model: agent.llm_model || null,
      };
      const endpoint = editingAgentId ? `${apiUrl}/api/agents/${editingAgentId}` : `${apiUrl}/api/agents`;
      const response = await apiFetch(endpoint, { method: editingAgentId ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(agentPayload) });
      const payload = await response.json(); if (!response.ok) throw new Error(payload.detail ?? t("error.agentSave"));
      setAgent(initialAgent); setSelectedTemplate(""); setEditingAgentId(null); setDomainRules(null); await loadSetup(); setNotice(t("notice.agentSaved", { name: payload.name }));
    } catch (error) { setNotice(error instanceof Error ? error.message : t("error.agentSave")); } finally { setBusy(false); }
  }

  function editAgent(item: Agent) {
    setEditingAgentId(item.id);
    setSelectedTemplate(item.template_key ?? "");
    setAgent({ name: item.name, role: item.role, theta_x: item.theta_x, theta_q: item.theta_q, theta_h: item.theta_h, theta_s: item.theta_s, theta_u: item.theta_u, llm_base_url: item.llm_base_url ?? "", llm_api_key: "", llm_model: item.llm_model ?? "", temperature: item.temperature, max_tokens: item.max_tokens });
    document.getElementById("setup")?.scrollIntoView({ behavior: "smooth" });
  }

  async function deleteAgent(item: Agent) {
    if (!window.confirm(t("confirm.deleteAgent", { name: item.name }))) return;
    const response = await apiFetch(`${apiUrl}/api/agents/${item.id}`, { method: "DELETE" });
    if (!response.ok) { const payload = await response.json(); setNotice(payload.detail ?? t("notice.agentDeleteFailed")); return; }
    setDomainRules(null); await loadSetup(); setNotice(t("notice.agentDeleted", { name: item.name }));
  }

  async function testConnection(item: Agent) {
    setConnectionTests((current) => ({ ...current, [item.id]: { ok: false, message: t("notice.testing") } }));
    const response = await apiFetch(`${apiUrl}/api/agents/${item.id}/test-connection`, { method: "POST" });
    const payload = await response.json();
    setConnectionTests((current) => ({ ...current, [item.id]: { ok: payload.ok, message: payload.ok ? `OK · ${formatNumber(payload.latency_ms, { maximumFractionDigits: 0 })}ms · ${payload.model}` : payload.error } }));
  }

  async function generateMandate() {
    if (selectedScenario === null || agents.length === 0) return;
    const scenarioId = selectedScenario;
    setRulesBusy(true);
    try {
      const response = await apiFetch(`${apiUrl}/api/scenarios/${scenarioId}/domain-rules`, { method: "POST" });
      if (!response.ok) {
        const rawError = await response.text();
        let message = rawError || t("error.mandateGenerate");
        try {
          const parsedError = JSON.parse(rawError);
          message = typeof parsedError.detail === "string" ? parsedError.detail : parsedError.detail?.message ?? parsedError.message ?? message;
        } catch {}
        throw new Error(message);
      }
      const rawPayload = await response.text();
      let payload: DomainRules;
      try {
        payload = JSON.parse(rawPayload) as DomainRules;
      } catch {
        throw new Error(rawPayload || t("error.mandateInvalid"));
      }
      if (!payload.generated) throw new Error(payload.detail ?? t("error.mandateNotGenerated"));
      setDomainRules({ ...payload, generated: true, stale: false });
      try {
        await refreshMandateData(scenarioId);
      } catch (refreshError) {
        setNotice(refreshError instanceof Error ? refreshError.message : t("error.refreshDelayed"));
      }
      setMandateLogs((current) => [...current, { stage: "MANDATE", level: payload.status === "partial" ? "WARNING" : "SUCCESS", message: payload.detail ?? t("notice.mandateGenerated", { count: formatNumber(payload.generated_count ?? 0) }) }]);
      setNotice(payload.detail ?? t("notice.mandateReady"));
    } catch (error) {
      const message = error instanceof Error ? error.message : t("error.mandateGenerate");
      setMandateLogs((current) => [...current, { stage: "MANDATE", level: "ERROR", message }]);
      setNotice(message);
    } finally {
      setRulesBusy(false);
    }
  }

  async function generateAgentMandate(agentId: number) {
    if (selectedScenario === null) return;
    const scenarioId = selectedScenario;
    setBusyAgentId(agentId);
    try {
      const response = await apiFetch(
        `${apiUrl}/api/scenarios/${scenarioId}/agents/${agentId}/domain-rules`,
        { method: "POST" },
      );
      const rawPayload = await response.text();
      let payload: AgentDomainRules | { detail?: string };
      try {
        payload = JSON.parse(rawPayload) as AgentDomainRules | { detail?: string };
      } catch {
        throw new Error(rawPayload || t("error.agentMandateGenerate"));
      }
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : t("error.agentMandateGenerate"));
      try {
        await refreshMandateData(scenarioId);
      } catch (refreshError) {
        setNotice(refreshError instanceof Error ? refreshError.message : t("error.refreshDelayed"));
      }
      const generated = payload as AgentDomainRules;
      setMandateLogs((current) => [...current, { stage: "MANDATE", level: generated.synthesis_status === "generated" ? "SUCCESS" : "WARNING", message: t("notice.agentMandateUpdated", { name: generated.name }) }]);
      setNotice(t("notice.agentMandateRefreshed", { name: generated.name }));
    } catch (error) {
      const message = error instanceof Error ? error.message : t("error.agentMandateGenerate");
      setMandateLogs((current) => [...current, { stage: "MANDATE", level: "ERROR", message }]);
      setNotice(message);
    } finally {
      setBusyAgentId(null);
    }
  }

  async function submitScenario(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true);
    try {
      const response = await apiFetch(`${apiUrl}/api/scenarios`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scenario) });
      const payload = await response.json(); if (!response.ok) throw new Error(payload.detail ?? t("notice.scenarioSaveFailed"));
      setScenario(initialScenario); await loadSetup(); setDomainRules(null); setDashboard(null); setGraph(null); setRun(null); setTrackerHistory([]); setSelectedScenario(payload.id); setNotice(t("notice.scenarioSaved", { id: payload.id }));
    } catch (error) { setNotice(error instanceof Error ? error.message : t("notice.scenarioSaveFailed")); } finally { setBusy(false); }
  }

  async function startRun() {
    if (selectedScenario === null) { setNotice(t("notice.selectRun")); return; }
    if (rulesAreStale) { setNotice(t("notice.rulesStale")); return; }
    let response: Response;
    try {
      response = await apiFetch(`${apiUrl}/api/scenarios/${selectedScenario}/runs`, { method: "POST" });
    } catch {
      setNotice(t("notice.queueFailed"));
      return;
    }
    const payload = await response.json().catch(() => null);
    if (!response.ok || !payload) { setNotice(t("notice.queueFailed")); return; }
    setRun({ task_id: payload.task_id, session_id: payload.session_id, scenario_id: payload.scenario_id, status: payload.status, logs: payload.logs, simulation_artifacts: [] });
    appendQueueLog(payload.task_id, payload.logs);
    setNotice(t("notice.queued", { task: payload.task_id.slice(0, 8) }));
    try {
      await loadRunGraph(payload.task_id, payload.scenario_id, payload.session_id);
    } catch {
      setNotice(t("notice.queuedGraphDelayed", { task: payload.task_id.slice(0, 8) }));
    }
  }

  async function exportManifest() {
    if (selectedScenario === null) { setNotice(t("notice.selectExport")); return; }
    const response = await apiFetch(`${apiUrl}/api/scenarios/${selectedScenario}/manifest`);
    if (!response.ok) { setNotice(t("notice.exportFailed")); return; }
    const blob = await response.blob(); const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `shcr-scenario-${selectedScenario}-manifest.json`; anchor.click(); URL.revokeObjectURL(url); setNotice(t("notice.exported"));
  }

  const latest = dashboard?.latest_metric ?? null;
  const activeLogs = trackerHistory.length ? trackerHistory : [{ stage: "IDLE", level: "INFO", message: t("notice.idle") }];
  const selectedTemplateData = templates.find((item) => item.key === selectedTemplate);
  const canGenerateRules = selectedScenario !== null && agents.length > 0;
  const rulesAreStale = canGenerateRules && (!domainRules || domainRules.scenario_id !== selectedScenario || !domainRules.generated || domainRules.stale || domainRules.agent_count !== agents.length);
  const activeInfluence = useMemo(() => [...(dashboard?.influence_observations ?? [])].sort((a, b) => (b.normalized_weight ?? 0) - (a.normalized_weight ?? 0)), [dashboard]);
  const simulationArtifacts = run ? run.simulation_artifacts ?? [] : dashboard?.simulation_artifacts ?? [];
  const agentBreakdown = run?.agent_breakdown?.length ? run.agent_breakdown : dashboard?.agent_breakdown ?? [];
  const narrativeDisagreements = run?.disagreements?.length ? run.disagreements : dashboard?.disagreements ?? [];
  const collectiveReasoning = run?.collective_reasoning ?? dashboard?.collective_reasoning ?? null;

  return <main className="shell dashboard-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">S</span><span>SHCR</span></div><div className="top-actions"><label className="language-select"><span>{t("language.label")}</span><select aria-label={t("language.label")} value={lang} onChange={(event) => setLang(event.target.value as "id" | "en")}><option value="id">ID · {t("language.id")}</option><option value="en">EN · {t("language.en")}</option></select></label><div className="top-meta"><span className="live-dot" /> {t("top.node")} <b>{t("top.phase")}</b></div></div></header>
    <section className="dashboard-hero"><div><div className="eyebrow">{t("hero.eyebrow")}</div><h1>{t("hero.title")}<br /><em>{t("hero.emphasis")}</em></h1><p>{t("hero.description")}</p></div><div className="hero-orbit"><span>DDR</span><b>→</b><span>CAR</span><b>→</b><span>SHCR</span></div></section>
    <div className="notice" role="status" aria-live="polite"><span className="notice-label">{t("notice.label")}</span><span>{notice || t("notice.ready")}</span></div>

    <section className="control-strip"><div className="scenario-overview"><span>{t("scenario.active")}</span><strong>{selectedScenario ? t("scenario.number", { id: selectedScenario }) : t("scenario.none")}</strong><p className={scenarioExpanded ? "expanded" : "collapsed"}>{scenarios.find((item) => item.id === selectedScenario)?.description ?? t("scenario.prompt")}</p><button type="button" onClick={() => setScenarioExpanded((value) => !value)}>{scenarioExpanded ? t("scenario.hide") : t("scenario.show")}</button><select aria-label={t("scenario.select")} value={selectedScenario ?? ""} onChange={(event) => { setSelectedScenario(Number(event.target.value)); setDomainRules(null); setDashboard(null); setGraph(null); setRun(null); setTrackerHistory([]); setMandateLogs([]); }}><option value="" disabled>{t("scenario.select")}</option>{scenarios.map((item) => <option key={item.id} value={item.id}>#{item.id} — {item.description.slice(0, 72)}</option>)}</select></div><button className="primary-button run-button" onClick={startRun} disabled={run?.status === "RUNNING" || run?.status === "QUEUED" || rulesAreStale}>{run?.status === "RUNNING" ? t("run.running") : run?.status === "QUEUED" ? t("run.queued") : t("run.start")}<span>↗</span></button></section>

    <nav className="menu-rail"><span className="menu-label">{t("nav.label")}</span><a href="#setup">{t("nav.setup")}</a><a href="#setup">{t("nav.scenario")}</a><a className="active" href="#tracker">{t("nav.tracker")}</a><a href="#agent-breakdown">{t("nav.agents")}</a><a href="#network">{t("nav.graph")}</a><a href="#ddr">{t("nav.ddr")}</a><a href="#analytics">{t("nav.analytics")}</a></nav>

    <section id="tracker" className="tracker-panel panel"><div className="section-header"><div><span className="section-number">03</span><h2>{t("tracker.title")}</h2></div><span className={`run-state ${run?.status?.toLowerCase() ?? "idle"}`}>{run?.status ?? "IDLE"}</span></div><div className="tracker-content"><div ref={trackerConsoleRef} className="progress-console max-h-[500px] overflow-y-auto">{activeLogs.map((log, index) => <div className={`console-line ${log.level.toLowerCase()}`} key={`${log.stage}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><i className={index === activeLogs.length - 1 && !["SUCCEEDED", "FAILED"].includes(run?.status ?? "") ? "pulse" : "done"} /><div><b>[{log.level}] {log.code ?? log.stage}</b><small>{log.messages?.[lang] ?? log.message}</small>{log.fallback?.used && <em>Fallback: {log.fallback.retained_artifact ?? log.fallback.kind} · {log.fallback.consensus_impact}</em>}</div></div>)}</div><div className="task-readout"><span>{t("tracker.task")}</span><strong>{run?.task_id ?? "—"}</strong><small>{run?.error ?? (run?.status === "SUCCEEDED" ? t("tracker.persisted") : t("tracker.polling"))}</small></div></div><AgentBreakdownPanel agents={agentBreakdown} disagreements={narrativeDisagreements} simulations={simulationArtifacts} />{simulationArtifacts.length > 0 && <div className="simulation-stack">{simulationArtifacts.map((artifact) => <SimulationResolutionPanel artifact={artifact} key={artifact.id} />)}</div>}</section>

    <section id="network" className="panel graph-panel"><div className="section-header"><div><span className="section-number">05</span><h2>{t("graph.title")}</h2></div><div className="graph-legend"><span><i className="pending" /> {t("status.pending")}</span><span><i className="running" /> {t("status.active")}</span><span><i className="succeeded" /> {t("status.complete")}</span><span><i className="warning" /> {t("status.dissent")}</span><span><i className="failed" /> {t("status.failed")}</span></div></div><RunGraph graph={graph} /></section>

    <section id="ddr" className="panel"><div className="section-header"><div><span className="section-number">06</span><h2>{t("ddr.title")}</h2></div><span className="panel-code">D<sub>ij</sub> / {t("ddr.components")}</span></div><DdrNetworkPanel disagreements={narrativeDisagreements} breakdown={agentBreakdown} /></section>


    <section id="analytics" className="analytics-section"><div className="section-header"><div><span className="section-number">07</span><h2>{t("analytics.title")}</h2></div><button className="export-button" onClick={exportManifest}>{t("analytics.export")}</button></div><CollectiveReasoningPanel analysis={collectiveReasoning} /><div className="convergence-banner"><div><span>{t("analytics.final")}</span><strong>{latest?.convergence_status ?? t("analytics.awaiting")}</strong></div><div className="convergence-meta"><span>{t("analytics.feasible")}</span><b>{latest?.feasible_alternatives_count === undefined || latest?.feasible_alternatives_count === null ? "—" : formatNumber(latest.feasible_alternatives_count)}</b></div></div><div className="metric-grid"><MetricCard label={t("analytics.metrics.constraint")} value={latest ? formatNumber(latest.hard_constraint_violation_rate, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"} unit="%" tone="rust" /><MetricCard label={t("analytics.metrics.provenance")} value={latest ? formatNumber(latest.provenance_completeness_percent, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"} unit="%" /><MetricCard label={t("analytics.metrics.schema")} value={dashboard ? formatNumber(dashboard.schema_validity_percent, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"} unit="%" tone="mint" /><MetricCard label={t("analytics.metrics.latency")} value={latest ? formatNumber(latest.latency_ms, { maximumFractionDigits: 0 }) : "—"} unit={latest ? `ms · ${formatNumber(latest.token_usage)} tok` : ""} /></div><div className="analytics-lower"><div className="history-block"><div className="subhead"><span>{t("analytics.metricHistory")}</span><b>{formatNumber(dashboard?.metric_history?.length ?? 0)} {t("common.runs")}</b></div>{dashboard?.metric_history?.length ? dashboard.metric_history.slice(0, 5).map((metric) => <div className="history-row" key={metric.id}><span>#{metric.id}</span><strong>{metric.convergence_status}</strong><i>{formatNumber(metric.provenance_completeness_percent, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}% {t("analytics.provenance")}</i><b>{formatDateTime(metric.created_at)}</b></div>) : <p className="empty">{t("analytics.noHistory")}</p>}</div><div className="influence-block"><div className="subhead"><span>{t("analytics.influence")}</span><b>{formatNumber(activeInfluence.length)} {t("common.observations")}</b></div>{activeInfluence.length ? activeInfluence.map((item, index) => <div className="weight-row" key={`${item.agent}-${item.proposition}`}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{item.agent}</strong><small>{item.proposition}</small></div><b>{item.normalized_weight === null ? "—" : `${formatNumber(item.normalized_weight * 100, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`}</b></div>) : <p className="empty">{t("analytics.noInfluence")}</p>}</div></div></section>

    <section id="setup" className="setup-section">
      <div className="section-header"><div><span className="section-number">01 / 02</span><h2>{t("setup.title")}</h2></div><button ref={templateButtonRef} type="button" className="template-load-button" onClick={openTemplateModal} disabled={busy}>{t("setup.loadTemplates")}</button></div>
      <p className="setup-intro">{t("setup.intro")}</p>
      <form className="agent-config-form" onSubmit={submitAgent}>
        <fieldset className="form-card">
          <legend><span>01</span> {t("setup.agentIdentity")}</legend>
          <div className="form-grid two-column">
            <label className="form-field"><strong>{t("form.template")}</strong><select value={selectedTemplate} onChange={(event) => applyTemplate(event.target.value)}><option value="">{t("setup.emptyTemplate")}</option>{templates.map((template) => <option key={template.key} value={template.key}>{template.name}</option>)}</select><small>{t("form.templateHint")}</small></label>
            <label className="form-field"><strong>{t("form.agentName")}</strong><input value={agent.name} onChange={(event) => setAgent({ ...agent, name: event.target.value })} required /><small>{t("form.agentNameHint")}</small></label>
            <label className="form-field full-width"><strong>{t("form.role")}</strong><input value={agent.role} onChange={(event) => setAgent({ ...agent, role: event.target.value })} required /><small>{t("form.roleHint")}</small></label>
          </div>
          {selectedTemplateData && <TemplateContractPreview template={selectedTemplateData} />}
        </fieldset>

        <fieldset className="form-card">
          <legend><span>02</span> {t("setup.llm")}</legend>
          <div className="form-grid two-column">
            <label className="form-field full-width"><strong>{t("form.apiBase")}</strong><input type="url" value={agent.llm_base_url ?? ""} onChange={(event) => setAgent({ ...agent, llm_base_url: event.target.value })} /><small>{t("form.baseHint")}</small></label>
            <label className="form-field"><strong>{t("form.apiKey")}</strong><input type="password" autoComplete="new-password" value={agent.llm_api_key} onChange={(event) => setAgent({ ...agent, llm_api_key: event.target.value })} /><small>{t("form.keyHint")}</small></label>
            <label className="form-field"><strong>{t("form.model")}</strong><input value={agent.llm_model ?? ""} onChange={(event) => setAgent({ ...agent, llm_model: event.target.value })} /><small>{t("form.modelHint")}</small></label>
            <label className="form-field"><strong>{t("form.temperature")}</strong><input type="number" min="0" max="2" step="0.01" value={agent.temperature} onChange={(event) => setAgent({ ...agent, temperature: Number(event.target.value) })} required /><small>{t("form.temperatureHint")}</small></label>
            <label className="form-field"><strong>{t("form.maxTokens")}</strong><input type="number" min="1" step="1" value={agent.max_tokens} onChange={(event) => setAgent({ ...agent, max_tokens: Number(event.target.value) })} required /><small>{t("form.tokensHint")}</small></label>
          </div>
        </fieldset>

        <fieldset className="form-card mandate-card">
          <legend><span>03</span> {t("setup.mandate")}</legend>
          <div className="automatic-rules"><div className="automatic-rules-intro"><div><strong>{selectedTemplate ? t("mandate.templateActive") : t("mandate.ready")}</strong><p>{t("mandate.description")}</p></div><button type="button" className="template-load-button" onClick={generateMandate} disabled={!canGenerateRules || rulesBusy || busyAgentId !== null}>{rulesBusy ? t("common.generating") : t("mandate.generateAll")}</button></div>{rulesAreStale && <small className="stale-rules">{t("mandate.stale")}</small>}{domainRules && !domainRules.stale && <><div className="rules-revision"><span>{t("mandate.revision")} {domainRules.revision}</span><b>{formatNumber(domainRules.agent_count)} {t("common.agents")}</b><small>{formatNumber(domainRules.rules.owned_checks?.length ?? 0)} {t("mandate.hardChecks")} · {formatNumber(domainRules.rules.hard_constraints?.length ?? 0)} {t("mandate.globalConstraints")}</small></div><div className="agent-mandate-grid">{(domainRules.agent_rules ?? []).map((rules, index) => <AgentMandateCard rules={rules} index={index} onGenerate={generateAgentMandate} busy={rulesBusy || busyAgentId !== null} key={rules.agent_id} />)}</div></>}{mandateLogs.length > 0 && <div className="mandate-console">{mandateLogs.map((log, index) => <div className={log.level.toLowerCase()} key={`${log.stage}-${index}`}><b>[{log.level}] {log.stage}</b><span>{log.message}</span></div>)}</div>}</div>
        </fieldset>

        <fieldset className="form-card">
          <legend><span>04</span> {t("setup.weights")}</legend>
          <div className="theta-form-grid">
            <NumericField label={t("theta.expertise")} value={agent.theta_x} hint={t("theta.expertiseHint")} onChange={(value) => setAgent({ ...agent, theta_x: value })} />
            <NumericField label={t("theta.evidence")} value={agent.theta_q} hint={t("theta.evidenceHint")} onChange={(value) => setAgent({ ...agent, theta_q: value })} />
            <NumericField label={t("theta.history")} value={agent.theta_h} hint={t("theta.historyHint")} onChange={(value) => setAgent({ ...agent, theta_h: value })} />
            <NumericField label={t("theta.relevance")} value={agent.theta_s} hint={t("theta.relevanceHint")} onChange={(value) => setAgent({ ...agent, theta_s: value })} />
            <NumericField label={t("theta.uncertainty")} value={agent.theta_u} hint={t("theta.uncertaintyHint")} onChange={(value) => setAgent({ ...agent, theta_u: value })} />
          </div>
        </fieldset>
        <button className="primary-button agent-submit" disabled={busy} type="submit">{editingAgentId ? t("setup.saveChanges") : t("setup.saveAgent")}<span>↗</span></button>
      </form>

      <div className="agent-register"><div className="subhead"><span>{t("setup.savedAgents")}</span><b>{formatNumber(agents.length)} {t("common.agents")}</b></div>{agents.length ? agents.map((item) => <div className="agent-register-row" key={item.id}><div><strong>{item.name}</strong><small>{item.role}</small></div><span>{item.template_key ? t("setup.templateTag") : t("setup.customTag")}</span><b>{item.llm_model ?? t("setup.envDefault")}</b><div className="agent-actions"><button type="button" onClick={() => generateAgentMandate(item.id)} disabled={rulesBusy || busyAgentId !== null || busyAgentId === item.id}>{busyAgentId === item.id ? t("common.generating") : t("common.generateMandate")}</button><button type="button" onClick={() => testConnection(item)}>{t("action.test")}</button><button type="button" onClick={() => editAgent(item)}>{t("action.edit")}</button><button type="button" className="danger" onClick={() => deleteAgent(item)}>{t("action.delete")}</button>{connectionTests[item.id] && <small className={connectionTests[item.id].ok ? "test-ok" : "test-error"}>{connectionTests[item.id].message}</small>}</div></div>) : <p className="empty">{t("setup.noAgents")}</p>}</div>

      <form className="scenario-config-form" onSubmit={submitScenario}><div><span className="section-number">{t("scenario.section")}</span><h3>{t("setup.scenario")}</h3><small>{t("form.scenarioHint")}</small></div><label className="form-field"><strong>{t("form.goal")}</strong><textarea value={scenario.description} onChange={(event) => setScenario({ ...scenario, description: event.target.value })} rows={4} required /><small>{t("form.goalHint")}</small></label><label className="form-field"><strong>{t("form.programCost")}</strong><input type="number" min="0" step="0.01" value={scenario.program_cost ?? ""} onChange={(event) => setScenario({ ...scenario, program_cost: event.target.value === "" ? null : Number(event.target.value) })} /><small>{t("form.costHint")}</small></label><button className="secondary-button" disabled={busy}>{t("action.saveScenario")}</button></form>
    </section>
    {templateModalOpen && <div className="modal-backdrop" role="presentation"><div ref={templateModalRef} className="template-modal" role="dialog" aria-modal="true" aria-labelledby="template-modal-title"><div className="section-header"><div><span className="section-number">LLM</span><h2 id="template-modal-title">{t("setup.modalTitle")}</h2></div><button type="button" className="modal-close" aria-label={t("modal.close")} onClick={closeTemplateModal}>×</button></div><p>{t("setup.modalDescription")}</p><div className="template-config-list">{templates.map((item) => { const config = templateConfigs[item.key]; if (!config) return null; return <fieldset className="template-config-card" key={item.key}><legend>{item.name}</legend><label><span>{t("form.apiBase")}</span><input value={config.llm_base_url} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_base_url: event.target.value } })} /></label><label><span>{t("form.apiKey")}</span><input type="password" value={config.llm_api_key} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_api_key: event.target.value } })} /></label><label><span>{t("form.model")}</span><input value={config.llm_model} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_model: event.target.value } })} /></label><label><span>{t("form.temperature")}</span><input type="number" min="0" max="2" step="0.01" value={config.temperature} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, temperature: Number(event.target.value) } })} /></label><label><span>{t("form.maxTokens")}</span><input type="number" min="1" value={config.max_tokens} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, max_tokens: Number(event.target.value) } })} /></label></fieldset>; })}</div><button type="button" className="primary-button" onClick={loadAllTemplates} disabled={busy}>{busy ? t("setup.saving") : t("setup.modalSave")}<span>↗</span></button></div></div>}
    <footer><span>{t("footer.instrument")}</span><span>SRR + (RAR → DAI) + DDR + CAR</span></footer>
  </main>;
}
