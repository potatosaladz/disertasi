"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import RunGraph, { type RunGraphPayload } from "./RunGraph";

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
  conflict_categories: { component: string; category: string; narrative: string; agent_i_artifacts?: unknown; agent_j_artifacts?: unknown }[];
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
  disagreements: Disagreement[];
  simulation_artifacts: SimulationArtifact[];
  influence_observations: Influence[];
};
type RunLog = { stage: string; level: string; message: string };
type RunState = { task_id: string | null; session_id: string; scenario_id: number; status: "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED"; logs: RunLog[]; simulation_artifacts?: SimulationArtifact[]; agent_breakdown?: AgentBreakdown[]; disagreements?: Disagreement[]; progress_stage?: string | null; result?: Record<string, unknown> | null; error?: string | null; created_at?: string | null; started_at?: string | null; completed_at?: string | null };
type AgentDomainRules = { agent_id: number; name: string; role: string; template_key: string | null; mandate: string | null; primary_sources: string[]; constraints: string[]; owned_checks: string[]; synthesis_status: "generated" | "fallback"; scenario_mandate: string | null; scenario_focus: string[]; priority_questions: string[]; required_evidence: string[]; epistemic_logic_traceability: string[]; structured_consensus_protocol: string[]; regulatory_compliance_alignment: string[]; llm_model: string | null; token_usage: number | null; error: { code: string; message: string } | null };
type DomainRules = { scenario_id: number; revision: string; generated: boolean; stale: boolean; agent_count: number; rules: { hard_constraints?: string[]; owned_checks?: string[]; principles?: string[]; primary_sources?: string[]; automatic_deficit_ceiling?: number }; agent_rules: AgentDomainRules[]; status: "success" | "partial" | "failed" | "stale" | "missing"; generated_count: number; failure_count: number; detail: string | null };
type AgentForm = Omit<Agent, "id" | "has_llm_api_key" | "template_key" | "system_prompt"> & { llm_api_key: string };
type ScenarioForm = Omit<Scenario, "id" | "max_deficit_constraint">;

const apiUrl = "";
const components = ["dE", "dA", "dP", "dR", "dU", "dO", "dC", "dREC"] as const;
const componentLabels: Record<(typeof components)[number], string> = { dE: "Evidence", dA: "Assumptions", dP: "Projection", dR: "Risk", dU: "Uncertainty", dO: "Objective", dC: "Constraint", dREC: "Recommendation" };
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
  return <div className="template-contract"><div><span>MANDATE</span><strong>{template.description}</strong></div><div><span>PRIMARY SOURCES</span><strong>{template.primary_sources.join(" · ")}</strong></div><div><span>CONSTRAINTS</span><strong>{template.constraints.join(" · ")}</strong></div><div><span>OWNED CHECKS</span><strong>{template.owned_checks.join(" · ")}</strong></div></div>;
}

function RuleList({ title, items, empty }: { title: string; items: string[]; empty: string }) {
  return <section className="rule-list"><span>{title}</span>{items.length ? <ul>{items.map((item) => <li key={item}>{item}</li>)}</ul> : <p>{empty}</p>}</section>;
}

function AgentMandateCard({ rules, index, onGenerate, busy }: { rules: AgentDomainRules; index: number; onGenerate: (agentId: number) => void; busy: boolean }) {
  return <details className="agent-mandate-card" open={index === 0}><summary><div><span>{String(index + 1).padStart(2, "0")} / {rules.template_key?.toUpperCase() ?? "CUSTOM"} / {rules.synthesis_status.toUpperCase()}</span><strong>{rules.name}</strong><small>{rules.role}</small></div><b>{rules.primary_sources.length} sources · {rules.constraints.length} constraints</b></summary><div className="agent-mandate-body"><div className="mandate-card-actions"><button type="button" className="template-load-button" onClick={() => onGenerate(rules.agent_id)} disabled={busy}>{busy ? "Generating…" : "Generate Mandat"}</button></div>{rules.scenario_mandate && <section className="mandate-copy"><span>LLM SCENARIO MANDATE</span><p>{rules.scenario_mandate}</p></section>}<section className="mandate-copy"><span>AUTHORITATIVE LOCAL SEED</span><p>{rules.mandate ?? "Mandat terstruktur belum tersedia untuk agen ini."}</p></section>{rules.error && <section className="mandate-copy"><span>{rules.error.code}</span><p>{rules.error.message}</p></section>}<div className="agent-rule-columns"><RuleList title="SCENARIO FOCUS" items={rules.scenario_focus} empty="LLM synthesis fallback aktif." /><RuleList title="PRIORITY QUESTIONS" items={rules.priority_questions} empty="Belum ada pertanyaan hasil sintesis." /><RuleList title="REQUIRED EVIDENCE" items={rules.required_evidence} empty="Belum ada bukti tambahan hasil sintesis." /><RuleList title="EPISTEMIC TRACEABILITY" items={rules.epistemic_logic_traceability} empty="Belum ada protokol jejak epistemik." /><RuleList title="STRUCTURED CONSENSUS" items={rules.structured_consensus_protocol} empty="Belum ada protokol konsensus terstruktur." /><RuleList title="REGULATORY ALIGNMENT" items={rules.regulatory_compliance_alignment} empty="Belum ada alignment kepatuhan fiskal." /><RuleList title="PRIMARY SOURCES" items={rules.primary_sources} empty="Tidak ada sumber primer terstruktur." /><RuleList title="CONSTRAINTS" items={rules.constraints} empty="Tidak ada constraint terstruktur." /><RuleList title="OWNED CHECKS" items={rules.owned_checks} empty="Tidak ada owned checks terstruktur." /></div></div></details>;
}

function decisionText(value: unknown, fallback = "Belum tersedia") {
  if (typeof value === "string" && value.trim()) return value;
  if (value && typeof value === "object") {
    const item = value as DecisionItem;
    return item.content ?? item.name ?? fallback;
  }
  return fallback;
}

function DecisionArtifactList({ items, empty }: { items: DecisionItem[]; empty: string }) {
  return items.length ? <ul>{items.map((item, index) => <li key={`${decisionText(item)}-${index}`}><span>{decisionText(item)}</span>{item.source_tag && <small>{item.source_tag}</small>}{item.deficit !== undefined && <small>defisit {item.deficit}% · utilitas {item.utility ?? "—"}</small>}</li>)}</ul> : <p>{empty}</p>;
}

function AgentBreakdownCard({ agent, index }: { agent: AgentBreakdown; index: number }) {
  const initialStage = agent.pre_arbitration?.stage ?? agent.deliberation_stages[0]?.stage ?? "FINAL";
  const initialRound = agent.pre_arbitration?.round_number ?? agent.deliberation_stages[0]?.round_number ?? 0;
  const [selectedKey, setSelectedKey] = useState(`${initialStage}:${initialRound}`);
  const selected = agent.deliberation_stages.find((stage) => `${stage.stage}:${stage.round_number}` === selectedKey) ?? agent.pre_arbitration ?? agent.final_position;
  const position: AgentPosition = selected ?? agent.pre_arbitration ?? agent.final_position ?? { stage: agent.position_stage, round_number: 0, agent_opinion: agent.agent_opinion, reasoning_summary: agent.reasoning_summary, constraints_considered: agent.constraints_considered, statutory_gates: agent.statutory_gates, recommendation: agent.recommendation, confidence: agent.confidence, evidence: agent.evidence, assumptions: agent.assumptions, predictions: agent.predictions, risks: agent.risks, uncertainties: agent.uncertainties, objectives: agent.objectives, alternatives: agent.alternatives };
  const confidence = typeof position.confidence === "number" ? `${Math.round(position.confidence * 100)}%` : "—";
  return <details className="agent-breakdown-card" open={index === 0}><summary><div><span>{String(index + 1).padStart(2, "0")} / {agent.template_key?.toUpperCase() ?? "SECTORAL"}</span><strong>{agent.agent_name}</strong><small>{agent.agent_role}</small></div><div className="agent-confidence"><b>{confidence}</b><small>confidence</small></div></summary><div className="agent-breakdown-body"><div className="position-tabs" role="tablist" aria-label={`Tahapan posisi ${agent.agent_name}`}>{agent.deliberation_stages.map((stage, stageIndex) => { const key = `${stage.stage}:${stage.round_number}`; const label = stage.stage === "INITIAL" ? "Argumen Awal" : stage.stage === "PRE_ARBITRATION" ? "Pra-Arbitrase" : stage.stage === "FINAL" ? "Posisi Final" : `Pasca-Simulasi ${stage.round_number}`; return <button type="button" role="tab" aria-selected={selectedKey === key} className={selectedKey === key ? "active" : ""} onClick={() => setSelectedKey(key)} key={`${key}-${stageIndex}`}>{label}</button>; })}</div><div className="agent-opinion"><span>PANDANGAN / ANALISIS AGEN</span><p>{position.reasoning_summary ?? position.agent_opinion ?? "Ringkasan opini belum tersedia pada tahap ini."}</p></div><div className="agent-decision-grid"><section className="statutory-gate"><span>CONSTRAINT &amp; STATUTORY GATE</span><DecisionArtifactList items={position.constraints_considered} empty="Tidak ada constraint terstruktur." /></section><section className="partial-recommendation"><span>REKOMENDASI KEBIJAKAN</span><strong>{decisionText(position.recommendation)}</strong><small>Tingkat keyakinan {confidence} · posisi {position.stage.replaceAll("_", " ").toLowerCase()}</small></section></div><div className="agent-artifact-grid"><section><span>ALTERNATIF PRA-ARBITRASE</span><DecisionArtifactList items={position.alternatives} empty="Tidak ada alternatif terstruktur." /></section><section><span>BUKTI &amp; PREDIKSI</span><DecisionArtifactList items={[...position.evidence, ...position.predictions]} empty="Tidak ada bukti atau prediksi terstruktur." /></section><section><span>RISIKO &amp; KETIDAKPASTIAN</span><DecisionArtifactList items={[...position.risks, ...position.uncertainties]} empty="Tidak ada risiko atau ketidakpastian terstruktur." /></section></div></div></details>;
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

function DisagreementInspector({ item, breakdown }: { item: Disagreement; breakdown: AgentBreakdown[] }) {
  const agentIPosition = breakdown.find((agent) => agent.agent_name === item.agent_i)?.pre_arbitration ?? null;
  const agentJPosition = breakdown.find((agent) => agent.agent_name === item.agent_j)?.pre_arbitration ?? null;
  const calculation = item.fiscal_calculation ?? {};
  const resolution = item.resolution_detail ?? {};
  const active = item.active_components ?? components.filter((component) => item[component]);
  return <div className="ddr-inspector"><div className="ddr-inspector-head"><span>INSPECTOR / #{item.id}</span><h3>{item.agent_i}<em> × </em>{item.agent_j}</h3><p>{resolution.conclusion ?? item.resolution_mechanism}</p><b className={`ddr-status ${active.length ? "conflict" : "clear"}`}>{active.length ? `KONFLIK AKTIF / ${active.length}` : "BERSIH / 0"}</b></div><div className="ddr-category-grid">{components.map((component) => <div className={item[component] ? "active" : "clear"} key={component}><span>{component}</span><strong>{componentLabels[component]}</strong><b>{item[component] ? "1" : "0"}</b></div>)}</div>{item.conflict_categories?.length > 0 && <div className="ddr-category-narrative">{item.conflict_categories.map((category) => <section key={category.component}><span>{category.component} / {category.category}</span><p>{category.narrative}</p><div className="ddr-category-compare"><div><b>{item.agent_i}</b>{Array.isArray(category.agent_i_artifacts) && category.agent_i_artifacts.length ? category.agent_i_artifacts.slice(0, 3).map((artifact, index) => <small key={index}>{artifactText(artifact)}</small>) : <small>—</small>}</div><div><b>{item.agent_j}</b>{Array.isArray(category.agent_j_artifacts) && category.agent_j_artifacts.length ? category.agent_j_artifacts.slice(0, 3).map((artifact, index) => <small key={index}>{artifactText(artifact)}</small>) : <small>—</small>}</div></div></section>)}</div>}<div className="ddr-analysis-grid"><section><span>KALKULASI DAMPAK DEFISIT</span><p>{calculation.calculation_note ?? "Perhitungan defisit menunggu data alternatif terstruktur."}</p><div className="ddr-formula"><b>Statutory ceiling</b><small>{calculation.statutory_deficit_ceiling_percent ?? "—"}% PDB</small><b>Formula headroom</b><small>{calculation.formula ?? "ceiling − projected deficit"}</small><b>Formula kompromi</b><small>{calculation.compromise_formula ?? "argmax utility dengan deficit ≤ ceiling"}</small></div>{calculation.selected_compromise && <div className="ddr-selected-compromise"><span>KOMPROMI TERPILIH · ROUND {resolution.simulation_round ?? "—"}</span><strong>{calculation.selected_compromise.name ?? "Modelled compromise"}</strong><small>proyeksi defisit {calculation.selected_compromise.deficit ?? "—"}% · utilitas {calculation.selected_compromise.utility ?? "—"} · headroom {calculation.selected_compromise.headroom_percent ?? "—"}%</small></div>}</section><section><span>LANDASAN HUKUM &amp; OTORITAS</span>{item.legal_basis?.length ? <ul className="ddr-legal-list">{item.legal_basis.map((basis) => <li key={basis.source_tag}><b>{basis.source_tag}</b><small>{basis.basis}</small></li>)}</ul> : <p>Tidak ada rujukan hukum terstruktur pada pasangan agen ini.</p>}</section></div><div className="ddr-agent-compare"><section><span>POSISI PRA-ARBITRASE / {item.agent_i}</span><p>{agentIPosition?.reasoning_summary ?? "Ringkasan posisi belum tersedia."}</p><b>Rekomendasi</b><small>{decisionText(agentIPosition?.recommendation ?? null)}</small><b>Alternatif</b><div className="ddr-alternative-list">{agentIPosition?.alternatives?.length ? agentIPosition.alternatives.slice(0, 3).map((alternative, index) => <small key={index}>{alternative.name ?? "Alternatif"} · defisit {alternative.deficit ?? "—"}% · utilitas {alternative.utility ?? "—"}</small>) : <small>—</small>}</div></section><section><span>POSISI PRA-ARBITRASE / {item.agent_j}</span><p>{agentJPosition?.reasoning_summary ?? "Ringkasan posisi belum tersedia."}</p><b>Rekomendasi</b><small>{decisionText(agentJPosition?.recommendation ?? null)}</small><b>Alternatif</b><div className="ddr-alternative-list">{agentJPosition?.alternatives?.length ? agentJPosition.alternatives.slice(0, 3).map((alternative, index) => <small key={index}>{alternative.name ?? "Alternatif"} · defisit {alternative.deficit ?? "—"}% · utilitas {alternative.utility ?? "—"}</small>) : <small>—</small>}</div></section></div><div className="ddr-resolution-block"><div><span>RESOLUSI ARBITER</span><strong>{resolution.arbiter_conclusion ?? resolution.route ?? item.resolution_mechanism}</strong><small>{resolution.simulation_summary ?? "Belum ada narasi simulasi untuk pasangan ini."}</small></div><div><span>STATUS &amp; SISA KONFLIK</span><strong>{resolution.status ?? "—"}</strong><small>{resolution.remaining_prediction_conflicts ?? "—"} prediksi tersisa · bobot influence gabungan {item.influence_context?.combined_weight ?? "—"}</small></div></div></div>;
}

function DdrNetworkPanel({ disagreements, breakdown }: { disagreements: Disagreement[]; breakdown: AgentBreakdown[] }) {
  const [selectedId, setSelectedId] = useState<number | null>(disagreements[0]?.id ?? null);
  const selected = disagreements.find((item) => item.id === selectedId) ?? disagreements[0] ?? null;
  if (!disagreements.length) return <div className="empty-state"><strong>No DDR vectors yet.</strong><span>Run a cycle with at least two schema-valid agents to populate the matrix.</span></div>;
  return <div className="ddr-layout"><div className="ddr-matrix-column"><div className="matrix-wrap"><table className="ddr-table"><caption>Matriks Divergensi DDR per pasangan agen</caption><thead><tr><th scope="col">AGENT PAIR</th>{components.map((component) => <th scope="col" key={component}>{component}</th>)}<th scope="col">RESOLUTION MECHANISM</th></tr></thead><tbody>{disagreements.map((item) => <tr className={selected?.id === item.id ? "selected" : ""} key={item.id}><th scope="row"><button type="button" className="ddr-pair-button" aria-expanded={selected?.id === item.id} aria-controls="ddr-inspector" onClick={() => setSelectedId(item.id)}><strong>{item.agent_i}</strong><small>× {item.agent_j}</small></button></th>{components.map((component) => <td key={component}><span className={`bool ${item[component] ? "conflict" : "clear"}`} aria-label={`${componentLabels[component]}: ${item[component] ? "aktif" : "bersih"}`}>{item[component] ? "1" : "0"}</span></td>)}<td className="resolution">{item.resolution_mechanism}</td></tr>)}</tbody></table></div><p className="ddr-matrix-hint">Pilih pasangan agen untuk membuka penjelasan kalkulasi defisit, formula kompromi, landasan hukum, dan kesimpulan arbiter.</p></div><aside id="ddr-inspector" className="ddr-inspector-column">{selected && <DisagreementInspector item={selected} breakdown={breakdown} />}</aside></div>;
}

function AgentBreakdownPanel({ agents, disagreements, simulations }: { agents: AgentBreakdown[]; disagreements: Disagreement[]; simulations: SimulationArtifact[] }) {
  const conflictCount = disagreements.filter((item) => components.some((component) => item[component])).length || simulations.reduce((total, artifact) => total + (artifact.input?.conflicts?.length ?? 0), 0);
  return <div id="agent-breakdown" className="agent-breakdown"><div className="deliberation-flow"><div><span>01 / INDIVIDUAL ARGUMENTS</span><strong>{agents.length} agen sektoral</strong><small>Opini, statutory gate, dan opsi kebijakan</small></div><i>→</i><div className={conflictCount ? "conflicted" : ""}><span>02 / DDR CONFLICT DETECTION</span><strong>{conflictCount || "—"} pasangan konflik</strong><small>Perbedaan E · A · P · R · U · O · C · REC</small></div><i>→</i><div className={simulations.length ? "resolved" : ""}><span>03 / SIMULATION ARBITRATION</span><strong>{simulations.length || "—"} ronde simulasi</strong><small>Resolusi dimodelkan dan dikembalikan ke agen</small></div></div><div className="agent-breakdown-heading"><div><span>TRANSPARANSI DELIBERASI</span><h3>Agent Breakdown View</h3></div><p>Pilih tahap pada setiap kartu untuk membandingkan argumen awal, rekomendasi pra-arbitrase, dan posisi setelah resolusi Simulation Agent.</p></div>{agents.length ? <div className="agent-breakdown-grid">{agents.map((agent, index) => <AgentBreakdownCard agent={agent} index={index} key={agent.agent_id} />)}</div> : <div className="empty-state"><strong>Argumen individual belum tersedia.</strong><span>Rincian agen akan muncul setelah keluaran SRR pertama tervalidasi.</span></div>}</div>;
}

function SimulationResolutionPanel({ artifact }: { artifact: SimulationArtifact }) {
  const conflicts = artifact.input?.conflicts ?? [];
  const output = artifact.output;
  return <section className="simulation-resolution"><div className="simulation-resolution-head"><div><span className="section-number">SIM / ROUND {artifact.round_number}</span><h3>Simulation Agent Resolution &amp; Impact Analysis</h3></div><span className={`simulation-status ${artifact.status.toLowerCase()}`}>{artifact.status}</span></div><div className="simulation-meta"><div><span>REQUEST</span><strong>{artifact.trigger}</strong><small>{conflicts.length} conflict pair{conflicts.length === 1 ? "" : "s"} · {conflicts.flatMap((item) => item.components ?? []).join(", ") || "awaiting conflict detail"}</small></div><div><span>ARBITER</span><strong>{output.agent_name ?? "Native Simulation Agent"}</strong><small>{artifact.simulation_version} · {output.evidence_status ?? "modelled"} evidence</small></div><div><span>FOLLOW-UP</span><strong>{output.follow_up_consensus_status ?? (artifact.status === "RUNNING" ? "in progress" : "not started")}</strong><small>{output.remaining_prediction_conflicts ?? "—"} prediction conflicts remaining</small></div></div><div className="simulation-resolution-body"><div className="simulation-resolution-copy"><span>WHAT WAS SIMULATED</span><p>{output.simulation_summary ?? (artifact.status === "RUNNING" ? output.message : "Structured sectoral positions were compared under the scenario deficit ceiling.")}</p><span>ARBITRATED RESOLUTION</span><p>{output.resolution ?? output.message ?? "Simulation request accepted; output is being assembled."}</p>{output.fallback_reason && <small className="simulation-fallback">Deterministic native sandbox used: {output.fallback_reason}</small>}</div>{output.alternatives?.length ? <div className="simulation-alternatives"><span>MODELLED ALTERNATIVES</span>{output.alternatives.slice(0, 4).map((alternative, index) => <div className="simulation-alternative" key={`${alternative.name}-${index}`}><strong>{alternative.name ?? `Alternative ${index + 1}`}</strong><small>deficit {alternative.deficit ?? "—"}% · utility {alternative.utility ?? "—"} · {alternative.evidence_status ?? "modelled"}</small></div>)}</div> : <div className="simulation-alternatives"><span>MODELLED ALTERNATIVES</span><p>{artifact.status === "RUNNING" ? "Awaiting structured alternatives…" : "No alternatives returned."}</p></div>}</div>{(output.risks?.length || output.uncertainties?.length || output.limitations?.length) ? <div className="simulation-impact-grid"><div><span>RISKS</span>{(output.risks ?? []).slice(0, 3).map((item, index) => <p key={`risk-${index}`}>{item.content}</p>)}</div><div><span>UNCERTAINTIES / LIMITATIONS</span>{[...(output.uncertainties ?? []).map((item) => item.content), ...(output.limitations ?? [])].slice(0, 4).map((item, index) => <p key={`uncertainty-${index}`}>{item}</p>)}</div></div> : null}</section>;
}

export default function Home() {
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
  const [notice, setNotice] = useState("Research console ready. Select a scenario to activate the tracker.");
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
    const response = await fetch("/api/agent-templates", { cache: "no-store" });
    if (!response.ok) throw new Error(`Template API returned ${response.status}`);
    const payload: AgentTemplate[] = await response.json();
    if (!Array.isArray(payload) || payload.length !== 5) throw new Error("Template API returned invalid data");
    setTemplates(payload);
  }

  async function loadSetup() {
    const [agentResponse, scenarioResponse] = await Promise.all([
      fetch(`${apiUrl}/api/agents`),
      fetch(`${apiUrl}/api/scenarios`),
    ]);
    if (!agentResponse.ok || !scenarioResponse.ok) throw new Error("API unavailable");
    const nextAgents: Agent[] = await agentResponse.json();
    const nextScenarios: Scenario[] = await scenarioResponse.json();
    setAgents(nextAgents);
    setScenarios(nextScenarios);
    if (selectedScenario === null && nextScenarios.length > 0) setSelectedScenario(nextScenarios[nextScenarios.length - 1].id);
  }

  async function loadDashboard(id: number, sessionId?: string) {
    const sessionQuery = sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : "";
    const response = await fetch(`${apiUrl}/api/scenarios/${id}/dashboard?refresh=${Date.now()}${sessionQuery}`, { cache: "no-store" });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Dashboard unavailable (${response.status})`);
    }
    const payload: Dashboard = await response.json();
    setDashboard(payload);
    setDomainRules(payload.domain_rules);
  }

  async function loadRunGraph(taskId: string | null, scenarioId: number, sessionId: string) {
    const endpoint = taskId
      ? `${apiUrl}/api/runs/${taskId}/graph`
      : `${apiUrl}/api/scenarios/${scenarioId}/runs/${sessionId}/graph`;
    const response = await fetch(`${endpoint}?refresh=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Graph API returned ${response.status}`);
    const payload: RunGraphPayload = await response.json();
    setGraph(payload);
  }

  async function loadLatestRun(id: number) {
    const response = await fetch(`${apiUrl}/api/scenarios/${id}/runs/latest?refresh=${Date.now()}`, { cache: "no-store" });
    if (response.status === 404) {
      setRun(null);
      setGraph(null);
      setTrackerHistory([]);
      return;
    }
    if (!response.ok) throw new Error(`Run history unavailable (${response.status})`);
    const payload: RunState = await response.json();
    setRun(payload);
    setTrackerHistory(payload.logs ?? []);
    if (payload.task_id) trackerLogCounts.current[payload.task_id] = payload.logs?.length ?? 0;
    await loadRunGraph(payload.task_id, payload.scenario_id, payload.session_id);
    if (payload.status === "SUCCEEDED" && payload.scenario_id === id) await loadDashboard(id, payload.session_id);
  }
  async function loadDomainRules(id: number) {
    const response = await fetch(`${apiUrl}/api/scenarios/${id}/domain-rules?refresh=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Mandate data unavailable (${response.status})`);
    }
    const payload: DomainRules = await response.json();
    setDomainRules(payload);
    return payload;
  }

  async function refreshMandateData(id: number) {
    const results = await Promise.allSettled([loadDashboard(id), loadDomainRules(id)]);
    if (results.every((result) => result.status === "rejected")) {
      throw new Error("Mandat tersimpan, tetapi dashboard belum dapat dimuat ulang.");
    }
  }
  useEffect(() => {
    loadTemplates().catch((error) => setNotice(`Template agen gagal dimuat: ${error.message}`));
    loadSetup().catch(() => setNotice("API connection pending. Check the backend URL."));
  }, []);
  useEffect(() => {
    if (selectedScenario !== null) {
      Promise.all([loadDashboard(selectedScenario), loadLatestRun(selectedScenario)]).catch(() => setNotice("No dashboard or run history data for this scenario yet."));
    }
  }, [selectedScenario]);
  useEffect(() => {
    const consoleElement = trackerConsoleRef.current;
    if (consoleElement) consoleElement.scrollTop = consoleElement.scrollHeight;
  }, [trackerHistory.length]);

  useEffect(() => {
    if (!run || ["SUCCEEDED", "FAILED"].includes(run.status)) return;
    const timer = window.setInterval(async () => {
      if (!run.task_id) {
        await loadLatestRun(run.scenario_id);
        return;
      }
      const response = await fetch(`${apiUrl}/api/runs/${run.task_id}`, { cache: "no-store" });
      if (!response.ok) {
        setNotice(`Tracker API error (${response.status}).`);
        return;
      }
      const nextRun: RunState = await response.json();
      setRun(nextRun);
      if (nextRun.task_id) appendTrackerLogs(nextRun.task_id, nextRun.logs);
      await loadRunGraph(nextRun.task_id, nextRun.scenario_id, nextRun.session_id);
      if (nextRun.status === "SUCCEEDED" && selectedScenario === nextRun.scenario_id) {
        await loadDashboard(nextRun.scenario_id, nextRun.session_id);
        setNotice("Cycle complete. Dashboard metrics refreshed from PostgreSQL.");
      } else if (nextRun.status === "FAILED") {
        setNotice(nextRun.error ?? "Cycle failed. Persisted logs remain available in Live Tracker.");
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [run, selectedScenario]);

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
      const response = await fetch(`${apiUrl}/api/agents/load-templates`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ configs }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Template agents could not be loaded");
      await Promise.all([loadSetup(), loadTemplates()]); setDomainRules(null); setTemplateModalOpen(false);
      setNotice(`${payload.created} template baru dimuat; ${payload.total} template APBN siap digunakan.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Template agents could not be loaded");
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
      const response = await fetch(endpoint, { method: editingAgentId ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(agentPayload) });
      const payload = await response.json(); if (!response.ok) throw new Error(payload.detail ?? "Agent could not be saved");
      setAgent(initialAgent); setSelectedTemplate(""); setEditingAgentId(null); setDomainRules(null); await loadSetup(); setNotice(`Agent ${payload.name} berhasil disimpan.`);
    } catch (error) { setNotice(error instanceof Error ? error.message : "Agent could not be saved"); } finally { setBusy(false); }
  }

  function editAgent(item: Agent) {
    setEditingAgentId(item.id);
    setSelectedTemplate(item.template_key ?? "");
    setAgent({ name: item.name, role: item.role, theta_x: item.theta_x, theta_q: item.theta_q, theta_h: item.theta_h, theta_s: item.theta_s, theta_u: item.theta_u, llm_base_url: item.llm_base_url ?? "", llm_api_key: "", llm_model: item.llm_model ?? "", temperature: item.temperature, max_tokens: item.max_tokens });
    document.getElementById("setup")?.scrollIntoView({ behavior: "smooth" });
  }

  async function deleteAgent(item: Agent) {
    if (!window.confirm(`Hapus agen ${item.name}?`)) return;
    const response = await fetch(`${apiUrl}/api/agents/${item.id}`, { method: "DELETE" });
    if (!response.ok) { const payload = await response.json(); setNotice(payload.detail ?? "Agent tidak dapat dihapus"); return; }
    setDomainRules(null); await loadSetup(); setNotice(`Agent ${item.name} dihapus.`);
  }

  async function testConnection(item: Agent) {
    setConnectionTests((current) => ({ ...current, [item.id]: { ok: false, message: "Testing connection…" } }));
    const response = await fetch(`${apiUrl}/api/agents/${item.id}/test-connection`, { method: "POST" });
    const payload = await response.json();
    setConnectionTests((current) => ({ ...current, [item.id]: { ok: payload.ok, message: payload.ok ? `OK · ${payload.latency_ms.toFixed(0)}ms · ${payload.model}` : payload.error } }));
  }

  async function generateMandate() {
    if (selectedScenario === null || agents.length === 0) return;
    const scenarioId = selectedScenario;
    setRulesBusy(true);
    try {
      const response = await fetch(`${apiUrl}/api/scenarios/${scenarioId}/domain-rules`, { method: "POST" });
      if (!response.ok) {
        const rawError = await response.text();
        let message = rawError || `Mandat gagal dibuat (${response.status})`;
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
        throw new Error(rawPayload || "Respons sukses generate mandat bukan JSON yang valid");
      }
      if (!payload.generated) throw new Error(payload.detail ?? "Backend tidak menandai mandat sebagai generated");
      setDomainRules({ ...payload, generated: true, stale: false });
      try {
        await refreshMandateData(scenarioId);
      } catch (refreshError) {
        setNotice(refreshError instanceof Error ? refreshError.message : "Mandat berhasil dibuat; refresh dashboard tertunda.");
      }
      setMandateLogs((current) => [...current, { stage: "MANDATE", level: payload.status === "partial" ? "WARNING" : "SUCCESS", message: payload.detail ?? `Generated ${payload.generated_count ?? 0} mandates.` }]);
      setNotice(payload.detail ?? "Mandat dinamis berhasil dibuat.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Mandat gagal dibuat";
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
      const response = await fetch(
        `${apiUrl}/api/scenarios/${scenarioId}/agents/${agentId}/domain-rules`,
        { method: "POST" },
      );
      const rawPayload = await response.text();
      let payload: AgentDomainRules | { detail?: string };
      try {
        payload = JSON.parse(rawPayload) as AgentDomainRules | { detail?: string };
      } catch {
        throw new Error(rawPayload || `Mandat agen gagal dibuat (${response.status})`);
      }
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "Mandat agen gagal dibuat");
      try {
        await refreshMandateData(scenarioId);
      } catch (refreshError) {
        setNotice(refreshError instanceof Error ? refreshError.message : "Mandat berhasil dibuat; refresh dashboard tertunda.");
      }
      const generated = payload as AgentDomainRules;
      setMandateLogs((current) => [...current, { stage: "MANDATE", level: generated.synthesis_status === "generated" ? "SUCCESS" : "WARNING", message: `Mandat ${generated.name} diperbarui.` }]);
      setNotice(`Mandat ${generated.name} diperbarui dan dashboard telah di-refresh.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Mandat agen gagal dibuat";
      setMandateLogs((current) => [...current, { stage: "MANDATE", level: "ERROR", message }]);
      setNotice(message);
    } finally {
      setBusyAgentId(null);
    }
  }

  async function submitScenario(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true);
    try {
      const response = await fetch(`${apiUrl}/api/scenarios`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scenario) });
      const payload = await response.json(); if (!response.ok) throw new Error(payload.detail ?? "Scenario could not be saved");
      setScenario(initialScenario); await loadSetup(); setDomainRules(null); setDashboard(null); setGraph(null); setRun(null); setTrackerHistory([]); setSelectedScenario(payload.id); setNotice(`Skenario ${payload.id} tersimpan; aturan hukum dan batas defisit diterapkan otomatis.`);
    } catch (error) { setNotice(error instanceof Error ? error.message : "Scenario could not be saved"); } finally { setBusy(false); }
  }

  async function startRun() {
    if (selectedScenario === null) { setNotice("Select a scenario before starting a cycle."); return; }
    if (rulesAreStale) { setNotice("Generate ulang mandat setelah perubahan agen sebelum memulai diskusi."); return; }
    const response = await fetch(`${apiUrl}/api/scenarios/${selectedScenario}/runs`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) { setNotice(payload.detail ?? "Could not queue cycle"); return; }
    setRun({ task_id: payload.task_id, session_id: payload.session_id, scenario_id: payload.scenario_id, status: payload.status, logs: payload.logs, simulation_artifacts: [] });
    appendQueueLog(payload.task_id, payload.logs);
    await loadRunGraph(payload.task_id, payload.scenario_id, payload.session_id);
    setNotice(`Task ${payload.task_id.slice(0, 8)} queued on Celery.`);
  }

  async function exportManifest() {
    if (selectedScenario === null) { setNotice("Select a scenario before exporting."); return; }
    const response = await fetch(`${apiUrl}/api/scenarios/${selectedScenario}/manifest`);
    if (!response.ok) { setNotice("Manifest could not be generated."); return; }
    const blob = await response.blob(); const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `shcr-scenario-${selectedScenario}-manifest.json`; anchor.click(); URL.revokeObjectURL(url); setNotice("Reproducibility manifest downloaded.");
  }

  const latest = dashboard?.latest_metric ?? null;
  const activeLogs = trackerHistory.length ? trackerHistory : [{ stage: "IDLE", level: "INFO", message: "Awaiting task dispatch." }];
  const selectedTemplateData = templates.find((item) => item.key === selectedTemplate);
  const canGenerateRules = selectedScenario !== null && agents.length > 0;
  const rulesAreStale = canGenerateRules && (!domainRules || domainRules.scenario_id !== selectedScenario || !domainRules.generated || domainRules.stale || domainRules.agent_count !== agents.length);
  const activeInfluence = useMemo(() => [...(dashboard?.influence_observations ?? [])].sort((a, b) => (b.normalized_weight ?? 0) - (a.normalized_weight ?? 0)), [dashboard]);
  const simulationArtifacts = run ? run.simulation_artifacts ?? [] : dashboard?.simulation_artifacts ?? [];
  const agentBreakdown = run?.agent_breakdown?.length ? run.agent_breakdown : dashboard?.agent_breakdown ?? [];
  const narrativeDisagreements = run?.disagreements?.length ? run.disagreements : dashboard?.disagreements ?? [];

  return <main className="shell dashboard-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">S</span><span>SHCR</span></div><div className="top-meta"><span className="live-dot" /> RESEARCH NODE / POSTGRES + CELERY <b>PHASE 05</b></div></header>
    <section className="dashboard-hero"><div><div className="eyebrow">PHASE 05 / OBSERVATION LAYER</div><h1>Make disagreement<br /><em>inspectable.</em></h1><p>A live evidence surface for tracking execution, divergence, hard constraints, and the convergence states produced by each fiscal experiment.</p></div><div className="hero-orbit"><span>DDR</span><b>→</b><span>CAR</span><b>→</b><span>SHCR</span></div></section>
    <div className="notice"><span className="notice-label">SYSTEM NOTE</span><span>{notice}</span></div>

    <section className="control-strip"><div className="scenario-overview"><span>ACTIVE SCENARIO / SCOPE</span><strong>{selectedScenario ? `Scenario #${selectedScenario}` : "Belum dipilih"}</strong><p className={scenarioExpanded ? "expanded" : "collapsed"}>{scenarios.find((item) => item.id === selectedScenario)?.description ?? "Pilih skenario untuk memulai."}</p><button type="button" onClick={() => setScenarioExpanded((value) => !value)}>{scenarioExpanded ? "Sembunyikan Detail" : "Tampilkan Detail / Expand"}</button><select aria-label="Select active scenario" value={selectedScenario ?? ""} onChange={(event) => { setSelectedScenario(Number(event.target.value)); setDomainRules(null); setDashboard(null); setGraph(null); setRun(null); setTrackerHistory([]); setMandateLogs([]); }}><option value="" disabled>Select scenario</option>{scenarios.map((item) => <option key={item.id} value={item.id}>#{item.id} — {item.description.slice(0, 72)}</option>)}</select></div><button className="primary-button run-button" onClick={startRun} disabled={run?.status === "RUNNING" || run?.status === "QUEUED" || rulesAreStale}>{run?.status === "RUNNING" ? "Diskusi berjalan…" : "Mulai Diskusi"}<span>↗</span></button></section>

    <nav className="menu-rail"><span className="menu-label">CONTROL MENUS</span><a href="#setup">01 / Setup</a><a href="#setup">02 / Scenario</a><a className="active" href="#tracker">03 / Live Tracker</a><a href="#agent-breakdown">04 / Agent Breakdown</a><a href="#network">05 / Graph Network</a><a href="#ddr">06 / DDR</a><a href="#analytics">07 / Analytics</a></nav>

    <section id="tracker" className="tracker-panel panel"><div className="section-header"><div><span className="section-number">03</span><h2>Live Tracker</h2></div><span className={`run-state ${run?.status?.toLowerCase() ?? "idle"}`}>{run?.status ?? "IDLE"}</span></div><div className="tracker-content"><div ref={trackerConsoleRef} className="progress-console max-h-[500px] overflow-y-auto">{activeLogs.map((log, index) => <div className={`console-line ${log.level.toLowerCase()}`} key={`${log.stage}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><i className={index === activeLogs.length - 1 && !["SUCCEEDED", "FAILED"].includes(run?.status ?? "") ? "pulse" : "done"} /><div><b>[{log.level}] {log.stage}</b><small>{log.message}</small></div></div>)}</div><div className="task-readout"><span>TASK IDENTIFIER</span><strong>{run?.task_id ?? "—"}</strong><small>{run?.error ?? (run?.status === "SUCCEEDED" ? "Result persisted" : "Polling every 1.2 seconds")}</small></div></div><AgentBreakdownPanel agents={agentBreakdown} disagreements={narrativeDisagreements} simulations={simulationArtifacts} />{simulationArtifacts.length > 0 && <div className="simulation-stack">{simulationArtifacts.map((artifact) => <SimulationResolutionPanel artifact={artifact} key={artifact.id} />)}</div>}</section>

    <section id="network" className="panel graph-panel"><div className="section-header"><div><span className="section-number">05</span><h2>Interactive Graph Network</h2></div><div className="graph-legend"><span><i className="pending" /> pending</span><span><i className="running" /> active</span><span><i className="succeeded" /> complete</span><span><i className="warning" /> dissent</span><span><i className="failed" /> failed</span></div></div><RunGraph graph={graph} /></section>

    <section id="ddr" className="panel"><div className="section-header"><div><span className="section-number">06</span><h2>DDR Network</h2></div><span className="panel-code">D<sub>ij</sub> / 8 COMPONENTS</span></div><DdrNetworkPanel disagreements={narrativeDisagreements} breakdown={agentBreakdown} /></section>


    <section id="analytics" className="analytics-section"><div className="section-header"><div><span className="section-number">07</span><h2>Decision Analytics</h2></div><button className="export-button" onClick={exportManifest}>↓ Export reproducibility manifest</button></div><div className="convergence-banner"><div><span>FINAL CONVERGENCE STATE</span><strong>{latest?.convergence_status ?? "AWAITING CYCLE"}</strong></div><div className="convergence-meta"><span>FEASIBLE ALTERNATIVES</span><b>{latest?.feasible_alternatives_count ?? "—"}</b></div></div><div className="metric-grid"><MetricCard label="Hard-constraint violation rate" value={latest ? latest.hard_constraint_violation_rate.toFixed(2) : "—"} unit="%" tone="rust" /><MetricCard label="Provenance completeness" value={latest ? latest.provenance_completeness_percent.toFixed(2) : "—"} unit="%" /><MetricCard label="Schema validity" value={dashboard ? dashboard.schema_validity_percent.toFixed(2) : "—"} unit="%" tone="mint" /><MetricCard label="Latency / token usage" value={latest ? latest.latency_ms.toFixed(0) : "—"} unit={latest ? `ms · ${latest.token_usage} tok` : ""} /></div><div className="analytics-lower"><div className="history-block"><div className="subhead"><span>METRIC SNAPSHOT HISTORY</span><b>{dashboard?.metric_history.length ?? 0} RUNS</b></div>{dashboard?.metric_history.length ? dashboard.metric_history.slice(0, 5).map((metric) => <div className="history-row" key={metric.id}><span>#{metric.id}</span><strong>{metric.convergence_status}</strong><i>{metric.provenance_completeness_percent.toFixed(1)}% provenance</i><b>{new Date(metric.created_at).toLocaleTimeString()}</b></div>) : <p className="empty">Metric history will appear after the first completed cycle.</p>}</div><div className="influence-block"><div className="subhead"><span>RAR-DAI INFLUENCE RANKING</span><b>{activeInfluence.length} OBSERVATIONS</b></div>{activeInfluence.length ? activeInfluence.map((item, index) => <div className="weight-row" key={`${item.agent}-${item.proposition}`}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{item.agent}</strong><small>{item.proposition}</small></div><b>{item.normalized_weight === null ? "—" : `${(item.normalized_weight * 100).toFixed(1)}%`}</b></div>) : <p className="empty">No influence observations recorded for this scenario.</p>}</div></div></section>

    <section id="setup" className="setup-section">
      <div className="section-header"><div><span className="section-number">01 / 02</span><h2>Heterogeneous Agent Studio</h2></div><button type="button" className="template-load-button" onClick={openTemplateModal} disabled={busy}>Muat 5 template APBN</button></div>
      <p className="setup-intro">Pilih template untuk mengisi mandat otomatis, lalu sesuaikan identitas, koneksi model, kebijakan generasi, dan bobot RAR-DAI sebelum menyimpan agen.</p>
      <form className="agent-config-form" onSubmit={submitAgent}>
        <fieldset className="form-card">
          <legend><span>01</span> Identitas &amp; Peran Agen</legend>
          <div className="form-grid two-column">
            <label className="form-field"><strong>Template Agen APBN</strong><select value={selectedTemplate} onChange={(event) => applyTemplate(event.target.value)}><option value="">Mulai dari konfigurasi kosong</option>{templates.map((template) => <option key={template.key} value={template.key}>{template.name}</option>)}</select><small>Pilih mandat standar APBN atau isi konfigurasi agen secara mandiri.</small></label>
            <label className="form-field"><strong>Nama Agen</strong><input value={agent.name} onChange={(event) => setAgent({ ...agent, name: event.target.value })} required /><small>Nama unik yang tampil pada deliberasi dan matriks DDR.</small></label>
            <label className="form-field full-width"><strong>Peran Fungsional</strong><input value={agent.role} onChange={(event) => setAgent({ ...agent, role: event.target.value })} required /><small>Contoh: Penerimaan Negara, Belanja Pemerintah, atau Stabilisasi Makro-Fiskal.</small></label>
          </div>
          {selectedTemplateData && <TemplateContractPreview template={selectedTemplateData} />}
        </fieldset>

        <fieldset className="form-card">
          <legend><span>02</span> Konfigurasi LLM Mandiri</legend>
          <div className="form-grid two-column">
            <label className="form-field full-width"><strong>Custom LLM Base URL</strong><input type="url" value={agent.llm_base_url ?? ""} onChange={(event) => setAgent({ ...agent, llm_base_url: event.target.value })} /><small>Masukkan base URL OpenAI-compatible; biarkan kosong untuk menggunakan default dari .env.</small></label>
            <label className="form-field"><strong>API Key Agen</strong><input type="password" autoComplete="new-password" value={agent.llm_api_key} onChange={(event) => setAgent({ ...agent, llm_api_key: event.target.value })} /><small>Kredensial khusus agen. Nilai tidak pernah dikembalikan oleh API.</small></label>
            <label className="form-field"><strong>Model Name</strong><input value={agent.llm_model ?? ""} onChange={(event) => setAgent({ ...agent, llm_model: event.target.value })} /><small>Nama model provider; kosong berarti memakai OPENAI_MODEL default.</small></label>
            <label className="form-field"><strong>Temperature</strong><input type="number" min="0" max="2" step="0.01" value={agent.temperature} onChange={(event) => setAgent({ ...agent, temperature: Number(event.target.value) })} required /><small>0 untuk stabilitas maksimum; rentang valid 0 sampai 2.</small></label>
            <label className="form-field"><strong>Max Tokens</strong><input type="number" min="1" step="1" value={agent.max_tokens} onChange={(event) => setAgent({ ...agent, max_tokens: Number(event.target.value) })} required /><small>Batas token keluaran untuk satu respons deliberasi agen.</small></label>
          </div>
        </fieldset>

        <fieldset className="form-card mandate-card">
          <legend><span>03</span> Mandat &amp; Aturan Domain Otomatis</legend>
          <div className="automatic-rules"><div className="automatic-rules-intro"><div><strong>{selectedTemplate ? "Kontrak domain template aktif" : "Mandat lintas agen siap dirakit"}</strong><p>Backend menyusun mandat, sumber primer, constraints, dan owned checks per agen tanpa mencampur kepemilikan aturan.</p></div><button type="button" className="template-load-button" onClick={generateMandate} disabled={!canGenerateRules || rulesBusy || busyAgentId !== null}>{rulesBusy ? "Generating…" : "Generate Otomatis Mandat"}</button></div>{rulesAreStale && <small className="stale-rules">Konfigurasi agen berubah; generate ulang wajib dilakukan sebelum diskusi.</small>}{domainRules && !domainRules.stale && <><div className="rules-revision"><span>REVISION {domainRules.revision}</span><b>{domainRules.agent_count} AGENTS</b><small>{domainRules.rules.owned_checks?.length ?? 0} hard checks · {domainRules.rules.hard_constraints?.length ?? 0} constraints global</small></div><div className="agent-mandate-grid">{domainRules.agent_rules.map((rules, index) => <AgentMandateCard rules={rules} index={index} onGenerate={generateAgentMandate} busy={rulesBusy || busyAgentId !== null} key={rules.agent_id} />)}</div></>}{mandateLogs.length > 0 && <div className="mandate-console">{mandateLogs.map((log, index) => <div className={log.level.toLowerCase()} key={`${log.stage}-${index}`}><b>[{log.level}] {log.stage}</b><span>{log.message}</span></div>)}</div>}</div>
        </fieldset>

        <fieldset className="form-card">
          <legend><span>04</span> Bobot RAR-DAI</legend>
          <div className="theta-form-grid">
            <NumericField label="Theta X — Expertise" value={agent.theta_x} hint="Bobot kompetensi domain." onChange={(value) => setAgent({ ...agent, theta_x: value })} />
            <NumericField label="Theta Q — Evidence" value={agent.theta_q} hint="Bobot kualitas bukti." onChange={(value) => setAgent({ ...agent, theta_q: value })} />
            <NumericField label="Theta H — History" value={agent.theta_h} hint="Bobot rekam historis." onChange={(value) => setAgent({ ...agent, theta_h: value })} />
            <NumericField label="Theta S — Relevance" value={agent.theta_s} hint="Bobot relevansi skenario." onChange={(value) => setAgent({ ...agent, theta_s: value })} />
            <NumericField label="Theta U — Uncertainty" value={agent.theta_u} hint="Penalti ketidakpastian; isi 0 untuk A6." onChange={(value) => setAgent({ ...agent, theta_u: value })} />
          </div>
        </fieldset>
        <button className="primary-button agent-submit" disabled={busy} type="submit">{editingAgentId ? "Simpan perubahan agen" : "Simpan konfigurasi agen"}<span>↗</span></button>
      </form>

      <div className="agent-register"><div className="subhead"><span>AGEN TERSIMPAN</span><b>{agents.length} AGENTS</b></div>{agents.length ? agents.map((item) => <div className="agent-register-row" key={item.id}><div><strong>{item.name}</strong><small>{item.role}</small></div><span>{item.template_key ? "TEMPLATE" : "CUSTOM"}</span><b>{item.llm_model ?? "ENV DEFAULT"}</b><div className="agent-actions"><button type="button" onClick={() => generateAgentMandate(item.id)} disabled={rulesBusy || busyAgentId !== null || busyAgentId === item.id}>{busyAgentId === item.id ? "Generating…" : "Generate Mandat"}</button><button type="button" onClick={() => testConnection(item)}>Test Connection</button><button type="button" onClick={() => editAgent(item)}>Edit</button><button type="button" className="danger" onClick={() => deleteAgent(item)}>Delete</button>{connectionTests[item.id] && <small className={connectionTests[item.id].ok ? "test-ok" : "test-error"}>{connectionTests[item.id].message}</small>}</div></div>) : <p className="empty">Belum ada agen. Muat template APBN atau buat agen baru.</p>}</div>

      <form className="scenario-config-form" onSubmit={submitScenario}><div><span className="section-number">SCENARIO</span><h3>Uji kebijakan fiskal</h3><small>Constraint hukum dan batas defisit diterapkan otomatis oleh agen.</small></div><label className="form-field"><strong>Goal / Deskripsi Kebijakan</strong><textarea value={scenario.description} onChange={(event) => setScenario({ ...scenario, description: event.target.value })} rows={4} required /><small>Jelaskan tujuan kebijakan, program yang diuji, horizon waktu, dan hasil yang diharapkan.</small></label><label className="form-field"><strong>Program Cost / Parameter Finansial</strong><input type="number" min="0" step="0.01" value={scenario.program_cost ?? ""} onChange={(event) => setScenario({ ...scenario, program_cost: event.target.value === "" ? null : Number(event.target.value) })} /><small>Opsional. Gunakan satuan fiskal yang konsisten dengan alternatif; kosongkan bila belum diketahui.</small></label><button className="secondary-button" disabled={busy}>Simpan skenario</button></form>
    </section>
    {templateModalOpen && <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Konfigurasi LLM template APBN"><div className="template-modal"><div className="section-header"><div><span className="section-number">LLM</span><h2>Konfigurasi 5 Template APBN</h2></div><button type="button" className="modal-close" onClick={() => setTemplateModalOpen(false)}>×</button></div><p>Atur koneksi tiap agen sebelum template disimpan. Kolom kosong memakai konfigurasi environment default.</p><div className="template-config-list">{templates.map((item) => { const config = templateConfigs[item.key]; if (!config) return null; return <fieldset className="template-config-card" key={item.key}><legend>{item.name}</legend><label><span>API Base URL</span><input value={config.llm_base_url} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_base_url: event.target.value } })} /></label><label><span>API Key</span><input type="password" value={config.llm_api_key} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_api_key: event.target.value } })} /></label><label><span>Model Name</span><input value={config.llm_model} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_model: event.target.value } })} /></label><label><span>Temperature</span><input type="number" min="0" max="2" step="0.01" value={config.temperature} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, temperature: Number(event.target.value) } })} /></label><label><span>Max Tokens</span><input type="number" min="1" value={config.max_tokens} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, max_tokens: Number(event.target.value) } })} /></label></fieldset>; })}</div><button type="button" className="primary-button" onClick={loadAllTemplates} disabled={busy}>{busy ? "Menyimpan…" : "Simpan dan Muat Semua Template"}<span>↗</span></button></div></div>}
    <footer><span>STRUCTURED CONSENSUS / RESEARCH INSTRUMENT</span><span>SRR + (RAR → DAI) + DDR + CAR</span></footer>
  </main>;
}
