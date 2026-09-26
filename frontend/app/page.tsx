"use client";

import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import RunGraph, { type RunGraphPayload } from "./RunGraph";
import { useI18n } from "./i18n";

type Agent = {
  id: number;
  agent_uuid: string;
  name: string;
  display_name?: string;
  role: string;
  template_key: string | null;
  scenario_id?: number | null;
  specialist_domain?: string | null;
  theta_x: number;
  theta_q: number;
  theta_h: number;
  theta_s: number;
  theta_u: number;
  rar_dai_weight_mode: "auto" | "manual";
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
type ScenarioInputs = {
  instrument: string | null;
  targeting: string | null;
  program_cost: number | null;
  duration_months: number | null;
  evaluation_trigger: string | null;
  program_cost_period: string | null;
  skip_llm_formulation: boolean | null;
  formulation_dry_run_only: boolean | null;
  single_year_deployment: boolean | null;
  proposed_reallocation: number | null;
  reallocation_from_education: boolean | null;
  proposed_additional_revenue: number | null;
  revenue_measure_type: string | null;
  proposed_debt_financing: number | null;
  debt_financing_mode: string | null;
  proposed_sal_use: number | null;
  sal_purpose: string | null;
  proposed_other_financing: number | null;
  appropriation_available: boolean | null;
  verified_reallocation_capacity: number | null;
  verified_revenue_offset_capacity: number | null;
  verified_debt_financing_headroom: number | null;
  verified_sal_available: number | null;
  verified_operational_cash_minimum: number | null;
  verified_projected_cash_after_policy: number | null;
  verified_cumulative_borrowing_pct_gdp: number | null;
  spending_reallocation_authorized: boolean | null;
  dpr_spending_adjustment_recommendation: boolean | null;
  finance_minister_sal_authorized: boolean | null;
  dpr_sal_approval_obtained: boolean | null;
  dpr_additional_sbn_approval_obtained: boolean | null;
  tax_measure_has_enacted_law: boolean | null;
  pnbp_measure_has_valid_tariff_instrument: boolean | null;
  output_outcome_documented: boolean | null;
  domestic_product_compliance_documented: boolean | null;
  growth_outlook: number | null;
  inflation_outlook: number | null;
  fx_outlook: number | null;
  sbn10y_yield_outlook: number | null;
  icp_outlook: number | null;
  oil_lifting_outlook: number | null;
  gas_lifting_outlook: number | null;
  tax_revenue_forecast: number | null;
};
type Scenario = ScenarioInputs & { id: number; description: string };
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
type VectorMetadata = { component: string; active: boolean; category?: string; meaning?: string; narrative?: string; narrative_i18n?: { id?: string; en?: string }; utility_metadata?: { task?: string; result?: string; why?: string; name_source?: string }; economic_impact?: string; formula?: string; agent_i?: { value?: unknown; normalized?: string[]; source_tags?: string[]; artifact_hash?: string }; agent_j?: { value?: unknown; normalized?: string[]; source_tags?: string[]; artifact_hash?: string }; calculation?: Record<string, unknown> & { status?: string; value?: number | null; not_calculated_reason?: string; confidence_gap?: number | null; deficit_range_gap_percent_gdp?: number | null; violation?: boolean }; resolution_path?: string; status?: string };
type Disagreement = {
  id: number;
  agent_i_id: number;
  agent_i: string;
  agent_i_display_name: string;
  agent_j_id: number;
  agent_j: string;
  agent_j_display_name: string;
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
type InfluenceDimension = { label?: string; value?: number; source?: string; status?: string };
type Influence = { agent: string; proposition: string; normalized_weight: number | null; raw_score: number | null; gate: number; interactions?: { peer_agent_name: string; active_components: string[]; agreement_ratio: number }[]; calculation?: { dimensions?: Record<string, InfluenceDimension>; interaction_count?: number; inputs?: Record<string, number> } };
type SimulationConflict = { agent_i?: string; agent_j?: string; components?: string[]; route?: string };
type SimulationAlternative = { name?: string; deficit?: number; utility?: number; source_tag?: string; evidence_status?: string };
type SimulationOutput = { agent_name?: string; resolution?: string; simulation_summary?: string; evidence_status?: string; follow_up_consensus_status?: string; remaining_prediction_conflicts?: number; fallback_reason?: string; limitations?: string[]; conflict_summary?: string[]; modelled_variables?: string[]; alternatives?: SimulationAlternative[]; risks?: { content?: string }[]; uncertainties?: { content?: string }[]; message?: string };
type SimulationArtifact = { id: number; session_id: string; scenario_id: number; trigger: string; round_number: number; status: string; simulation_version: string; input?: { conflicts?: SimulationConflict[]; sectoral_inputs?: { agent?: string; role?: string; predictions?: { content?: string }[]; alternatives?: SimulationAlternative[]; recommendation?: unknown }[]; scenario?: Partial<ScenarioInputs> & { description?: string; statutory_deficit_ceiling_percent?: number } }; output: SimulationOutput; latency_ms: number; token_usage: number; created_at: string };
type DecisionItem = { content?: string; source_tag?: string; name?: string; deficit?: number; utility?: number };
type FallbackMetadata = { kind?: string; label?: string; label_i18n?: { id?: string; en?: string }; reason?: string; reason_i18n?: { id?: string; en?: string }; decision_status?: "evidence_required" };
type AgentPosition = { stage: string; round_number: number; agent_opinion: string | null; reasoning_summary: string | null; constraints_considered: DecisionItem[]; statutory_gates: string[]; recommendation: DecisionItem | string | null; confidence: number | null; evidence: DecisionItem[]; assumptions: DecisionItem[]; predictions: DecisionItem[]; risks: DecisionItem[]; uncertainties: DecisionItem[]; objectives: DecisionItem[]; alternatives: DecisionItem[]; fallback_metadata?: FallbackMetadata | null };
type AgentBreakdown = { agent_id: number; agent_name: string; display_name?: string; agent_role: string; role: string; template_key: string | null; schema_valid: boolean | null; provenance_count: number; position_stage: string; agent_opinion: string | null; reasoning_summary: string | null; constraints_considered: DecisionItem[]; statutory_gates: string[]; recommendation: DecisionItem | string | null; confidence: number | null; evidence: DecisionItem[]; assumptions: DecisionItem[]; predictions: DecisionItem[]; risks: DecisionItem[]; uncertainties: DecisionItem[]; objectives: DecisionItem[]; alternatives: DecisionItem[]; fallback_metadata?: FallbackMetadata | null; utility_metadata?: { task?: string; result?: string; why?: string; name_source?: string }; pre_arbitration: AgentPosition | null; final_position: AgentPosition | null; deliberation_stages: AgentPosition[] };
type CollectiveClaim = { agent_id?: number; agent_name?: string; display_name?: string; agent_role?: string; schema_valid?: boolean; position_stage?: string; main_claim?: string | null; recommendation?: DecisionItem | string | null; confidence?: number | null; constraints?: DecisionItem[]; evidence?: DecisionItem[]; predictions?: DecisionItem[] };
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
type PollingState = "PENDING" | "PROCESSING" | "TEMPORARY_HYDRATION_DELAY" | "SUCCEEDED" | "FAILED";
type RunState = { task_id: string | null; session_id: string; scenario_id: number; status: "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED"; polling_state?: PollingState; should_poll?: boolean; terminal?: boolean; logs: RunLog[]; simulation_artifacts?: SimulationArtifact[]; agent_breakdown?: AgentBreakdown[]; collective_reasoning?: CollectiveReasoning | null; disagreements?: Disagreement[]; progress_stage?: string | null; result?: { task?: Record<string, unknown>; result?: Record<string, unknown>; why?: Record<string, unknown>; car?: { hard_stop?: { triggered?: boolean; reason?: string; simulation_bypassed?: boolean; llm_compromise_bypassed?: boolean }; rejected_alternatives?: RejectedAlternative[]; hard_constraints?: HardConstraint[] } } | null; error?: string | null; created_at?: string | null; started_at?: string | null; completed_at?: string | null };
type AgentDomainRules = { agent_id: number; name: string; display_name?: string; role: string; template_key: string | null; mandate: string | null; primary_sources: string[]; constraints: string[]; owned_checks: string[]; synthesis_status: "generated" | "fallback"; scenario_mandate: string | null; scenario_focus: string[]; priority_questions: string[]; required_evidence: string[]; epistemic_logic_traceability: string[]; structured_consensus_protocol: string[]; regulatory_compliance_alignment: string[]; utility_metadata?: { task?: string; result?: string; why?: string; name_source?: string }; llm_model: string | null; token_usage: number | null; error: { code: string; message: string } | null };
type DomainRules = { scenario_id: number; revision: string; generated: boolean; stale: boolean; agent_count: number; rules: { hard_constraints?: string[]; owned_checks?: string[]; principles?: string[]; primary_sources?: string[]; automatic_deficit_ceiling?: number }; agent_rules: AgentDomainRules[]; status: "success" | "partial" | "failed" | "stale" | "missing"; generated_count: number; failure_count: number; detail: string | null };
type AgentForm = Omit<Agent, "id" | "agent_uuid" | "display_name" | "has_llm_api_key" | "template_key" | "system_prompt"> & { llm_api_key: string };
type OrchestratorConfigForm = { api_base_url: string; api_key: string; model_name: string; temperature: number; max_tokens: number };
type LatestRunState = "idle" | "loading" | "empty" | "available" | "error";
type ScenarioForm = Omit<Scenario, "id">;
type UtilityMetadata = { task?: string; result?: string; why?: string; name_source?: string };
type HardConstraint = { code?: string; formula?: string; status?: string; reason?: string; ceiling_percent_gdp?: number; requested_ceiling_percent_gdp?: number; source_tags?: string[]; calculation_status?: string };
type RejectedAlternative = { name?: string; projected_deficit_percent_gdp?: number | null; ceiling_percent_gdp?: number; excess_percent_gdp?: number | null; violated_constraints?: string[]; reason?: string };
type GlobalConfig = { llm_base_url: string | null; llm_model: string | null; temperature: number; max_tokens: number; apply_to_all: boolean; revision: number; has_llm_api_key: boolean };
type GlobalConfigForm = { llm_base_url: string; llm_api_key: string; llm_model: string; temperature: number; max_tokens: number; apply_to_all: boolean };

const initialGlobalConfig: GlobalConfigForm = { llm_base_url: "", llm_api_key: "", llm_model: "", temperature: 0.2, max_tokens: 4000, apply_to_all: false };
const initialOrchestratorConfig: OrchestratorConfigForm = { api_base_url: "", api_key: "", model_name: "", temperature: 0.2, max_tokens: 4000 };

const apiUrl = "";
const scenarioLabels: Record<"id" | "en", { edit: string; save: string; deleted: string; updated: string; deleteFailed: string; confirmDelete: (id: number) => string }> = {
  id: {
    edit: "Ubah",
    save: "Perbarui Skenario",
    deleted: "Skenario {id} dihapus.",
    updated: "Skenario {id} diperbarui; aturan lama ditandai kedaluwarsa.",
    deleteFailed: "Skenario tidak dapat dihapus.",
    confirmDelete: (id: number) => `Hapus skenario ${id} beserta seluruh agen dan riwayat prosesnya?`,
  },
  en: {
    edit: "Edit",
    save: "Update Scenario",
    deleted: "Scenario {id} deleted.",
    updated: "Scenario {id} updated; previous mandates are now marked stale.",
    deleteFailed: "Scenario could not be deleted.",
    confirmDelete: (id: number) => `Delete scenario ${id} with all of its agents and runs?`,
  },
};
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
  rar_dai_weight_mode: "auto",
  llm_base_url: "",
  llm_api_key: "",
  llm_model: "",
  temperature: 0.2,
  max_tokens: 4000,
};
const initialScenario: ScenarioForm = {
  description: "",
  instrument: null,
  targeting: null,
  program_cost: null,
  duration_months: null,
  evaluation_trigger: null,
  program_cost_period: null,
  skip_llm_formulation: null,
  formulation_dry_run_only: null,
  single_year_deployment: null,
  proposed_reallocation: null,
  reallocation_from_education: null,
  proposed_additional_revenue: null,
  revenue_measure_type: null,
  proposed_debt_financing: null,
  debt_financing_mode: null,
  proposed_sal_use: null,
  sal_purpose: null,
  proposed_other_financing: null,
  appropriation_available: null,
  verified_reallocation_capacity: null,
  verified_revenue_offset_capacity: null,
  verified_debt_financing_headroom: null,
  verified_sal_available: null,
  verified_operational_cash_minimum: null,
  verified_projected_cash_after_policy: null,
  verified_cumulative_borrowing_pct_gdp: null,
  spending_reallocation_authorized: null,
  dpr_spending_adjustment_recommendation: null,
  finance_minister_sal_authorized: null,
  dpr_sal_approval_obtained: null,
  dpr_additional_sbn_approval_obtained: null,
  tax_measure_has_enacted_law: null,
  pnbp_measure_has_valid_tariff_instrument: null,
  output_outcome_documented: null,
  domestic_product_compliance_documented: null,
  growth_outlook: null,
  inflation_outlook: null,
  fx_outlook: null,
  sbn10y_yield_outlook: null,
  icp_outlook: null,
  oil_lifting_outlook: null,
  gas_lifting_outlook: null,
  tax_revenue_forecast: null,
};

function NumericField({ label, value, onChange, hint, disabled = false }: { label: string; value: number; onChange: (value: number) => void; hint: string; disabled?: boolean }) {
  return <label className="form-field range-field"><span className="range-label"><strong>{label}</strong><output>{value.toFixed(2)}</output></span><input type="range" min="0" max="2" step="0.01" value={value} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))} /><small>{hint}</small></label>;
}

function OptionalTextField({ label, value, onChange, multiline = false }: { label: string; value: string | null; onChange: (value: string | null) => void; multiline?: boolean }) {
  const control = multiline
    ? <textarea value={value ?? ""} onChange={(event) => onChange(event.target.value || null)} rows={2} />
    : <input value={value ?? ""} onChange={(event) => onChange(event.target.value || null)} />;
  return <label className="form-field"><strong>{label}</strong>{control}</label>;
}

function OptionalNumberField({ label, value, onChange, min, step = "any" }: { label: string; value: number | null; onChange: (value: number | null) => void; min?: number; step?: number | "any" }) {
  return <label className="form-field"><strong>{label}</strong><input type="number" min={min} step={step} value={value ?? ""} onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))} /></label>;
}

function ScenarioParameterSections({ value, onChange, lang }: { value: ScenarioForm; onChange: (value: ScenarioForm) => void; lang: "id" | "en" }) {
  const set = <K extends keyof ScenarioForm>(field: K, next: ScenarioForm[K]) => onChange({ ...value, [field]: next });
  const section = lang === "id"
    ? {
        policy: "Ruang Lingkup Kebijakan", proposals: "Usulan Pembiayaan", evidence: "Bukti Terverifikasi", macro: "Prognosis Makro", yes: "Ya", no: "Tidak", unknown: "Belum ditentukan",
        direct: "Eksekusi Langsung / Lewati Formulasi LLM", directHelp: "Lewati peer-review formulasi LLM; artefak SRR awal yang tervalidasi langsung masuk ke RAR-DAI, DDR, dan CAR.",
        dryRun: "Dry-Run Formulasi Saja", dryRunHelp: "Jalankan dan validasi formulasi SRR, lalu berhenti sebelum RAR-DAI, DDR, simulasi, dan CAR.",
        singleYear: "Deployment Penuh Satu Tahun", singleYearHelp: "Evaluasi kebijakan sebagai deployment satu tahun tanpa ronde simulasi bertahap; batas defisit CAR 3% tetap berlaku.",
      }
    : {
        policy: "Policy Scope", proposals: "Financing Proposals", evidence: "Verified Evidence", macro: "Macro Outlook", yes: "Yes", no: "No", unknown: "Not specified",
        direct: "Direct Execution / Skip LLM Formulation", directHelp: "Skip the LLM peer-review formulation stage; initial validated SRR artifacts proceed directly to RAR-DAI, DDR, and CAR.",
        dryRun: "Formulation Dry-Run Only", dryRunHelp: "Generate and validate SRR formulation, then stop before RAR-DAI, DDR, simulation, and CAR.",
        singleYear: "Single-Year Full Deployment", singleYearHelp: "Evaluate a one-year full deployment without iterative simulation rounds; the statutory 3% CAR ceiling still applies.",
      };
  const setBoolean = (field: keyof ScenarioInputs, next: boolean | null) => {
    const updated = { ...value, [field]: next };
    if (next === true && field === "skip_llm_formulation") updated.formulation_dry_run_only = false;
    if (next === true && field === "formulation_dry_run_only") updated.skip_llm_formulation = false;
    onChange(updated);
  };
  const booleanField = (field: keyof ScenarioInputs, label: string, help?: string) => <label className="form-field" key={field}><strong>{label}</strong><select value={value[field] === null ? "" : String(value[field])} onChange={(event) => setBoolean(field, event.target.value === "" ? null : event.target.value === "true")}><option value="">{section.unknown}</option><option value="true">{section.yes}</option><option value="false">{section.no}</option></select>{help && <small>{help}</small>}</label>;
  return <div className="scenario-parameter-sections">
    <details open><summary>{section.policy}</summary><div className="form-grid two-column"><OptionalTextField label="Instrument" value={value.instrument} onChange={(next) => set("instrument", next)} /><OptionalTextField label="Targeting" value={value.targeting} onChange={(next) => set("targeting", next)} multiline /><OptionalNumberField label="Program Cost" value={value.program_cost} onChange={(next) => set("program_cost", next)} min={0} /><OptionalNumberField label="Duration (months)" value={value.duration_months} onChange={(next) => set("duration_months", next)} min={1} step={1} /><OptionalTextField label="Evaluation Trigger" value={value.evaluation_trigger} onChange={(next) => set("evaluation_trigger", next)} multiline /><OptionalTextField label="Program Cost Period" value={value.program_cost_period} onChange={(next) => set("program_cost_period", next)} />{booleanField("skip_llm_formulation", section.direct, section.directHelp)}{booleanField("formulation_dry_run_only", section.dryRun, section.dryRunHelp)}{booleanField("single_year_deployment", section.singleYear, section.singleYearHelp)}</div></details>
    <details><summary>{section.proposals}</summary><div className="form-grid two-column"><OptionalNumberField label="Proposed Reallocation" value={value.proposed_reallocation} onChange={(next) => set("proposed_reallocation", next)} min={0} />{booleanField("reallocation_from_education", "Reallocation From Education") }<OptionalNumberField label="Proposed Additional Revenue" value={value.proposed_additional_revenue} onChange={(next) => set("proposed_additional_revenue", next)} min={0} /><OptionalTextField label="Revenue Measure Type" value={value.revenue_measure_type} onChange={(next) => set("revenue_measure_type", next)} /><OptionalNumberField label="Proposed Debt Financing" value={value.proposed_debt_financing} onChange={(next) => set("proposed_debt_financing", next)} min={0} /><OptionalTextField label="Debt Financing Mode" value={value.debt_financing_mode} onChange={(next) => set("debt_financing_mode", next)} /><OptionalNumberField label="Proposed SAL Use" value={value.proposed_sal_use} onChange={(next) => set("proposed_sal_use", next)} min={0} /><OptionalTextField label="SAL Purpose" value={value.sal_purpose} onChange={(next) => set("sal_purpose", next)} multiline /><OptionalNumberField label="Proposed Other Financing" value={value.proposed_other_financing} onChange={(next) => set("proposed_other_financing", next)} min={0} /></div></details>
    <details><summary>{section.evidence}</summary><div className="form-grid two-column">{booleanField("appropriation_available", "Appropriation Available")}<OptionalNumberField label="Verified Reallocation Capacity" value={value.verified_reallocation_capacity} onChange={(next) => set("verified_reallocation_capacity", next)} min={0} /><OptionalNumberField label="Verified Revenue Offset Capacity" value={value.verified_revenue_offset_capacity} onChange={(next) => set("verified_revenue_offset_capacity", next)} min={0} /><OptionalNumberField label="Verified Debt Financing Headroom" value={value.verified_debt_financing_headroom} onChange={(next) => set("verified_debt_financing_headroom", next)} min={0} /><OptionalNumberField label="Verified SAL Available" value={value.verified_sal_available} onChange={(next) => set("verified_sal_available", next)} min={0} /><OptionalNumberField label="Verified Operational Cash Minimum" value={value.verified_operational_cash_minimum} onChange={(next) => set("verified_operational_cash_minimum", next)} min={0} /><OptionalNumberField label="Verified Projected Cash After Policy" value={value.verified_projected_cash_after_policy} onChange={(next) => set("verified_projected_cash_after_policy", next)} min={0} /><OptionalNumberField label="Verified Cumulative Borrowing (% GDP)" value={value.verified_cumulative_borrowing_pct_gdp} onChange={(next) => set("verified_cumulative_borrowing_pct_gdp", next)} min={0} />{booleanField("spending_reallocation_authorized", "Spending Reallocation Authorized")}{booleanField("dpr_spending_adjustment_recommendation", "DPR Spending Adjustment Recommendation")}{booleanField("finance_minister_sal_authorized", "Finance Minister SAL Authorization")}{booleanField("dpr_sal_approval_obtained", "DPR SAL Approval")}{booleanField("dpr_additional_sbn_approval_obtained", "DPR Additional SBN Approval")}{booleanField("tax_measure_has_enacted_law", "Tax Measure Has Enacted Law")}{booleanField("pnbp_measure_has_valid_tariff_instrument", "PNBP Measure Has Valid Tariff Instrument")}{booleanField("output_outcome_documented", "Output / Outcome Documented")}{booleanField("domestic_product_compliance_documented", "Domestic Product Compliance Documented")}</div></details>
    <details><summary>{section.macro}</summary><div className="form-grid two-column"><OptionalNumberField label="Growth Outlook (%)" value={value.growth_outlook} onChange={(next) => set("growth_outlook", next)} /><OptionalNumberField label="Inflation Outlook (%)" value={value.inflation_outlook} onChange={(next) => set("inflation_outlook", next)} /><OptionalNumberField label="FX Outlook" value={value.fx_outlook} onChange={(next) => set("fx_outlook", next)} min={0.01} /><OptionalNumberField label="SBN 10Y Yield Outlook (%)" value={value.sbn10y_yield_outlook} onChange={(next) => set("sbn10y_yield_outlook", next)} min={0} /><OptionalNumberField label="ICP Outlook" value={value.icp_outlook} onChange={(next) => set("icp_outlook", next)} min={0} /><OptionalNumberField label="Oil Lifting Outlook" value={value.oil_lifting_outlook} onChange={(next) => set("oil_lifting_outlook", next)} min={0} /><OptionalNumberField label="Gas Lifting Outlook" value={value.gas_lifting_outlook} onChange={(next) => set("gas_lifting_outlook", next)} min={0} /><OptionalNumberField label="Tax Revenue Forecast" value={value.tax_revenue_forecast} onChange={(next) => set("tax_revenue_forecast", next)} min={0} /></div></details>
  </div>;
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

function UtilityMetadataPanel({ metadata, open = false }: { metadata?: UtilityMetadata; open?: boolean }) {
  const { t } = useI18n();
  if (!metadata || ![metadata.task, metadata.result, metadata.why, metadata.name_source].some(Boolean)) return null;
  return <details className="utility-metadata" open={open}><summary>{t("utility.title")}</summary><dl>{metadata.task && <div><dt>{t("utility.task")}</dt><dd>{metadata.task}</dd></div>}{metadata.result && <div><dt>{t("utility.result")}</dt><dd>{metadata.result}</dd></div>}{metadata.why && <div><dt>{t("utility.why")}</dt><dd>{metadata.why}</dd></div>}{metadata.name_source && <div><dt>{t("utility.nameSource")}</dt><dd>{metadata.name_source}</dd></div>}</dl></details>;
}


function metadataEntries(value: unknown): Array<[string, string]> {
  if (!value || typeof value !== "object") return [];
  return Object.entries(value as Record<string, unknown>).map(([key, item]) => [
    key.replaceAll("_", " "),
    item === null || item === undefined
      ? "—"
      : typeof item === "object"
        ? JSON.stringify(item)
        : String(item),
  ]);
}


function RunMetadataPanel({ run }: { run: RunState | null }) {
  const { t } = useI18n();
  const sections = [
    [t("utility.task"), metadataEntries(run?.result?.task)],
    [t("utility.result"), metadataEntries(run?.result?.result)],
    [t("utility.why"), metadataEntries(run?.result?.why)],
  ] as const;
  if (!sections.some(([, entries]) => entries.length > 0)) return null;
  return <div className="run-metadata" aria-label={t("utility.title")}>{sections.map(([title, entries]) => entries.length > 0 && <section key={title}><h3>{title}</h3><dl>{entries.map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl></section>)}</div>;
}

function AgentMandateCard({ rules, index, onGenerate, busy }: { rules: AgentDomainRules; index: number; onGenerate: (agentId: number) => void; busy: boolean }) {
  const { t, formatNumber } = useI18n();
  return <details className="agent-mandate-card" open={index === 0}><summary><div><span>{String(index + 1).padStart(2, "0")} / {rules.template_key?.toUpperCase() ?? "CUSTOM"} / {rules.synthesis_status.toUpperCase()}</span><strong>{rules.display_name ?? rules.name}</strong><small>{rules.role}</small></div><b>{formatNumber((rules.primary_sources ?? []).length)} {t("common.sources")} · {formatNumber((rules.constraints ?? []).length)} {t("common.constraints")}</b></summary><div className="agent-mandate-body"><div className="mandate-card-actions"><button type="button" className="template-load-button" onClick={() => onGenerate(rules.agent_id)} disabled={busy}>{busy ? t("common.generating") : t("common.generateMandate")}</button></div>{rules.scenario_mandate && <section className="mandate-copy"><span>{t("mandate.scenario")}</span><p>{rules.scenario_mandate}</p></section>}<section className="mandate-copy"><span>{t("mandate.seed")}</span><p>{rules.mandate ?? t("mandate.unavailable")}</p></section><UtilityMetadataPanel metadata={rules.utility_metadata} />{rules.error && <section className="mandate-copy"><span>{rules.error.code}</span><p>{rules.error.message}</p></section>}<div className="agent-rule-columns"><RuleList title={t("mandate.focus")} items={rules.scenario_focus} empty={t("mandate.empty.focus")} /><RuleList title={t("mandate.questions")} items={rules.priority_questions} empty={t("mandate.empty.questions")} /><RuleList title={t("mandate.evidence")} items={rules.required_evidence} empty={t("mandate.empty.evidence")} /><RuleList title={t("mandate.traceability")} items={rules.epistemic_logic_traceability} empty={t("mandate.empty.traceability")} /><RuleList title={t("mandate.consensus")} items={rules.structured_consensus_protocol} empty={t("mandate.empty.consensus")} /><RuleList title={t("mandate.regulatory")} items={rules.regulatory_compliance_alignment} empty={t("mandate.empty.regulatory")} /><RuleList title={t("template.sources")} items={rules.primary_sources} empty={t("mandate.empty.sources")} /><RuleList title={t("template.constraints")} items={rules.constraints} empty={t("mandate.empty.constraints")} /><RuleList title="OWNED CHECKS" items={rules.owned_checks} empty="Tidak ada owned checks terstruktur." /></div></div></details>;
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

function FallbackMetadataCard({ metadata }: { metadata?: FallbackMetadata | null }) {
  const { lang, t } = useI18n();
  if (!metadata) return null;
  const label = metadata.label_i18n?.[lang] ?? metadata.label ?? t("agent.evidenceRequired");
  const reason = metadata.reason_i18n?.[lang] ?? metadata.reason ?? t("agent.evidenceRequiredReason");
  return <div className="fallback-metadata" role="note"><strong>{label}</strong><p>{reason}</p><small>{t("agent.notDecisionEligible")}</small></div>;
}


function AgentBreakdownCard({ agent, index }: { agent: AgentBreakdown; index: number }) {
  const { t, formatNumber } = useI18n();
  const stages = agent.deliberation_stages ?? [];
  const initialStage = agent.pre_arbitration?.stage ?? stages[0]?.stage ?? "FINAL";
  const initialRound = agent.pre_arbitration?.round_number ?? stages[0]?.round_number ?? 0;
  const [selectedKey, setSelectedKey] = useState(`${initialStage}:${initialRound}`);
  const selected = stages.find((stage) => `${stage.stage}:${stage.round_number}` === selectedKey) ?? agent.pre_arbitration ?? agent.final_position;
  const position: AgentPosition = selected ?? agent.pre_arbitration ?? agent.final_position ?? { stage: agent.position_stage, round_number: 0, agent_opinion: agent.agent_opinion, reasoning_summary: agent.reasoning_summary, constraints_considered: agent.constraints_considered, statutory_gates: agent.statutory_gates, recommendation: agent.recommendation, confidence: agent.confidence, evidence: agent.evidence, assumptions: agent.assumptions, predictions: agent.predictions, risks: agent.risks, uncertainties: agent.uncertainties, objectives: agent.objectives, alternatives: agent.alternatives, fallback_metadata: agent.fallback_metadata };
  const confidence = typeof position.confidence === "number" ? `${formatNumber(position.confidence * 100, { maximumFractionDigits: 0 })}%` : "—";
  return <details className="agent-breakdown-card" open={index === 0}><summary><div><span>{String(index + 1).padStart(2, "0")} / {agent.template_key?.toUpperCase() ?? "SECTORAL"}</span><strong>{agent.display_name ?? agent.agent_name}</strong><small>{agent.agent_role}</small></div><div className="agent-confidence"><b>{confidence}</b><small>{t("common.confidence")}</small></div></summary><div className="agent-breakdown-body"><UtilityMetadataPanel metadata={agent.utility_metadata} open={index === 0} /><div className="position-tabs" role="group" aria-label={t("agent.positionStages", { name: agent.agent_name })}>{stages.map((stage, stageIndex) => { const key = `${stage.stage}:${stage.round_number}`; const label = stage.stage === "INITIAL" ? t("agent.initial") : stage.stage === "PRE_ARBITRATION" ? t("agent.pre") : stage.stage === "POST_SIMULATION" ? `${t("agent.post")} · ${t("common.round")} ${formatNumber(stage.round_number)}` : t("agent.final"); return <button type="button" aria-pressed={selectedKey === key} className={selectedKey === key ? "active" : ""} onClick={() => setSelectedKey(key)} key={`${key}-${stageIndex}`}>{label}</button>; })}</div><div className="agent-opinion"><span>{t("agent.opinion")}</span><p>{position.reasoning_summary ?? position.agent_opinion ?? t("agent.noSummary")}</p></div><div className="agent-decision-grid"><section className="statutory-gate"><span>{t("agent.constraints")}</span><DecisionArtifactList items={position.constraints_considered} empty={t("agent.noConstraints")} /></section><section className="partial-recommendation"><span>{t("agent.recommendation")}</span><strong>{decisionText(position.recommendation, t("common.unknown"))}</strong><small>{t("agent.confidenceLine", { confidence, position: position.stage.replaceAll("_", " ").toLowerCase() })}</small></section></div><div className="agent-artifact-grid"><section><span>{t("agent.preAlternatives")}</span><DecisionArtifactList items={position.alternatives} empty={position.fallback_metadata ? "" : t("agent.noAlternatives")} /><FallbackMetadataCard metadata={position.fallback_metadata} /></section><section><span>{t("agent.evidencePredictions")}</span><DecisionArtifactList items={[...(position.evidence ?? []), ...(position.predictions ?? [])]} empty={t("agent.noEvidencePredictions")} /></section><section><span>{t("agent.riskUncertainty")}</span><DecisionArtifactList items={[...(position.risks ?? []), ...(position.uncertainties ?? [])]} empty={t("agent.noRiskUncertainty")} /></section></div></div></details>;
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
  return <details className={`vector-detail ${metadata.active ? "active" : "clear"}`}><summary><span>{metadata.component}</span><strong>{metadata.category ?? componentLabels[metadata.component as keyof typeof componentLabels]?.[lang] ?? metadata.component}</strong><b>{metadata.active ? "1" : "0"}</b></summary><div><section><span>{t("ddr.formula")}</span><code>{metadata.formula ?? "—"}</code><p>{calculationValue !== undefined && calculationValue !== null ? formatNumber(calculationValue) : calculation.not_calculated_reason ?? t("ddr.notCalculated")}</p></section><section><span>{t("ddr.meaning")}</span><p>{metadata.narrative_i18n?.[lang] ?? metadata.narrative ?? metadata.meaning ?? "—"}</p></section><UtilityMetadataPanel metadata={metadata.utility_metadata} /><section><span>{t("ddr.impact")}</span><p>{metadata.economic_impact ?? "—"}</p></section><section><span>{t("ddr.routePath")}</span><p>{metadata.resolution_path ?? "—"}</p></section></div></details>;
}

function DisagreementInspector({ item, breakdown }: { item: Disagreement; breakdown: AgentBreakdown[] }) {
  const { lang, t, formatNumber } = useI18n();
  const agentILabel = item.agent_i_display_name ?? item.agent_i;
  const agentJLabel = item.agent_j_display_name ?? item.agent_j;
  const agentIPosition = breakdown.find((agent) => agent.agent_id === item.agent_i_id || agent.agent_name === item.agent_i)?.pre_arbitration ?? null;
  const agentJPosition = breakdown.find((agent) => agent.agent_id === item.agent_j_id || agent.agent_name === item.agent_j)?.pre_arbitration ?? null;
  const calculation = item.fiscal_calculation ?? {};
  const resolution = item.resolution_detail ?? {};
  const active = item.active_components ?? components.filter((component) => item[component]);
  const conflictNarratives = active.map((component) => item.vector_metadata?.[component]?.narrative_i18n?.[lang] ?? item.vector_metadata?.[component]?.narrative ?? item.conflict_categories?.find((category) => category.component === component)?.narrative).filter((value): value is string => Boolean(value));
  return <div className="ddr-inspector" aria-live="polite"><div className="ddr-inspector-head"><span>{t("ddr.inspector")} / #{item.id}</span><h3>{agentILabel}<em> × </em>{agentJLabel}</h3><p>{resolution.conclusion ?? item.resolution_mechanism}</p><b className={`ddr-status ${active.length ? "conflict" : "clear"}`}>{active.length ? `${t("ddr.activeConflict")} / ${formatNumber(active.length)}` : `${t("ddr.clean")} / 0`}</b></div>{active.length > 0 && <section className="conflict-narrative" aria-label={t("ddr.conflictOccurred")}><span>{t("ddr.conflictOccurred")}</span>{conflictNarratives.length ? conflictNarratives.map((narrative, index) => <p key={`${narrative}-${index}`}>{narrative}</p>) : <p>{resolution.conclusion ?? item.resolution_mechanism}</p>}</section>}<div className="ddr-vector-details">{components.map((component) => <VectorDetail metadata={item.vector_metadata?.[component] ?? { component, active: item[component], status: item[component] ? "detected" : "no-divergence" }} key={component} />)}</div>{item.conflict_categories?.length > 0 && <div className="ddr-category-narrative">{item.conflict_categories.map((category) => <section key={category.component}><span>{category.component} / {category.category}</span><p>{category.narrative}</p>{category.impact && <p><b>{t("ddr.impact")}:</b> {category.impact}</p>}<div className="ddr-category-compare"><div><b>{agentILabel}</b>{Array.isArray(category.agent_i_artifacts) && category.agent_i_artifacts.length ? category.agent_i_artifacts.slice(0, 3).map((artifact, index) => <small key={index}>{artifactText(artifact)}</small>) : <small>—</small>}</div><div><b>{agentJLabel}</b>{Array.isArray(category.agent_j_artifacts) && category.agent_j_artifacts.length ? category.agent_j_artifacts.slice(0, 3).map((artifact, index) => <small key={index}>{artifactText(artifact)}</small>) : <small>—</small>}</div></div></section>)}</div>}<div className="ddr-analysis-grid"><section><span>{t("ddr.deficitCalculation")}</span><p>{calculation.calculation_note ?? t("ddr.notCalculated")}</p><div className="ddr-formula"><b>{t("ddr.ceiling")}</b><small>{calculation.statutory_deficit_ceiling_percent === undefined ? "—" : formatNumber(calculation.statutory_deficit_ceiling_percent)}% {t("common.gdp")}</small><b>{t("ddr.headroomFormula")}</b><small>{calculation.formula ?? "ceiling − projected deficit"}</small><b>{t("ddr.compromiseFormula")}</b><small>{calculation.compromise_formula ?? "argmax utility subject to deficit ≤ ceiling"}</small></div>{calculation.selected_compromise && <div className="ddr-selected-compromise"><span>{t("ddr.selectedCompromise")} · {t("common.round")} {resolution.simulation_round === undefined ? "—" : formatNumber(resolution.simulation_round)}</span><strong>{calculation.selected_compromise.name ?? t("analytics.unnamedAlternative")}</strong><small>{calculation.selected_compromise.deficit === undefined ? "—" : formatNumber(calculation.selected_compromise.deficit)}% {t("common.gdp")} · {t("common.utility")} {calculation.selected_compromise.utility === undefined ? "—" : formatNumber(calculation.selected_compromise.utility)} · {t("ddr.headroom")} {calculation.selected_compromise.headroom_percent === null || calculation.selected_compromise.headroom_percent === undefined ? "—" : formatNumber(calculation.selected_compromise.headroom_percent)} pp</small></div>}</section><section><span>{t("ddr.legal")}</span>{item.legal_basis?.length ? <ul className="ddr-legal-list">{item.legal_basis.map((basis) => <li key={basis.source_tag}><b>{basis.source_tag}</b><small>{basis.basis}</small></li>)}</ul> : <p>{t("ddr.noLegal")}</p>}</section></div><div className="ddr-agent-compare"><section><span>{t("ddr.prePosition")} / {item.agent_i}</span><p>{agentIPosition?.reasoning_summary ?? t("common.unknown")}</p><b>{t("agent.recommendation")}</b><small>{decisionText(agentIPosition?.recommendation ?? null, t("common.unknown"))}</small></section><section><span>{t("ddr.prePosition")} / {item.agent_j}</span><p>{agentJPosition?.reasoning_summary ?? t("common.unknown")}</p><b>{t("agent.recommendation")}</b><small>{decisionText(agentJPosition?.recommendation ?? null, t("common.unknown"))}</small></section></div><div className="ddr-resolution-block"><div><span>{t("ddr.arbiterResolution")}</span><strong>{resolution.arbiter_conclusion ?? resolution.route ?? item.resolution_mechanism}</strong><small>{resolution.simulation_summary ?? t("common.unknown")}</small></div><div><span>{t("ddr.residual")}</span><strong>{resolution.status ?? "—"}</strong><small>{resolution.remaining_prediction_conflicts === undefined ? "—" : formatNumber(resolution.remaining_prediction_conflicts)} · RAR-DAI {item.influence_context?.combined_weight === undefined ? "—" : formatNumber(item.influence_context.combined_weight)}</small></div></div></div>;
}

function DdrNetworkPanel({ disagreements, breakdown }: { disagreements: Disagreement[]; breakdown: AgentBreakdown[] }) {
  const { lang, t } = useI18n();
  const [selectedId, setSelectedId] = useState<number | null>(disagreements[0]?.id ?? null);
  const selected = disagreements.find((item) => item.id === selectedId) ?? disagreements[0] ?? null;
  if (!disagreements.length) return <div className="empty-state"><strong>{t("ddr.none")}</strong><span>{t("ddr.noneDetail")}</span></div>;
  return <div className="ddr-layout"><div className="ddr-matrix-column"><div className="matrix-wrap"><table className="ddr-table"><caption>{t("ddr.caption")}</caption><thead><tr><th scope="col">{t("ddr.pair")}</th>{components.map((component) => <th scope="col" title={componentLabels[component][lang]} key={component}>{component}</th>)}<th scope="col">{t("ddr.route")}</th></tr></thead><tbody>{disagreements.map((item) => <tr className={selected?.id === item.id ? "selected" : ""} key={item.id}><th scope="row"><button type="button" className="ddr-pair-button" aria-pressed={selected?.id === item.id} onClick={() => setSelectedId(item.id)}><strong>{item.agent_i_display_name ?? item.agent_i}</strong><small>× {item.agent_j_display_name ?? item.agent_j}</small></button></th>{components.map((component) => <td key={component}><span className={`bool ${item[component] ? "conflict" : "clear"}`} title={item.vector_metadata?.[component]?.meaning} aria-label={`${componentLabels[component][lang]}: ${item[component] ? t("ddr.active") : t("ddr.clear")}`}>{item[component] ? "1" : "0"}</span></td>)}<td className="resolution">{item.resolution_mechanism}</td></tr>)}</tbody></table></div><p className="ddr-matrix-hint">{t("ddr.hint")}</p></div><aside id="ddr-inspector" className="ddr-inspector-column">{selected && <DisagreementInspector item={selected} breakdown={breakdown} />}</aside></div>;
}

function AgentBreakdownPanel({ agents, disagreements, simulations }: { agents: AgentBreakdown[]; disagreements: Disagreement[]; simulations: SimulationArtifact[] }) {
  const { t, formatNumber } = useI18n();
  const safeAgents = agents ?? [];
  const safeDisagreements = disagreements ?? [];
  const safeSimulations = simulations ?? [];
  const conflictCount = safeDisagreements.filter((item) => components.some((component) => item[component])).length || safeSimulations.reduce((total, artifact) => total + (artifact.input?.conflicts?.length ?? 0), 0);
  return <div id="agent-breakdown" className="agent-breakdown"><div className="deliberation-flow"><div><span>{t("agent.flow.arguments")}</span><strong>{t("agent.flow.argumentCount", { count: formatNumber(safeAgents.length) })}</strong><small>{t("agent.flow.argumentDetail")}</small></div><i>→</i><div className={conflictCount ? "conflicted" : ""}><span>{t("agent.flow.conflicts")}</span><strong>{t("agent.flow.conflictCount", { count: conflictCount ? formatNumber(conflictCount) : "—" })}</strong><small>{t("agent.flow.conflictDetail")}</small></div><i>→</i><div className={safeSimulations.length ? "resolved" : ""}><span>{t("agent.flow.simulation")}</span><strong>{t("agent.flow.simulationCount", { count: safeSimulations.length ? formatNumber(safeSimulations.length) : "—" })}</strong><small>{t("agent.flow.simulationDetail")}</small></div></div><div className="agent-breakdown-heading"><div><span>{t("agent.transparency")}</span><h3>{t("agent.breakdownTitle")}</h3></div><p>{t("agent.breakdownDescription")}</p></div>{safeAgents.length ? <div className="agent-breakdown-grid">{safeAgents.map((agent, index) => <AgentBreakdownCard agent={agent} index={index} key={agent.agent_id} />)}</div> : <div className="empty-state"><strong>{t("agent.empty")}</strong><span>{t("agent.emptyDetail")}</span></div>}</div>;
}

function InfluencePanel({ observations }: { observations: Influence[] }) {
  const { t, formatNumber } = useI18n();
  if (!observations.length) return <div className="collective-empty"><strong>{t("analytics.influence")}</strong><span>{t("analytics.noInfluence")}</span></div>;
  return <section className="influence-panel"><header><span>{t("analytics.influence")}</span><p>{t("analytics.influenceFormula")}</p></header><div>{observations.map((observation, index) => <article key={`${observation.agent}-${index}`}><div><strong>{observation.agent}</strong><b>{observation.normalized_weight === null ? "—" : `${formatNumber(observation.normalized_weight * 100, { maximumFractionDigits: 2 })}%`}</b></div><small>{observation.proposition}</small><dl>{Object.entries(observation.calculation?.dimensions ?? {}).map(([key, dimension]) => <div key={key}><dt>{key} · {dimension.label ?? key}</dt><dd>{typeof dimension.value === "number" ? formatNumber(dimension.value, { maximumFractionDigits: 4 }) : "—"}</dd></div>)}</dl></article>)}</div></section>;
}


function CollectiveReasoningPanel({ analysis }: { analysis: CollectiveReasoning | null }) {
  const { lang, t, formatNumber } = useI18n();
  if (!analysis) return <div className="collective-empty"><strong>{t("analytics.noNarrative")}</strong><span>{t("analytics.noNarrativeDetail")}</span></div>;
  const recommendation = analysis.recommendation_and_follow_up ?? { status: "—", round_number: 0, modelled_alternatives: [], limitations: [] };
  const why = analysis.why_and_how ?? { summary: "", divergence_points: [], convergence_signals: [] };
  const normative = analysis.normative_evaluation ?? { summary: "", principles: [] };
  return <section className="collective-reasoning"><div className="collective-method"><span>{t("analytics.dossier")}</span><strong>{analysis.methodology}</strong><small>{t("analytics.runStatus")}: {analysis.run_status}</small></div><section className="collective-section claims"><header><span>{t("analytics.claims")}</span><h3>{t("analytics.claimsTitle")}</h3><p>{t("analytics.claimsDescription")}</p></header><div className="collective-claim-grid">{(analysis.claims_and_positions ?? []).map((claim, index) => <article className={claim.schema_valid ? "valid" : "invalid"} key={`${claim.agent_id}-${index}`}><div><span>{String(index + 1).padStart(2, "0")} / {claim.position_stage?.replaceAll("_", " ") ?? t("analytics.noPosition")}</span><b>{typeof claim.confidence === "number" ? `${formatNumber(claim.confidence * 100, { maximumFractionDigits: 0 })}%` : "—"}</b></div><h4>{claim.display_name ?? claim.agent_name ?? t("analytics.unknownAgent")}</h4><small>{claim.agent_role}</small><p>{claim.main_claim ?? t("analytics.invalidClaim")}</p><section><span>{t("analytics.recommendationLabel")}</span><strong>{decisionText(claim.recommendation, t("common.unknown"))}</strong></section></article>)}</div></section><section className="collective-section rationale"><header><span>{t("analytics.why")}</span><h3>{t("analytics.whyTitle")}</h3><p>{why.summary_i18n?.[lang] ?? why.summary}</p></header><div className="rationale-layout"><div className="divergence-ledger">{(why.divergence_points ?? []).map((point) => <article key={point.component}><span>{point.component}</span><div><strong>{point.category ?? t("analytics.divergence")}</strong><p>{point.narrative}</p><small>{formatNumber(point.pair_count)} {t("analytics.pairs")} · {(point.agent_pairs ?? []).slice(0, 3).join(" · ")}</small></div></article>)}</div><div className="convergence-ledger"><span>{t("analytics.measuredAgreement")}</span>{(why.convergence_signals ?? []).map((signal) => <article key={signal.signal}><b>{signal.coverage}</b><div><strong>{signal.signal}</strong><p>{signal.narrative}</p></div></article>)}</div></div></section><section className="collective-section resolution"><header><span>{t("analytics.recommendation")}</span><h3>{t("analytics.recommendationTitle")}</h3><p>{recommendation.simulation_summary ?? t("analytics.noSimulation")}</p></header><div className="resolution-verdict"><div><span>{t("analytics.finalResolution")}</span><strong>{recommendation.final_resolution ?? t("analytics.noResolution")}</strong><small>{recommendation.arbiter ?? t("simulation.native")} · {t("common.round")} {formatNumber(recommendation.round_number)} · {recommendation.status}</small></div><div><span>{t("analytics.followUp")}</span><strong>{recommendation.follow_up_consensus_status ?? t("analytics.notRequired")}</strong><small>{recommendation.remaining_prediction_conflicts === null || recommendation.remaining_prediction_conflicts === undefined ? "—" : formatNumber(recommendation.remaining_prediction_conflicts)} {t("analytics.remainingConflicts")}</small></div></div><div className="modelled-option-grid">{(recommendation.modelled_alternatives ?? []).length ? (recommendation.modelled_alternatives ?? []).map((alternative, index) => <article key={`${alternative.name}-${index}`}><span>{t("analytics.modelledOption")} {String(index + 1).padStart(2, "0")}</span><strong>{alternative.name ?? t("analytics.unnamedAlternative")}</strong><div><b>{alternative.deficit === undefined ? "—" : formatNumber(alternative.deficit)}%</b><small>{t("analytics.projectedDeficit")}</small><b>{alternative.utility === undefined ? "—" : formatNumber(alternative.utility)}</b><small>{t("analytics.utility")}</small></div><small>{alternative.source_tag ?? "SIMULATION_MODELLED"}</small></article>) : <p className="empty">{t("analytics.noAlternatives")}</p>}</div>{(recommendation.limitations ?? []).length > 0 && <div className="analytics-limitations"><span>{t("analytics.limitations")}</span>{(recommendation.limitations ?? []).map((limitation) => <p key={limitation}>{limitation}</p>)}</div>}</section><section className="collective-section normative"><header><span>{t("analytics.normative")}</span><h3>{t("analytics.normativeTitle")}</h3><p>{normative.summary}</p></header><div className="normative-grid">{(normative.principles ?? []).map((principle) => <article key={principle.principle}><div><span>{principle.principle}</span><b className={principle.status.toLowerCase()}>{principle.status.replaceAll("_", " ")}</b></div><p>{principle.evaluation}</p><small>{(principle.legal_sources ?? []).length ? (principle.legal_sources ?? []).join(" · ") : t("analytics.defaultMethod")}</small></article>)}</div></section></section>;
}

function SimulationResolutionPanel({ artifact }: { artifact: SimulationArtifact }) {
  const { t, formatNumber } = useI18n();
  const conflicts = artifact.input?.conflicts ?? [];
  const output = artifact.output ?? {};
  const alternatives = output.alternatives ?? [];
  const limitations = output.limitations ?? [];
  return <section className="simulation-resolution"><div className="simulation-resolution-head"><div><span className="section-number">SIM / {t("simulation.round")} {formatNumber(artifact.round_number)}</span><h3>{t("simulation.title")}</h3></div><span className={`simulation-status ${artifact.status.toLowerCase()}`}>{artifact.status}</span></div><div className="simulation-meta"><div><span>{t("simulation.request")}</span><strong>{artifact.trigger}</strong><small>{formatNumber(conflicts.length)} {t("simulation.conflictPair")} · {conflicts.flatMap((item) => item.components ?? []).join(", ") || t("simulation.awaitingDetail")}</small></div><div><span>{t("simulation.arbiter")}</span><strong>{output.agent_name ?? t("simulation.native")}</strong><small>{artifact.simulation_version} · {output.evidence_status ?? t("simulation.modelledEvidence")}</small></div><div><span>{t("simulation.followUp")}</span><strong>{output.follow_up_consensus_status ?? (artifact.status === "RUNNING" ? t("simulation.inProgress") : t("simulation.notStarted"))}</strong><small>{output.remaining_prediction_conflicts === undefined ? "—" : formatNumber(output.remaining_prediction_conflicts)} {t("simulation.predictionRemaining")}</small></div></div><div className="simulation-resolution-body"><div className="simulation-resolution-copy"><span>{t("simulation.what")}</span><p>{output.simulation_summary ?? (artifact.status === "RUNNING" ? output.message : t("simulation.defaultSummary"))}</p><span>{t("simulation.resolution")}</span><p>{output.resolution ?? output.message ?? t("simulation.accepted")}</p>{output.fallback_reason && <small className="simulation-fallback">{t("simulation.fallback")}: {output.fallback_reason}</small>}</div>{alternatives.length ? <div className="simulation-alternatives"><span>{t("simulation.alternatives")}</span>{alternatives.slice(0, 4).map((alternative, index) => <div className="simulation-alternative" key={`${alternative.name}-${index}`}><strong>{alternative.name ?? t("simulation.alternative", { index: formatNumber(index + 1) })}</strong><small>{t("common.deficit")} {alternative.deficit === undefined ? "—" : formatNumber(alternative.deficit)}% · {t("common.utility")} {alternative.utility === undefined ? "—" : formatNumber(alternative.utility)}</small><em>{alternative.source_tag ?? "SIMULATION_MODELLED"}</em></div>)}</div> : null}</div><div className="simulation-impact-grid"><section><span>{t("simulation.riskUncertainty")}</span>{[...(output.risks ?? []), ...(output.uncertainties ?? [])].slice(0, 6).map((item, index) => <p key={index}>{item.content}</p>)}</section><section><span>{t("analytics.limitations")}</span>{limitations.length ? limitations.map((item, index) => <p key={index}>{item}</p>) : <p>{output.evidence_status ?? t("simulation.noLimitations")}</p>}</section></div></section>;
}

function ConstraintCard({ constraint }: { constraint: HardConstraint }) {
  const { t, formatNumber } = useI18n();
  const title = constraint.code === "DEFICIT_3PCT"
    ? t("constraint.deficit")
    : constraint.code === "SCENARIO_DEFICIT_CEILING"
      ? t("constraint.scenarioCeiling")
      : constraint.code === "EDUCATION_20PCT"
        ? t("constraint.education")
        : constraint.code === "DDR_DC_HARD_STOP"
          ? t("constraint.hardStop")
          : constraint.code?.replaceAll("_", " ") ?? t("constraint.unknown");
  const formula = constraint.formula?.replaceAll("<=", "≤").replaceAll(">=", "≥");
  return <article className={`constraint-card ${constraint.status ?? "unknown"}`}><header><strong>{title}</strong><span>{constraint.status ?? constraint.calculation_status ?? "—"}</span></header>{formula && <code>{formula}</code>}{typeof constraint.ceiling_percent_gdp === "number" && <p>{t("constraint.ceiling")}: <b>≤ {formatNumber(constraint.ceiling_percent_gdp)}% {t("common.gdp")}</b></p>}{constraint.source_tags?.length ? <small>{t("constraint.sources")}: {constraint.source_tags.join(" · ")}</small> : null}{constraint.reason && <div className="constraint-alert" role="note">{constraint.reason}</div>}</article>;
}


function RejectedAlternativeCard({ alternative }: { alternative: RejectedAlternative }) {
  const { t, formatNumber } = useI18n();
  return <article className="rejected-alternative-card"><strong>{alternative.name ?? t("common.unknown")}</strong>{typeof alternative.projected_deficit_percent_gdp === "number" && <p>{t("constraint.projectedDeficit")}: {formatNumber(alternative.projected_deficit_percent_gdp)}% {t("common.gdp")}</p>}{typeof alternative.ceiling_percent_gdp === "number" && <p>{t("constraint.ceiling")}: ≤ {formatNumber(alternative.ceiling_percent_gdp)}% {t("common.gdp")}</p>}{alternative.violated_constraints?.length ? <small>{alternative.violated_constraints.join(" · ")}</small> : null}{alternative.reason && <div className="constraint-alert" role="note">{alternative.reason}</div>}</article>;
}


function InfeasiblePanel({ metric, hardStop, message, rejectedAlternatives, hardConstraints }: { metric: Metric | null; hardStop?: { triggered?: boolean; reason?: string; simulation_bypassed?: boolean; llm_compromise_bypassed?: boolean }; message?: string; rejectedAlternatives?: RejectedAlternative[]; hardConstraints?: HardConstraint[] }) {
  const { t, formatNumber } = useI18n();
  const status = metric?.convergence_status?.toUpperCase() ?? "";
  if (!hardStop?.triggered && !status.includes("INFEASIBLE")) return null;
  const violationRate = metric?.hard_constraint_violation_rate;
  return <section className="infeasible-panel" aria-labelledby="infeasible-title"><div><span>{t("infeasible.eyebrow")}</span><h3 id="infeasible-title">INFEASIBLE</h3></div><div className="infeasible-copy"><strong>{t("infeasible.title")}</strong><p>{message?.trim() || hardStop?.reason || t("infeasible.reason")}</p><p>{violationRate === undefined ? t("infeasible.rateUnavailable") : t("infeasible.rate", { rate: formatNumber(violationRate, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) })}</p><b>{t("infeasible.conclusion")}</b></div>{Boolean(rejectedAlternatives?.length || hardConstraints?.length) && <div className="infeasible-evidence">{Boolean(hardConstraints?.length) && <section><span>{t("infeasible.constraints")}</span>{hardConstraints?.map((item, index) => <ConstraintCard constraint={item} key={`${item.code ?? "constraint"}-${index}`} />)}</section>}{Boolean(rejectedAlternatives?.length) && <section><span>{t("infeasible.rejected")}</span>{rejectedAlternatives?.map((item, index) => <RejectedAlternativeCard alternative={item} key={`${item.name ?? "alternative"}-${index}`} />)}</section>}</div>}</section>;
}

function SimulationEmptyState({ hardStop, message }: { hardStop?: { triggered?: boolean; reason?: string; simulation_bypassed?: boolean; llm_compromise_bypassed?: boolean }; message?: string }) {
  const { t } = useI18n();
  if (hardStop?.triggered) return <div className="simulation-empty hard-stop-state"><span>{t("simulation.hardStop")}</span><strong>{t("simulation.hardStopTitle")}</strong><p>{message?.trim() || hardStop.reason || t("simulation.hardStopReason")}</p><small>{hardStop.simulation_bypassed ? t("simulation.bypassed") : t("simulation.noArtifact")} · {hardStop.llm_compromise_bypassed ? t("simulation.compromiseBypassed") : t("simulation.noCompromise")}</small></div>;
  return <div className="simulation-empty"><span>{t("simulation.noSimulationLabel")}</span><strong>{t("simulation.noSimulationTitle")}</strong><p>{t("simulation.noSimulationDetail")}</p></div>;
}

function WizardControls({ current, onChange }: { current: number; onChange: (index: number) => void }) {
  const { t } = useI18n();
  return <nav className="wizard-controls" aria-label={t("wizard.controls")}><button type="button" onClick={() => onChange(current - 1)} disabled={current === 0}>← {t("wizard.back")}</button><span>{t("wizard.progress", { current: current + 1, total: 5 })}</span><button type="button" onClick={() => onChange(current + 1)} disabled={current === 4}>{t("wizard.next")} →</button></nav>;
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
  const [editingScenarioId, setEditingScenarioId] = useState<number | null>(null);
  const [scenarioBusy, setScenarioBusy] = useState(false);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [graph, setGraph] = useState<RunGraphPayload | null>(null);
  const [run, setRun] = useState<RunState | null>(null);
  const [notice, setNotice] = useState("");
  const [domainRules, setDomainRules] = useState<DomainRules | null>(null);
  const [mandateReadyScenarioId, setMandateReadyScenarioId] = useState<number | null>(null);
  const [rulesBusy, setRulesBusy] = useState(false);
  const [busyAgentId, setBusyAgentId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [globalConfig, setGlobalConfig] = useState<GlobalConfigForm>(initialGlobalConfig);
  const [globalConfigMeta, setGlobalConfigMeta] = useState<{ revision: number | string; has_llm_api_key: boolean }>({ revision: "—", has_llm_api_key: false });
  const [globalConfigBusy, setGlobalConfigBusy] = useState(false);
  const [globalConfigLoading, setGlobalConfigLoading] = useState(true);
  const [globalConfigLoadError, setGlobalConfigLoadError] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const [editingAgentId, setEditingAgentId] = useState<number | null>(null);
  const [scenarioExpanded, setScenarioExpanded] = useState(false);
  const [templateModalOpen, setTemplateModalOpen] = useState(false);
  const [orchestratorConfig, setOrchestratorConfig] = useState<OrchestratorConfigForm>(initialOrchestratorConfig);
  const [orchestratorHasSavedKey, setOrchestratorHasSavedKey] = useState(false);
  const [orchestratorConfigBusy, setOrchestratorConfigBusy] = useState(false);
  const [latestRunState, setLatestRunState] = useState<LatestRunState>("idle");
  const [templateConfigs, setTemplateConfigs] = useState<Record<string, { llm_base_url: string; llm_api_key: string; llm_model: string; temperature: number; max_tokens: number }>>({});
  const [connectionTests, setConnectionTests] = useState<Record<number, { ok: boolean; message: string; pending?: boolean }>>({});
  const [runHydrationError, setRunHydrationError] = useState<string | null>(null);
  const [runActionBusy, setRunActionBusy] = useState(false);
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
    if (!Array.isArray(payload) || payload.length !== 7) throw new Error(t("error.templateInvalid"));
    setTemplates(payload);
  }

  async function loadAgents(scenarioId?: number | null) {
    const endpoint = scenarioId === null || scenarioId === undefined
      ? `${apiUrl}/api/agents`
      : `${apiUrl}/api/agents?scenario_id=${scenarioId}`;
    const response = await apiFetch(endpoint, { cache: "no-store" });
    if (!response.ok) throw new Error(t("error.apiUnavailable"));
    const payload: Agent[] = await response.json();
    setAgents(payload);
    return payload;
  }

  async function loadSetup() {
    const [nextAgents, scenarioResponse] = await Promise.all([
      loadAgents(selectedScenario),
      apiFetch(`${apiUrl}/api/scenarios`),
    ]);
    if (!scenarioResponse.ok) throw new Error(t("error.apiUnavailable"));
    const nextScenarios: Scenario[] = await scenarioResponse.json();
    setAgents(nextAgents);
    setScenarios(nextScenarios);
    if (selectedScenario === null && nextScenarios.length > 0) setSelectedScenario(nextScenarios[nextScenarios.length - 1].id);
  }

  async function loadGlobalConfig() {
    setGlobalConfigLoading(true);
    setGlobalConfigLoadError(false);
    try {
      const response = await apiFetch(`${apiUrl}/api/global-config`, { cache: "no-store" });
      if (!response.ok) throw new Error(t("error.globalConfigLoad", { status: response.status }));
      const payload = await response.json() as Partial<GlobalConfig>;
      setGlobalConfig({
        llm_base_url: typeof payload.llm_base_url === "string" ? payload.llm_base_url : "",
        llm_api_key: "",
        llm_model: typeof payload.llm_model === "string" ? payload.llm_model : "",
        temperature: typeof payload.temperature === "number" ? payload.temperature : initialGlobalConfig.temperature,
        max_tokens: typeof payload.max_tokens === "number" ? payload.max_tokens : initialGlobalConfig.max_tokens,
        apply_to_all: payload.apply_to_all === true,
      });
      setGlobalConfigMeta({ revision: typeof payload.revision === "number" ? payload.revision : "—", has_llm_api_key: payload.has_llm_api_key === true });
    } catch (error) {
      setGlobalConfigLoadError(true);
      throw error;
    } finally {
      setGlobalConfigLoading(false);
    }
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
    setMandateReadyScenarioId(
      payload.domain_rules.generated && !payload.domain_rules.stale ? id : null
    );
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

  async function hydrateRunData(payload: RunState): Promise<boolean> {
    if (payload.status === "FAILED") return true;
    const requests: Promise<void>[] = [
      loadRunGraph(payload.task_id, payload.scenario_id, payload.session_id),
    ];
    if (payload.status === "SUCCEEDED") {
      requests.push(loadDashboard(payload.scenario_id, payload.session_id));
    }
    const results = await Promise.allSettled(requests);
    const failed = results.some((result) => result.status === "rejected");
    setRunHydrationError(failed ? t("notice.hydrationRefetch") : null);
    return !failed;
  }


  async function loadLatestRun(id: number) {
    setLatestRunState("loading");
    try {
      const response = await apiFetch(`${apiUrl}/api/scenarios/${id}/runs/latest`, { cache: "no-store" });
      if (response.status === 404) {
        setRun(null);
        setGraph(null);
        setTrackerHistory([]);
        setRunHydrationError(null);
        setLatestRunState("empty");
        await loadDashboard(id);
        return;
      }
      if (!response.ok) throw new Error(t("error.history", { status: response.status }));
      const payload: RunState = await response.json();
      setRun(payload);
      setLatestRunState("available");
      setTrackerHistory(payload.logs ?? []);
      if (payload.task_id) trackerLogCounts.current[payload.task_id] = payload.logs?.length ?? 0;
      if (payload.status === "FAILED") {
        setRunHydrationError(null);
        setNotice(payload.error ?? t("notice.failed"));
        return;
      }
      await hydrateRunData(payload);
    } catch (error) {
      setLatestRunState("error");
      throw error;
    }
  }

  async function loadOrchestratorConfig(id: number) {
    const response = await apiFetch(`${apiUrl}/api/scenarios/${id}/orchestrator-config`, { cache: "no-store" });
    if (!response.ok) throw new Error(t("error.orchestratorConfig"));
    const payload = await response.json() as { api_base_url: string | null; model_name: string | null; temperature: number; max_tokens: number; has_api_key: boolean };
    setOrchestratorConfig({
      api_base_url: payload.api_base_url ?? "",
      api_key: "",
      model_name: payload.model_name ?? "",
      temperature: payload.temperature,
      max_tokens: payload.max_tokens,
    });
    setOrchestratorHasSavedKey(payload.has_api_key);
  }

  async function loadDomainRules(id: number) {
    const response = await apiFetch(`${apiUrl}/api/scenarios/${id}/domain-rules`, { cache: "no-store" });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || t("error.mandateData", { status: response.status }));
    }
    const payload: DomainRules = await response.json();
    setDomainRules(payload);
    setMandateReadyScenarioId(payload.generated && !payload.stale ? id : null);
    return payload;
  }

  async function refreshMandateData(id: number) {
    const results = await Promise.allSettled([loadDashboard(id), loadDomainRules(id)]);
    if (results.every((result) => result.status === "rejected")) {
      throw new Error(t("error.mandateRefresh"));
    }
    return results;
  }
  useEffect(() => {
    loadTemplates().catch((error) => setNotice(t("notice.templatesFailed", { message: error.message })));
    loadSetup().catch(() => setNotice(t("notice.apiPending")));
    loadGlobalConfig().catch(() => setNotice(t("notice.globalConfigPending")));
  }, []);
  useEffect(() => {
    if (selectedScenario !== null) {
      Promise.all([
        loadAgents(selectedScenario),
        loadDashboard(selectedScenario),
        loadLatestRun(selectedScenario),
        loadOrchestratorConfig(selectedScenario),
      ]).catch(() => setNotice(t("notice.noDashboard")));
    }
  }, [selectedScenario, lang]);
  useEffect(() => {
    const consoleElement = trackerConsoleRef.current;
    if (consoleElement) consoleElement.scrollTop = consoleElement.scrollHeight;
  }, [trackerHistory.length]);

  useEffect(() => {
    if (
      !run
      || run.polling_state === "FAILED"
      || run.status === "FAILED"
      || run.should_poll !== true
      || !run.polling_state
      || !["PENDING", "PROCESSING", "TEMPORARY_HYDRATION_DELAY"].includes(
        run.polling_state
      )
    ) return;
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      let continuePolling = run.should_poll === true;
      try {
        const endpoint = run.task_id
          ? `${apiUrl}/api/runs/${run.task_id}`
          : `${apiUrl}/api/scenarios/${run.scenario_id}/runs/latest`;
        const response = await apiFetch(endpoint, { cache: "no-store" });
        if (!response.ok) {
          if (!cancelled) setNotice(t("notice.trackerError", { status: response.status }));
        } else {
          const nextRun: RunState = await response.json();
          if (cancelled) return;
          setRun(nextRun);
          if (nextRun.task_id) appendTrackerLogs(nextRun.task_id, nextRun.logs ?? []);
          if (nextRun.polling_state === "FAILED") {
            continuePolling = false;
            setRunHydrationError(null);
            setNotice(nextRun.error ?? t("notice.failed"));
          } else if (nextRun.polling_state === "SUCCEEDED") {
            continuePolling = false;
            const hydrated = await hydrateRunData(nextRun);
            if (!cancelled) setNotice(hydrated ? t("notice.complete") : t("notice.hydrationRefetch"));
          } else {
            continuePolling = nextRun.should_poll === true && [
              "PENDING",
              "PROCESSING",
              "TEMPORARY_HYDRATION_DELAY",
            ].includes(nextRun.polling_state ?? "");
            if (nextRun.polling_state === "TEMPORARY_HYDRATION_DELAY") {
              setNotice(t("notice.temporaryHydration"));
            }
          }
        }
      } catch {
        if (!cancelled) setNotice(t("notice.apiPending"));
      } finally {
        if (!cancelled && continuePolling) timer = window.setTimeout(poll, 1200);
      }
    };
    timer = window.setTimeout(poll, 1200);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [run?.task_id, run?.status, run?.polling_state, run?.should_poll, run?.scenario_id, selectedScenario, lang]);

  useEffect(() => {
    if (!run || run.status !== "SUCCEEDED" || runHydrationError) return;
    setNotice(t("notice.complete"));
  }, [run?.status, run?.session_id, runHydrationError, lang]);


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
      rar_dai_weight_mode: "auto",
    });
  }

  function openTemplateModal() {
    setTemplateConfigs(Object.fromEntries(templates.map((item) => [item.key, { llm_base_url: "", llm_api_key: "", llm_model: "", temperature: item.temperature, max_tokens: item.max_tokens }])));
    setTemplateModalOpen(true);
  }

  async function loadAllTemplates() {
    if (selectedScenario === null) {
      setNotice(t("notice.selectRun"));
      return;
    }
    if (!orchestratorConfig.api_base_url || !orchestratorConfig.model_name || (!orchestratorConfig.api_key && !orchestratorHasSavedKey)) {
      setNotice(t("error.orchestratorRequired"));
      return;
    }
    setBusy(true);
    setOrchestratorConfigBusy(true);
    try {
      const configs = Object.fromEntries(Object.entries(templateConfigs).map(([key, value]) => [key, { ...value, llm_base_url: value.llm_base_url || null, llm_api_key: value.llm_api_key || null, llm_model: value.llm_model || null }]));
      const requestBody = {
        api_base_url: orchestratorConfig.api_base_url,
        api_key: orchestratorConfig.api_key || null,
        model_name: orchestratorConfig.model_name,
        temperature: orchestratorConfig.temperature,
        max_tokens: orchestratorConfig.max_tokens,
        configs,
      };
      const response = await apiFetch(`${apiUrl}/api/scenarios/${selectedScenario}/orchestrate-agents`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestBody) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? t("error.templatesLoad"));
      const refreshedAgents = await loadAgents(selectedScenario);
      setAgents(refreshedAgents);
      setOrchestratorConfig((current) => ({ ...current, api_key: "" }));
      setOrchestratorHasSavedKey(payload.orchestrator_config?.has_api_key === true);
      setDomainRules(null);
      setMandateReadyScenarioId(null);
      closeTemplateModal();
      setNotice(t("notice.templatesLoaded", { created: formatNumber(payload.created), total: formatNumber(payload.total) }));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : t("error.templatesLoad"));
    } finally {
      setBusy(false);
      setOrchestratorConfigBusy(false);
    }
  }

  async function submitGlobalConfig(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (globalConfigLoading) return;
    setGlobalConfigBusy(true);
    try {
      const payload = {
        llm_base_url: globalConfig.llm_base_url || null,
        llm_model: globalConfig.llm_model || null,
        temperature: globalConfig.temperature,
        max_tokens: globalConfig.max_tokens,
        apply_to_all: globalConfig.apply_to_all,
        ...(globalConfig.llm_api_key ? { llm_api_key: globalConfig.llm_api_key } : {}),
      };
      const response = await apiFetch(`${apiUrl}/api/global-config`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const result = await response.json().catch(() => ({})) as Partial<GlobalConfig> & { detail?: string };
      if (!response.ok) throw new Error(result.detail ?? t("error.globalConfigSave"));
      setGlobalConfig((current) => ({
        ...current,
        llm_api_key: "",
        llm_base_url: typeof result.llm_base_url === "string" ? result.llm_base_url : current.llm_base_url,
        llm_model: typeof result.llm_model === "string" ? result.llm_model : current.llm_model,
        temperature: typeof result.temperature === "number" ? result.temperature : current.temperature,
        max_tokens: typeof result.max_tokens === "number" ? result.max_tokens : current.max_tokens,
        apply_to_all: typeof result.apply_to_all === "boolean" ? result.apply_to_all : current.apply_to_all,
      }));
      setGlobalConfigMeta((current) => ({ revision: typeof result.revision === "number" ? result.revision : current.revision, has_llm_api_key: typeof result.has_llm_api_key === "boolean" ? result.has_llm_api_key : current.has_llm_api_key || Boolean(globalConfig.llm_api_key) }));
      if (globalConfig.apply_to_all) await loadSetup();
      setDomainRules(null); setMandateReadyScenarioId(null);
      setNotice(t("notice.globalConfigSaved"));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : t("error.globalConfigSave"));
    } finally {
      setGlobalConfigBusy(false);
    }
  }

  async function submitAgent(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true);
    try {
      const agentPayload = {
        ...agent,
        ...(editingAgentId === null ? { scenario_id: selectedScenario } : {}),
        template_key: selectedTemplate || null,
        llm_base_url: agent.llm_base_url || null,
        llm_api_key: agent.llm_api_key || undefined,
        llm_model: agent.llm_model || null,
      };
      const endpoint = editingAgentId ? `${apiUrl}/api/agents/${editingAgentId}` : `${apiUrl}/api/agents`;
      const response = await apiFetch(endpoint, { method: editingAgentId ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(agentPayload) });
      const payload = await response.json(); if (!response.ok) throw new Error(payload.detail ?? t("error.agentSave"));
      setAgent(initialAgent); setSelectedTemplate(""); setEditingAgentId(null); setDomainRules(null); setMandateReadyScenarioId(null); await loadSetup(); setNotice(t("notice.agentSaved", { name: payload.name }));
    } catch (error) { setNotice(error instanceof Error ? error.message : t("error.agentSave")); } finally { setBusy(false); }
  }

  function editAgent(item: Agent) {
    setEditingAgentId(item.id);
    setSelectedTemplate(item.template_key ?? "");
    setAgent({ name: item.name, role: item.role, theta_x: item.theta_x, theta_q: item.theta_q, theta_h: item.theta_h, theta_s: item.theta_s, theta_u: item.theta_u, rar_dai_weight_mode: item.rar_dai_weight_mode, llm_base_url: item.llm_base_url ?? "", llm_api_key: "", llm_model: item.llm_model ?? "", temperature: item.temperature, max_tokens: item.max_tokens });
    selectTab(0);
    window.requestAnimationFrame(() => document.getElementById("wizard-panel-0")?.scrollIntoView({ behavior: "smooth" }));
  }

  async function deleteAgent(item: Agent) {
    if (!window.confirm(t("confirm.deleteAgent", { name: item.name }))) return;
    try {
      const response = await apiFetch(`${apiUrl}/api/agents/${item.id}`, { method: "DELETE" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({})) as { detail?: string };
        throw new Error(payload.detail ?? t("notice.agentDeleteFailed"));
      }
      setAgents((prev) => prev.filter((agentRow) => agentRow.id !== item.id));
      setDomainRules(null); setMandateReadyScenarioId(null);
      if (editingAgentId === item.id) { setEditingAgentId(null); setAgent(initialAgent); setSelectedTemplate(""); }
      setNotice(t("notice.agentDeleted", { name: item.name }));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : t("notice.agentDeleteFailed"));
    }
  }

  async function testConnection(item: Agent) {
    setConnectionTests((current) => ({ ...current, [item.id]: { ok: false, message: t("notice.testing"), pending: true } }));
    try {
      const response = await apiFetch(`${apiUrl}/api/agents/${item.id}/test-connection`, { method: "POST" });
      const payload = await response.json().catch(() => ({})) as { ok?: boolean; latency_ms?: number; model?: string; error?: string; detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? payload.error ?? t("error.connectionTest"));
      setConnectionTests((current) => ({ ...current, [item.id]: { ok: payload.ok === true, message: payload.ok ? `OK · ${formatNumber(payload.latency_ms ?? 0, { maximumFractionDigits: 0 })}ms · ${payload.model ?? "—"}` : payload.error ?? t("error.connectionTest") } }));
    } catch (error) {
      setConnectionTests((current) => ({ ...current, [item.id]: { ok: false, message: error instanceof Error ? error.message : t("error.connectionTest") } }));
    }
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
      const freshRules: DomainRules = { ...payload, scenario_id: scenarioId, generated: true, stale: false };
      setDomainRules(freshRules);
      setMandateReadyScenarioId(scenarioId);
      refreshMandateData(scenarioId).catch((refreshError) => {
        setNotice(refreshError instanceof Error ? refreshError.message : t("error.refreshDelayed"));
      });
      setMandateLogs((current) => [...current, { stage: "MANDATE", level: payload.status === "success" ? "SUCCESS" : "WARNING", message: payload.detail ?? t("notice.mandateGenerated", { count: formatNumber(payload.generated_count ?? 0) }) }]);
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
    event.preventDefault();
    setBusy(true);
    setScenarioBusy(true);
    try {
      const scenarioPayload: Partial<ScenarioForm> = editingScenarioId !== null
        ? Object.fromEntries(
            (Object.entries(scenario) as [keyof ScenarioForm, ScenarioForm[keyof ScenarioForm]][]).filter(([field, next]) => {
              const current = scenarios.find((item) => item.id === editingScenarioId);
              return current ? current[field as keyof Scenario] !== next : true;
            }),
          )
        : { ...scenario };
      const endpoint = editingScenarioId !== null ? `${apiUrl}/api/scenarios/${editingScenarioId}` : `${apiUrl}/api/scenarios`;
      const response = await apiFetch(endpoint, {
        method: editingScenarioId !== null ? "PATCH" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scenarioPayload),
      });
      const payload = await response.json().catch(() => ({})) as Scenario & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? t("notice.scenarioSaveFailed"));
      const saved: Scenario = payload;
      setScenarios((current) => {
        const exists = current.some((item) => item.id === saved.id);
        return exists
          ? current.map((item) => (item.id === saved.id ? saved : item))
          : [...current, saved];
      });
      const wasEditing = editingScenarioId !== null;
      setScenario(initialScenario);
      setEditingScenarioId(null);
      setSelectedScenario(saved.id);
      setDomainRules(null); setMandateReadyScenarioId(null);
      setDashboard(null);
      setGraph(null);
      setRun(null);
      setTrackerHistory([]);
      setMandateLogs([]);
      if (wasEditing) {
        setNotice(scenarioLabels[lang].updated.replace("{id}", String(saved.id)));
        return;
      }
      setNotice(t("notice.scenarioSaved", { id: saved.id }));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : t("notice.scenarioSaveFailed"));
    } finally {
      setBusy(false);
      setScenarioBusy(false);
    }
  }

  function editScenario(item: Scenario) {
    setEditingScenarioId(item.id);
    const fields = structuredClone(item) as ScenarioForm & { id?: number };
    delete fields.id;
    setScenario(fields);
    selectTab(0);
    window.requestAnimationFrame(() => document.getElementById("wizard-panel-0")?.scrollIntoView({ behavior: "smooth" }));
  }

  async function deleteScenario(item: Scenario) {
    if (!window.confirm(scenarioLabels[lang].confirmDelete(item.id))) return;
    setScenarioBusy(true);
    try {
      const response = await apiFetch(`${apiUrl}/api/scenarios/${item.id}`, { method: "DELETE" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({})) as { detail?: string };
        throw new Error(payload.detail ?? scenarioLabels[lang].deleteFailed);
      }
      setScenarios((current) => current.filter((row) => row.id !== item.id));
      if (editingScenarioId === item.id) { setEditingScenarioId(null); setScenario(initialScenario); }
      if (selectedScenario === item.id) {
        setDomainRules(null); setMandateReadyScenarioId(null);
        setDashboard(null);
        setGraph(null);
        setRun(null);
        setTrackerHistory([]);
        setMandateLogs([]);
        setAgents([]);
        setSelectedScenario(null);
      }
      setNotice(scenarioLabels[lang].deleted.replace("{id}", String(item.id)));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : scenarioLabels[lang].deleteFailed);
    } finally {
      setScenarioBusy(false);
    }
  }

  async function refetchRunStatus() {
    if (!run?.task_id) return;
    setRunActionBusy(true);
    try {
      const response = await apiFetch(`${apiUrl}/api/runs/${run.task_id}`, { cache: "no-store" });
      if (!response.ok) throw new Error(t("error.history", { status: response.status }));
      const payload: RunState = await response.json();
      setRun(payload);
      setTrackerHistory(payload.logs ?? []);
      if (payload.status === "FAILED") {
        setRunHydrationError(null);
        setNotice(payload.error ?? t("notice.failed"));
      } else {
        const hydrated = await hydrateRunData(payload);
        setNotice(payload.should_poll ? t("notice.pollingResumed") : hydrated ? t("notice.refetched") : t("notice.hydrationRefetch"));
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : t("error.history", { status: 0 }));
    } finally {
      setRunActionBusy(false);
    }
  }


  async function refetchRunData() {
    if (!run) return;
    setRunActionBusy(true);
    try {
      const hydrated = await hydrateRunData(run);
      setNotice(hydrated ? t("notice.refetched") : t("notice.hydrationRefetch"));
    } finally {
      setRunActionBusy(false);
    }
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
    setRunHydrationError(null);
    setDashboard(null);
    setGraph(null);
    setLatestRunState("available");
    setRun({ task_id: payload.task_id, session_id: payload.session_id, scenario_id: payload.scenario_id, status: payload.status, polling_state: payload.polling_state, should_poll: payload.should_poll, terminal: payload.terminal, logs: payload.logs, simulation_artifacts: [], agent_breakdown: [], collective_reasoning: null, disagreements: [] });
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

  const runDashboard = !run || dashboard?.session_id === run.session_id ? dashboard : null;
  const latest = runDashboard?.latest_metric ?? null;
  const currentMetric = latest;
  const currentGraph = !run || graph?.session_id === run.session_id ? graph : null;
  const finalResultMessage = typeof run?.result?.result?.message === "string"
    ? run.result.result.message
    : undefined;
  const activeLogs = trackerHistory.length ? trackerHistory : [{ stage: "IDLE", level: "INFO", message: t("notice.idle") }];
  const selectedTemplateData = templates.find((item) => item.key === selectedTemplate);
  const canGenerateRules = selectedScenario !== null && agents.length > 0;
  const canStartDiscussion = canGenerateRules && mandateReadyScenarioId === selectedScenario && domainRules?.scenario_id === selectedScenario && domainRules.generated && !domainRules.stale;
  const rulesAreStale = canGenerateRules && !canStartDiscussion;
  const runStateLabel = latestRunState === "empty"
    ? t("run.readyInitial")
    : latestRunState === "loading"
      ? t("run.checking")
      : run?.status ?? "IDLE";
  const activeInfluence = useMemo(() => {
    return [...(runDashboard?.influence_observations ?? [])].sort(
      (a, b) => (b.normalized_weight ?? 0) - (a.normalized_weight ?? 0)
    );
  }, [runDashboard]);
  const simulationArtifacts = run ? run.simulation_artifacts ?? [] : runDashboard?.simulation_artifacts ?? [];
  const agentBreakdown = run ? run.agent_breakdown ?? [] : runDashboard?.agent_breakdown ?? [];
  const narrativeDisagreements = run ? run.disagreements ?? [] : runDashboard?.disagreements ?? [];
  const collectiveReasoning = run ? run.collective_reasoning ?? null : runDashboard?.collective_reasoning ?? null;
  const hardStop = run?.result?.car?.hard_stop;
  const rejectedAlternatives = run?.result?.car?.rejected_alternatives;
  const hardConstraints = run?.result?.car?.hard_constraints;
  const tabs = [t("wizard.setup"), t("wizard.phase1"), t("wizard.phase2"), t("wizard.phase3"), t("wizard.final")];
  const selectTab = (index: number, focus = false) => {
    const next = Math.max(0, Math.min(4, index));
    setActiveTab(next);
    if (focus) window.requestAnimationFrame(() => tabRefs.current[next]?.focus());
  };
  const handleTabKeyDown = (event: ReactKeyboardEvent<HTMLButtonElement>, index: number) => {
    const next = event.key === "ArrowRight" ? (index + 1) % tabs.length : event.key === "ArrowLeft" ? (index - 1 + tabs.length) % tabs.length : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : null;
    if (next === null) return;
    event.preventDefault();
    selectTab(next, true);
  };

  return <main className="shell dashboard-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">S</span><span>SHCR</span></div><div className="top-actions"><label className="language-select"><span>{t("language.label")}</span><select aria-label={t("language.label")} value={lang} onChange={(event) => setLang(event.target.value as "id" | "en")}><option value="id">ID · {t("language.id")}</option><option value="en">EN · {t("language.en")}</option></select></label><div className="top-meta"><span className="live-dot" /> {t("top.node")} <b>{t("top.phase")}</b></div></div></header>
    <section className="dashboard-hero"><div><div className="eyebrow">{t("hero.eyebrow")}</div><h1>{t("hero.title")}<br /><em>{t("hero.emphasis")}</em></h1><p>{t("hero.description")}</p></div><div className="hero-orbit"><span>DDR</span><b>→</b><span>CAR</span><b>→</b><span>SHCR</span></div></section>
    <div className="notice" role="status" aria-live="polite"><span className="notice-label">{t("notice.label")}</span><span>{notice || t("notice.ready")}</span></div>

    <section className="control-strip"><div className="scenario-overview"><span>{t("scenario.active")}</span><strong>{selectedScenario ? t("scenario.number", { id: selectedScenario }) : t("scenario.none")}</strong><p className={scenarioExpanded ? "expanded" : "collapsed"}>{scenarios.find((item) => item.id === selectedScenario)?.description ?? t("scenario.prompt")}</p><button type="button" onClick={() => setScenarioExpanded((value) => !value)}>{scenarioExpanded ? t("scenario.hide") : t("scenario.show")}</button><select aria-label={t("scenario.select")} value={selectedScenario ?? ""} onChange={(event) => { setSelectedScenario(Number(event.target.value)); setDomainRules(null); setMandateReadyScenarioId(null); setDashboard(null); setGraph(null); setRun(null); setLatestRunState("loading"); setTrackerHistory([]); setMandateLogs([]); }}><option value="" disabled>{t("scenario.select")}</option>{scenarios.map((item) => <option key={item.id} value={item.id}>#{item.id} — {item.description.slice(0, 72)}</option>)}</select></div><button className="primary-button run-button" onClick={startRun} disabled={run?.status === "RUNNING" || run?.status === "QUEUED" || !canStartDiscussion}>{run?.status === "RUNNING" ? t("run.running") : run?.status === "QUEUED" ? t("run.queued") : t("run.start")}<span>↗</span></button></section>

    <nav className="wizard-tabs" role="tablist" aria-label={t("wizard.label")}>{tabs.map((label, index) => <button ref={(element) => { tabRefs.current[index] = element; }} id={`wizard-tab-${index}`} type="button" role="tab" aria-selected={activeTab === index} aria-controls={`wizard-panel-${index}`} tabIndex={activeTab === index ? 0 : -1} className={activeTab === index ? "active" : ""} onClick={() => selectTab(index)} onKeyDown={(event) => handleTabKeyDown(event, index)} key={label}><span>{String(index + 1).padStart(2, "0")}</span><strong>{label}</strong></button>)}</nav>

    <section id="wizard-panel-1" role="tabpanel" aria-labelledby="wizard-tab-1" tabIndex={0} hidden={activeTab !== 1} className="tracker-panel panel"><div className="section-header"><div><span className="section-number">03</span><h2>{t("tracker.title")}</h2></div><span className={`run-state ${latestRunState === "empty" ? "ready" : run?.status?.toLowerCase() ?? "idle"}`}>{runStateLabel}</span></div><div className="tracker-content"><div ref={trackerConsoleRef} className="progress-console max-h-[500px] overflow-y-auto">{activeLogs.map((log, index) => <div className={`console-line ${log.level.toLowerCase()}`} key={`${log.stage}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><i className={index === activeLogs.length - 1 && !["SUCCEEDED", "FAILED"].includes(run?.status ?? "") ? "pulse" : "done"} /><div><b>[{log.level}] {log.code ?? log.stage}</b><small>{log.messages?.[lang] ?? log.message}</small>{log.fallback?.used && <em>Fallback: {log.fallback.retained_artifact ?? log.fallback.kind} · {log.fallback.consensus_impact}</em>}</div></div>)}</div><div className="task-readout"><span>{t("tracker.task")}</span><strong>{run?.task_id ?? "—"}</strong><small>{run?.error ?? (latestRunState === "empty" ? t("run.readyInitialDetail") : run?.status === "SUCCEEDED" ? t("tracker.persisted") : run?.status === "FAILED" ? t("notice.failed") : t("tracker.polling"))}</small><RunMetadataPanel run={run} /></div></div>{run?.status === "FAILED" && <section className="run-failure" role="alert"><strong>{t("notice.failed")}</strong><p>{run.error ?? t("notice.failedDetail")}</p><div><button type="button" onClick={refetchRunStatus} disabled={runActionBusy}>{runActionBusy ? t("notice.refetching") : t("notice.refetch")}</button><button type="button" onClick={startRun} disabled={runActionBusy || rulesAreStale}>{t("notice.retryRun")}</button></div></section>}{runHydrationError && run?.status !== "FAILED" && <section className="run-failure" role="alert"><p>{runHydrationError}</p><button type="button" onClick={refetchRunData} disabled={runActionBusy}>{runActionBusy ? t("notice.refetching") : t("notice.refetch")}</button></section>}<AgentBreakdownPanel agents={agentBreakdown} disagreements={narrativeDisagreements} simulations={simulationArtifacts} /><WizardControls current={activeTab} onChange={selectTab} /></section>

    <section id="wizard-panel-2" role="tabpanel" aria-labelledby="wizard-tab-2" tabIndex={0} hidden={activeTab !== 2} className="panel"><div className="section-header"><div><span className="section-number">06</span><h2>{t("ddr.title")}</h2></div><span className="panel-code">D<sub>ij</sub> / {t("ddr.components")}</span></div><DdrNetworkPanel disagreements={narrativeDisagreements} breakdown={agentBreakdown} /><WizardControls current={activeTab} onChange={selectTab} /></section>

    <section id="wizard-panel-3" role="tabpanel" aria-labelledby="wizard-tab-3" tabIndex={0} hidden={activeTab !== 3} className="panel simulation-phase"><div className="section-header"><div><span className="section-number">03</span><h2>{t("wizard.phase3")}</h2></div><span className="panel-code">CAR / SIM</span></div>{simulationArtifacts.length > 0 ? <div className="simulation-stack">{simulationArtifacts.map((artifact) => <SimulationResolutionPanel artifact={artifact} key={artifact.id} />)}</div> : <SimulationEmptyState hardStop={hardStop} message={finalResultMessage} />}<WizardControls current={activeTab} onChange={selectTab} /></section>


    <section id="wizard-panel-4" role="tabpanel" aria-labelledby="wizard-tab-4" tabIndex={0} hidden={activeTab !== 4} className="final-panel"><section id="analytics" className="analytics-section"><div className="section-header"><div><span className="section-number">07</span><h2>{t("analytics.title")}</h2></div><button className="export-button" onClick={exportManifest}>{t("analytics.export")}</button></div><CollectiveReasoningPanel analysis={collectiveReasoning} /><InfluencePanel observations={activeInfluence} /><InfeasiblePanel metric={currentMetric} hardStop={hardStop} message={finalResultMessage} rejectedAlternatives={rejectedAlternatives} hardConstraints={hardConstraints} /><section className="panel graph-panel"><div className="section-header"><div><span className="section-number">GRAPH</span><h2>{t("graph.title")}</h2></div><div className="graph-legend"><span><i className="pending" /> {t("status.pending")}</span><span><i className="running" /> {t("status.active")}</span><span><i className="succeeded" /> {t("status.complete")}</span><span><i className="warning" /> {t("status.dissent")}</span><span><i className="failed" /> {t("status.failed")}</span></div></div><RunGraph graph={currentGraph} /></section><div className="convergence-banner"><div><span>{t("analytics.final")}</span><strong>{currentMetric?.convergence_status ?? t("analytics.awaiting")}</strong></div><div className="convergence-meta"><span>{t("analytics.feasible")}</span><b>{currentMetric?.feasible_alternatives_count === undefined || currentMetric?.feasible_alternatives_count === null ? "—" : formatNumber(currentMetric.feasible_alternatives_count)}</b></div></div><div className="metric-grid"><MetricCard label={t("analytics.metrics.constraint")} value={currentMetric ? formatNumber(currentMetric.hard_constraint_violation_rate, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"} unit="%" tone="rust" /><MetricCard label={t("analytics.metrics.provenance")} value={currentMetric ? formatNumber(currentMetric.provenance_completeness_percent, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"} unit="%" /><MetricCard label={t("analytics.metrics.schema")} value={dashboard ? formatNumber(dashboard.schema_validity_percent, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"} unit="%" tone="mint" /><MetricCard label={t("analytics.metrics.latency")} value={latest ? formatNumber(latest.latency_ms, { maximumFractionDigits: 0 }) : "—"} unit={latest ? `ms · ${formatNumber(latest.token_usage)} tok` : ""} /></div><div className="analytics-lower"><div className="history-block"><div className="subhead"><span>{t("analytics.metricHistory")}</span><b>{formatNumber(dashboard?.metric_history?.length ?? 0)} {t("common.runs")}</b></div>{dashboard?.metric_history?.length ? dashboard.metric_history.slice(0, 5).map((metric) => <div className="history-row" key={metric.id}><span>#{metric.id}</span><strong>{metric.convergence_status}</strong><i>{formatNumber(metric.provenance_completeness_percent, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}% {t("analytics.provenance")}</i><b>{formatDateTime(metric.created_at)}</b></div>) : <p className="empty">{t("analytics.noHistory")}</p>}</div><div className="influence-block"><div className="subhead"><span>{t("analytics.influence")}</span><b>{formatNumber(activeInfluence.length)} {t("common.observations")}</b></div>{activeInfluence.length ? activeInfluence.map((item, index) => <div className="weight-row" key={`${item.agent}-${item.proposition}`}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{item.agent}</strong><small>{item.proposition}</small></div><b>{item.normalized_weight === null ? "—" : `${formatNumber(item.normalized_weight * 100, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`}</b></div>) : <p className="empty">{t("analytics.noInfluence")}</p>}</div></div></section><WizardControls current={activeTab} onChange={selectTab} /></section>

    <section id="wizard-panel-0" role="tabpanel" aria-labelledby="wizard-tab-0" tabIndex={0} hidden={activeTab !== 0} className="setup-section">
      <div className="section-header"><div><span className="section-number">01 / 02</span><h2>{t("setup.title")}</h2></div></div>
      <p className="setup-intro">{t("setup.intro")}</p>
      <section className="orchestrator-config-card" aria-busy={orchestratorConfigBusy}>
        <div className="orchestrator-config-heading"><div><span className="section-number">01/a</span><h3>{t("orchestrator.title")}</h3><p>{t("orchestrator.description")}</p></div><div className={`config-indicator ${orchestratorHasSavedKey ? "configured" : "unconfigured"}`}><i aria-hidden="true" /><strong>{orchestratorHasSavedKey ? t("orchestrator.configured") : t("orchestrator.notConfigured")}</strong></div></div>
        <div className="orchestrator-config-grid">
          <label className="form-field"><strong>{t("form.apiBase")}</strong><input type="url" placeholder="https://ai.zytroapi.my.id/v1" value={orchestratorConfig.api_base_url} onChange={(event) => setOrchestratorConfig({ ...orchestratorConfig, api_base_url: event.target.value })} /></label>
          <label className="form-field"><strong>{t("form.apiKey")}</strong><input type="password" autoComplete="new-password" placeholder={orchestratorHasSavedKey ? t("orchestrator.keyPlaceholder") : ""} value={orchestratorConfig.api_key} onChange={(event) => setOrchestratorConfig({ ...orchestratorConfig, api_key: event.target.value })} /></label>
          <label className="form-field"><strong>{t("form.model")}</strong><input placeholder="deepseek-v4.1-flash / gpt-4o" value={orchestratorConfig.model_name} onChange={(event) => setOrchestratorConfig({ ...orchestratorConfig, model_name: event.target.value })} /></label>
          <label className="form-field range-field"><span className="range-label"><strong>{t("form.temperature")}</strong><output>{orchestratorConfig.temperature.toFixed(2)}</output></span><input type="range" min="0" max="2" step="0.01" value={orchestratorConfig.temperature} onChange={(event) => setOrchestratorConfig({ ...orchestratorConfig, temperature: Number(event.target.value) })} /></label>
          <label className="form-field"><strong>{t("form.maxTokens")}</strong><input type="number" min="1" step="1" value={orchestratorConfig.max_tokens} onChange={(event) => setOrchestratorConfig({ ...orchestratorConfig, max_tokens: Number(event.target.value) })} /></label>
        </div>
        <div className="orchestrator-actions"><button ref={templateButtonRef} type="button" className="secondary-button" onClick={openTemplateModal} disabled={busy || selectedScenario === null}>{t("orchestrator.domainOverrides")}</button><button type="button" className="template-load-button" onClick={loadAllTemplates} disabled={busy || selectedScenario === null}>{orchestratorConfigBusy ? t("common.generating") : t("orchestrator.generate")}</button></div>
      </section>
      <form className="global-config-form" aria-busy={globalConfigLoading || globalConfigBusy} onSubmit={submitGlobalConfig}><fieldset className="global-config-fieldset" disabled={globalConfigLoading || globalConfigBusy || globalConfigLoadError}><div className="global-config-heading"><div><span className="section-number">GLOBAL / LLM</span><h3>{t("global.title")}</h3><p>{t("global.description")}</p></div><div className={`config-indicator ${globalConfigMeta.has_llm_api_key ? "configured" : "unconfigured"}`}><i aria-hidden="true" /><strong>{globalConfigMeta.has_llm_api_key ? t("global.configured") : t("global.notConfigured")}</strong><small>{t("global.revision", { revision: globalConfigMeta.revision })}</small></div></div><div className="global-config-grid"><label className="form-field"><strong>{t("form.apiBase")}</strong><input type="url" value={globalConfig.llm_base_url} onChange={(event) => setGlobalConfig({ ...globalConfig, llm_base_url: event.target.value })} /><small>{t("global.baseHint")}</small></label><label className="form-field"><strong>{t("form.apiKey")}</strong><input type="password" autoComplete="new-password" value={globalConfig.llm_api_key} placeholder={globalConfigMeta.has_llm_api_key ? t("global.keyPlaceholder") : ""} onChange={(event) => setGlobalConfig({ ...globalConfig, llm_api_key: event.target.value })} /><small>{t("global.keyHint")}</small></label><label className="form-field"><strong>{t("form.model")}</strong><input value={globalConfig.llm_model} onChange={(event) => setGlobalConfig({ ...globalConfig, llm_model: event.target.value })} /></label><label className="form-field"><strong>{t("form.maxTokens")}</strong><input type="number" min="1" step="1" required value={globalConfig.max_tokens} onChange={(event) => setGlobalConfig({ ...globalConfig, max_tokens: Number(event.target.value) })} /></label><label className="form-field"><strong>{t("form.temperature")}</strong><input type="number" min="0" max="2" step="0.01" required value={globalConfig.temperature} onChange={(event) => setGlobalConfig({ ...globalConfig, temperature: Number(event.target.value) })} /></label><label className="toggle-field"><input type="checkbox" checked={globalConfig.apply_to_all} onChange={(event) => setGlobalConfig({ ...globalConfig, apply_to_all: event.target.checked })} /><span aria-hidden="true" /><strong>{t("global.applyAll")}</strong><small>{t("global.applyAllHint")}</small></label></div><button className="primary-button global-config-submit" disabled={globalConfigBusy}>{globalConfigBusy ? t("setup.saving") : globalConfigLoading ? t("global.loading") : t("global.save")}<span>↗</span></button></fieldset></form>{globalConfigLoadError && <div className="config-load-error" role="alert"><span>{t("notice.globalConfigPending")}</span><button type="button" onClick={() => loadGlobalConfig().catch(() => setNotice(t("notice.globalConfigPending")))}>{t("notice.refetch")}</button></div>}
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
          <label className="form-field weight-mode-field"><strong>{t("weights.mode")}</strong><select value={agent.rar_dai_weight_mode} onChange={(event) => setAgent({ ...agent, rar_dai_weight_mode: event.target.value as "auto" | "manual" })}><option value="auto">{t("weights.auto")}</option><option value="manual">{t("weights.manual")}</option></select><small>{t("weights.modeHint")}</small></label>
          {agent.rar_dai_weight_mode === "manual" && <h3 className="manual-weight-title">{t("weights.manual")}</h3>}
          <div className={`theta-form-grid ${agent.rar_dai_weight_mode === "auto" ? "auto-weight-grid" : ""}`}>
            <NumericField label={t("theta.expertise")} value={agent.theta_x} hint={t("theta.expertiseHint")} disabled={agent.rar_dai_weight_mode === "auto"} onChange={(value) => setAgent({ ...agent, theta_x: value })} />
            <NumericField label={t("theta.evidence")} value={agent.theta_q} hint={t("theta.evidenceHint")} disabled={agent.rar_dai_weight_mode === "auto"} onChange={(value) => setAgent({ ...agent, theta_q: value })} />
            <NumericField label={t("theta.history")} value={agent.theta_h} hint={t("theta.historyHint")} disabled={agent.rar_dai_weight_mode === "auto"} onChange={(value) => setAgent({ ...agent, theta_h: value })} />
            <NumericField label={t("theta.relevance")} value={agent.theta_s} hint={t("theta.relevanceHint")} disabled={agent.rar_dai_weight_mode === "auto"} onChange={(value) => setAgent({ ...agent, theta_s: value })} />
            <NumericField label={t("theta.uncertainty")} value={agent.theta_u} hint={t("theta.uncertaintyHint")} disabled={agent.rar_dai_weight_mode === "auto"} onChange={(value) => setAgent({ ...agent, theta_u: value })} />
          </div>
        </fieldset>
        <button className="primary-button agent-submit" disabled={busy} type="submit">{editingAgentId ? t("setup.saveChanges") : t("setup.saveAgent")}<span>↗</span></button>
      </form>

      <div className="agent-register"><div className="subhead"><span>{t("setup.savedAgents")}</span><b>{formatNumber(agents.length)} {t("common.agents")}</b></div>{agents.length ? agents.map((item) => <div className="agent-register-row" key={item.id}><div><strong>{item.display_name ?? item.name}</strong><small>{item.role}</small></div><span>{item.template_key ? t("setup.templateTag") : t("setup.customTag")}</span><b>{item.llm_model ?? t("setup.envDefault")}</b><div className="agent-actions"><button type="button" onClick={() => generateAgentMandate(item.id)} disabled={rulesBusy || busyAgentId !== null || busyAgentId === item.id}>{busyAgentId === item.id ? t("common.generating") : t("common.generateMandate")}</button><button type="button" onClick={() => testConnection(item)} disabled={connectionTests[item.id]?.pending === true}>{connectionTests[item.id]?.pending ? t("notice.testing") : t("action.test")}</button><button type="button" onClick={() => editAgent(item)}>{t("action.edit")}</button><button type="button" className="danger" onClick={() => deleteAgent(item)}>{t("action.delete")}</button>{connectionTests[item.id] && <small role="status" aria-live="polite" className={connectionTests[item.id].ok ? "test-ok" : "test-error"}>{connectionTests[item.id].message}</small>}</div></div>) : <p className="empty">{t("setup.noAgents")}</p>}</div>

      <div className="agent-register"><div className="subhead"><span>{t("scenario.section")}</span><b>{formatNumber(scenarios.length)}</b></div>{scenarios.length ? scenarios.map((item) => <div className={`agent-register-row ${selectedScenario === item.id ? "active" : ""}`} key={item.id}><div><strong>#{item.id}</strong><small>{item.description}</small></div><span>{item.instrument ?? "APBN"}</span><div className="agent-actions"><button type="button" onClick={() => { setSelectedScenario(item.id); setDomainRules(null); setMandateReadyScenarioId(null); setDashboard(null); setGraph(null); setRun(null); setLatestRunState("loading"); setTrackerHistory([]); setMandateLogs([]); }}>{t("scenario.select")}</button><button type="button" onClick={() => editScenario(item)} disabled={scenarioBusy}>{scenarioLabels[lang].edit}</button><button type="button" className="danger" onClick={() => deleteScenario(item)} disabled={scenarioBusy}>{t("action.delete")}</button></div></div>) : <p className="empty">{t("scenario.none")}</p>}</div>

      <form className="scenario-config-form" onSubmit={submitScenario}><div><span className="section-number">{t("scenario.section")}</span><h3>{t("setup.scenario")}</h3><small>{t("form.scenarioHint")}</small>{editingScenarioId !== null && <strong className="scenario-editing">{t("scenario.active")} #{editingScenarioId}</strong>}</div><label className="form-field"><strong>{t("form.goal")}</strong><textarea value={scenario.description} onChange={(event) => setScenario({ ...scenario, description: event.target.value })} rows={4} required /><small>{t("form.goalHint")}</small></label><ScenarioParameterSections value={scenario} onChange={setScenario} lang={lang} /><div className="scenario-form-actions"><button className="secondary-button" disabled={busy || scenarioBusy}>{editingScenarioId !== null ? scenarioLabels[lang].save : t("action.saveScenario")}</button>{editingScenarioId !== null && <button type="button" className="secondary-button" disabled={scenarioBusy} onClick={() => { setEditingScenarioId(null); setScenario(initialScenario); }}>{t("action.cancel")}</button>}</div></form><WizardControls current={activeTab} onChange={selectTab} />
    </section>
    {templateModalOpen && <div className="modal-backdrop" role="presentation"><div ref={templateModalRef} className="template-modal" role="dialog" aria-modal="true" aria-labelledby="template-modal-title"><div className="section-header"><div><span className="section-number">LLM</span><h2 id="template-modal-title">{t("setup.modalTitle")}</h2></div><button type="button" className="modal-close" aria-label={t("modal.close")} onClick={closeTemplateModal}>×</button></div><p>{t("setup.modalDescription")}</p><div className="template-config-list">{templates.map((item) => { const config = templateConfigs[item.key]; if (!config) return null; return <fieldset className="template-config-card" key={item.key}><legend>{item.name}</legend><label><span>{t("form.apiBase")}</span><input value={config.llm_base_url} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_base_url: event.target.value } })} /></label><label><span>{t("form.apiKey")}</span><input type="password" value={config.llm_api_key} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_api_key: event.target.value } })} /></label><label><span>{t("form.model")}</span><input value={config.llm_model} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_model: event.target.value } })} /></label><label><span>{t("form.temperature")}</span><input type="number" min="0" max="2" step="0.01" value={config.temperature} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, temperature: Number(event.target.value) } })} /></label><label><span>{t("form.maxTokens")}</span><input type="number" min="1" value={config.max_tokens} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, max_tokens: Number(event.target.value) } })} /></label></fieldset>; })}</div><button type="button" className="primary-button" onClick={loadAllTemplates} disabled={busy || selectedScenario === null}>{busy ? t("setup.saving") : t("setup.modalSave")}<span>↗</span></button></div></div>}
    <footer><span>{t("footer.instrument")}</span><span>SRR + (RAR → DAI) + DDR + CAR</span></footer>
  </main>;
}
