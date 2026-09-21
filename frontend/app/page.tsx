"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

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
  resolution_mechanism: string;
};
type Influence = { agent: string; proposition: string; normalized_weight: number | null; raw_score: number | null; gate: number };
type Dashboard = {
  scenario: Scenario;
  latest_metric: Metric | null;
  metric_history: Metric[];
  schema_validity_percent: number;
  reasoning_log_count: number;
  disagreements: Disagreement[];
  influence_observations: Influence[];
};
type RunLog = { stage: string; level: string; message: string };
type RunState = { task_id: string; status: string; logs: RunLog[]; result?: { metric_snapshot_id: number } | null; error?: string };
type AgentDomainRules = { agent_id: number; name: string; role: string; template_key: string | null; mandate: string | null; primary_sources: string[]; constraints: string[]; owned_checks: string[]; synthesis_status: "generated" | "fallback"; scenario_mandate: string | null; scenario_focus: string[]; priority_questions: string[]; required_evidence: string[]; llm_model: string | null; token_usage: number | null; error: { code: string; message: string } | null };
type DomainRules = { scenario_id: number; revision: string; generated: boolean; stale: boolean; agent_count: number; rules: { hard_constraints?: string[]; owned_checks?: string[]; principles?: string[]; primary_sources?: string[]; automatic_deficit_ceiling?: number }; agent_rules: AgentDomainRules[]; status: "success" | "partial" | "failed" | "stale" | "missing"; generated_count: number; failure_count: number; detail: string | null };
type AgentForm = Omit<Agent, "id" | "has_llm_api_key" | "template_key" | "system_prompt"> & { llm_api_key: string };
type ScenarioForm = Omit<Scenario, "id" | "max_deficit_constraint">;

const apiUrl = "";
const components = ["dE", "dA", "dP", "dR", "dU", "dO", "dC", "dREC"] as const;
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

function AgentMandateCard({ rules, index }: { rules: AgentDomainRules; index: number }) {
  return <details className="agent-mandate-card" open={index === 0}><summary><div><span>{String(index + 1).padStart(2, "0")} / {rules.template_key?.toUpperCase() ?? "CUSTOM"} / {rules.synthesis_status.toUpperCase()}</span><strong>{rules.name}</strong><small>{rules.role}</small></div><b>{rules.primary_sources.length} sources · {rules.constraints.length} constraints</b></summary><div className="agent-mandate-body">{rules.scenario_mandate && <section className="mandate-copy"><span>LLM SCENARIO MANDATE</span><p>{rules.scenario_mandate}</p></section>}<section className="mandate-copy"><span>AUTHORITATIVE LOCAL SEED</span><p>{rules.mandate ?? "Mandat terstruktur belum tersedia untuk agen ini."}</p></section>{rules.error && <section className="mandate-copy"><span>{rules.error.code}</span><p>{rules.error.message}</p></section>}<div className="agent-rule-columns"><RuleList title="SCENARIO FOCUS" items={rules.scenario_focus} empty="LLM synthesis fallback aktif." /><RuleList title="PRIORITY QUESTIONS" items={rules.priority_questions} empty="Belum ada pertanyaan hasil sintesis." /><RuleList title="REQUIRED EVIDENCE" items={rules.required_evidence} empty="Belum ada bukti tambahan hasil sintesis." /><RuleList title="PRIMARY SOURCES" items={rules.primary_sources} empty="Tidak ada sumber primer terstruktur." /><RuleList title="CONSTRAINTS" items={rules.constraints} empty="Tidak ada constraint terstruktur." /><RuleList title="OWNED CHECKS" items={rules.owned_checks} empty="Tidak ada owned checks terstruktur." /></div></div></details>;
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
  const [run, setRun] = useState<RunState | null>(null);
  const [notice, setNotice] = useState("Research console ready. Select a scenario to activate the tracker.");
  const [domainRules, setDomainRules] = useState<DomainRules | null>(null);
  const [rulesBusy, setRulesBusy] = useState(false);
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

  async function loadDashboard(id: number) {
    const response = await fetch(`${apiUrl}/api/scenarios/${id}/dashboard`, { cache: "no-store" });
    if (!response.ok) throw new Error("Dashboard unavailable");
    setDashboard(await response.json());
  }

  useEffect(() => {
    loadTemplates().catch((error) => setNotice(`Template agen gagal dimuat: ${error.message}`));
    loadSetup().catch(() => setNotice("API connection pending. Check the backend URL."));
  }, []);
  useEffect(() => { if (selectedScenario !== null) loadDashboard(selectedScenario).catch(() => setNotice("No dashboard data for this scenario yet.")); }, [selectedScenario]);
  useEffect(() => {
    const consoleElement = trackerConsoleRef.current;
    if (consoleElement) consoleElement.scrollTop = consoleElement.scrollHeight;
  }, [trackerHistory.length]);

  useEffect(() => {
    if (!run || ["SUCCEEDED", "FAILED"].includes(run.status)) return;
    const timer = window.setInterval(async () => {
      const response = await fetch(`${apiUrl}/api/runs/${run.task_id}`, { cache: "no-store" });
      if (!response.ok) {
        setNotice(`Tracker API error (${response.status}).`);
        return;
      }
      const nextRun: RunState = await response.json();
      setRun(nextRun);
      appendTrackerLogs(nextRun.task_id, nextRun.logs);
      if (nextRun.status === "SUCCEEDED" && selectedScenario !== null) {
        await loadDashboard(selectedScenario);
        setNotice("Cycle complete. Dashboard metrics refreshed from PostgreSQL.");
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
    setRulesBusy(true);
    try {
      const response = await fetch(`${apiUrl}/api/scenarios/${selectedScenario}/domain-rules`, { method: "POST" });
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

  async function submitScenario(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true);
    try {
      const response = await fetch(`${apiUrl}/api/scenarios`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scenario) });
      const payload = await response.json(); if (!response.ok) throw new Error(payload.detail ?? "Scenario could not be saved");
      setScenario(initialScenario); await loadSetup(); setSelectedScenario(payload.id); setNotice(`Skenario ${payload.id} tersimpan; aturan hukum dan batas defisit diterapkan otomatis.`);
    } catch (error) { setNotice(error instanceof Error ? error.message : "Scenario could not be saved"); } finally { setBusy(false); }
  }

  async function startRun() {
    if (selectedScenario === null) { setNotice("Select a scenario before starting a cycle."); return; }
    if (rulesAreStale) { setNotice("Generate ulang mandat setelah perubahan agen sebelum memulai diskusi."); return; }
    const response = await fetch(`${apiUrl}/api/scenarios/${selectedScenario}/runs`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) { setNotice(payload.detail ?? "Could not queue cycle"); return; }
    setRun({ task_id: payload.task_id, status: payload.status, logs: payload.logs });
    appendTrackerLogs(payload.task_id, payload.logs);
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
  const rulesAreStale = canGenerateRules && (!domainRules || !domainRules.generated || domainRules.stale || domainRules.agent_count !== agents.length);
  const activeInfluence = useMemo(() => [...(dashboard?.influence_observations ?? [])].sort((a, b) => (b.normalized_weight ?? 0) - (a.normalized_weight ?? 0)), [dashboard]);

  return <main className="shell dashboard-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">S</span><span>SHCR</span></div><div className="top-meta"><span className="live-dot" /> RESEARCH NODE / POSTGRES + CELERY <b>PHASE 05</b></div></header>
    <section className="dashboard-hero"><div><div className="eyebrow">PHASE 05 / OBSERVATION LAYER</div><h1>Make disagreement<br /><em>inspectable.</em></h1><p>A live evidence surface for tracking execution, divergence, hard constraints, and the convergence states produced by each fiscal experiment.</p></div><div className="hero-orbit"><span>DDR</span><b>→</b><span>CAR</span><b>→</b><span>SHCR</span></div></section>
    <div className="notice"><span className="notice-label">SYSTEM NOTE</span><span>{notice}</span></div>

    <section className="control-strip"><div className="scenario-overview"><span>ACTIVE SCENARIO / SCOPE</span><strong>{selectedScenario ? `Scenario #${selectedScenario}` : "Belum dipilih"}</strong><p className={scenarioExpanded ? "expanded" : "collapsed"}>{scenarios.find((item) => item.id === selectedScenario)?.description ?? "Pilih skenario untuk memulai."}</p><button type="button" onClick={() => setScenarioExpanded((value) => !value)}>{scenarioExpanded ? "Sembunyikan Detail" : "Tampilkan Detail / Expand"}</button><select aria-label="Select active scenario" value={selectedScenario ?? ""} onChange={(event) => { setSelectedScenario(Number(event.target.value)); setDomainRules(null); }}><option value="" disabled>Select scenario</option>{scenarios.map((item) => <option key={item.id} value={item.id}>#{item.id} — {item.description.slice(0, 72)}</option>)}</select></div><button className="primary-button run-button" onClick={startRun} disabled={run?.status === "RUNNING" || run?.status === "QUEUED" || rulesAreStale}>{run?.status === "RUNNING" ? "Diskusi berjalan…" : "Mulai Diskusi"}<span>↗</span></button></section>

    <nav className="menu-rail"><span className="menu-label">CONTROL MENUS</span><a href="#setup">01 / Setup</a><a href="#setup">02 / Scenario</a><a className="active" href="#tracker">03 / Live Tracker</a><a href="#ddr">04 / DDR Network</a><a href="#analytics">05 / Analytics</a></nav>

    <section id="tracker" className="tracker-panel panel"><div className="section-header"><div><span className="section-number">03</span><h2>Live Tracker</h2></div><span className={`run-state ${run?.status?.toLowerCase() ?? "idle"}`}>{run?.status ?? "IDLE"}</span></div><div className="tracker-content"><div ref={trackerConsoleRef} className="progress-console max-h-[500px] overflow-y-auto">{activeLogs.map((log, index) => <div className={`console-line ${log.level.toLowerCase()}`} key={`${log.stage}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><i className={index === activeLogs.length - 1 && !["SUCCEEDED", "FAILED"].includes(run?.status ?? "") ? "pulse" : "done"} /><div><b>[{log.level}] {log.stage}</b><small>{log.message}</small></div></div>)}</div><div className="task-readout"><span>TASK IDENTIFIER</span><strong>{run?.task_id ?? "—"}</strong><small>{run?.error ?? (run?.status === "SUCCEEDED" ? "Result persisted" : "Polling every 1.2 seconds")}</small></div></div></section>

    <section id="ddr" className="panel"><div className="section-header"><div><span className="section-number">04</span><h2>DDR Network</h2></div><span className="panel-code">D<sub>ij</sub> / 8 COMPONENTS</span></div><div className="matrix-wrap">{dashboard?.disagreements.length ? <table className="ddr-table"><thead><tr><th>AGENT PAIR</th>{components.map((component) => <th key={component}>{component}</th>)}<th>RESOLUTION MECHANISM</th></tr></thead><tbody>{dashboard.disagreements.map((item) => <tr key={item.id}><td><strong>{item.agent_i}</strong><small>× {item.agent_j}</small></td>{components.map((component) => <td key={component}><span className={`bool ${item[component] ? "conflict" : "clear"}`}>{item[component] ? "1" : "0"}</span></td>)}<td className="resolution">{item.resolution_mechanism}</td></tr>)}</tbody></table> : <div className="empty-state"><strong>No DDR vectors yet.</strong><span>Run a cycle with at least two schema-valid agents to populate the matrix.</span></div>}</div></section>

    <section id="analytics" className="analytics-section"><div className="section-header"><div><span className="section-number">05</span><h2>Decision Analytics</h2></div><button className="export-button" onClick={exportManifest}>↓ Export reproducibility manifest</button></div><div className="convergence-banner"><div><span>FINAL CONVERGENCE STATE</span><strong>{latest?.convergence_status ?? "AWAITING CYCLE"}</strong></div><div className="convergence-meta"><span>FEASIBLE ALTERNATIVES</span><b>{latest?.feasible_alternatives_count ?? "—"}</b></div></div><div className="metric-grid"><MetricCard label="Hard-constraint violation rate" value={latest ? latest.hard_constraint_violation_rate.toFixed(2) : "—"} unit="%" tone="rust" /><MetricCard label="Provenance completeness" value={latest ? latest.provenance_completeness_percent.toFixed(2) : "—"} unit="%" /><MetricCard label="Schema validity" value={dashboard ? dashboard.schema_validity_percent.toFixed(2) : "—"} unit="%" tone="mint" /><MetricCard label="Latency / token usage" value={latest ? latest.latency_ms.toFixed(0) : "—"} unit={latest ? `ms · ${latest.token_usage} tok` : ""} /></div><div className="analytics-lower"><div className="history-block"><div className="subhead"><span>METRIC SNAPSHOT HISTORY</span><b>{dashboard?.metric_history.length ?? 0} RUNS</b></div>{dashboard?.metric_history.length ? dashboard.metric_history.slice(0, 5).map((metric) => <div className="history-row" key={metric.id}><span>#{metric.id}</span><strong>{metric.convergence_status}</strong><i>{metric.provenance_completeness_percent.toFixed(1)}% provenance</i><b>{new Date(metric.created_at).toLocaleTimeString()}</b></div>) : <p className="empty">Metric history will appear after the first completed cycle.</p>}</div><div className="influence-block"><div className="subhead"><span>RAR-DAI INFLUENCE RANKING</span><b>{activeInfluence.length} OBSERVATIONS</b></div>{activeInfluence.length ? activeInfluence.slice(0, 4).map((item, index) => <div className="weight-row" key={`${item.agent}-${item.proposition}`}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{item.agent}</strong><small>{item.proposition}</small></div><b>{item.normalized_weight === null ? "—" : `${(item.normalized_weight * 100).toFixed(1)}%`}</b></div>) : <p className="empty">No influence observations recorded for this scenario.</p>}</div></div></section>

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
          <div className="automatic-rules"><div className="automatic-rules-intro"><div><strong>{selectedTemplate ? "Kontrak domain template aktif" : "Mandat lintas agen siap dirakit"}</strong><p>Backend menyusun mandat, sumber primer, constraints, dan owned checks per agen tanpa mencampur kepemilikan aturan.</p></div><button type="button" className="template-load-button" onClick={generateMandate} disabled={!canGenerateRules || rulesBusy}>{rulesBusy ? "Generating…" : "Generate Otomatis Mandat"}</button></div>{rulesAreStale && <small className="stale-rules">Konfigurasi agen berubah; generate ulang wajib dilakukan sebelum diskusi.</small>}{domainRules && !domainRules.stale && <><div className="rules-revision"><span>REVISION {domainRules.revision}</span><b>{domainRules.agent_count} AGENTS</b><small>{domainRules.rules.owned_checks?.length ?? 0} hard checks · {domainRules.rules.hard_constraints?.length ?? 0} constraints global</small></div><div className="agent-mandate-grid">{domainRules.agent_rules.map((rules, index) => <AgentMandateCard rules={rules} index={index} key={rules.agent_id} />)}</div></>}{mandateLogs.length > 0 && <div className="mandate-console">{mandateLogs.map((log, index) => <div className={log.level.toLowerCase()} key={`${log.stage}-${index}`}><b>[{log.level}] {log.stage}</b><span>{log.message}</span></div>)}</div>}</div>
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

      <div className="agent-register"><div className="subhead"><span>AGEN TERSIMPAN</span><b>{agents.length} AGENTS</b></div>{agents.length ? agents.map((item) => <div className="agent-register-row" key={item.id}><div><strong>{item.name}</strong><small>{item.role}</small></div><span>{item.template_key ? "TEMPLATE" : "CUSTOM"}</span><b>{item.llm_model ?? "ENV DEFAULT"}</b><div className="agent-actions"><button type="button" onClick={() => testConnection(item)}>Test Connection</button><button type="button" onClick={() => editAgent(item)}>Edit</button><button type="button" className="danger" onClick={() => deleteAgent(item)}>Delete</button>{connectionTests[item.id] && <small className={connectionTests[item.id].ok ? "test-ok" : "test-error"}>{connectionTests[item.id].message}</small>}</div></div>) : <p className="empty">Belum ada agen. Muat template APBN atau buat agen baru.</p>}</div>

      <form className="scenario-config-form" onSubmit={submitScenario}><div><span className="section-number">SCENARIO</span><h3>Uji kebijakan fiskal</h3><small>Constraint hukum dan batas defisit diterapkan otomatis oleh agen.</small></div><label className="form-field"><strong>Goal / Deskripsi Kebijakan</strong><textarea value={scenario.description} onChange={(event) => setScenario({ ...scenario, description: event.target.value })} rows={4} required /><small>Jelaskan tujuan kebijakan, program yang diuji, horizon waktu, dan hasil yang diharapkan.</small></label><label className="form-field"><strong>Program Cost / Parameter Finansial</strong><input type="number" min="0" step="0.01" value={scenario.program_cost ?? ""} onChange={(event) => setScenario({ ...scenario, program_cost: event.target.value === "" ? null : Number(event.target.value) })} /><small>Opsional. Gunakan satuan fiskal yang konsisten dengan alternatif; kosongkan bila belum diketahui.</small></label><button className="secondary-button" disabled={busy}>Simpan skenario</button></form>
    </section>
    {templateModalOpen && <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Konfigurasi LLM template APBN"><div className="template-modal"><div className="section-header"><div><span className="section-number">LLM</span><h2>Konfigurasi 5 Template APBN</h2></div><button type="button" className="modal-close" onClick={() => setTemplateModalOpen(false)}>×</button></div><p>Atur koneksi tiap agen sebelum template disimpan. Kolom kosong memakai konfigurasi environment default.</p><div className="template-config-list">{templates.map((item) => { const config = templateConfigs[item.key]; if (!config) return null; return <fieldset className="template-config-card" key={item.key}><legend>{item.name}</legend><label><span>API Base URL</span><input value={config.llm_base_url} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_base_url: event.target.value } })} /></label><label><span>API Key</span><input type="password" value={config.llm_api_key} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_api_key: event.target.value } })} /></label><label><span>Model Name</span><input value={config.llm_model} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, llm_model: event.target.value } })} /></label><label><span>Temperature</span><input type="number" min="0" max="2" step="0.01" value={config.temperature} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, temperature: Number(event.target.value) } })} /></label><label><span>Max Tokens</span><input type="number" min="1" value={config.max_tokens} onChange={(event) => setTemplateConfigs({ ...templateConfigs, [item.key]: { ...config, max_tokens: Number(event.target.value) } })} /></label></fieldset>; })}</div><button type="button" className="primary-button" onClick={loadAllTemplates} disabled={busy}>{busy ? "Menyimpan…" : "Simpan dan Muat Semua Template"}<span>↗</span></button></div></div>}
    <footer><span>STRUCTURED CONSENSUS / RESEARCH INSTRUMENT</span><span>SRR + (RAR → DAI) + DDR + CAR</span></footer>
  </main>;
}
