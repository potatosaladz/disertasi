"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Agent = {
  id: number;
  name: string;
  role: string;
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
type Scenario = { id: number; description: string; max_deficit_constraint: number };
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
type RunState = { task_id: string; status: string; logs: string[]; result?: { metric_snapshot_id: number } | null; error?: string };
type AgentForm = Omit<Agent, "id" | "has_llm_api_key"> & { llm_api_key: string };
type ScenarioForm = Omit<Scenario, "id">;

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
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
  system_prompt: "",
  temperature: 0.2,
  max_tokens: 4000,
};
const initialScenario: ScenarioForm = { description: "", max_deficit_constraint: 3 };

function NumericField({ label, value, onChange, hint }: { label: string; value: number; onChange: (value: number) => void; hint: string }) {
  return <label className="field"><span>{label}</span><input type="number" min="0" step="0.01" value={value} onChange={(event) => onChange(Number(event.target.value))} required /><small>{hint}</small></label>;
}

function MetricCard({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: string }) {
  return <div className={`metric-card ${tone ?? ""}`}><span>{label}</span><strong>{value}<small>{unit}</small></strong></div>;
}

export default function Home() {
  const [agent, setAgent] = useState<AgentForm>(initialAgent);
  const [scenario, setScenario] = useState<ScenarioForm>(initialScenario);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<number | null>(null);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [run, setRun] = useState<RunState | null>(null);
  const [notice, setNotice] = useState("Research console ready. Select a scenario to activate the tracker.");
  const [busy, setBusy] = useState(false);

  async function loadSetup() {
    const [agentResponse, scenarioResponse] = await Promise.all([fetch(`${apiUrl}/api/agents`), fetch(`${apiUrl}/api/scenarios`)]);
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

  useEffect(() => { loadSetup().catch(() => setNotice("API connection pending. Check the backend URL.")); }, []);
  useEffect(() => { if (selectedScenario !== null) loadDashboard(selectedScenario).catch(() => setNotice("No dashboard data for this scenario yet.")); }, [selectedScenario]);

  useEffect(() => {
    if (!run || ["SUCCEEDED", "FAILED"].includes(run.status)) return;
    const timer = window.setInterval(async () => {
      const response = await fetch(`${apiUrl}/api/runs/${run.task_id}`, { cache: "no-store" });
      if (!response.ok) return;
      const nextRun: RunState = await response.json();
      setRun(nextRun);
      if (nextRun.status === "SUCCEEDED" && selectedScenario !== null) {
        await loadDashboard(selectedScenario);
        setNotice("Cycle complete. Dashboard metrics refreshed from PostgreSQL.");
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [run, selectedScenario]);

  async function submitAgent(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true);
    try {
      const agentPayload = {
        ...agent,
        llm_base_url: agent.llm_base_url || null,
        llm_api_key: agent.llm_api_key || null,
        llm_model: agent.llm_model || null,
        system_prompt: agent.system_prompt || null,
      };
      const response = await fetch(`${apiUrl}/api/agents`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(agentPayload) });
      const payload = await response.json(); if (!response.ok) throw new Error(payload.detail ?? "Agent could not be saved");
      setAgent(initialAgent); await loadSetup(); setNotice(`Agent ${payload.name} committed. ΘU = ${payload.theta_u}.`);
    } catch (error) { setNotice(error instanceof Error ? error.message : "Agent could not be saved"); } finally { setBusy(false); }
  }

  async function submitScenario(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true);
    try {
      const response = await fetch(`${apiUrl}/api/scenarios`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scenario) });
      const payload = await response.json(); if (!response.ok) throw new Error(payload.detail ?? "Scenario could not be saved");
      setScenario(initialScenario); await loadSetup(); setSelectedScenario(payload.id); setNotice(`Scenario ${payload.id} committed with C_H = ${payload.max_deficit_constraint}%.`);
    } catch (error) { setNotice(error instanceof Error ? error.message : "Scenario could not be saved"); } finally { setBusy(false); }
  }

  async function startRun() {
    if (selectedScenario === null) { setNotice("Select a scenario before starting a cycle."); return; }
    const response = await fetch(`${apiUrl}/api/scenarios/${selectedScenario}/runs`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) { setNotice(payload.detail ?? "Could not queue cycle"); return; }
    setRun({ task_id: payload.task_id, status: payload.status, logs: payload.logs });
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
  const activeLogs = run?.logs ?? ["Awaiting task dispatch."];
  const activeInfluence = useMemo(() => [...(dashboard?.influence_observations ?? [])].sort((a, b) => (b.normalized_weight ?? 0) - (a.normalized_weight ?? 0)), [dashboard]);

  return <main className="shell dashboard-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">S</span><span>SHCR</span></div><div className="top-meta"><span className="live-dot" /> RESEARCH NODE / POSTGRES + CELERY <b>PHASE 05</b></div></header>
    <section className="dashboard-hero"><div><div className="eyebrow">PHASE 05 / OBSERVATION LAYER</div><h1>Make disagreement<br /><em>inspectable.</em></h1><p>A live evidence surface for tracking execution, divergence, hard constraints, and the convergence states produced by each fiscal experiment.</p></div><div className="hero-orbit"><span>DDR</span><b>→</b><span>CAR</span><b>→</b><span>SHCR</span></div></section>
    <div className="notice"><span className="notice-label">SYSTEM NOTE</span><span>{notice}</span></div>

    <section className="control-strip"><div className="scenario-select"><label className="field"><span>Active scenario / experiment scope</span><select value={selectedScenario ?? ""} onChange={(event) => setSelectedScenario(Number(event.target.value))}><option value="" disabled>Select scenario</option>{scenarios.map((item) => <option key={item.id} value={item.id}>#{item.id} — {item.description}</option>)}</select></label></div><button className="primary-button run-button" onClick={startRun} disabled={run?.status === "RUNNING" || run?.status === "QUEUED"}>{run?.status === "RUNNING" ? "Cycle running…" : "Run SHCR cycle"}<span>↗</span></button></section>

    <nav className="menu-rail"><span className="menu-label">CONTROL MENUS</span><a href="#setup">01 / Setup</a><a href="#setup">02 / Scenario</a><a className="active" href="#tracker">03 / Live Tracker</a><a href="#ddr">04 / DDR Network</a><a href="#analytics">05 / Analytics</a></nav>

    <section id="tracker" className="tracker-panel panel"><div className="section-header"><div><span className="section-number">03</span><h2>Live Tracker</h2></div><span className={`run-state ${run?.status?.toLowerCase() ?? "idle"}`}>{run?.status ?? "IDLE"}</span></div><div className="tracker-content"><div className="progress-console">{activeLogs.map((log, index) => <div className="console-line" key={`${log}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><i className={index === activeLogs.length - 1 && run?.status !== "SUCCEEDED" ? "pulse" : "done"} />{log}</div>)}</div><div className="task-readout"><span>TASK IDENTIFIER</span><strong>{run?.task_id ?? "—"}</strong><small>{run?.error ?? (run?.status === "SUCCEEDED" ? "Result persisted" : "Polling every 1.2 seconds")}</small></div></div></section>

    <section id="ddr" className="panel"><div className="section-header"><div><span className="section-number">04</span><h2>DDR Network</h2></div><span className="panel-code">D<sub>ij</sub> / 8 COMPONENTS</span></div><div className="matrix-wrap">{dashboard?.disagreements.length ? <table className="ddr-table"><thead><tr><th>AGENT PAIR</th>{components.map((component) => <th key={component}>{component}</th>)}<th>RESOLUTION MECHANISM</th></tr></thead><tbody>{dashboard.disagreements.map((item) => <tr key={item.id}><td><strong>{item.agent_i}</strong><small>× {item.agent_j}</small></td>{components.map((component) => <td key={component}><span className={`bool ${item[component] ? "conflict" : "clear"}`}>{item[component] ? "1" : "0"}</span></td>)}<td className="resolution">{item.resolution_mechanism}</td></tr>)}</tbody></table> : <div className="empty-state"><strong>No DDR vectors yet.</strong><span>Run a cycle with at least two schema-valid agents to populate the matrix.</span></div>}</div></section>

    <section id="analytics" className="analytics-section"><div className="section-header"><div><span className="section-number">05</span><h2>Decision Analytics</h2></div><button className="export-button" onClick={exportManifest}>↓ Export reproducibility manifest</button></div><div className="convergence-banner"><div><span>FINAL CONVERGENCE STATE</span><strong>{latest?.convergence_status ?? "AWAITING CYCLE"}</strong></div><div className="convergence-meta"><span>FEASIBLE ALTERNATIVES</span><b>{latest?.feasible_alternatives_count ?? "—"}</b></div></div><div className="metric-grid"><MetricCard label="Hard-constraint violation rate" value={latest ? latest.hard_constraint_violation_rate.toFixed(2) : "—"} unit="%" tone="rust" /><MetricCard label="Provenance completeness" value={latest ? latest.provenance_completeness_percent.toFixed(2) : "—"} unit="%" /><MetricCard label="Schema validity" value={dashboard ? dashboard.schema_validity_percent.toFixed(2) : "—"} unit="%" tone="mint" /><MetricCard label="Latency / token usage" value={latest ? latest.latency_ms.toFixed(0) : "—"} unit={latest ? `ms · ${latest.token_usage} tok` : ""} /></div><div className="analytics-lower"><div className="history-block"><div className="subhead"><span>METRIC SNAPSHOT HISTORY</span><b>{dashboard?.metric_history.length ?? 0} RUNS</b></div>{dashboard?.metric_history.length ? dashboard.metric_history.slice(0, 5).map((metric) => <div className="history-row" key={metric.id}><span>#{metric.id}</span><strong>{metric.convergence_status}</strong><i>{metric.provenance_completeness_percent.toFixed(1)}% provenance</i><b>{new Date(metric.created_at).toLocaleTimeString()}</b></div>) : <p className="empty">Metric history will appear after the first completed cycle.</p>}</div><div className="influence-block"><div className="subhead"><span>RAR-DAI INFLUENCE RANKING</span><b>{activeInfluence.length} OBSERVATIONS</b></div>{activeInfluence.length ? activeInfluence.slice(0, 4).map((item, index) => <div className="weight-row" key={`${item.agent}-${item.proposition}`}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{item.agent}</strong><small>{item.proposition}</small></div><b>{item.normalized_weight === null ? "—" : `${(item.normalized_weight * 100).toFixed(1)}%`}</b></div>) : <p className="empty">No influence observations recorded for this scenario.</p>}</div></div></section>

    <section id="setup" className="setup-summary"><div><span className="section-number">01 / 02</span><h2>Heterogeneous Agent Setup</h2><p>Each agent can use its own OpenAI-compatible endpoint, model, mandate, and generation policy.</p></div><div className="setup-actions"><form onSubmit={submitAgent}><input placeholder="Agent name" value={agent.name} onChange={(event) => setAgent({ ...agent, name: event.target.value })} required /><input placeholder="Role / APBN mandate" value={agent.role} onChange={(event) => setAgent({ ...agent, role: event.target.value })} required /><input placeholder="Custom LLM Base URL" value={agent.llm_base_url ?? ""} onChange={(event) => setAgent({ ...agent, llm_base_url: event.target.value })} /><input placeholder="Model name" value={agent.llm_model ?? ""} onChange={(event) => setAgent({ ...agent, llm_model: event.target.value })} /><input placeholder="API key (stored securely)" type="password" value={agent.llm_api_key} onChange={(event) => setAgent({ ...agent, llm_api_key: event.target.value })} /><input aria-label="Temperature" type="number" min="0" max="2" step="0.01" value={agent.temperature} onChange={(event) => setAgent({ ...agent, temperature: Number(event.target.value) })} /><input aria-label="Max tokens" type="number" min="1" step="1" value={agent.max_tokens} onChange={(event) => setAgent({ ...agent, max_tokens: Number(event.target.value) })} /><input aria-label="Theta U" type="number" min="0" step="0.01" value={agent.theta_u} onChange={(event) => setAgent({ ...agent, theta_u: Number(event.target.value) })} /><textarea className="mandate-input" placeholder="System prompt / mandate / domain principles" value={agent.system_prompt ?? ""} onChange={(event) => setAgent({ ...agent, system_prompt: event.target.value })} rows={4} /><button className="secondary-button" disabled={busy}>Add heterogeneous agent</button></form><form onSubmit={submitScenario}><input placeholder="Scenario description" value={scenario.description} onChange={(event) => setScenario({ ...scenario, description: event.target.value })} required /><input aria-label="Maximum deficit" type="number" min="0" step="0.01" value={scenario.max_deficit_constraint} onChange={(event) => setScenario({ ...scenario, max_deficit_constraint: Number(event.target.value) })} /><button className="secondary-button" disabled={busy}>Add scenario / C_H</button></form></div></section>
    <footer><span>STRUCTURED CONSENSUS / RESEARCH INSTRUMENT</span><span>SRR + (RAR → DAI) + DDR + CAR</span></footer>
  </main>;
}
