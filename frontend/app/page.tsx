"use client";

import { FormEvent, useEffect, useState } from "react";

type Agent = {
  id: number;
  name: string;
  role: string;
  theta_x: number;
  theta_q: number;
  theta_h: number;
  theta_s: number;
  theta_u: number;
};

type Scenario = {
  id: number;
  description: string;
  max_deficit_constraint: number;
};

type AgentForm = Omit<Agent, "id">;
type ScenarioForm = Omit<Scenario, "id">;

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const initialAgent: AgentForm = {
  name: "",
  role: "",
  theta_x: 1,
  theta_q: 1,
  theta_h: 1,
  theta_s: 1,
  theta_u: 1,
};
const initialScenario: ScenarioForm = {
  description: "",
  max_deficit_constraint: 3,
};

function NumericField({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  hint: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        min="0"
        max="100"
        step="0.01"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        required
      />
      <small>{hint}</small>
    </label>
  );
}

export default function Home() {
  const [agent, setAgent] = useState<AgentForm>(initialAgent);
  const [scenario, setScenario] = useState<ScenarioForm>(initialScenario);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [notice, setNotice] = useState("Ready for a controlled experiment.");
  const [busy, setBusy] = useState(false);

  async function loadData() {
    const [agentResponse, scenarioResponse] = await Promise.all([
      fetch(`${apiUrl}/api/agents`),
      fetch(`${apiUrl}/api/scenarios`),
    ]);
    if (!agentResponse.ok || !scenarioResponse.ok) throw new Error("API unavailable");
    setAgents(await agentResponse.json());
    setScenarios(await scenarioResponse.json());
  }

  useEffect(() => {
    loadData().catch(() => setNotice("API connection pending. Check the backend URL."));
  }, []);

  async function submitAgent(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!agent.name.trim() || !agent.role.trim()) {
      setNotice("Agent name and role are required.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${apiUrl}/api/agents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(agent),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Agent could not be saved");
      setAgents((current) => [...current, payload]);
      setAgent(initialAgent);
      setNotice(`Agent ${payload.name} committed. Theta_U = ${payload.theta_u}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Agent could not be saved");
    } finally {
      setBusy(false);
    }
  }

  async function submitScenario(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!scenario.description.trim() || scenario.max_deficit_constraint < 0) {
      setNotice("Add a scenario description and non-negative deficit limit.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${apiUrl}/api/scenarios`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scenario),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Scenario could not be saved");
      setScenarios((current) => [...current, payload]);
      setScenario(initialScenario);
      setNotice(`Scenario ${payload.id} committed with C_H = ${payload.max_deficit_constraint}%.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Scenario could not be saved");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">S</span><span>SHCR</span></div>
        <div className="top-meta"><span className="live-dot" /> LOCAL RESEARCH NODE <b>v0.4</b></div>
      </header>

      <section className="hero">
        <div className="eyebrow">PHASE 04 / EXPERIMENT CONTROL</div>
        <h1>Shape the agents.<br /><em>Frame the decision.</em></h1>
        <p>Configure heterogeneous reasoning agents and the fiscal boundary conditions they must respect before a consensus cycle begins.</p>
        <div className="status-line"><span className="status-chip">MENU 01 — AGENT STUDIO</span><span>→</span><span className="status-chip muted">MENU 02 — SCENARIO BUILDER</span></div>
      </section>

      <div className="notice"><span className="notice-label">SYSTEM NOTE</span><span>{notice}</span></div>

      <section className="workspace">
        <article className="panel agent-panel">
          <div className="panel-heading"><div><span className="section-number">01</span><h2>Agent Studio</h2></div><span className="panel-code">RAR-DAI / Θ</span></div>
          <p className="panel-intro">Every coefficient is exposed for ablation. Set all values to <strong>1.0</strong> for B5 baseline, or isolate a mechanism by setting its theta to <strong>0</strong>.</p>
          <form onSubmit={submitAgent}>
            <div className="two-up">
              <label className="field"><span>Agent name</span><input value={agent.name} onChange={(e) => setAgent({ ...agent, name: e.target.value })} placeholder="e.g. Fiscal Analyst" required /></label>
              <label className="field"><span>Role / domain</span><input value={agent.role} onChange={(e) => setAgent({ ...agent, role: e.target.value })} placeholder="e.g. Macro policy" required /></label>
            </div>
            <div className="theta-heading"><span>Influence coefficients</span><span>softmax inputs / gates downstream</span></div>
            <div className="theta-grid">
              <NumericField label="ΘX · Expertise" value={agent.theta_x} hint="X" onChange={(value) => setAgent({ ...agent, theta_x: value })} />
              <NumericField label="ΘQ · Evidence" value={agent.theta_q} hint="Q" onChange={(value) => setAgent({ ...agent, theta_q: value })} />
              <NumericField label="ΘH · History" value={agent.theta_h} hint="H" onChange={(value) => setAgent({ ...agent, theta_h: value })} />
              <NumericField label="ΘS · Relevance" value={agent.theta_s} hint="S" onChange={(value) => setAgent({ ...agent, theta_s: value })} />
              <NumericField label="ΘU · Uncertainty" value={agent.theta_u} hint="U / penalty" onChange={(value) => setAgent({ ...agent, theta_u: value })} />
            </div>
            <div className="ablation-callout"><span className="callout-icon">A6</span><span><strong>Uncertainty ablation</strong><br />Set ΘU to 0 to remove the explicit uncertainty penalty.</span></div>
            <button className="primary-button" disabled={busy} type="submit">{busy ? "Committing…" : "Commit agent"}<span>↗</span></button>
          </form>
        </article>

        <article className="panel scenario-panel">
          <div className="panel-heading"><div><span className="section-number">02</span><h2>Scenario Builder</h2></div><span className="panel-code">CAR / C<sub>H</sub></span></div>
          <p className="panel-intro">Define the decision context and the hard boundary that no reconciliation layer or language model may override.</p>
          <form onSubmit={submitScenario}>
            <label className="field"><span>Scenario description</span><textarea value={scenario.description} onChange={(e) => setScenario({ ...scenario, description: e.target.value })} placeholder="Describe the fiscal decision, horizon, and strategic tension…" rows={7} required /></label>
            <div className="constraint-box"><div className="constraint-label"><span>HARD CONSTRAINT</span><b>C<sub>H</sub></b></div><NumericField label="Maximum deficit (%)" value={scenario.max_deficit_constraint} hint="Alternative fails when deficit > this limit" onChange={(value) => setScenario({ ...scenario, max_deficit_constraint: value })} /><div className="constraint-rule" /></div>
            <button className="primary-button dark" disabled={busy} type="submit">{busy ? "Committing…" : "Commit scenario"}<span>↗</span></button>
          </form>
        </article>
      </section>

      <section className="ledger-grid">
        <div className="ledger"><div className="ledger-head"><span>REGISTERED AGENTS</span><b>{String(agents.length).padStart(2, "0")}</b></div>{agents.length === 0 ? <p className="empty">No agents committed yet.</p> : agents.map((item) => <div className="ledger-row" key={item.id}><span className="row-index">{String(item.id).padStart(2, "0")}</span><div><strong>{item.name}</strong><small>{item.role}</small></div><span className="theta-value">ΘU {item.theta_u.toFixed(2)}</span></div>)}</div>
        <div className="ledger"><div className="ledger-head"><span>SCENARIO REGISTER</span><b>{String(scenarios.length).padStart(2, "0")}</b></div>{scenarios.length === 0 ? <p className="empty">No scenarios committed yet.</p> : scenarios.map((item) => <div className="ledger-row" key={item.id}><span className="row-index">{String(item.id).padStart(2, "0")}</span><div><strong>{item.description}</strong><small>C<sub>H</sub> maximum deficit</small></div><span className="theta-value">{item.max_deficit_constraint.toFixed(2)}%</span></div>)}</div>
      </section>

      <footer><span>STRUCTURED CONSENSUS / RESEARCH INSTRUMENT</span><span>SRR + (RAR → DAI) + DDR + CAR</span></footer>
    </main>
  );
}
