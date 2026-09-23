"use client";

import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type EdgeMouseHandler,
  type Node,
  type NodeMouseHandler,
  type NodeProps,
} from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";

export type GraphStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "WARNING" | "FAILED" | "SKIPPED";
export type RunGraphNode = { id: string; kind: "scenario" | "stage" | "decision" | "agent" | "simulation" | "consensus"; label: string; subtitle?: string; status: GraphStatus; position: { x: number; y: number }; details: Record<string, unknown> };
export type RunGraphEdge = { id: string; source: string; target: string; kind: string; status: GraphStatus; label: string; animated?: boolean; details?: Record<string, unknown> };
export type RunGraphPayload = { schema_version: string; task_id: string | null; session_id: string; scenario_id: number; run_status: string; progress_stage: string | null; updated_at: string; nodes: RunGraphNode[]; edges: RunGraphEdge[]; summary: Record<string, unknown> };

type GraphNodeData = RunGraphNode & { [key: string]: unknown };
type SelectedItem = { id: string; type: "node" | "edge"; kind: string; title: string; subtitle: string; status: GraphStatus; details: Record<string, unknown> };
type DetailLog = { index?: number; stage?: string; level?: string; message?: string };

const statusColors: Record<GraphStatus, string> = {
  PENDING: "#c7c8c1",
  RUNNING: "#d5ee70",
  SUCCEEDED: "#6bb582",
  WARNING: "#e6bf62",
  FAILED: "#d96b48",
  SKIPPED: "#8d9791",
};

function NetworkNode({ data, selected }: NodeProps<Node<GraphNodeData>>) {
  return <div className={`network-node ${data.kind} ${data.status.toLowerCase()} ${selected ? "selected" : ""}`}><Handle type="target" position={Position.Top} /><span>{data.kind.toUpperCase()}</span><strong>{data.label}</strong>{data.subtitle && <small>{data.subtitle}</small>}<b>{data.status}</b><Handle type="source" position={Position.Bottom} /></div>;
}

const nodeTypes = { network: NetworkNode };

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isLogList(value: unknown): value is DetailLog[] {
  return Array.isArray(value) && value.every((item) => isRecord(item) && "level" in item && "message" in item);
}

function labelText(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function scalarText(value: unknown) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Ya" : "Tidak";
  return String(value);
}

function GraphLogList({ logs }: { logs: DetailLog[] }) {
  const successCount = logs.filter((log) => log.level?.toUpperCase() === "SUCCESS").length;
  const warningCount = logs.filter((log) => log.level?.toUpperCase() === "WARNING").length;
  const errorCount = logs.filter((log) => log.level?.toUpperCase() === "ERROR").length;
  const latest = logs.at(-1);
  return <div className="graph-log-block"><div className="graph-phase-summary"><span>RINGKASAN FASE</span><strong>{successCount} sukses · {warningCount} peringatan · {errorCount} error</strong><p>{latest?.message ?? "Belum ada peristiwa pada fase ini."}</p></div><ol className="graph-log-list" aria-label="Log eksekusi fase">{logs.map((log, index) => { const level = (log.level ?? "INFO").toLowerCase(); return <li className={`graph-log-card ${level}`} key={`${log.index ?? index}-${log.stage}-${index}`}><span>{String((log.index ?? index) + 1).padStart(2, "0")}</span><i aria-hidden="true" /><div><b>[{log.level ?? "INFO"}] {log.stage ?? "PHASE"}</b><p>{log.message ?? "Tidak ada pesan log."}</p></div></li>; })}</ol></div>;
}

function DetailValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (isLogList(value)) return <GraphLogList logs={value} />;
  if (Array.isArray(value)) {
    if (!value.length) return <p className="graph-empty-value">Tidak ada data.</p>;
    return <div className="graph-detail-array">{value.map((item, index) => <article key={index}><b>{String(index + 1).padStart(2, "0")}</b><DetailValue value={item} depth={depth + 1} /></article>)}</div>;
  }
  if (isRecord(value)) {
    const entries = Object.entries(value);
    if (!entries.length) return <p className="graph-empty-value">Tidak ada data.</p>;
    return <dl className={`graph-detail-object depth-${Math.min(depth, 2)}`}>{entries.map(([key, item]) => <div key={key}><dt>{labelText(key)}</dt><dd><DetailValue value={item} depth={depth + 1} /></dd></div>)}</dl>;
  }
  return <p className="graph-detail-scalar">{scalarText(value)}</p>;
}

export default function RunGraph({ graph }: { graph: RunGraphPayload | null }) {
  const [selected, setSelected] = useState<SelectedItem | null>(null);
  const graphNodes = useMemo<Node<GraphNodeData>[]>(() => (graph?.nodes ?? []).map((item) => ({ id: item.id, type: "network", position: item.position, data: item })), [graph]);
  const graphEdges = useMemo<Edge[]>(() => (graph?.edges ?? []).map((item) => ({ id: item.id, source: item.source, target: item.target, label: item.label, animated: item.animated, markerEnd: { type: MarkerType.ArrowClosed }, className: `network-edge ${item.kind} ${item.status.toLowerCase()}`, data: item })), [graph]);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<GraphNodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  useEffect(() => {
    setNodes((current) => graphNodes.map((node) => ({ ...node, position: current.find((item) => item.id === node.id)?.position ?? node.position })));
    setEdges(graphEdges);
  }, [graphEdges, graphNodes, setEdges, setNodes]);
  useEffect(() => setSelected(null), [graph?.session_id]);

  const onNodeClick: NodeMouseHandler<Node<GraphNodeData>> = (_event, node) => {
    setSelected({ id: node.id, type: "node", kind: node.data.kind, title: node.data.label, subtitle: node.data.subtitle ?? node.data.kind, status: node.data.status, details: node.data.details });
  };
  const onEdgeClick: EdgeMouseHandler = (_event, edge) => {
    const data = edge.data as RunGraphEdge | undefined;
    setSelected({ id: edge.id, type: "edge", kind: data?.kind ?? "transition", title: String(edge.label ?? data?.kind ?? "Graph transition"), subtitle: `${edge.source} → ${edge.target}`, status: data?.status ?? "PENDING", details: data?.details ?? {} });
  };

  if (!graph) return <div className="graph-empty"><strong>Graph awaiting an active run.</strong><span>Start or restore a scenario run to map live SHCR transitions.</span></div>;

  return <div className="network-layout"><div className="network-canvas"><ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} nodeTypes={nodeTypes} onNodeClick={onNodeClick} onEdgeClick={onEdgeClick} fitView fitViewOptions={{ padding: 0.16 }} minZoom={0.28} maxZoom={1.8} nodesDraggable elementsSelectable><MiniMap nodeColor={(node) => statusColors[(node.data as GraphNodeData).status]} maskColor="rgba(16, 26, 22, .72)" /><Controls showInteractive={false} /><Background color="rgba(93,104,98,.28)" gap={22} size={1} /></ReactFlow></div><aside className={`network-drawer ${selected ? "open" : ""}`} aria-labelledby={selected ? "graph-detail-title" : undefined} aria-describedby={selected ? "graph-detail-subtitle" : undefined}>{selected ? <><button type="button" aria-label="Tutup detail graph" onClick={() => setSelected(null)}>CLOSE ×</button><span>{selected.type.toUpperCase()} / {selected.kind.toUpperCase()} / {selected.id}</span><h3 id="graph-detail-title">{selected.title}</h3><small id="graph-detail-subtitle">{selected.subtitle}</small><b className={`graph-detail-status ${selected.status.toLowerCase()}`}>STATUS: {selected.status}</b><div className="network-detail-list">{Object.entries(selected.details).map(([key, value]) => <section className={key === "logs" ? "logs" : ""} key={key}><span>{labelText(key)}</span><DetailValue value={value} /></section>)}</div></> : <><span>INSPECTOR</span><h3>Select a node or edge</h3><p>Inspect mandates, decision artifacts, disagreement routes, simulation inputs, outputs, and transition state.</p></>}</aside></div>;
}
