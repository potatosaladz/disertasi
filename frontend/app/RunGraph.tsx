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
  type OnSelectionChangeFunc,
} from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import { useI18n } from "./i18n";

export type GraphStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "WARNING" | "FAILED" | "SKIPPED";
export type RunGraphNode = { id: string; kind: "scenario" | "stage" | "decision" | "agent" | "simulation" | "consensus"; label: string; subtitle?: string; status: GraphStatus; position: { x: number; y: number }; details: Record<string, unknown> };
export type RunGraphEdge = { id: string; source: string; target: string; kind: string; status: GraphStatus; label: string; animated?: boolean; details?: Record<string, unknown> };
export type RunGraphPayload = { schema_version: string; task_id: string | null; session_id: string; scenario_id: number; run_status: string; progress_stage: string | null; updated_at: string; nodes: RunGraphNode[]; edges: RunGraphEdge[]; summary: Record<string, unknown> };

type GraphNodeData = RunGraphNode & { [key: string]: unknown };
type SelectedItem = { id: string; type: "node" | "edge"; kind: string; title: string; subtitle: string; status: GraphStatus; details: Record<string, unknown> };
type DetailLog = { index?: number; stage?: string; level?: string; message?: string; code?: string; messages?: { id?: string; en?: string } };

const statusColors: Record<GraphStatus, string> = {
  PENDING: "#c7c8c1",
  RUNNING: "#d5ee70",
  SUCCEEDED: "#6bb582",
  WARNING: "#e6bf62",
  FAILED: "#d96b48",
  SKIPPED: "#8d9791",
};

function NetworkNode({ data, selected }: NodeProps<Node<GraphNodeData>>) {
  const { t } = useI18n();
  return <div className={`network-node ${data.kind} ${data.status.toLowerCase()} ${selected ? "selected" : ""}`}><Handle type="target" position={Position.Top} aria-label={t("graph.incoming")} /><span>{data.kind.toUpperCase()}</span><strong>{data.label}</strong>{data.subtitle && <small>{data.subtitle}</small>}<b>{data.status}</b><Handle type="source" position={Position.Bottom} aria-label={t("graph.outgoing")} /></div>;
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

function scalarText(value: unknown, formatNumber: (value: number) => string, formatDateTime: (value: string) => string) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}T/.test(value)) return formatDateTime(value);
  return String(value);
}

function GraphLogList({ logs }: { logs: DetailLog[] }) {
  const { lang, t, formatNumber } = useI18n();
  const successCount = logs.filter((log) => log.level?.toUpperCase() === "SUCCESS").length;
  const warningCount = logs.filter((log) => log.level?.toUpperCase() === "WARNING").length;
  const errorCount = logs.filter((log) => log.level?.toUpperCase() === "ERROR").length;
  const latest = logs.at(-1);
  const message = (log: DetailLog) => log.messages?.[lang] ?? log.message ?? t("graph.noEvents");
  return <div className="graph-log-block"><div className="graph-phase-summary"><span>{t("graph.phaseSummary")}</span><strong>{formatNumber(successCount)} {t("graph.success")} · {formatNumber(warningCount)} {t("graph.warning")} · {formatNumber(errorCount)} {t("graph.error")}</strong><p>{latest ? message(latest) : t("graph.noEvents")}</p></div><ol className="graph-log-list" aria-label={t("graph.logs")}>{logs.map((log, index) => { const level = (log.level ?? "INFO").toLowerCase(); return <li className={`graph-log-card ${level}`} key={`${log.index ?? index}-${log.stage}-${index}`}><span>{String((log.index ?? index) + 1).padStart(2, "0")}</span><i aria-hidden="true" /><div><b>[{log.level ?? "INFO"}] {log.code ?? log.stage ?? "PHASE"}</b><p>{message(log)}</p></div></li>; })}</ol></div>;
}

function DetailValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  const { t, formatNumber, formatDateTime } = useI18n();
  if (isLogList(value)) return <GraphLogList logs={value} />;
  if (Array.isArray(value)) {
    if (!value.length) return <p className="graph-empty-value">{t("graph.noData")}</p>;
    return <div className="graph-detail-array">{value.map((item, index) => <article key={index}><b>{String(index + 1).padStart(2, "0")}</b><DetailValue value={item} depth={depth + 1} /></article>)}</div>;
  }
  if (isRecord(value)) {
    const entries = Object.entries(value).filter(([key]) => !key.endsWith("_i18n") && key !== "messages");
    if (!entries.length) return <p className="graph-empty-value">{t("graph.noData")}</p>;
    return <dl className={`graph-detail-object depth-${Math.min(depth, 2)}`}>{entries.map(([key, item]) => <div key={key}><dt>{labelText(key)}</dt><dd><DetailValue value={item} depth={depth + 1} /></dd></div>)}</dl>;
  }
  return <p className="graph-detail-scalar">{value === true ? t("graph.yes") : value === false ? t("graph.no") : scalarText(value, formatNumber, formatDateTime)}</p>;
}

export default function RunGraph({ graph }: { graph: RunGraphPayload | null }) {
  const { t } = useI18n();
  const [selected, setSelected] = useState<SelectedItem | null>(null);
  const graphNodes = useMemo<Node<GraphNodeData>[]>(() => (graph?.nodes ?? []).map((item) => ({ id: item.id, type: "network", position: item.position, data: item })), [graph]);
  const graphEdges = useMemo<Edge[]>(() => (graph?.edges ?? []).map((item) => ({ id: item.id, source: item.source, target: item.target, label: item.label, ariaLabel: `${item.label}: ${item.source} → ${item.target}`, animated: item.animated, markerEnd: { type: MarkerType.ArrowClosed }, className: `network-edge ${item.kind} ${item.status.toLowerCase()}`, data: item })), [graph]);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<GraphNodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  useEffect(() => {
    setNodes((current) => graphNodes.map((node) => ({ ...node, position: current.find((item) => item.id === node.id)?.position ?? node.position })));
    setEdges(graphEdges);
  }, [graphEdges, graphNodes, setEdges, setNodes]);
  useEffect(() => setSelected(null), [graph?.session_id]);

  const selectNode = (node: Node<GraphNodeData>) => {
    setSelected({ id: node.id, type: "node", kind: node.data.kind, title: node.data.label, subtitle: node.data.subtitle ?? node.data.kind, status: node.data.status, details: node.data.details });
  };
  const selectEdge = (edge: Edge) => {
    const data = edge.data as RunGraphEdge | undefined;
    setSelected({ id: edge.id, type: "edge", kind: data?.kind ?? "transition", title: String(edge.label ?? data?.kind ?? t("graph.transition")), subtitle: `${edge.source} → ${edge.target}`, status: data?.status ?? "PENDING", details: data?.details ?? {} });
  };
  const onNodeClick: NodeMouseHandler<Node<GraphNodeData>> = (_event, node) => selectNode(node);
  const onEdgeClick: EdgeMouseHandler = (_event, edge) => selectEdge(edge);
  const onSelectionChange: OnSelectionChangeFunc<Node<GraphNodeData>, Edge> = ({ nodes: selectedNodes, edges: selectedEdges }) => {
    if (selectedNodes[0]) selectNode(selectedNodes[0]);
    else if (selectedEdges[0]) selectEdge(selectedEdges[0]);
    else setSelected(null);
  };

  if (!graph) return <div className="graph-empty"><strong>{t("graph.awaiting")}</strong><span>{t("graph.awaitingDetail")}</span></div>;

  return <div className="network-layout"><div className="network-canvas"><ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} nodeTypes={nodeTypes} onNodeClick={onNodeClick} onEdgeClick={onEdgeClick} onSelectionChange={onSelectionChange} fitView fitViewOptions={{ padding: 0.16 }} minZoom={0.28} maxZoom={1.8} nodesDraggable elementsSelectable ariaLabelConfig={{ "controls.ariaLabel": t("graph.controls"), "minimap.ariaLabel": t("graph.minimap"), "node.a11yDescription.keyboardDisabled": t("graph.keyboard") }}><MiniMap nodeColor={(node) => statusColors[(node.data as GraphNodeData).status]} maskColor="rgba(16, 26, 22, .72)" ariaLabel={t("graph.minimap")} /><Controls showInteractive={false} aria-label={t("graph.controls")} /><Background color="rgba(93,104,98,.28)" gap={22} size={1} /></ReactFlow></div><aside className={`network-drawer ${selected ? "open" : ""}`} aria-live="polite" aria-atomic="false" aria-labelledby={selected ? "graph-detail-title" : undefined} aria-describedby={selected ? "graph-detail-subtitle" : undefined}>{selected ? <><button type="button" aria-label={t("graph.close")} onClick={() => setSelected(null)}>{t("common.close").toUpperCase()} ×</button><span>{selected.type.toUpperCase()} / {selected.kind.toUpperCase()} / {selected.id}</span><h3 id="graph-detail-title">{selected.title}</h3><small id="graph-detail-subtitle">{selected.subtitle}</small><b className={`graph-detail-status ${selected.status.toLowerCase()}`}>{t("graph.status")}: {selected.status}</b><div className="network-detail-list">{Object.entries(selected.details ?? {}).map(([key, value]) => <section className={key === "logs" ? "logs" : ""} key={key}><span>{labelText(key)}</span><DetailValue value={value} /></section>)}</div></> : <><span>{t("graph.inspector")}</span><h3>{t("graph.select")}</h3><p>{t("graph.inspect")}</p></>}</aside></div>;
}
