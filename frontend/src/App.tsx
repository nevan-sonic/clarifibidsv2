import React, { useState, useRef, useLayoutEffect, useCallback } from 'react';
import { 
  ShieldCheck, 
  Send, 
  BookOpen, 
  CheckCircle2, 
  AlertCircle, 
  LogOut, 
  FileCheck, 
  AlertTriangle, 
  Trash2,
  Upload,
  FileText,
  X,
  ShieldAlert
} from 'lucide-react';

interface Source {
  source: string;
  file_name?: string;
  item_number: number;
  page_number?: number;
  question: string;
  similarity_score: number;
  content_snippet?: string;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  role?: string;
  classification?: {
    category: string;
    subcategory: string;
    intent: string;
    depth_level: string;
    confidence_score: number;
    entities: string[];
  };
  sources?: Source[];
  evaluation?: {
    is_relevant: boolean;
    is_sufficient: boolean;
    coverage_score: number;
    action: string;
    reason: string;
  };
  latency_ms?: number;
  strategy?: string;
}

const ROLE_QUERIES: Record<string, string[]> = {
  'Bidder': [
    "What are the prerequisites, conditions, and relationship steps for bid submission lifecycle and DSC token configuration?",
    "What are the requirements of the client machine to access the e-Procurement site ?",
    "What is the default Date and Time format used in the System ?",
    "Can I access my eProcurement Account from a different client system ?",
    "What is the validity period of a Digital Signature Certificate ?"
  ],
  'Foreign Bidder': [
    "Can a foreign bidder or consultant get DSC from their respective country ?",
    "Can international consultant get DSC online without physical presence in India ?",
    "Whether Can I Upload BoQ with Currencies other than Indian Rupees ?",
    "Can International firm upload proposals on e-procurement portal with DSC ?"
  ],
  'Department User': [
    "Explain the multi-hop relationship and required quorum between Bid Opener Decryption Failure and Multi-party Bid Decryption Procedure",
    "What is the best practice in renewal of DSC pertaining to Department User ?",
    "What to do if Bid Opener name is not visible in bid openers list during tender creation ?",
    "Who can assign or deassign the role of bid opener and what is the procedure ?"
  ]
};


// ─── Dynamic Knowledge Graph Modal ───────────────────────────────────────────
interface GraphNode { id: string; name: string; node_type: string; description?: string; }
interface GraphEdge { source: string; target: string; relationship_type: string; }
interface GraphData { total_nodes?: number; total_edges?: number; nodes: GraphNode[]; edges: GraphEdge[]; }

const REL_COLOR: Record<string, string> = {
  TRIGGERS_PROCEDURE: '#ef4444',
  REQUIRES_PREREQUISITE: '#38bdf8',
  GOVERNED_BY_CONDITION: '#f59e0b',
  SUPPORTS_EVIDENCE: '#10b981',
};

function getRelColor(rel: string): string {
  for (const [key, color] of Object.entries(REL_COLOR)) {
    if (rel.includes(key) || rel.includes(key.split('_')[0])) return color;
  }
  return '#94a3b8';
}

function getNodeColor(type: string): string {
  if (type === 'ISSUE') return '#dc2626';
  if (type === 'PROCEDURE') return '#0284c7';
  if (type === 'RULE') return '#10b981';
  return '#f59e0b'; // CONDITION
}

function getNodeLetter(type: string): string {
  if (type === 'ISSUE') return 'I';
  if (type === 'PROCEDURE') return 'P';
  if (type === 'RULE') return 'R';
  return 'C';
}

interface ArrowLine {
  x1: number; y1: number; x2: number; y2: number;
  color: string; dashed: boolean; rel: string;
}

function KnowledgeGraphModal({ graphData, loadingGraph, selectedGraphNode, setSelectedGraphNode, onClose }: {
  graphData: GraphData | null;
  loadingGraph: boolean;
  selectedGraphNode: GraphNode | null;
  setSelectedGraphNode: (n: GraphNode) => void;
  onClose: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const nodeRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const [arrows, setArrows] = useState<ArrowLine[]>([]);

  // Groups for 3-column layout
  const issueNodes = graphData?.nodes.filter(n => n.node_type === 'ISSUE').slice(0, 6) ?? [];
  const procedureNodes = graphData?.nodes.filter(n => n.node_type === 'PROCEDURE').slice(0, 6) ?? [];
  const conditionNodes = graphData?.nodes.filter(n => n.node_type === 'CONDITION' || n.node_type === 'RULE').slice(0, 6) ?? [];

  const setNodeRef = useCallback((id: string, el: HTMLDivElement | null) => {
    if (el) nodeRefs.current.set(id, el);
    else nodeRefs.current.delete(id);
  }, []);

  // Recompute arrows after nodes render
  useLayoutEffect(() => {
    if (!graphData || !containerRef.current) return;
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();

    const newArrows: ArrowLine[] = [];

    for (const edge of graphData.edges) {
      const srcEl = nodeRefs.current.get(edge.source);
      const tgtEl = nodeRefs.current.get(edge.target);
      if (!srcEl || !tgtEl) continue;

      const srcRect = srcEl.getBoundingClientRect();
      const tgtRect = tgtEl.getBoundingClientRect();

      // Relative to the SVG container
      const x1 = srcRect.right - rect.left;
      const y1 = srcRect.top + srcRect.height / 2 - rect.top;
      const x2 = tgtRect.left - rect.left;
      const y2 = tgtRect.top + tgtRect.height / 2 - rect.top;

      // If same column (vertical) use center-to-center vertical
      const srcCx = srcRect.left + srcRect.width / 2 - rect.left;
      const tgtCx = tgtRect.left + tgtRect.width / 2 - rect.left;
      const isSameCol = Math.abs(srcCx - tgtCx) < 80;

      newArrows.push({
        x1: isSameCol ? srcRect.left + srcRect.width / 2 - rect.left : x1,
        y1: isSameCol ? srcRect.bottom - rect.top : y1,
        x2: isSameCol ? tgtRect.left + tgtRect.width / 2 - rect.left : x2,
        y2: isSameCol ? tgtRect.top - rect.top : y2,
        color: getRelColor(edge.relationship_type),
        dashed: edge.relationship_type.includes('PREREQUISITE') || edge.relationship_type.includes('EVIDENCE'),
        rel: edge.relationship_type,
      });
    }

    setArrows(newArrows);
  }, [graphData, selectedGraphNode]);

  // Re-measure on resize
  useLayoutEffect(() => {
    const handleResize = () => {
      if (!graphData || !containerRef.current) return;
      const container = containerRef.current;
      const rect = container.getBoundingClientRect();
      const newArrows: ArrowLine[] = [];
      for (const edge of graphData.edges) {
        const srcEl = nodeRefs.current.get(edge.source);
        const tgtEl = nodeRefs.current.get(edge.target);
        if (!srcEl || !tgtEl) continue;
        const srcRect = srcEl.getBoundingClientRect();
        const tgtRect = tgtEl.getBoundingClientRect();
        const x1 = srcRect.right - rect.left;
        const y1 = srcRect.top + srcRect.height / 2 - rect.top;
        const x2 = tgtRect.left - rect.left;
        const y2 = tgtRect.top + tgtRect.height / 2 - rect.top;
        const srcCx = srcRect.left + srcRect.width / 2 - rect.left;
        const tgtCx = tgtRect.left + tgtRect.width / 2 - rect.left;
        const isSameCol = Math.abs(srcCx - tgtCx) < 80;
        newArrows.push({
          x1: isSameCol ? srcRect.left + srcRect.width / 2 - rect.left : x1,
          y1: isSameCol ? srcRect.bottom - rect.top : y1,
          x2: isSameCol ? tgtRect.left + tgtRect.width / 2 - rect.left : x2,
          y2: isSameCol ? tgtRect.top - rect.top : y2,
          color: getRelColor(edge.relationship_type),
          dashed: edge.relationship_type.includes('PREREQUISITE') || edge.relationship_type.includes('EVIDENCE'),
          rel: edge.relationship_type,
        });
      }
      setArrows(newArrows);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [graphData]);

  function NodeCard({ node, col }: { node: GraphNode; col: number }) {
    const isSelected = selectedGraphNode?.id === node.id;
    const color = getNodeColor(node.node_type);
    const letter = getNodeLetter(node.node_type);
    const borderStyle = node.name.includes('Lifecycle') ? 'dashed' : 'solid';
    const selBg = col === 0 ? '#1e1b4b' : col === 1 ? '#0c4a6e' : node.node_type === 'RULE' ? '#064e3b' : '#78350f';

    return (
      <div
        ref={(el) => setNodeRef(node.id, el)}
        onClick={() => setSelectedGraphNode(node)}
        style={{
          display: 'flex', alignItems: 'center', gap: '0.65rem',
          padding: '0.65rem 0.85rem', borderRadius: '8px',
          background: isSelected ? selBg : '#0f172a',
          border: `2px ${borderStyle} ${isSelected ? color : '#334155'}`,
          cursor: 'pointer',
          boxShadow: isSelected ? `0 0 20px ${color}99` : '0 2px 4px rgba(0,0,0,0.4)',
          transition: 'all 0.2s ease', minHeight: '52px', position: 'relative', zIndex: 2,
        }}
      >
        <div style={{
          width: '34px', height: '34px', borderRadius: '50%', background: color,
          border: `2px solid ${color}44`, color: '#fff', display: 'flex',
          alignItems: 'center', justifyContent: 'center', fontWeight: 800,
          fontSize: '0.85rem', flexShrink: 0,
        }}>
          {letter}
        </div>
        <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#f8fafc', lineHeight: 1.3 }}>
          {node.name.length > 32 ? node.name.slice(0, 32) + '...' : node.name}
        </span>
      </div>
    );
  }

  const outgoing = graphData?.edges.filter(e => e.source === selectedGraphNode?.id) ?? [];
  const incoming = graphData?.edges.filter(e => e.target === selectedGraphNode?.id) ?? [];

  return (
    <div
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
        backgroundColor: 'rgba(5, 10, 24, 0.90)', backdropFilter: 'blur(8px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 9999, padding: '1rem',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div style={{
        maxWidth: '1300px', width: '97%', height: '93vh',
        display: 'flex', flexDirection: 'column',
        backgroundColor: '#080f1e', borderRadius: '16px',
        boxShadow: '0 25px 60px rgba(0,0,0,0.85), 0 0 0 1px rgba(255,255,255,0.08)',
        border: '1px solid #1e293b', padding: '1.25rem 1.5rem',
        overflow: 'hidden', color: '#f8fafc',
      }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '0.85rem', marginBottom: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ background: '#0284c7', color: '#fff', borderRadius: '8px', padding: '0.45rem', display: 'flex' }}>
              <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>
            </div>
            <div>
              <h3 style={{ margin: 0, fontSize: '1.15rem', color: '#f8fafc', fontWeight: 800, letterSpacing: '-0.01em' }}>
                ClarifiBids Knowledge Graph &bull; Graph RAG Flow
              </h3>
              <p style={{ margin: '0.15rem 0 0', fontSize: '0.78rem', color: '#94a3b8' }}>
                PostgreSQL 18 Entity-Relationship Network &bull; Multi-hop traversal &bull; {graphData?.total_nodes ?? 0} nodes · {graphData?.total_edges ?? 0} edges live
              </p>
            </div>
          </div>
          <button type="button" onClick={onClose} style={{
            background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '6px', cursor: 'pointer', color: '#94a3b8', padding: '0.4rem',
            display: 'flex', alignItems: 'center',
          }}>
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>

        {loadingGraph ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
            <div className="spinner" style={{ width: '32px', height: '32px', borderColor: '#38bdf8', borderTopColor: 'transparent', marginBottom: '1rem' }}></div>
            <p style={{ fontSize: '0.95rem', fontWeight: 600 }}>Loading Knowledge Graph from PostgreSQL 18...</p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.25rem', flex: 1, minHeight: 0 }}>

            {/* Graph Canvas */}
            <div
              ref={containerRef}
              style={{
                display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.75rem',
                backgroundColor: '#070d19', borderRadius: '12px',
                padding: '1.25rem 1.5rem', border: '1px solid rgba(255,255,255,0.08)',
                overflowY: 'auto', position: 'relative', alignContent: 'start',
              }}
            >
              {/* Dynamic SVG Arrow Overlay */}
              <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 1, overflow: 'visible' }}>
                <defs>
                  {Object.entries(REL_COLOR).map(([rel, color]) => (
                    <marker key={rel} id={`arrow-${rel}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                      <path d="M 0 1 L 10 5 L 0 9 z" fill={color} />
                    </marker>
                  ))}
                  <marker id="arrow-default" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#94a3b8" />
                  </marker>
                </defs>

                {arrows.map((a, i) => {
                  const relKey = Object.keys(REL_COLOR).find(k => a.rel.includes(k)) ?? 'default';
                  const dx = a.x2 - a.x1;
                  const dy = a.y2 - a.y1;
                  const isSameCol = Math.abs(dx) < 80;
                  // Cubic bezier: horizontal cross-column arrows curve outward, vertical same-col arrows are straight
                  const cp1x = isSameCol ? a.x1 : a.x1 + dx * 0.4;
                  const cp1y = isSameCol ? a.y1 + dy * 0.5 : a.y1;
                  const cp2x = isSameCol ? a.x2 : a.x2 - dx * 0.4;
                  const cp2y = isSameCol ? a.y2 - dy * 0.5 : a.y2;
                  const d = isSameCol
                    ? `M ${a.x1} ${a.y1} L ${a.x2} ${a.y2}`
                    : `M ${a.x1} ${a.y1} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${a.x2} ${a.y2}`;

                  // Highlight arrows connected to selected node
                  const isHighlighted = selectedGraphNode &&
                    graphData?.edges[i] &&
                    (graphData.edges[i]?.source === selectedGraphNode.id || graphData.edges[i]?.target === selectedGraphNode.id);

                  return (
                    <path
                      key={i}
                      d={d}
                      stroke={a.color}
                      strokeWidth={isHighlighted ? 3 : 2}
                      strokeDasharray={a.dashed ? '5,4' : undefined}
                      fill="none"
                      markerEnd={`url(#arrow-${relKey})`}
                      opacity={selectedGraphNode ? (isHighlighted ? 1 : 0.25) : 0.75}
                      style={{ transition: 'opacity 0.2s ease, stroke-width 0.2s ease' }}
                    />
                  );
                })}
              </svg>

              {/* Column 1 Header + Issues */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem', zIndex: 2 }}>
                <div style={{
                  padding: '0.4rem 0.75rem', borderRadius: '6px',
                  background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.35)',
                  color: '#fca5a5', fontSize: '0.73rem', fontWeight: 800,
                  textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.05em',
                }}>1. Issues / Blockers</div>
                {issueNodes.map(n => <NodeCard key={n.id} node={n} col={0} />)}
              </div>

              {/* Column 2 Header + Procedures */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem', zIndex: 2 }}>
                <div style={{
                  padding: '0.4rem 0.75rem', borderRadius: '6px',
                  background: 'rgba(2,132,199,0.18)', border: '1px solid rgba(2,132,199,0.4)',
                  color: '#7dd3fc', fontSize: '0.73rem', fontWeight: 800,
                  textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.05em',
                }}>2. Procedures / Actions</div>
                {procedureNodes.map(n => <NodeCard key={n.id} node={n} col={1} />)}
              </div>

              {/* Column 3 Header + Conditions */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem', zIndex: 2 }}>
                <div style={{
                  padding: '0.4rem 0.75rem', borderRadius: '6px',
                  background: 'rgba(245,158,11,0.15)', border: '1px solid rgba(245,158,11,0.35)',
                  color: '#fcd34d', fontSize: '0.73rem', fontWeight: 800,
                  textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.05em',
                }}>3. Conditions &amp; Policies</div>
                {conditionNodes.map(n => <NodeCard key={n.id} node={n} col={2} />)}
              </div>
            </div>

            {/* Right Panel: Node Inspector */}
            <div style={{
              backgroundColor: 'rgba(15,23,42,0.9)', borderRadius: '12px', padding: '1.25rem',
              border: '1px solid rgba(255,255,255,0.08)', display: 'flex',
              flexDirection: 'column', overflowY: 'auto', gap: '0',
            }}>
              {selectedGraphNode ? (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <span style={{
                      fontSize: '0.7rem', padding: '0.15rem 0.5rem', borderRadius: '4px',
                      fontWeight: 800, textTransform: 'uppercase',
                      background: getNodeColor(selectedGraphNode.node_type), color: '#fff',
                    }}>{selectedGraphNode.node_type}</span>
                    <span style={{ fontSize: '0.72rem', color: '#64748b' }}>ID: {selectedGraphNode.id.substring(0, 8)}...</span>
                  </div>
                  <h3 style={{ margin: '0 0 0.5rem', fontSize: '1.05rem', color: '#f8fafc', fontWeight: 800, lineHeight: 1.35 }}>
                    {selectedGraphNode.name}
                  </h3>
                  <p style={{ margin: '0 0 1rem', fontSize: '0.8rem', color: '#94a3b8', lineHeight: 1.5 }}>
                    {selectedGraphNode.description || 'Official entity registered in GePNIC knowledge base graph traversal schema.'}
                  </p>

                  {/* Outgoing */}
                  <div style={{ marginBottom: '1rem' }}>
                    <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', fontWeight: 800, marginBottom: '0.4rem' }}>
                      Outgoing Multi-Hop Chains ({outgoing.length})
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                      {outgoing.map((e, idx) => {
                        const target = graphData?.nodes.find(n => n.id === e.target);
                        const c = getRelColor(e.relationship_type);
                        return (
                          <div key={idx} onClick={() => target && setSelectedGraphNode(target)}
                            style={{ background: 'rgba(30,41,59,0.6)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '6px', padding: '0.5rem 0.75rem', cursor: 'pointer' }}
                            onMouseEnter={el => el.currentTarget.style.borderColor = c}
                            onMouseLeave={el => el.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'}
                          >
                            <div style={{ fontSize: '0.68rem', fontWeight: 800, color: c, textTransform: 'uppercase', marginBottom: '0.1rem' }}>{e.relationship_type}</div>
                            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#f8fafc' }}>{target?.name ?? e.target} &rarr;</div>
                          </div>
                        );
                      })}
                      {outgoing.length === 0 && <div style={{ fontSize: '0.75rem', color: '#64748b', fontStyle: 'italic' }}>No outgoing relationships from this node.</div>}
                    </div>
                  </div>

                  {/* Incoming */}
                  <div style={{ marginBottom: '1rem' }}>
                    <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#a855f7', fontWeight: 800, marginBottom: '0.4rem' }}>
                      Triggered By / Prerequisites ({incoming.length})
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                      {incoming.map((e, idx) => {
                        const source = graphData?.nodes.find(n => n.id === e.source);
                        return (
                          <div key={idx} onClick={() => source && setSelectedGraphNode(source)}
                            style={{ background: 'rgba(30,41,59,0.6)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '6px', padding: '0.5rem 0.75rem', cursor: 'pointer' }}
                            onMouseEnter={el => el.currentTarget.style.borderColor = '#a855f7'}
                            onMouseLeave={el => el.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'}
                          >
                            <div style={{ fontSize: '0.68rem', fontWeight: 800, color: '#c084fc', textTransform: 'uppercase', marginBottom: '0.1rem' }}>{e.relationship_type}</div>
                            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#f8fafc' }}>&larr; {source?.name ?? e.source}</div>
                          </div>
                        );
                      })}
                      {incoming.length === 0 && <div style={{ fontSize: '0.75rem', color: '#64748b', fontStyle: 'italic' }}>Root entity (no incoming triggers).</div>}
                    </div>
                  </div>

                  {/* Legend */}
                  <div style={{ marginTop: 'auto', paddingTop: '0.75rem', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                    <div style={{ fontSize: '0.7rem', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', marginBottom: '0.4rem' }}>Relationship Key</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontSize: '0.72rem' }}>
                      {Object.entries(REL_COLOR).map(([rel, color]) => (
                        <span key={rel} style={{ color }}>&bull; {rel.replace(/_/g, ' ')}</span>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ textAlign: 'center', margin: 'auto', color: '#64748b', fontSize: '0.85rem' }}>
                  Click any node in the canvas to inspect its multi-hop Graph RAG chains.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const [currentUser, setCurrentUser] = useState<{ id: string; username: string; email: string; role: string } | null>(() => {
    const saved = localStorage.getItem('clarifibids_user');
    return saved ? JSON.parse(saved) : null;
  });
  const [authToken, setAuthToken] = useState<string | null>(() => localStorage.getItem('clarifibids_token'));
  
  // Auth Form State
  const [authRole, setAuthRole] = useState('Bidder');
  const [authError, setAuthError] = useState('');
  const [authSubmitting, setAuthSubmitting] = useState(false);

  const [queryText, setQueryText] = useState('');
  const [loading, setLoading] = useState(false);

  // Admin Knowledge Upload & Documents List State
  const [adminDocs, setAdminDocs] = useState<any[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadSourceName, setUploadSourceName] = useState('');
  const [uploadSubmitting, setUploadSubmitting] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [uploadError, setUploadError] = useState('');

  // Interactive Document Page Preview State
  const [showGraphModal, setShowGraphModal] = useState(false);
  const [graphData, setGraphData] = useState<{ total_nodes?: number; total_edges?: number; nodes: any[]; edges: any[] } | null>(null);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [selectedGraphNode, setSelectedGraphNode] = useState<any>(null);

  const fetchGraphData = async () => {
    setLoadingGraph(true);
    setShowGraphModal(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/graph/visualize');
      if (res.ok) {
        const data = await res.json();
        setGraphData(data);
        const defaultProc = data.nodes.find((n: any) => n.name.includes('Bid Submission Lifecycle')) || data.nodes[0];
        setSelectedGraphNode(defaultProc);
      }
    } catch (e) {
      console.error('Failed to load graph data', e);
    } finally {
      setLoadingGraph(false);
    }
  };

  const [previewModal, setPreviewModal] = useState<{
    isOpen: boolean;
    fileName: string;
    pageNumber: number;
    totalPages: number;
    highlightClause: string;
    pageText: string;
    downloadUrl: string;
    loading: boolean;
    error: string | null;
  }>({
    isOpen: false,
    fileName: '',
    pageNumber: 1,
    totalPages: 1,
    highlightClause: '',
    pageText: '',
    downloadUrl: '',
    loading: false,
    error: null,
  });

  const openDocumentPreview = async (fileName: string, pageNumber: number = 1, itemNumber?: number) => {
    setPreviewModal({
      isOpen: true,
      fileName,
      pageNumber,
      totalPages: 1,
      highlightClause: '',
      pageText: '',
      downloadUrl: `http://127.0.0.1:8000/api/v1/documents/raw/${fileName}`,
      loading: true,
      error: null,
    });

    try {
      let url = `http://127.0.0.1:8000/api/v1/documents/preview?file_name=${encodeURIComponent(fileName)}&page_number=${pageNumber}`;
      if (itemNumber !== undefined && itemNumber !== null) {
        url += `&item_number=${itemNumber}`;
      }
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Failed to load document preview (HTTP ${res.status})`);
      }
      const data = await res.json();
      setPreviewModal({
        isOpen: true,
        fileName: data.file_name,
        pageNumber: data.page_number,
        totalPages: data.total_pages,
        highlightClause: data.highlight_clause || '',
        pageText: data.page_text || '',
        downloadUrl: data.download_url,
        loading: false,
        error: null,
      });
    } catch (err: any) {
      setPreviewModal(prev => ({
        ...prev,
        loading: false,
        error: err.message || 'Error loading preview',
      }));
    }
  };

  const fetchAdminDocs = async () => {
    const token = authToken || localStorage.getItem('clarifibids_token');
    if (!token) return;
    setLoadingDocs(true);
    try {
      const res = await fetch('http://localhost:8000/api/v1/admin/documents', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (res.ok) {
        const data = await res.json();
        setAdminDocs(data.documents || []);
      }
    } catch (e) {
      console.error("Failed to load documents", e);
    } finally {
      setLoadingDocs(false);
    }
  };

  React.useEffect(() => {
    if (currentUser?.role === 'ADMIN') {
      fetchAdminDocs();
    }
  }, [currentUser?.role, authToken]);



  const handleDocumentUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    setUploadSubmitting(true);
    setUploadError('');
    setUploadResult(null);

    const formData = new FormData();
    formData.append('file', uploadFile);
    if (uploadSourceName) formData.append('source_name', uploadSourceName);

    try {
      const headers: Record<string, string> = {};
      if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

      const res = await fetch('http://localhost:8000/api/v1/admin/documents/upload', {
        method: 'POST',
        headers,
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed with status ${res.status}`);
      }

      const resData = await res.json();
      setUploadResult(resData.data);
      setUploadFile(null);
      setUploadSourceName('');
      fetchAdminDocs(); // Refresh documents table
    } catch (err: any) {
      setUploadError(err.message);
    } finally {
      setUploadSubmitting(false);
    }
  };


  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    const savedUser = localStorage.getItem('clarifibids_user');
    if (savedUser) {
      try {
        const u = JSON.parse(savedUser);
        const savedMsgs = localStorage.getItem(`clarifibids_msgs_${u.id}`);
        return savedMsgs ? JSON.parse(savedMsgs) : [];
      } catch (e) {
        return [];
      }
    }
    return [];
  });

  // Whenever user changes or logs in, load that user's specific thread
  React.useEffect(() => {
    if (currentUser?.id) {
      const saved = localStorage.getItem(`clarifibids_msgs_${currentUser.id}`);
      setMessages(saved ? JSON.parse(saved) : []);
    } else {
      setMessages([]);
    }
  }, [currentUser?.id]);

  // Whenever messages change, persist to this user's thread
  React.useEffect(() => {
    if (currentUser?.id) {
      localStorage.setItem(`clarifibids_msgs_${currentUser.id}`, JSON.stringify(messages));
    }
  }, [messages, currentUser?.id]);

  const handleLogout = () => {
    localStorage.removeItem('clarifibids_token');
    localStorage.removeItem('clarifibids_user');
    setCurrentUser(null);
    setAuthToken(null);
    setMessages([]);
  };

  const handleClearChat = () => {
    if (currentUser?.id) {
      localStorage.removeItem(`clarifibids_msgs_${currentUser.id}`);
    }
    setMessages([]);
  };

  const handleQuickLogin = async (selectedRole: string) => {
    setAuthError('');
    setAuthSubmitting(true);
    try {
      const res = await fetch('http://localhost:8000/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: selectedRole })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Login failed');
      }

      const data = await res.json();
      localStorage.setItem('clarifibids_token', data.access_token);
      localStorage.setItem('clarifibids_user', JSON.stringify(data.user));
      setAuthToken(data.access_token);

      const userSavedMsgs = localStorage.getItem(`clarifibids_msgs_${data.user.id}`);
      setMessages(userSavedMsgs ? JSON.parse(userSavedMsgs) : []);
      setCurrentUser(data.user);
    } catch (err: any) {
      setAuthError(err.message || 'Login failed');
    } finally {
      setAuthSubmitting(false);
    }
  };

  const handleSend = async (textToSend?: string) => {
    const q = textToSend || queryText;
    if (!q.trim() || loading || !currentUser) return;

    const lockedRole = currentUser.role;
    const userMsgId = Date.now().toString();
    const newMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: q,
      role: lockedRole
    };

    setMessages(prev => [...prev, newMsg]);
    if (!textToSend) setQueryText('');
    setLoading(true);

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }

      const res = await fetch('http://localhost:8000/api/v1/chat/query', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          query_text: q,
          user_role: lockedRole,
          session_id: currentUser.id
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server responded with status ${res.status}`);
      }

      const data = await res.json();
      
      // Clean up raw markdown source references from text body since we have dedicated official cards
      let cleanedAnswer = (data.answer || '')
        .replace(/\*\*Source Attribution\*\*:[^\n]+/gi, '')
        .replace(/\*\*Source Reference\*\*:[^\n]+/gi, '')
        .replace(/\*+\s*GePNIC FAQ Repository, Item #[0-9]+/gi, '')
        .trim();

      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: cleanedAnswer,
        classification: data.classification,
        sources: data.sources,
        evaluation: data.evaluation,
        latency_ms: data.latency_ms,
        strategy: data.strategy
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: `Error connecting to backend: ${err.message}. Please ensure the FastAPI backend is running on http://localhost:8000.`
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  // If user is not authenticated, render the dedicated Login / Authorization Page
  if (!currentUser) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: '#f8fafc' }}>
        {/* Government of India Official Top Bar */}
        <div className="gov-tricolor-bar">
          <div className="gov-stripe-saffron"></div>
          <div className="gov-stripe-white"></div>
          <div className="gov-stripe-green"></div>
        </div>

        <div className="gov-top-header">
          <div className="gov-top-left">
            <span>भारत सरकार | Government of India</span>
            <span>•</span>
            <span>राष्ट्रीय सूचना-विज्ञान केंद्र | National Informatics Centre (NIC)</span>
          </div>
          <div className="gov-top-right">
            <span>eProcurement System of India (GePNIC)</span>
            <span>•</span>
            <span>Central Public Procurement Portal (CPPP)</span>
          </div>
        </div>

        <div className="auth-page-wrapper" style={{ flex: 1 }}>
          {/* Left Side: Domain & Branding */}
          <div className="auth-brand-side">
            <div>
              <div className="brand-header-area">
                <div className="brand-large-icon" style={{ background: '#ffffff', border: '2px solid #ff9933' }}>
                  <ShieldCheck size={36} color="#0a2540" />
                </div>
                <div>
                  <div style={{ fontSize: '0.78rem', color: '#fef08a', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 700 }}>
                    e-Procurement Portal (GePNIC)
                  </div>
                  <h1 className="brand-title" style={{ fontSize: '2.2rem' }}>CLARIFIBIDS</h1>
                  <p className="brand-subtitle" style={{ color: '#bae6fd' }}>
                    Instant, Grounded Assistance for GePNIC e-Tendering & Bidding
                  </p>
                </div>
              </div>

              <div className="brand-feature-list">
                <div className="feature-item">
                  <div className="feature-icon-badge" style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#fca5a5' }}>
                    <AlertTriangle size={22} />
                  </div>
                  <div>
                    <div className="feature-title">Eliminate Technical Bid Rejections</div>
                    <div className="feature-desc">
                      Solve critical blockers before tender deadlines—get instant, step-by-step guidance on <strong>DSC token recognition</strong>, <strong>Java JRE 32/64-bit errors</strong>, <strong>browser compatibility</strong>, and <strong>BoQ template validation</strong>.
                    </div>
                  </div>
                </div>

                <div className="feature-item">
                  <div className="feature-icon-badge" style={{ background: 'rgba(2, 132, 199, 0.2)', color: '#7dd3fc' }}>
                    <BookOpen size={22} />
                  </div>
                  <div>
                    <div className="feature-title">Official Page-by-Page Document Citations</div>
                    <div className="feature-desc">
                      No guesswork or AI hallucinations. Every answer points you directly to the <strong>exact page number and item number</strong> in the official GePNIC Procurement Manual so you can verify policies with confidence.
                    </div>
                  </div>
                </div>

                <div className="feature-item">
                  <div className="feature-icon-badge" style={{ background: 'rgba(19, 136, 8, 0.2)', color: '#86efac' }}>
                    <FileCheck size={22} />
                  </div>
                  <div>
                    <div className="feature-title">Role-Specific Procurement Guidance</div>
                    <div className="feature-desc">
                      Personalized workflows for <strong>Domestic Bidders</strong> (EMD exemption & vendor enrollment), <strong>Foreign Bidders</strong> (overseas DSC & forex currencies), and <strong>Department Users</strong> (tender publishing & bid openers).
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="brand-footer-note">
              Empowering government suppliers and procurement officers with fast, authoritative answers. Tender deadlines operate on Indian Standard Time (IST, GMT+5:30).
            </div>
          </div>

        {/* Right Side: Authentication Gateway Card */}
        <div className="auth-form-side">
          <div className="auth-form-container">
            <h2 className="auth-title">
              Sign In to ClarifiBids
            </h2>
            <p className="auth-desc">
              Select your procurement role to enter the authorized intelligence workspace immediately.
            </p>

            {authError && <div className="auth-error">{authError}</div>}

            <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, color: '#1e293b', marginBottom: '0.4rem', display: 'block' }}>
                  Select Business / Organizational Role
                </label>
                <select 
                  className="form-input"
                  value={authRole}
                  onChange={e => setAuthRole(e.target.value)}
                  style={{ fontSize: '0.92rem', padding: '0.65rem 0.85rem', fontWeight: 600 }}
                >
                  <option value="Bidder">Bidder (Domestic Indian Procurement)</option>
                  <option value="Foreign Bidder">Foreign Bidder (International Vendor)</option>
                  <option value="Department User">Department User (Procuring Entity)</option>
                  <option value="ADMIN">ADMIN (System & Document Management)</option>
                </select>
              </div>

              <div className="role-badge-explainer" style={{ margin: '0.2rem 0 0.5rem 0' }}>
                <ShieldCheck size={16} color="#0284c7" />
                <span>Enforces strict role-based policy isolation and prevents unauthorized document leakage.</span>
              </div>

              <button 
                type="button" 
                className="submit-auth-btn" 
                disabled={authSubmitting}
                onClick={() => handleQuickLogin(authRole)}
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  gap: '0.6rem',
                  fontSize: '0.98rem',
                  padding: '0.8rem',
                  fontWeight: 700
                }}
              >
                {authSubmitting ? (
                  <>
                    <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }}></div>
                    <span>Accessing Workspace...</span>
                  </>
                ) : (
                  <>
                    <span>Enter Workspace as {authRole}</span>
                    <ShieldCheck size={18} />
                  </>
                )}
              </button>

              <div style={{ marginTop: '0.85rem', paddingTop: '0.85rem', borderTop: '1px dashed #cbd5e1' }}>
                <span style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#64748b', fontWeight: 700, display: 'block', marginBottom: '0.5rem' }}>
                  Quick Role Switcher
                </span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                  {[
                    { label: 'Domestic Bidder', role: 'Bidder' },
                    { label: 'Foreign Bidder', role: 'Foreign Bidder' },
                    { label: 'Department User', role: 'Department User' },
                    { label: 'Admin Portal', role: 'ADMIN' },
                  ].map(btn => (
                    <button
                      key={btn.role}
                      type="button"
                      disabled={authSubmitting}
                      onClick={() => handleQuickLogin(btn.role)}
                      style={{
                        background: '#f8fafc',
                        border: '1px solid #cbd5e1',
                        borderRadius: '6px',
                        padding: '0.45rem 0.6rem',
                        fontSize: '0.76rem',
                        fontWeight: 600,
                        color: '#1e293b',
                        cursor: 'pointer',
                        textAlign: 'center',
                        transition: 'all 0.15s ease'
                      }}
                      onMouseEnter={e => (e.currentTarget.style.borderColor = '#0284c7')}
                      onMouseLeave={e => (e.currentTarget.style.borderColor = '#cbd5e1')}
                    >
                      {btn.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      </div>
    );
  }

  return (
    <div className="clarifibids-container">
      {/* Government of India Official Top Bar */}
      <div className="gov-tricolor-bar">
        <div className="gov-stripe-saffron"></div>
        <div className="gov-stripe-white"></div>
        <div className="gov-stripe-green"></div>
      </div>

      <div className="gov-top-header">
        <div className="gov-top-left">
          <span>भारत सरकार | Government of India</span>
          <span>•</span>
          <span>राष्ट्रीय सूचना-विज्ञान केंद्र | National Informatics Centre (NIC)</span>
        </div>
        <div className="gov-top-right">
          <span>eProcurement System of India (GePNIC)</span>
          <span>•</span>
          <span>IST: {new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' })} (GMT+5:30)</span>
        </div>
      </div>

      {/* Top Banner & Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo-badge" style={{ background: '#0a2540', border: '2px solid #ff9933' }}>
            <ShieldCheck className="icon-shield" color="#ff9933" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <h1 className="title" style={{ color: '#0a2540' }}>CLARIFIBIDS</h1>
              <span style={{ fontSize: '0.68rem', background: '#e0f2fe', color: '#0369a1', padding: '0.15rem 0.45rem', borderRadius: '4px', fontWeight: 700, border: '1px solid #bae6fd' }}>
                GePNIC G2B ASSIST
              </span>
            </div>
            <p className="subtitle">Official Intelligence & Procedural Clarification Layer for eProcurement</p>
          </div>
        </div>

        {/* User Account & Locked Active Role Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {/* Locked Active Role Badge (No manual switching) */}
          <div className="role-selector-box" style={{ background: '#f8fafc', padding: '0.45rem 0.9rem' }}>
            <span className="role-label" style={{ color: '#475569' }}>Authorized Scope:</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 700, color: 'var(--primary)', fontSize: '0.85rem' }}>
              <ShieldCheck size={16} color="#0284c7" />
              <span>{currentUser?.role || 'Bidder'}</span>
            </div>
          </div>

          {/* View Knowledge Graph Button */}
          <button 
            onClick={fetchGraphData}
            className="download-doc-btn"
            style={{ background: '#f0fdf4', borderColor: '#86efac', color: '#166534', fontWeight: 700 }}
            title="Inspect relational Graph RAG entities and edges"
          >
            <BookOpen size={15} color="#166534" />
            <span>Knowledge Graph</span>
          </button>

          {/* Direct Upload Button for ADMIN */}
          {currentUser?.role === 'ADMIN' && (
            <button 
              onClick={() => { setShowUploadModal(true); setUploadResult(null); setUploadError(''); }}
              className="submit-auth-btn"
              style={{ width: 'auto', padding: '0.45rem 1rem', display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}
              title="Upload new knowledge base documents"
            >
              <Upload size={15} />
              <span>Upload Document</span>
            </button>
          )}


          {/* New / Reset Conversation Button */}
          {messages.length > 0 && (
            <button 
              onClick={handleClearChat}
              className="download-doc-btn"
              style={{ background: '#fef2f2', borderColor: '#fecaca', color: '#b91c1c' }}
              title="Clear current session chat history"
            >
              <Trash2 size={15} color="#b91c1c" />
              <span>Clear Thread</span>
            </button>
          )}

          {/* User Auth Profile & Sign Out */}
          <div className="user-auth-section">
            {currentUser && (
              <div className="user-profile-badge">
                <div className="user-avatar">{currentUser.username.charAt(0).toUpperCase()}</div>
                <div className="user-info">
                  <span className="user-name">{currentUser.username}</span>
                  <span className="user-role-sub">{currentUser.email}</span>
                </div>
                <button className="logout-icon-btn" onClick={handleLogout} title="Sign Out">
                  <LogOut size={16} />
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Workspace (Admin Panel for ADMIN; Chat Interface for all other roles) */}
      {currentUser?.role === 'ADMIN' ? (
        <main className="main-content" style={{ overflowY: 'auto', background: '#f8fafc' }}>

          <div className="admin-panel-container">
            {/* Admin Header Card */}
            <div className="admin-header-card">
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                  <ShieldAlert size={20} color="#0284c7" />
                  <span style={{ fontSize: '0.8rem', color: '#0284c7', fontWeight: 700, textTransform: 'uppercase' }}>
                    Central Knowledge Base Administration
                  </span>
                </div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#0f172a' }}>Admin Control Panel</h2>
                <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
                  Restricted to system administrators. Upload, chunk, and intelligently index domain documents across roles.
                </p>
              </div>

              <button 
                onClick={() => { setShowUploadModal(true); setUploadResult(null); setUploadError(''); }}
                className="submit-auth-btn"
                style={{ width: 'auto', padding: '0.6rem 1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              >
                <Upload size={16} />
                <span>Upload New Document</span>
              </button>
            </div>

            {/* Quick Metrics */}
            <div className="admin-stats-grid">
              <div className="admin-stat-card">
                <div style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 600 }}>Total Knowledge Sources</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#0f172a', marginTop: '0.25rem' }}>
                  {adminDocs.length} Documents
                </div>
              </div>
              <div className="admin-stat-card">
                <div style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 600 }}>Total Indexed Chunks</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#0284c7', marginTop: '0.25rem' }}>
                  {adminDocs.reduce((acc, d) => acc + (d.chunk_count || 0), 0)} Chunks
                </div>
              </div>
              <div className="admin-stat-card">
                <div style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 600 }}>Embedding Dimension</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#16a34a', marginTop: '0.25rem' }}>
                  384 dims (BAAI)
                </div>
              </div>
              <div className="admin-stat-card">
                <div style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 600 }}>Role RBAC Policies</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#9333ea', marginTop: '0.25rem' }}>
                  3 Isolated Roles
                </div>
              </div>
            </div>

            {/* Ingested Documents List */}
            <div className="admin-docs-table-card">
              <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#0f172a' }}>Ingested Knowledge Documents</h3>
                <button 
                  onClick={fetchAdminDocs} 
                  className="download-doc-btn"
                  style={{ fontSize: '0.78rem' }}
                >
                  Refresh List
                </button>
              </div>

              {loadingDocs ? (
                <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                  Loading knowledge base records...
                </div>
              ) : adminDocs.length === 0 ? (
                <div style={{ padding: '3rem', textAlign: 'center', color: '#64748b' }}>
                  No documents found. Click "Upload New Document" above to index your first file.
                </div>
              ) : (
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>Document Title</th>
                      <th>File Name</th>
                      <th>Format</th>
                      <th>Total Chunks</th>
                      <th>Indexing Status</th>
                      <th>Date Indexed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {adminDocs.map((doc, idx) => (
                      <tr key={doc.id || idx}>
                        <td style={{ fontWeight: 600, color: '#0f172a' }}>{doc.title}</td>
                        <td style={{ color: '#475569' }}>{doc.file_name}</td>
                        <td>
                          <span style={{ fontSize: '0.72rem', background: '#e2e8f0', padding: '0.2rem 0.5rem', borderRadius: '4px', fontWeight: 700 }}>
                            {doc.document_type}
                          </span>
                        </td>
                        <td style={{ fontWeight: 700, color: '#0284c7' }}>{doc.chunk_count} chunks</td>
                        <td>
                          <span style={{ 
                            fontSize: '0.72rem', 
                            background: doc.status === 'INDEXED' ? '#dcfce7' : '#fef3c7', 
                            color: doc.status === 'INDEXED' ? '#15803d' : '#b45309', 
                            padding: '0.2rem 0.55rem', 
                            borderRadius: '999px',
                            fontWeight: 700
                          }}>
                            {doc.status}
                          </span>
                        </td>
                        <td style={{ color: '#64748b', fontSize: '0.8rem' }}>
                          {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </main>
      ) : (
        <>
          {/* Main Workspace */}
          <main className="main-content">
            {/* Welcome Card if no messages */}

        {messages.length === 0 && (
          <div className="welcome-card">
            <div className="welcome-icon">
              <ShieldCheck className="icon-book" color="#0284c7" />
            </div>
            <h2>GePNIC Procedural & Technical Helpdesk</h2>
            <p>
              Ask any question about <strong>DSC token setup</strong>, <strong>bid submission deadlines</strong>, <strong>EMD fee payment & exemptions</strong>, <strong>BoQ preparation</strong>, or <strong>tender evaluation</strong>. ClarifiBids provides instant, policy-verified guidance cited directly from official government manuals.
            </p>
            <div className="example-prompts">
              <p className="example-title">Frequently Asked Questions for {currentUser?.role}:</p>
              <div className="pills-grid">
                {(ROLE_QUERIES[currentUser?.role || 'Bidder'] || ROLE_QUERIES['Bidder']).map((ex, i) => (
                  <button key={i} className="pill-btn" onClick={() => handleSend(ex)}>
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Message Feed */}
        <div className="message-feed">
          {messages.map(msg => (
            <div key={msg.id} className={`message-row ${msg.sender}`}>
              <div className="message-bubble">
                {msg.sender === 'user' ? (
                  <div className="user-message">
                    <span className="user-role-tag">{msg.role}</span>
                    <p>{msg.text}</p>
                  </div>
                ) : (
                  <div className="assistant-message">
                    {msg.strategy && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}>
                        <span style={{ 
                          fontSize: '0.7rem', 
                          fontWeight: 800, 
                          textTransform: 'uppercase', 
                          padding: '0.2rem 0.55rem', 
                          borderRadius: '4px',
                          background: msg.strategy.includes('GRAPH') ? '#f0fdf4' : msg.strategy.includes('WEB') ? '#fff7ed' : '#f0f9ff',
                          color: msg.strategy.includes('GRAPH') ? '#166534' : msg.strategy.includes('WEB') ? '#c2410c' : '#0369a1',
                          border: `1px solid ${msg.strategy.includes('GRAPH') ? '#86efac' : msg.strategy.includes('WEB') ? '#fed7aa' : '#bae6fd'}`
                        }}>
                          ⚡ Active Routing: {msg.strategy}
                        </span>
                      </div>
                    )}
                    <div className="response-text">
                      {msg.text.split('\n').map((line, idx) => (
                        <p key={idx}>{line}</p>
                      ))}
                    </div>

                    {/* Official Policy Citations Section */}
                    {msg.sources && msg.sources.length > 0 ? (
                      <div className="sources-container">
                        <div className="official-source-banner">
                          <BookOpen size={16} color="#0284c7" />
                          <span style={{ fontWeight: 700, fontSize: '0.85rem', color: '#0a2540' }}>
                            Official Government Document Reference
                          </span>
                          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem', color: '#16a34a', fontWeight: 600 }}>
                              <CheckCircle2 size={14} /> Verified Policy
                            </span>
                          </div>
                        </div>
                        <div className="sources-list" style={{ marginTop: '0.6rem' }}>
                          {msg.sources.map((s, idx) => (
                            <div key={idx} className="source-card">
                              <div className="source-meta">
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                                  <span className="source-title" style={{ fontWeight: 700, color: '#0a2540', fontSize: '0.82rem' }}>
                                    📄 {s.file_name || 'FAQ.docx'} (GePNIC Procurement Manual)
                                  </span>
                                  <span className="badge-item-number" style={{ background: '#f8fafc', border: '1px solid #cbd5e1', padding: '0.2rem 0.55rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 700, color: '#1e293b' }}>
                                    Item #{s.item_number}
                                  </span>
                                  <span 
                                    className="badge-page-reference cursor-pointer" 
                                    style={{ 
                                      background: '#ecfdf5', 
                                      border: '1px solid #10b981', 
                                      padding: '0.25rem 0.75rem', 
                                      borderRadius: '4px', 
                                      fontSize: '0.75rem', 
                                      fontWeight: 800, 
                                      color: '#065f46',
                                      cursor: 'pointer',
                                      display: 'inline-flex',
                                      alignItems: 'center',
                                      gap: '0.35rem',
                                      boxShadow: '0 1px 2px rgba(16, 185, 129, 0.15)',
                                      transition: 'all 0.2s ease'
                                    }}
                                    title="Click to open exact page of the actual document in browser"
                                    onClick={() => {
                                      const page = s.page_number || 1;
                                      const docName = s.file_name || 'FAQ.docx';
                                      window.open(`http://127.0.0.1:8000/api/v1/documents/view/${encodeURIComponent(docName)}#page=${page}`, '_blank');
                                    }}
                                  >
                                    📖 Reference: Page {s.page_number || 1}
                                    <span style={{ fontSize: '0.65rem', background: '#059669', color: '#fff', padding: '0.1rem 0.35rem', borderRadius: '3px', textTransform: 'uppercase' }}>
                                      Open Exact Page ↗
                                    </span>
                                  </span>

                                  <button
                                    type="button"
                                    style={{
                                      background: '#f0f9ff',
                                      border: '1px solid #bae6fd',
                                      color: '#0284c7',
                                      padding: '0.25rem 0.6rem',
                                      borderRadius: '4px',
                                      fontSize: '0.74rem',
                                      fontWeight: 700,
                                      cursor: 'pointer',
                                      display: 'inline-flex',
                                      alignItems: 'center',
                                      gap: '0.25rem'
                                    }}
                                    title="Open quick text preview popup"
                                    onClick={() => openDocumentPreview(
                                      s.file_name || 'FAQ.docx',
                                      s.page_number || ((s.item_number - 1) % 4) + 1,
                                      s.item_number
                                    )}
                                  >
                                    Quick Preview
                                  </button>
                                </div>
                                <span className="similarity-badge">
                                  Confidence: {(s.similarity_score * 100).toFixed(0)}%
                                </span>
                              </div>
                              <p className="source-q" style={{ margin: '0.4rem 0 0.3rem 0', color: '#1e3a8a', fontWeight: 600 }}>
                                <strong>Official Clause:</strong> {s.question}
                              </p>
                              {s.content_snippet && (
                                <div style={{ fontSize: '0.8rem', color: '#334155', background: '#f8fafc', padding: '0.55rem 0.8rem', borderRadius: '4px', borderLeft: '3px solid #0284c7', margin: '0.3rem 0 0 0', lineHeight: 1.45 }}>
                                  <strong style={{ color: '#0369a1' }}>Excerpt from Manual:</strong> "{s.content_snippet}..."
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="sources-container" style={{ background: '#fef2f2', border: '1px solid #fee2e2', borderRadius: '8px', padding: '0.8rem 1rem', marginTop: '0.8rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#b91c1c', fontWeight: 600, fontSize: '0.85rem' }}>
                          <AlertCircle size={16} />
                          <span>General Notice / Access Restricted</span>
                        </div>
                        <p style={{ fontSize: '0.78rem', color: '#7f1d1d', margin: '0.4rem 0 0 0', lineHeight: 1.4 }}>
                          The inquiry does not match active public GePNIC tender documents or is restricted under role access control.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <span>Classifying query and querying GePNIC Knowledge Base...</span>
            </div>
          )}
        </div>
      </main>

      {/* Input Bar */}
      <footer className="footer-input">
        <form 
          className="input-form"
          onSubmit={e => {
            e.preventDefault();
            handleSend();
          }}
        >
          <input
            type="text"
            className="chat-input"
            placeholder={`Ask a question as authorized ${currentUser?.role || 'Bidder'}...`}
            value={queryText}
            onChange={e => setQueryText(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="send-btn" disabled={loading || !queryText.trim()}>
            <Send className="send-icon" />
          </button>
        </form>
        <p className="disclaimer">
          ClarifiBids is an intelligent assistance layer and does not modify GePNIC procurement workflows.
        </p>
      </footer>
      </>
      )}


      {/* Admin Document Upload Modal */}
      {showUploadModal && (
        <div className="modal-backdrop" onClick={() => setShowUploadModal(false)}>
          <div className="modal-dialog" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Upload size={18} color="#0284c7" />
                <h3 className="modal-title">Upload & Index Knowledge Document</h3>
              </div>
              <button className="modal-close-btn" onClick={() => setShowUploadModal(false)}>
                <X size={18} />
              </button>
            </div>

            <div className="modal-body">
              <div style={{ fontSize: '0.85rem', color: '#475569', lineHeight: 1.5 }}>
                Upload any procurement document (<strong>.docx</strong>, <strong>.pdf</strong>, <strong>.txt</strong>).
                ClarifiBids will chunk the document and use <strong>Groq AI</strong> to intelligently classify each section into:
                <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                  <span className="role-badge-tag role-badge-general">GENERAL (Common)</span>
                  <span className="role-badge-tag role-badge-bidder">Bidder (Domestic)</span>
                  <span className="role-badge-tag role-badge-foreign">Foreign Bidder</span>
                  <span className="role-badge-tag role-badge-department">Department User</span>
                </div>
              </div>

              {uploadError && (
                <div className="auth-error-banner" style={{ margin: 0 }}>
                  <AlertCircle size={16} />
                  <span>{uploadError}</span>
                </div>
              )}

              {uploadResult && (
                <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '8px', padding: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#166534', fontWeight: 700, marginBottom: '0.5rem' }}>
                    <CheckCircle2 size={16} />
                    <span>Document Indexed Successfully!</span>
                  </div>
                  <div style={{ fontSize: '0.85rem', color: '#14532d' }}>
                    <strong>{uploadResult.file_name}</strong> was split into <strong>{uploadResult.total_chunks} chunks</strong> with embeddings.
                  </div>
                  <div style={{ marginTop: '0.75rem', fontSize: '0.8rem', color: '#14532d' }}>
                    <strong>Role Breakdown:</strong>
                    <ul style={{ paddingLeft: '1.25rem', marginTop: '0.25rem' }}>
                      <li>Common (GENERAL): {uploadResult.role_breakdown?.GENERAL || 0} chunks</li>
                      <li>Domestic Bidder: {uploadResult.role_breakdown?.Bidder || 0} chunks</li>
                      <li>Foreign Bidder: {uploadResult.role_breakdown?.['Foreign Bidder'] || 0} chunks</li>
                      <li>Department User: {uploadResult.role_breakdown?.['Department User'] || 0} chunks</li>
                    </ul>
                  </div>
                </div>
              )}

              <form onSubmit={handleDocumentUpload} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>Knowledge Source Title (Optional)</label>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="e.g. GePNIC Tender Submission Manual 2026"
                    value={uploadSourceName}
                    onChange={e => setUploadSourceName(e.target.value)}
                  />
                </div>

                <div 
                  className="file-dropzone"
                  onClick={() => document.getElementById('file-upload-input')?.click()}
                >
                  <FileText size={32} color="#0284c7" style={{ margin: '0 auto 0.5rem auto' }} />
                  <div style={{ fontWeight: 600, color: '#0f172a', fontSize: '0.9rem' }}>
                    {uploadFile ? uploadFile.name : 'Click to select a document (.docx, .pdf, .txt)'}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.25rem' }}>
                    {uploadFile ? `${(uploadFile.size / 1024).toFixed(1)} KB` : 'Maximum file size: 25 MB'}
                  </div>
                  <input
                    id="file-upload-input"
                    type="file"
                    accept=".docx,.pdf,.txt,.md"
                    style={{ display: 'none' }}
                    onChange={e => {
                      if (e.target.files && e.target.files[0]) {
                        setUploadFile(e.target.files[0]);
                      }
                    }}
                  />
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '0.5rem' }}>
                  <button 
                    type="button" 
                    className="download-doc-btn" 
                    onClick={() => setShowUploadModal(false)}
                    disabled={uploadSubmitting}
                  >
                    Close
                  </button>
                  <button 
                    type="submit" 
                    className="submit-auth-btn" 
                    style={{ width: 'auto', padding: '0.5rem 1.25rem' }}
                    disabled={!uploadFile || uploadSubmitting}
                  >
                    {uploadSubmitting ? 'Chunking & Classifying with AI...' : 'Upload & Auto-Classify'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Knowledge Graph Visual Flow Explorer Modal */}
      {showGraphModal && (
        <KnowledgeGraphModal
          graphData={graphData}
          loadingGraph={loadingGraph}
          selectedGraphNode={selectedGraphNode}
          setSelectedGraphNode={setSelectedGraphNode}
          onClose={() => setShowGraphModal(false)}
        />
      )}

      {/* Interactive Document Page Preview Modal */}

      {/* Interactive Document Page Preview Modal */}
      {previewModal.isOpen && (
        <div 
          className="modal-overlay" 
          style={{ 
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.75)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: '1.5rem'
          }}
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setPreviewModal(prev => ({ ...prev, isOpen: false }));
            }
          }}
        >
          <div 
            className="modal-card" 
            style={{ 
              maxWidth: '820px', 
              width: '100%', 
              maxHeight: '90vh', 
              display: 'flex', 
              flexDirection: 'column',
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.35)',
              border: '1px solid #cbd5e1',
              padding: '1.5rem',
              overflow: 'hidden'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #e2e8f0', paddingBottom: '0.85rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ background: '#0284c7', color: '#fff', borderRadius: '8px', padding: '0.5rem', display: 'flex' }}>
                  <BookOpen size={22} />
                </div>
                <div>
                  <h3 style={{ margin: 0, fontSize: '1.1rem', color: '#0a2540', fontWeight: 700 }}>
                    Official Document Page Preview
                  </h3>
                  <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.8rem', color: '#64748b' }}>
                    📄 {previewModal.fileName} • Page {previewModal.pageNumber} of {previewModal.totalPages}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setPreviewModal(prev => ({ ...prev, isOpen: false }))}
                style={{ 
                  background: '#f1f5f9', 
                  border: '1px solid #cbd5e1', 
                  borderRadius: '6px',
                  cursor: 'pointer', 
                  color: '#475569', 
                  padding: '0.4rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
                title="Close Preview"
              >
                <X size={18} />
              </button>
            </div>

            <div style={{ padding: '1rem 0', flex: 1, overflowY: 'auto' }}>
              {previewModal.loading ? (
                <div style={{ textAlign: 'center', padding: '3.5rem 0', color: '#64748b' }}>
                  <div className="spinner" style={{ margin: '0 auto 1rem auto' }}></div>
                  <p style={{ fontSize: '0.9rem', fontWeight: 600 }}>Extracting exact page contents from official document...</p>
                </div>
              ) : previewModal.error ? (
                <div style={{ background: '#fef2f2', border: '1px solid #fecaca', padding: '1.25rem', borderRadius: '8px', color: '#991b1b' }}>
                  <AlertCircle size={22} style={{ marginBottom: '0.5rem' }} />
                  <p style={{ fontWeight: 700 }}>Failed to load document preview</p>
                  <p style={{ fontSize: '0.85rem', marginTop: '0.35rem' }}>{previewModal.error}</p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {previewModal.highlightClause && (
                    <div style={{ background: '#f0fdf4', border: '1px solid #86efac', borderRadius: '8px', padding: '0.85rem 1rem' }}>
                      <span style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#166534', fontWeight: 800 }}>
                        Highlighted Query Clause
                      </span>
                      <h4 style={{ margin: '0.35rem 0 0 0', color: '#14532d', fontSize: '0.95rem', fontWeight: 700 }}>
                        {previewModal.highlightClause}
                      </h4>
                    </div>
                  )}

                  <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '1.25rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '0.5rem' }}>
                      <span style={{ fontSize: '0.75rem', fontWeight: 800, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                        Document Page Extracted Content
                      </span>
                      <span style={{ fontSize: '0.75rem', color: '#0284c7', fontWeight: 600 }}>
                        GePNIC Authoritative Source
                      </span>
                    </div>
                    <pre style={{ 
                      whiteSpace: 'pre-wrap', 
                      wordBreak: 'break-word', 
                      fontFamily: 'inherit', 
                      fontSize: '0.88rem', 
                      lineHeight: 1.65, 
                      color: '#0f172a', 
                      maxHeight: '380px', 
                      overflowY: 'auto',
                      margin: 0
                    }}>
                      {previewModal.pageText}
                    </pre>
                  </div>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid #e2e8f0', paddingTop: '0.85rem', marginTop: 'auto' }}>
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>
                National Informatics Centre (NIC) • eProcurement Project
              </span>
              <button
                type="button"
                className="submit-auth-btn"
                style={{ width: 'auto', padding: '0.5rem 1.4rem', fontSize: '0.88rem' }}
                onClick={() => setPreviewModal(prev => ({ ...prev, isOpen: false }))}
              >
                Close Preview
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

