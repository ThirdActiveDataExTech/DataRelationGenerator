import { useEffect, useRef, useMemo } from 'react'
import { Network } from 'vis-network'
import type { Data, Options } from 'vis-network'
import './RelationGraph.css'

export interface DatasetNodeInput {
  table_name: string
  domain?: string
  record_count?: number
  column_count?: number
}

export interface RelationEdgeInput {
  table_a: string
  table_b: string
  score: number
  method: string
  relation_type: string
}

interface Props {
  datasets: DatasetNodeInput[]
  relations: RelationEdgeInput[]
}

/** 관계 유형별 색상 (엣지 색상 = 관계 유형 기준) */
const RELATION_TYPE_STYLE: Record<string, { color: string; label: string }> = {
  JOINABLE: { color: '#0284c7', label: 'JOINABLE (결합 가능)' },
  SEMANTICALLY_RELATED: { color: '#16a34a', label: 'SEMANTICALLY_RELATED (의미 연관)' },
  COMPLEMENTARY: { color: '#a855f7', label: 'COMPLEMENTARY (보완)' },
}
const DEFAULT_TYPE_STYLE = { color: '#94a3b8', label: '기타' }

/** 데이터셋 도메인별 노드 색상 */
const DOMAIN_STYLE: Record<string, { background: string; border: string }> = {
  medical: { background: '#e0f2fe', border: '#0284c7' },
  broadcast: { background: '#fef9c3', border: '#eab308' },
  general: { background: '#f0fdf4', border: '#22c55e' },
}
const DEFAULT_DOMAIN_STYLE = { background: '#f1f5f9', border: '#94a3b8' }

const METHOD_LABEL: Record<string, string> = {
  rule: 'Tier 1 규칙',
  embedding: 'Tier 2 임베딩',
  llm: 'Tier 3 LLM',
}

/** 백엔드는 "표준유형:원본라벨" 형태로 저장하므로 `:` 앞 표준 유형으로 매칭 */
function baseType(t: string) {
  return t.split(':')[0]
}

function typeStyle(t: string) {
  return RELATION_TYPE_STYLE[baseType(t)] ?? DEFAULT_TYPE_STYLE
}

function edgeTooltip(r: RelationEdgeInput): string {
  const ts = typeStyle(r.relation_type)
  return `<div style="font-family: -apple-system, system-ui, sans-serif; font-size: 12px; max-width: 280px; line-height: 1.5;">
    <div style="font-weight: 700; color: #1a1d29; margin-bottom: 6px;">
      ${r.table_a} &harr; ${r.table_b}
    </div>
    <div style="margin: 3px 0;"><span style="color:#94a3b8;">유형:</span>
      <span style="color:${ts.color}; font-weight:600;">${r.relation_type}</span></div>
    <div style="margin: 3px 0;"><span style="color:#94a3b8;">탐지:</span>
      ${METHOD_LABEL[r.method] ?? r.method}</div>
    <div style="margin: 3px 0;"><span style="color:#94a3b8;">score:</span>
      ${r.score.toFixed(2)}</div>
  </div>`
}

function nodeTooltip(d: DatasetNodeInput): string {
  const parts: string[] = []
  if (d.domain) parts.push(`<span style="color:#94a3b8;">도메인:</span> ${d.domain}`)
  if (d.record_count != null) parts.push(`<span style="color:#94a3b8;">레코드:</span> ${d.record_count.toLocaleString()}건`)
  if (d.column_count != null) parts.push(`<span style="color:#94a3b8;">컬럼:</span> ${d.column_count}개`)
  return `<div style="font-family: -apple-system, system-ui, sans-serif; font-size: 12px; line-height: 1.5;">
    <div style="font-weight: 700; color: #1a1d29; margin-bottom: 6px;">${d.table_name}</div>
    ${parts.map(p => `<div style="margin:3px 0;">${p}</div>`).join('')}
  </div>`
}

export default function RelationGraph({ datasets, relations }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<Network | null>(null)

  // 화면에 실제로 등장하는 관계 유형만 범례에 표시
  const presentTypes = useMemo(() => {
    const set = new Set(relations.map(r => baseType(r.relation_type)))
    return Array.from(set)
  }, [relations])

  useEffect(() => {
    if (!containerRef.current) return

    // 노드: 등록된 데이터셋 + 관계에만 등장하는 테이블도 포함
    const nodeMap = new Map<string, DatasetNodeInput>()
    datasets.forEach(d => nodeMap.set(d.table_name, d))
    relations.forEach(r => {
      if (!nodeMap.has(r.table_a)) nodeMap.set(r.table_a, { table_name: r.table_a })
      if (!nodeMap.has(r.table_b)) nodeMap.set(r.table_b, { table_name: r.table_b })
    })

    // 노드별 연결 수(degree) 계산 → 크기 반영
    const degree = new Map<string, number>()
    relations.forEach(r => {
      degree.set(r.table_a, (degree.get(r.table_a) ?? 0) + 1)
      degree.set(r.table_b, (degree.get(r.table_b) ?? 0) + 1)
    })

    const nodes = Array.from(nodeMap.values()).map(d => {
      const style = DOMAIN_STYLE[d.domain ?? ''] ?? DEFAULT_DOMAIN_STYLE
      const deg = degree.get(d.table_name) ?? 0
      return {
        id: d.table_name,
        label: d.table_name,
        title: nodeTooltip(d),
        shape: 'dot',
        size: 14 + deg * 3,
        color: { background: style.background, border: style.border, highlight: { background: style.background, border: style.border } },
        font: { size: 13, color: '#1a1d29', face: 'system-ui' },
        borderWidth: 2,
      }
    })

    // 엣지: 관계 (양방향 ↔ 이므로 화살표 없음), 색상 = 관계 유형
    const edges = relations.map((r, i) => {
      const ts = typeStyle(r.relation_type)
      return {
        id: `e${i}`,
        from: r.table_a,
        to: r.table_b,
        title: edgeTooltip(r),
        color: { color: ts.color, highlight: ts.color, hover: ts.color, opacity: 0.85 },
        width: Math.max(1.5, r.score * 4),
        smooth: { enabled: true, type: 'dynamic', roundness: 0.5 },
      }
    })

    const data: Data = { nodes, edges }

    const options: Options = {
      physics: {
        enabled: true,
        barnesHut: {
          gravitationalConstant: -9000,
          centralGravity: 0.3,
          springLength: 170,
          springConstant: 0.04,
        },
        stabilization: { iterations: 200 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        navigationButtons: true,
        keyboard: false,
      },
      nodes: { shadow: { enabled: true, size: 6, x: 0, y: 2, color: 'rgba(0,0,0,0.08)' } },
    }

    if (networkRef.current) networkRef.current.destroy()
    networkRef.current = new Network(containerRef.current, data, options)

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy()
        networkRef.current = null
      }
    }
  }, [datasets, relations])

  const hasData = relations.length > 0

  return (
    <div className="rg-wrap">
      <div className="rg-toolbar">
        <div className="rg-legend">
          {(presentTypes.length ? presentTypes : Object.keys(RELATION_TYPE_STYLE)).map(t => (
            <span key={t} className="rg-legend-item">
              <span className="rg-legend-line" style={{ background: typeStyle(t).color }} />
              {typeStyle(t).label}
            </span>
          ))}
        </div>
        <div className="rg-hint">노드 크기 = 연결 수 · 선 굵기 = score · 마우스를 올리면 상세정보</div>
      </div>

      {hasData ? (
        <div ref={containerRef} className="rg-canvas" />
      ) : (
        <div className="rg-empty">
          <p>표시할 관계가 없습니다.</p>
          <p className="rg-empty-hint">아래 ‘관계 생성 파이프라인’을 실행하면 그래프가 나타납니다.</p>
        </div>
      )}
    </div>
  )
}
