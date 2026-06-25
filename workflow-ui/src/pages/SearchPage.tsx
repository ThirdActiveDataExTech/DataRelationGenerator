import { useState, useEffect } from 'react'
import GraphViewer from '../components/GraphViewer'
import './SearchPage.css'

interface AppConfig {
  x_token: string
}

interface DatasetResult {
  table_name: string
  domain: string
  description: string
  table_summary: string
  score: number
}

// 백엔드 /dataset/nl_search·/subgraph 의 부분 그래프 형식
interface DatasetSubgraph {
  nodes: { id: string; type: string; seed: boolean }[]
  edges: { source: string; target: string; relation_type: string; score: number }[]
}

// GraphViewer(vis-network)가 기대하는 형식으로 변환
function toViewerGraph(g: DatasetSubgraph | null) {
  if (!g) return null
  return {
    nodes: g.nodes.map(n => ({
      id: n.id,
      labels: ['Dataset'],
      properties: { name: n.id, seed: n.seed },
    })),
    relationships: g.edges.map(e => ({
      start: e.source,
      end: e.target,
      type: e.relation_type,
    })),
  }
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [intent, setIntent] = useState('')
  const [results, setResults] = useState<DatasetResult[]>([])
  const [graph, setGraph] = useState<DatasetSubgraph | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [config, setConfig] = useState<AppConfig | null>(null)

  useEffect(() => {
    fetch('/api/config')
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => console.error('Failed to load config:', err))
  }, [])

  const handleSearch = async () => {
    if (!query || !config) return

    setIsLoading(true)
    try {
      const response = await fetch('/dataset/nl_search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-token': config.x_token,
        },
        body: JSON.stringify({ query, top_k: topK }),
      })

      const data = await response.json()
      const result = data.result || {}
      setIntent(result.intent || '')
      setResults(result.datasets || [])
      setGraph(result.graph || null)
    } catch (error) {
      console.error('Search error:', error)
      alert('검색 중 오류가 발생했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  // 노드 클릭 → 해당 데이터셋 중심 1-hop 부분 그래프로 재구성
  const handleNodeClick = async (nodeId: string) => {
    if (!config) return

    try {
      const response = await fetch('/dataset/subgraph', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-token': config.x_token,
        },
        body: JSON.stringify({ table_names: [nodeId] }),
      })

      const data = await response.json()
      if (data.result && data.result.nodes.length > 0) {
        setGraph(data.result)
      }
    } catch (error) {
      console.error('Node explore error:', error)
      alert('노드 탐색 중 오류가 발생했습니다.')
    }
  }

  return (
    <div className="search-page">
      <div className="search-layout">
        <div className="search-sidebar">
          <div className="search-panel">
          <h2>자연어 검색</h2>
          <div className="search-input-group">
            <input
              type="text"
              className="search-input"
              placeholder="검색어를 입력하세요 (예: 서울의 대형 병원과 연관된 데이터)"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleSearch()}
            />
            <div className="search-option">
              <label>
                Top K Results
                <input
                  type="number"
                  value={topK}
                  onChange={e => setTopK(Number(e.target.value))}
                  min={1}
                  max={20}
                />
              </label>
            </div>
            <button
              className="search-button"
              onClick={handleSearch}
              disabled={!query || isLoading || !config}
            >
              {isLoading ? 'Searching...' : 'Search'}
            </button>
            {intent && (
              <div className="search-intent">
                검색 의도: <strong>{intent}</strong>
              </div>
            )}
          </div>
          </div>

          <div className="results-panel">
            <h3>데이터셋 검색 결과</h3>
            {results.length === 0 ? (
              <div className="empty-state">
                {isLoading ? 'Searching...' : '질의를 입력하고 Search를 누르세요'}
              </div>
            ) : (
              <div className="results-list">
                {results.map((ds, idx) => (
                  <div key={idx} className="result-card">
                    <div className="result-header">
                      <span className="result-title">{ds.table_name}</span>
                      <span className="result-score">{(ds.score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="result-content">
                      <span className="result-domain">[{ds.domain}]</span>{' '}
                      {ds.description || ds.table_summary || '설명 없음'}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="graph-section">
          <GraphViewer graph={toViewerGraph(graph)} onNodeClick={handleNodeClick} />
        </div>
      </div>
    </div>
  )
}
