import { useState, useEffect } from 'react'
import GraphViewer from '../components/GraphViewer'
import './SearchPage.css'

interface AppConfig {
  x_token: string
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(3)
  const [results, setResults] = useState<any[]>([])
  const [graph, setGraph] = useState<any>(null)
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
      const response = await fetch('/graph/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-token': config.x_token,
        },
        body: JSON.stringify({ query, top_k: topK }),
      })

      const data = await response.json()
      setResults(data.documents || [])
      setGraph(data.graph || null)
    } catch (error) {
      console.error('Search error:', error)
      alert('검색 중 오류가 발생했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleNodeClick = async (nodeId: string) => {
    if (!config) return

    try {
      const response = await fetch('/graph/explore', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-token': config.x_token,
        },
        body: JSON.stringify({ node_id: nodeId }),
      })

      const data = await response.json()

      if (data.graph && data.graph.nodes.length > 0) {
        // 클릭한 노드 중심으로 새로운 그래프 표시
        setGraph(data.graph)
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
          <h2>Graph Search</h2>
          <div className="search-input-group">
            <input
              type="text"
              className="search-input"
              placeholder="검색어를 입력하세요 (예: 병상수가 100개 이상인 병원)"
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
          </div>
          </div>

          <div className="results-panel">
            <h3>Search Results</h3>
            {results.length === 0 ? (
              <div className="empty-state">
                {isLoading ? 'Searching...' : 'Enter a query and click Search'}
              </div>
            ) : (
              <div className="results-list">
                {results.map((doc, idx) => (
                  <div key={idx} className="result-card">
                    <div className="result-header">
                      <span className="result-title">{doc.doc_id}</span>
                      <span className="result-score">{(doc.score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="result-content">
                      {doc.description || doc.text?.substring(0, 200) || 'No description'}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="graph-section">
          <GraphViewer graph={graph} onNodeClick={handleNodeClick} />
        </div>
      </div>
    </div>
  )
}
