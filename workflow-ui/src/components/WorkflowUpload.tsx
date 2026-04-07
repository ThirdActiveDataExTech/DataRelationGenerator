import { useState, useEffect } from 'react'
import WorkflowVisualization from './WorkflowVisualization'
import './WorkflowUpload.css'

interface WorkflowEvent {
  step: number
  status: 'processing' | 'completed' | 'error'
  message: string
  timestamp: number
  data?: Record<string, any>
}

interface AppConfig {
  x_token: string
}

export default function WorkflowUpload() {
  const [file, setFile] = useState<File | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [events, setEvents] = useState<WorkflowEvent[]>([])
  const [autoDetectSchema, setAutoDetectSchema] = useState(true)
  const [sampleSize, setSampleSize] = useState(5)
  const [similarityThreshold, setSimilarityThreshold] = useState(0.85)
  const [minClusterSize, setMinClusterSize] = useState(2)
  const [config, setConfig] = useState<AppConfig | null>(null)

  useEffect(() => {
    // Load config on mount
    fetch('/api/config')
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => console.error('Failed to load config:', err))
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async () => {
    if (!file || !config) {
      if (!config) {
        alert('설정을 불러오는 중입니다. 잠시 후 다시 시도하세요.')
      }
      return
    }

    setIsRunning(true)
    setEvents([])

    const formData = new FormData()
    formData.append('file', file)
    formData.append('auto_detect_schema', String(autoDetectSchema))
    formData.append('sample_size', String(sampleSize))
    formData.append('similarity_threshold', String(similarityThreshold))
    formData.append('min_cluster_size', String(minClusterSize))

    try {
      const response = await fetch('/workflow/execute', {
        method: 'POST',
        headers: {
          'x-token': config.x_token,
        },
        body: formData,
      })

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('Stream reader not available')
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.substring(6)
            const event: WorkflowEvent = JSON.parse(data)
            setEvents(prev => [...prev, event])
          }
        }
      }
    } catch (error) {
      console.error('Workflow error:', error)
      setEvents(prev => [
        ...prev,
        {
          step: 0,
          status: 'error',
          message: `❌ 오류 발생: ${error}`,
          timestamp: Date.now(),
        },
      ])
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="workflow-container">
      <div className="upload-panel">
        <h2>📁 파일 업로드</h2>

        <div className="file-input-wrapper">
          <input
            type="file"
            accept=".json"
            onChange={handleFileChange}
            disabled={isRunning}
          />
          {file && <p className="file-name">{file.name}</p>}
        </div>

        <div className="options">
          <h3>⚙️ 옵션</h3>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={autoDetectSchema}
              onChange={e => setAutoDetectSchema(e.target.checked)}
              disabled={isRunning}
            />
            스키마 자동 감지
          </label>

          {autoDetectSchema && (
            <div className="option-group">
              <label>
                샘플 크기:
                <input
                  type="number"
                  value={sampleSize}
                  onChange={e => setSampleSize(Number(e.target.value))}
                  min={1}
                  max={20}
                  disabled={isRunning}
                />
              </label>
            </div>
          )}

          <div className="option-group">
            <label>
              유사도 임계값:
              <input
                type="number"
                value={similarityThreshold}
                onChange={e => setSimilarityThreshold(Number(e.target.value))}
                min={0}
                max={1}
                step={0.05}
                disabled={isRunning}
              />
            </label>
          </div>

          <div className="option-group">
            <label>
              최소 클러스터 크기:
              <input
                type="number"
                value={minClusterSize}
                onChange={e => setMinClusterSize(Number(e.target.value))}
                min={1}
                max={10}
                disabled={isRunning}
              />
            </label>
          </div>
        </div>

        <button
          className="submit-button"
          onClick={handleSubmit}
          disabled={!file || isRunning || !config}
        >
          {!config ? '⏳ 설정 로딩 중...' : isRunning ? '⏳ 실행 중...' : '🚀 워크플로우 실행'}
        </button>
      </div>

      <div className="visualization-panel">
        <WorkflowVisualization events={events} isRunning={isRunning} />
      </div>
    </div>
  )
}
