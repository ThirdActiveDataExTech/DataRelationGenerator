import { useEffect, useRef } from 'react'
import './WorkflowVisualization.css'

interface WorkflowEvent {
  step: number
  status: 'processing' | 'completed' | 'error'
  message: string
  timestamp: number
  data?: Record<string, any>
}

interface Props {
  events: WorkflowEvent[]
  isRunning: boolean
}

const WORKFLOW_STEPS = [
  { id: 1, name: '입력 데이터 수신', icon: '📥' },
  { id: 2, name: '스키마 분석', icon: '🔍' },
  { id: 3, name: 'Key-Value 추출\n& 그래프 생성', icon: '⚙️' },
  { id: 4, name: 'Value 정규화', icon: '🔄' },
  { id: 5, name: '완료', icon: '🎉' },
]

export default function WorkflowVisualization({ events, isRunning }: Props) {
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events])

  const getStepStatus = (stepId: number): 'idle' | 'processing' | 'completed' | 'error' => {
    const stepEvents = events.filter(e => e.step === stepId)
    if (stepEvents.length === 0) return 'idle'

    const lastEvent = stepEvents[stepEvents.length - 1]
    return lastEvent.status
  }

  const getStepData = (stepId: number) => {
    const stepEvents = events.filter(e => e.step === stepId)
    if (stepEvents.length === 0) return null

    const completedEvent = stepEvents.find(e => e.status === 'completed')
    return completedEvent?.data || null
  }

  return (
    <div className="workflow-visualization">
      <div className="workflow-header">
        <h2>🔄 워크플로우 진행 상태</h2>
        {isRunning && (
          <div className="running-indicator">
            <span className="pulse-dot"></span>
            실행 중
          </div>
        )}
      </div>

      <div className="workflow-steps">
        {WORKFLOW_STEPS.map((step, index) => {
          const status = getStepStatus(step.id)
          const data = getStepData(step.id)

          return (
            <div key={step.id} className="step-wrapper">
              <div className={`step-card step-${status}`}>
                <div className="step-icon">{step.icon}</div>
                <div className="step-content">
                  <div className="step-title">{step.name}</div>
                  <div className="step-status-badge">
                    {status === 'idle' && '대기 중'}
                    {status === 'processing' && (
                      <span className="processing">⏳ 진행 중...</span>
                    )}
                    {status === 'completed' && <span className="completed">✅ 완료</span>}
                    {status === 'error' && <span className="error">❌ 오류</span>}
                  </div>
                  {data && (
                    <div className="step-data">
                      {Object.entries(data).map(([key, value]) => (
                        <div key={key} className="data-item">
                          <span className="data-key">{key}:</span>
                          <span className="data-value">
                            {typeof value === 'object'
                              ? JSON.stringify(value)
                              : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              {index < WORKFLOW_STEPS.length - 1 && (
                <div className={`step-arrow ${status === 'completed' ? 'active' : ''}`}>
                  ↓
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="logs-panel">
        <h3>📋 실행 로그</h3>
        <div className="logs">
          {events.length === 0 ? (
            <div className="empty-logs">
              워크플로우를 실행하면 로그가 표시됩니다.
            </div>
          ) : (
            <>
              {events.map((event, index) => (
                <div key={index} className={`log-entry log-${event.status}`}>
                  <span className="log-time">
                    {new Date(event.timestamp * 1000).toLocaleTimeString()}
                  </span>
                  <span className="log-message">{event.message}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
