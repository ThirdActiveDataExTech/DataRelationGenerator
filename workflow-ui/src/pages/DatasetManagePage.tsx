import { useState, useEffect, useRef, useCallback } from 'react'
import './DatasetManagePage.css'

interface AppConfig {
  x_token: string
}

interface DatasetProfile {
  table_name: string
  domain: string
  record_count: number
  column_count: number
  column_names: string[]
  description: string
  table_summary: string
  keywords: string[]
  created_at: string
  kv_document_count?: number
  kv_processed?: boolean
  raw_available?: boolean
}

interface SyncItem {
  table_name: string
  in_opensearch: boolean
  in_neptune: boolean
  synced: boolean
}

interface Relation {
  table_a: string
  table_b: string
  score: number
  method: string
  relation_type: string
  created_at: string
}

interface Stats {
  dataset_count: number
  relation_count: number
  domain_counts: Record<string, number>
  method_counts: Record<string, number>
}

interface PipelineResult {
  table_a: string
  table_b: string
  tier1: { relation_type: string }[]
  tier2_candidates: { score: number }[]
  tier3_verified: { relation_type: string; confidence: number; reason: string }[]
  saved_relations: { relation_type: string; method: string; score: number }[]
}

interface PipelineJob {
  status: 'running' | 'done' | 'error'
  started_at: string
  finished_at?: string
  results?: PipelineResult[]
  total_saved?: number
  error?: string
}

const ACCEPTED_FORMATS = '.json,.jsonl,.csv'

const METHOD_LABEL: Record<string, string> = { rule: 'Tier 1 규칙', embedding: 'Tier 2 임베딩', llm: 'Tier 3 LLM' }
const METHOD_CLASS: Record<string, string> = { rule: 'dm-tier1', embedding: 'dm-tier2', llm: 'dm-tier3' }

export default function DatasetManagePage() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [datasets, setDatasets] = useState<DatasetProfile[]>([])
  const [syncItems, setSyncItems] = useState<SyncItem[]>([])
  const [syncLoading, setSyncLoading] = useState(false)
  const [relations, setRelations] = useState<Relation[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [expandedDataset, setExpandedDataset] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [tier2Threshold, setTier2Threshold] = useState(0.4)
  const [tier3Threshold, setTier3Threshold] = useState(0.7)

  // upload form
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [tableName, setTableName] = useState('')
  const [domain, setDomain] = useState('medical')
  const fileInputRef = useRef<HTMLInputElement>(null)

  // pipeline job
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<PipelineJob | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // KV 그래프 처리 작업 (데이터셋별)
  const [kvBusy, setKvBusy] = useState<Record<string, boolean>>({})
  const kvPollRef = useRef<Record<string, ReturnType<typeof setInterval>>>({})

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(setConfig).catch(() => {})
  }, [])

  useEffect(() => {
    if (config) {
      loadAll()
    }
  }, [config])

  useEffect(() => {
    if (!jobId || !config) return
    pollRef.current = setInterval(async () => {
      const res = await fetch(`/dataset/relation/pipeline/${jobId}`, { headers: { 'x-token': config.x_token } })
      const data = await res.json()
      const j: PipelineJob = data.result
      setJob(j)
      if (j.status !== 'running') {
        clearInterval(pollRef.current!)
        pollRef.current = null
        loadAll()
      }
    }, 2000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [jobId, config])

  // 언마운트 시 KV 폴링 정리
  useEffect(() => {
    const timers = kvPollRef.current
    return () => { Object.values(timers).forEach(clearInterval) }
  }, [])

  const headers = useCallback(() => ({ 'x-token': config!.x_token }), [config])

  const loadAll = async () => {
    await Promise.all([loadDatasets(), loadRelations(), loadStats()])
  }

  const loadDatasets = async () => {
    try {
      const res = await fetch('/dataset/list', { headers: headers() })
      const data = await res.json()
      setDatasets(data.result ?? [])
    } catch { setDatasets([]) }
  }

  const loadRelations = async () => {
    try {
      const res = await fetch('/dataset/relations', { headers: headers() })
      const data = await res.json()
      setRelations(data.result ?? [])
    } catch { setRelations([]) }
  }

  const loadStats = async () => {
    try {
      const res = await fetch('/dataset/stats', { headers: headers() })
      const data = await res.json()
      setStats(data.result ?? null)
    } catch { setStats(null) }
  }

  const loadSyncStatus = async () => {
    setSyncLoading(true)
    try {
      const res = await fetch('/dataset/sync-status', { headers: headers() })
      const data = await res.json()
      setSyncItems(data.result ?? [])
    } catch { setSyncItems([]) }
    finally { setSyncLoading(false) }
  }

  const handleUpload = async () => {
    if (!uploadFile || !tableName.trim() || !config) return
    setUploading(true)
    setError(null)
    const form = new FormData()
    form.append('file', uploadFile)
    form.append('table_name', tableName.trim())
    form.append('domain', domain)
    try {
      const res = await fetch('/dataset/ingest', { method: 'POST', headers: headers(), body: form })
      const text = await res.text()
      let data: Record<string, unknown> = {}
      try { data = JSON.parse(text) } catch {
        throw new Error(`서버 응답 파싱 실패 (HTTP ${res.status}): ${text.slice(0, 200)}`)
      }
      if (!res.ok || (data.result as Record<string, unknown>)?.error)
        throw new Error((data.description as string) ?? '업로드 실패')
      await loadAll()
      setUploadFile(null); setTableName('')
      if (fileInputRef.current) fileInputRef.current.value = ''
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '업로드 실패')
    } finally { setUploading(false) }
  }

  const handleBuildGraph = async (ds: DatasetProfile) => {
    if (!config) return
    const name = ds.table_name
    if (!ds.raw_available) {
      setError(
        `'${name}'은(는) 보관된 원본 파일이 없어 재처리할 수 없습니다. ` +
        `아래 '데이터셋 등록'에서 같은 테이블명(${name})으로 파일을 다시 업로드하면 KV 처리가 가능해집니다. ` +
        `(원본 저장 기능 도입 이전에 등록되었거나, 파일 없이 등록된 데이터셋입니다.)`
      )
      return
    }
    setKvBusy(prev => ({ ...prev, [name]: true }))
    try {
      const res = await fetch(`/dataset/${encodeURIComponent(name)}/build-graph`, {
        method: 'POST',
        headers: headers(),
      })
      const data = await res.json()
      const kvJobId = data.result?.job_id
      if (!kvJobId) throw new Error(data.description ?? 'KV 처리 시작 실패')

      // 작업 상태 폴링
      kvPollRef.current[name] = setInterval(async () => {
        const r = await fetch(`/dataset/build-graph/${kvJobId}`, { headers: headers() })
        const d = await r.json()
        const status = d.result?.status
        if (status && status !== 'running') {
          clearInterval(kvPollRef.current[name])
          delete kvPollRef.current[name]
          setKvBusy(prev => ({ ...prev, [name]: false }))
          if (status === 'error') setError(`'${name}' KV 처리 실패: ${d.result?.error ?? ''}`)
          await loadDatasets()
        }
      }, 2000)
    } catch (e: unknown) {
      setKvBusy(prev => ({ ...prev, [name]: false }))
      setError(e instanceof Error ? e.message : 'KV 처리 실패')
    }
  }

  const handleDelete = async (name: string) => {
    if (!config || !confirm(`'${name}' 데이터셋을 삭제할까요?`)) return
    await fetch(`/dataset/${encodeURIComponent(name)}`, { method: 'DELETE', headers: headers() })
    await loadAll()
  }

  const handleRunPipeline = async () => {
    if (!config || datasets.length < 2) return
    setJob({ status: 'running', started_at: new Date().toISOString() })
    setJobId(null)
    try {
      const res = await fetch('/dataset/relation/pipeline', {
        method: 'POST',
        headers: { ...headers(), 'Content-Type': 'application/json' },
        body: JSON.stringify({
          table_names: datasets.map(d => d.table_name),
          tier2_threshold: tier2Threshold,
          tier3_confidence_threshold: tier3Threshold,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.description ?? '파이프라인 시작 실패')
      setJobId(data.result.job_id)
    } catch (e: unknown) {
      setJob({ status: 'error', started_at: new Date().toISOString(), error: e instanceof Error ? e.message : '실패' })
    }
  }

  const syncBadge = (name: string) => {
    const item = syncItems.find(s => s.table_name === name)
    if (!item) return null
    if (item.synced) return <span className="dm-sync-badge dm-sync-ok">Neptune ✓</span>
    if (item.in_opensearch && !item.in_neptune) return <span className="dm-sync-badge dm-sync-warn">Neptune 미등록</span>
    return <span className="dm-sync-badge dm-sync-err">불일치</span>
  }

  const fileLabel = uploadFile
    ? `${uploadFile.name} (${(uploadFile.size / 1024).toFixed(1)} KB)`
    : '파일 선택 (JSON / JSONL / CSV)'

  return (
    <div className="dataset-manage-page">

      {/* ── 현황 요약 ── */}
      {stats && (
        <section className="dm-stats-row">
          <div className="dm-stat-card">
            <span className="dm-stat-value">{stats.dataset_count}</span>
            <span className="dm-stat-label">등록된 데이터셋</span>
          </div>
          <div className="dm-stat-card">
            <span className="dm-stat-value">{stats.relation_count}</span>
            <span className="dm-stat-label">생성된 관계</span>
          </div>
          <div className="dm-stat-card">
            <span className="dm-stat-value">{stats.method_counts['rule'] ?? 0}</span>
            <span className="dm-stat-label">Tier 1 (규칙)</span>
          </div>
          <div className="dm-stat-card">
            <span className="dm-stat-value">{stats.method_counts['embedding'] ?? 0}</span>
            <span className="dm-stat-label">Tier 2 (임베딩)</span>
          </div>
          <div className="dm-stat-card">
            <span className="dm-stat-value">{stats.method_counts['llm'] ?? 0}</span>
            <span className="dm-stat-label">Tier 3 (LLM)</span>
          </div>
          {Object.entries(stats.domain_counts).map(([d, n]) => (
            <div key={d} className="dm-stat-card dm-stat-domain">
              <span className="dm-stat-value">{n}</span>
              <span className="dm-stat-label">{d}</span>
            </div>
          ))}
        </section>
      )}

      {/* ── 등록된 데이터셋 + 관계 현황 (2컬럼) ── */}
      <div className="dm-overview-grid">

        {/* 데이터셋 목록 */}
        <section className="dm-card">
          <div className="dm-section-header">
            <h2 className="dm-section-title">
              등록된 데이터셋 <span className="dm-badge">{datasets.length}</span>
            </h2>
            <div className="dm-section-actions">
              <button className="dm-btn dm-btn-ghost" onClick={loadSyncStatus} disabled={syncLoading}>
                {syncLoading ? '확인 중…' : '동기화 확인'}
              </button>
              <button className="dm-btn dm-btn-ghost" onClick={loadAll}>새로고침</button>
            </div>
          </div>

          {datasets.length === 0 ? (
            <p className="dm-empty">등록된 데이터셋이 없습니다.</p>
          ) : (
            <ul className="dm-dataset-list">
              {datasets.map(ds => (
                <li key={ds.table_name} className="dm-dataset-item">
                  <div
                    className="dm-dataset-summary"
                    onClick={() => setExpandedDataset(expandedDataset === ds.table_name ? null : ds.table_name)}
                  >
                    <div className="dm-dataset-name">{ds.table_name}</div>
                    <span className="dm-tag">{ds.domain}</span>
                    <span className="dm-meta">{ds.record_count.toLocaleString()}건 · {ds.column_count}컬럼</span>
                    {syncBadge(ds.table_name)}
                    {kvBusy[ds.table_name] ? (
                      <span className="dm-kv-badge dm-kv-running"><span className="dm-spinner" />KV 처리 중…</span>
                    ) : ds.kv_processed ? (
                      <span className="dm-kv-badge dm-kv-ok">KV 처리됨 ({ds.kv_document_count}건)</span>
                    ) : (
                      <span className="dm-kv-badge dm-kv-none">KV 미처리</span>
                    )}
                    <div className="dm-dataset-actions">
                      {!ds.kv_processed && (
                        <button
                          className="dm-btn dm-btn-ghost"
                          disabled={kvBusy[ds.table_name]}
                          title={ds.raw_available ? 'Key-Value 그래프 생성' : '원본 파일 없음 — 클릭하면 안내 표시'}
                          onClick={e => { e.stopPropagation(); handleBuildGraph(ds) }}
                        >{kvBusy[ds.table_name] ? '처리 중…' : 'KV 처리'}</button>
                      )}
                      <button
                        className="dm-btn dm-btn-danger-ghost"
                        onClick={e => { e.stopPropagation(); handleDelete(ds.table_name) }}
                      >삭제</button>
                    </div>
                  </div>
                  {expandedDataset === ds.table_name && (
                    <div className="dm-dataset-detail">
                      <p className="dm-detail-desc">{ds.description || ds.table_summary}</p>
                      {ds.keywords?.length > 0 && (
                        <div className="dm-keywords">
                          {ds.keywords.map(kw => <span key={kw} className="dm-keyword-chip">{kw}</span>)}
                        </div>
                      )}
                      <div className="dm-columns">
                        {ds.column_names.map(c => <span key={c} className="dm-col-chip">{c}</span>)}
                      </div>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* 관계 목록 */}
        <section className="dm-card">
          <div className="dm-section-header">
            <h2 className="dm-section-title">
              생성된 관계 <span className="dm-badge">{relations.length}</span>
            </h2>
            <button className="dm-btn dm-btn-ghost" onClick={loadRelations}>새로고침</button>
          </div>

          {relations.length === 0 ? (
            <p className="dm-empty">저장된 관계가 없습니다.<br />파이프라인을 실행해 관계를 생성하세요.</p>
          ) : (
            <ul className="dm-relation-list">
              {relations.map((r, i) => (
                <li key={i} className="dm-relation-item">
                  <div className="dm-relation-tables">
                    <span className="dm-relation-table">{r.table_a}</span>
                    <span className="dm-relation-arrow">↔</span>
                    <span className="dm-relation-table">{r.table_b}</span>
                  </div>
                  <div className="dm-relation-meta">
                    <span className={`dm-tier-badge ${METHOD_CLASS[r.method] ?? ''}`}>
                      {METHOD_LABEL[r.method] ?? r.method}
                    </span>
                    <span className="dm-relation-type">{r.relation_type}</span>
                    <span className="dm-relation-score">score {r.score.toFixed(2)}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      {/* ── 업로드 ── */}
      <section className="dm-card">
        <h2 className="dm-section-title">데이터셋 등록</h2>
        <p className="dm-desc">JSON 배열, JSONL(줄-구분), CSV 파일을 지원합니다. 설명과 키워드는 LLM이 자동 생성합니다.</p>

        <div className="dm-upload-grid">
          <div className="dm-field dm-field-file">
            <label>파일 *</label>
            <div
              className={`dm-file-drop ${uploadFile ? 'dm-file-drop--has-file' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) setUploadFile(f) }}
            >
              <span className="dm-file-label">{fileLabel}</span>
              <span className="dm-file-hint">.json .jsonl .csv</span>
            </div>
            <input ref={fileInputRef} type="file" accept={ACCEPTED_FORMATS} style={{ display: 'none' }}
              onChange={e => setUploadFile(e.target.files?.[0] ?? null)} />
          </div>

          <div className="dm-field">
            <label>테이블명 *</label>
            <input type="text" placeholder="예: patient_info" value={tableName}
              onChange={e => setTableName(e.target.value)} />
          </div>

          <div className="dm-field">
            <label>도메인</label>
            <select value={domain} onChange={e => setDomain(e.target.value)}>
              <option value="medical">medical</option>
              <option value="broadcast">broadcast</option>
              <option value="general">general</option>
            </select>
          </div>


        </div>

        {error && <p className="dm-error">{error}</p>}
        <button className="dm-btn dm-btn-primary"
          disabled={!uploadFile || !tableName.trim() || uploading} onClick={handleUpload}>
          {uploading ? '등록 중…' : '등록'}
        </button>
      </section>

      {/* ── 관계 생성 파이프라인 ── */}
      <section className="dm-card">
        <h2 className="dm-section-title">관계 생성 파이프라인</h2>
        <p className="dm-desc">
          등록된 데이터셋들 간의 관계를 자동으로 분석하여 Neptune에 저장합니다.<br />
          <strong>Tier 1</strong>(동일 컬럼명) → <strong>Tier 2</strong>(description 임베딩 + 키워드 임베딩 결합) → <strong>Tier 3</strong>(LLM 검증)
        </p>

        <div className="dm-threshold-row">
          <label>Tier 2 임계값
            <input type="number" min={0} max={1} step={0.05} value={tier2Threshold}
              onChange={e => setTier2Threshold(Number(e.target.value))} />
          </label>
          <label>Tier 3 확신도 임계값
            <input type="number" min={0} max={1} step={0.05} value={tier3Threshold}
              onChange={e => setTier3Threshold(Number(e.target.value))} />
          </label>
        </div>

        <button className="dm-btn dm-btn-primary"
          disabled={datasets.length < 2 || job?.status === 'running'} onClick={handleRunPipeline}>
          {job?.status === 'running' ? '실행 중…' : `관계 생성 실행 (${datasets.length}개 테이블)`}
        </button>

        {datasets.length < 2 && <p className="dm-hint-inline">관계 생성에는 데이터셋이 2개 이상 필요합니다.</p>}

        {job && (
          <div className={`dm-job-status dm-job-${job.status}`}>
            {job.status === 'running' && <span className="dm-spinner" />}
            <span className="dm-job-label">
              {job.status === 'running' && '파이프라인 실행 중 (2초마다 갱신)…'}
              {job.status === 'done' && `완료: ${job.total_saved ?? 0}개 관계 저장됨`}
              {job.status === 'error' && `오류: ${job.error}`}
            </span>
          </div>
        )}

        {job?.status === 'done' && job.results && job.results.length > 0 && (
          <div className="dm-pipeline-results">
            <div className="dm-pipeline-summary">
              총 <strong>{job.results.length}</strong>쌍 처리,&nbsp;
              <strong>{job.total_saved}</strong>개 관계 저장됨
            </div>
            {job.results.map((r, i) => (
              <div key={i} className={`dm-pair-card ${r.saved_relations.length > 0 ? 'dm-pair-saved' : ''}`}>
                <div className="dm-pair-header">
                  <span className="dm-pair-name">{r.table_a}</span>
                  <span className="dm-pair-arrow">↔</span>
                  <span className="dm-pair-name">{r.table_b}</span>
                </div>
                <div className="dm-pair-tiers">
                  {r.tier1.length > 0 && <span className="dm-tier-badge dm-tier1">Tier1: {r.tier1[0].relation_type}</span>}
                  {r.tier2_candidates.length > 0 && <span className="dm-tier-badge dm-tier2">Tier2: {r.tier2_candidates[0].score.toFixed(3)}</span>}
                  {r.tier3_verified.length > 0 && <span className="dm-tier-badge dm-tier3">Tier3: {r.tier3_verified[0].relation_type} ({r.tier3_verified[0].confidence.toFixed(2)})</span>}
                  {r.saved_relations.length === 0 && <span className="dm-tier-badge dm-tier-none">관계 없음</span>}
                </div>
                {r.tier3_verified.length > 0 && <p className="dm-pair-reason">{r.tier3_verified[0].reason}</p>}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
