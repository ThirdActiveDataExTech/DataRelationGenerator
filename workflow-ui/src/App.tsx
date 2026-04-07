import './App.css'
import WorkflowUpload from './components/WorkflowUpload'

function App() {
  return (
    <div className="app">
      <header className="header">
        <div className="logo">Active Metadata Graph - Workflow</div>
        <nav className="nav">
          <a href="/">검색</a>
          <a href="/workflow" className="active">워크플로우</a>
          <a href="/docs">문서</a>
        </nav>
      </header>
      <main className="main-container">
        <WorkflowUpload />
      </main>
    </div>
  )
}

export default App
