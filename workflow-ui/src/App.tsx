import { Routes, Route, Link, useLocation } from 'react-router-dom'
import './App.css'
import SearchPage from './pages/SearchPage'
import WorkflowUpload from './components/WorkflowUpload'

function App() {
  const location = useLocation()

  return (
    <div className="app">
      <header className="header">
        <div className="logo">Active Metadata Graph</div>
        <nav className="nav">
          <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
            Search
          </Link>
          <Link to="/pipeline" className={location.pathname === '/pipeline' ? 'active' : ''}>
            Pipeline
          </Link>
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
            Documentation
          </a>
        </nav>
      </header>
      <main className="main-container">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/pipeline" element={<WorkflowUpload />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
