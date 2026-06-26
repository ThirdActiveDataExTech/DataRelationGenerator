import { Routes, Route, Link, useLocation } from 'react-router-dom'
import './App.css'
import SearchPage from './pages/SearchPage'
import RelationDiscoveryPage from './pages/RelationDiscoveryPage'
import DatasetManagePage from './pages/DatasetManagePage'

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
          <Link to="/datasets" className={location.pathname === '/datasets' ? 'active' : ''}>
            데이터셋 관리
          </Link>
          <Link to="/relation-discovery" className={location.pathname === '/relation-discovery' ? 'active' : ''}>
            Relation Discovery
          </Link>
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
            API Docs
          </a>
        </nav>
      </header>
      <main className="main-container">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/datasets" element={<DatasetManagePage />} />
          <Route path="/relation-discovery" element={<RelationDiscoveryPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
