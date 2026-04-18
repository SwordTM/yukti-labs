import React from 'react'

export default function Header({ currentPage, onNavigate }) {
  const handleClick = (e, page) => {
    e.preventDefault()
    onNavigate(page)
  }

  return (
    <header className="header">
      <div className="header-container">
        <h1 className="logo">ML Lens</h1>
        <nav className="nav">
          <a
            href="#"
            className={`nav-link ${currentPage === 'dashboard' ? 'active' : ''}`}
            onClick={(e) => handleClick(e, 'dashboard')}
          >
            Dashboard
          </a>
          <a
            href="#"
            className={`nav-link ${currentPage === 'ingestion' ? 'active' : ''}`}
            onClick={(e) => handleClick(e, 'ingestion')}
          >
            Ingestion
          </a>
          <a
            href="#"
            className={`nav-link ${currentPage === 'sandbox' ? 'active' : ''}`}
            onClick={(e) => handleClick(e, 'sandbox')}
          >
            Sandbox
          </a>
        </nav>
      </div>
    </header>
  )
}
