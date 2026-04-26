import React, { useState, useRef, useEffect } from 'react'
import LoadingBar from './LoadingBar'
import LoadingDots from './LoadingDots'
import MarkdownMessage from './MarkdownMessage'

const API_BASE = 'http://localhost:8000'

export default function ChatPanel({ onAction, manifest, expanded, setExpanded }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: "Hi! I'm here to help you understand the model components in the sandbox. Ask me anything about how they work or why the model behaves the way it does.",
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMsg = { id: Date.now(), role: 'user', content: text }
    const nextMessages = [...messages, userMsg]

    setMessages(nextMessages)
    setInput('')
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: nextMessages
            .filter((m) => m.role === 'user' || m.role === 'assistant')
            .map(({ role, content }) => ({ role, content })),
          manifest // Pass current manifest to the agent!
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }))
        throw new Error(err.detail || 'Request failed')
      }

      const data = await res.json()
      
      // If the backend returns an action, append it to the message object
      setMessages((prev) => [
        ...prev,
        { 
          id: Date.now() + 1, 
          role: 'assistant', 
          content: data.content,
          action: data.action // Agentic action block
        },
      ])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <aside className={`chat-panel ${expanded ? 'expanded' : 'collapsed'}`}>
      <LoadingBar loading={loading} label="Thinking…" />

      <div className="chat-panel-header">
        <div className="chat-panel-title-wrap">
          <span className="chat-panel-title">Model Chat</span>
          <span className="badge badge-completed">Transformer</span>
        </div>
        <button 
          className="chat-toggle-btn" 
          onClick={() => setExpanded(!expanded)}
          title={expanded ? "Collapse Chat" : "Expand Chat"}
        >
          {expanded ? '«' : '»'}
        </button>
      </div>

      <div className="chat-content-wrap">
        <div className="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`chat-bubble-container ${msg.role}`}>
              <div className={`chat-bubble ${msg.role}`}>
                {msg.role === 'assistant'
                  ? <MarkdownMessage content={msg.content} />
                  : <p>{msg.content}</p>
                }
              </div>
              
              {msg.action && (
                <div className="chat-action-block">
                  <div className="action-block-header">
                    <div className="action-block-icon">🧬</div>
                    <div className="action-block-info">
                      <span className="action-block-title">Proposed Modification</span>
                      <span className="action-block-subtitle">{msg.action.payload.name}</span>
                    </div>
                  </div>
                  
                  {msg.action.payload.depends_on && (
                    <div className="action-block-wiring">
                      <span className="wiring-label">Proposed Connections:</span>
                      <div className="wiring-path">
                        {msg.action.payload.depends_on.join(', ') || 'Root'} ➡ New Block
                      </div>
                    </div>
                  )}

                  <div className="action-block-footer">
                    <button 
                      className="btn-action-primary"
                      onClick={() => {
                        onAction(msg.action)
                        setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, action: null, content: m.content + " (Applied)" } : m))
                      }}
                    >
                      Apply to Sandbox
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="chat-bubble assistant">
              <LoadingDots />
            </div>
          )}
          {error && (
            <div className="chat-bubble assistant chat-bubble-error">
              <p>Something went wrong: {error}</p>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="chat-input-bar">
          <textarea
            className="chat-textarea"
            rows={2}
            placeholder="Ask about model behaviour…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            disabled={loading}
          />
          <button
            className="btn-primary chat-send-btn"
            onClick={send}
            disabled={loading || !input.trim()}
          >
            Send
          </button>
        </div>
      </div>
      
      {!expanded && (
        <div className="collapsed-chat-indicator" onClick={() => setExpanded(true)}>
          <span className="collapsed-chat-label">Chat</span>
        </div>
      )}
    </aside>
  )
}
