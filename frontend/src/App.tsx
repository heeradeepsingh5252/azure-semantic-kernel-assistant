import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent, FormEvent } from "react";
import "./App.css";

type Citation = {
  number: number;
  source: string;
  page?: number | null;
  chunk_id?: string;
  text?: string;
};

type ChatResponse = {
  session_id: string;
  answer: string;
  citations?: Citation[];
  pii_detected?: string[];
};

type Message = {
  id: string;
  role: "assistant" | "user";
  text: string;
  citations?: Citation[];
  pii?: string[];
  failed?: boolean;
};

type DocumentSource = {
  filename: string;
  pages: number;
  chunk_count: number;
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const PUBLIC_DEMO_READ_ONLY = import.meta.env.VITE_PUBLIC_DEMO_READ_ONLY === "true";

async function api(path: string, init?: RequestInit) {
  return fetch(`${API_BASE}${path}`, init);
}

async function explainError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.detail ?? body.message ?? JSON.stringify(body);
  } catch {
    return `Request failed with HTTP ${response.status}`;
  }
}

function makeSessionId() {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `session-${Date.now()}`;
}

function WelcomeMessage() {
  return (
    <div className="welcome-state">
      <div className="welcome-orb">✳</div>
      <p className="eyebrow">YOUR DOCUMENTS, UNDERSTOOD</p>
      <h2>What would you like to know?</h2>
      <p className="welcome-copy">
        Ask questions about your indexed documents. Answers include source
        evidence so you can check where the information came from.
      </p>
      <div className="suggestion-grid">
        <button
          type="button"
          className="suggestion"
          onClick={() =>
            window.dispatchEvent(
              new CustomEvent("semantic-prompt", {
                detail: "Summarize the key information in my documents.",
              }),
            )
          }
        >
          <span>✧</span>
          Summarize my documents
          <span className="suggestion-arrow">↗</span>
        </button>
        <button
          type="button"
          className="suggestion"
          onClick={() =>
            window.dispatchEvent(
              new CustomEvent("semantic-prompt", {
                detail: "What are the most important facts in my documents?",
              }),
            )
          }
        >
          <span>◈</span>
          Find important facts
          <span className="suggestion-arrow">↗</span>
        </button>
      </div>
    </div>
  );
}

function App() {
  const [health, setHealth] = useState<"checking" | "online" | "offline">(
    "checking",
  );
  const [sources, setSources] = useState<string[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(makeSessionId);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [busySource, setBusySource] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  const loadDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    try {
      const response = await api("/documents");
      if (!response.ok) throw new Error(await explainError(response));
      const body: { sources: string[] } = await response.json();
      setSources(body.sources ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load documents.");
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;

    void api("/health")
      .then((response) => {
        if (!response.ok) throw new Error("Health check failed");
        if (active) setHealth("online");
      })
      .catch(() => {
        if (active) setHealth("offline");
      });

    void loadDocuments();

    const handlePrompt = (event: Event) => {
      const customEvent = event as CustomEvent<string>;
      setInput(customEvent.detail);
    };

    window.addEventListener("semantic-prompt", handlePrompt);

    return () => {
      active = false;
      window.removeEventListener("semantic-prompt", handlePrompt);
    };
  }, [loadDocuments]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  async function uploadFile(file?: File) {
    if (!file) return;
    if (PUBLIC_DEMO_READ_ONLY) {
      setError("Uploads are disabled in the public read-only demo.");
      return;
    }

    const validExtension = /\.(pdf|txt)$/i.test(file.name);
    if (!validExtension) {
      setError("Choose a PDF or TXT file.");
      return;
    }

    setUploading(true);
    setError("");
    setNotice("");

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await api("/ingest", {
        method: "POST",
        body: form,
      });

      if (!response.ok) throw new Error(await explainError(response));

      const result: DocumentSource = await response.json();
      setNotice(
        `${result.filename} indexed successfully · ${result.chunk_count} chunk${result.chunk_count === 1 ? "" : "s"}`,
      );
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Document upload failed.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function deleteDocument(source: string) {
    if (PUBLIC_DEMO_READ_ONLY) {
      setError("Document removal is disabled in the public read-only demo.");
      return;
    }
    if (!window.confirm(`Remove "${source}" from the document index?`)) return;

    setBusySource(source);
    setError("");
    setNotice("");

    try {
      const response = await api(`/documents/${encodeURIComponent(source)}`, {
        method: "DELETE",
      });

      if (!response.ok) throw new Error(await explainError(response));

      setNotice(`${source} was removed.`);
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Document removal failed.");
    } finally {
      setBusySource("");
    }
  }

  async function sendMessage(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const question = input.trim();

    if (!question || sending) return;

    setInput("");
    setSending(true);
    setError("");

    setMessages((previous) => [
      ...previous,
      { id: `${Date.now()}-user`, role: "user", text: question },
    ]);

    try {
      const response = await api("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          question,
          top_k: 5,
        }),
      });

      if (!response.ok) throw new Error(await explainError(response));

      const result: ChatResponse = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          id: `${Date.now()}-assistant`,
          role: "assistant",
          text: result.answer,
          citations: result.citations ?? [],
          pii: result.pii_detected ?? [],
        },
      ]);
    } catch (err) {
      setMessages((previous) => [
        ...previous,
        {
          id: `${Date.now()}-error`,
          role: "assistant",
          text: err instanceof Error ? err.message : "The chat request failed.",
          failed: true,
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    void uploadFile(event.dataTransfer.files?.[0]);
  }

  function startNewChat() {
    setSessionId(makeSessionId());
    setMessages([]);
    setInput("");
    setError("");
    setNotice("Started a new conversation.");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Semantic AI home">
          <span className="brand-mark">S<span>.</span></span>
          <span className="brand-wordmark">semantic<span>ai</span></span>
        </a>

        <div className="workspace-label">WORKSPACE</div>

        <button className="nav-item active" type="button">
          <span className="nav-icon">◫</span> Document intelligence
        </button>

        <button className="nav-item" type="button" onClick={startNewChat}>
          <span className="nav-icon">✳</span> New conversation
          <span className="nav-shortcut">↵</span>
        </button>

        <div className="sidebar-section-title">
          <span>YOUR KNOWLEDGE</span>
          <span className="count-pill">{sources.length}</span>
        </div>

        <div className="sidebar-documents">
          {documentsLoading ? (
            <p className="sidebar-muted">Loading documents…</p>
          ) : sources.length === 0 ? (
            <p className="sidebar-muted">Your library is empty.</p>
          ) : (
            sources.map((source) => (
              <div className="sidebar-document" key={source} title={source}>
                <span className="file-mini">TXT</span>
                <span className="sidebar-document-name">{source}</span>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-bottom">
          <div className={`connection-card ${health}`}>
            <span className="status-dot" />
            <div>
              <strong>
                {health === "online"
                  ? "API connected"
                  : health === "checking"
                    ? "Checking connection"
                    : "API unavailable"}
              </strong>
              <span>
                {health === "online"
                  ? "Ready for requests"
                  : health === "checking"
                    ? "Connecting to backend…"
                    : "Check FastAPI on port 8000"}
              </span>
            </div>
          </div>
          <div className="sidebar-footer">
            <span className="azure-mark">◆</span> Powered by Azure AI
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <strong>Document intelligence</strong>
          </div>
          <div className="topbar-right">
            <span className="secure-label"><span>◆</span> Grounded AI</span>
            <button className="new-chat-button" onClick={startNewChat} type="button">
              <span>＋</span> New chat
            </button>
          </div>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> AI KNOWLEDGE WORKSPACE</div>
              <h1>Make your documents <span>talk.</span></h1>
              <p>
                Ask questions, uncover insights, and trace every answer back to
                its source.
              </p>
            </div>
            <div className="model-badge"><span className="model-spark">✳</span> Semantic Kernel <span className="badge-divider" /> Azure AI</div>
          </section>

          {notice && (
            <div className="notice-banner" role="status">
              <span>✓</span> {notice}
              <button type="button" aria-label="Dismiss notice" onClick={() => setNotice("")}>×</button>
            </div>
          )}

          {error && (
            <div className="error-banner" role="alert">
              <span>!</span> {error}
              <button type="button" aria-label="Dismiss error" onClick={() => setError("")}>×</button>
            </div>
          )}

          <section className="workspace-grid">
            <div className="left-column">
              <article className="panel upload-panel">
                <div className="panel-heading">
                  <div className="panel-icon violet">↥</div>
                  <div>
                    <h2>{PUBLIC_DEMO_READ_ONLY ? "Demo access" : "Add knowledge"}</h2>
                    <p>{PUBLIC_DEMO_READ_ONLY ? "Explore the indexed demo documents" : "Upload documents to your AI workspace"}</p>
                  </div>
                </div>

                {PUBLIC_DEMO_READ_ONLY && (
                  <div className="read-only-card">
                    <span className="read-only-icon">⌑</span>
                    <strong>Public demo · read-only</strong>
                    <p>Explore synthetic documents and cited answers. Uploads and removal are disabled.</p>
                  </div>
                )}
                <div
                  className={`dropzone ${dragging ? "dragging" : ""} ${uploading ? "uploading" : ""} ${PUBLIC_DEMO_READ_ONLY ? "readonly-hidden" : ""}`}
                  onDragOver={(event) => {
                    event.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={handleDrop}
                >
                  <input
                    ref={fileInputRef}
                    className="visually-hidden"
                    type="file"
                    accept=".pdf,.txt,application/pdf,text/plain"
                    aria-label="Choose a PDF or TXT document"
                    onChange={(event) => void uploadFile(event.target.files?.[0])}
                    disabled={uploading || PUBLIC_DEMO_READ_ONLY}
                  />
                  <div className="upload-symbol">{uploading ? "◌" : "↑"}</div>
                  <strong>{uploading ? "Processing document…" : "Drop your files here"}</strong>
                  <p>or select from your computer</p>
                  <button
                    type="button"
                    className="browse-button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                  >
                    {uploading ? "Please wait…" : "Browse files"}
                  </button>
                  <span className="file-hint">PDF or TXT <span>·</span> One file at a time</span>
                </div>

                <div className="privacy-note">
                  <span>⌑</span>
                  {PUBLIC_DEMO_READ_ONLY ? "Uploads and removal are disabled in this public demo." : "Documents are indexed for retrieval by your configured API."}
                </div>
              </article>

              <article className="panel library-panel">
                <div className="library-heading">
                  <div className="panel-heading">
                    <div className="panel-icon mint">▤</div>
                    <div>
                      <h2>Knowledge library</h2>
                      <p>Your indexed document sources</p>
                    </div>
                  </div>
                  <button
                    className="icon-button"
                    type="button"
                    title="Refresh document list"
                    aria-label="Refresh document list"
                    onClick={() => void loadDocuments()}
                  >
                    ↻
                  </button>
                </div>

                <div className="library-summary">
                  <span>INDEXED SOURCES</span>
                  <strong>{sources.length.toString().padStart(2, "0")}</strong>
                </div>

                <div className="document-list">
                  {documentsLoading ? (
                    <div className="empty-library">Loading your library…</div>
                  ) : sources.length === 0 ? (
                    <div className="empty-library">
                      <span>▤</span>
                      <strong>No documents yet</strong>
                      <p>Upload a PDF or TXT file to get started.</p>
                    </div>
                  ) : (
                    sources.map((source) => (
                      <div className="document-row" key={source}>
                        <div className="document-type">{/\.pdf$/i.test(source) ? "PDF" : "TXT"}</div>
                        <div className="document-info">
                          <strong title={source}>{source}</strong>
                          <span>Available for retrieval</span>
                        </div>
                        <button
                          className={`remove-document ${PUBLIC_DEMO_READ_ONLY ? "readonly-hidden" : ""}`}
                          type="button"
                          aria-label={`Remove ${source}`}
                          title="Remove document"
                          disabled={busySource === source}
                          onClick={() => void deleteDocument(source)}
                        >
                          {busySource === source ? "…" : "×"}
                        </button>
                      </div>
                    ))
                  )}
                </div>

                <div className="library-footnote">
                  <span className="tiny-check">✓</span>
                  Source list synced from your API
                </div>
              </article>
            </div>

            <article className="panel chat-panel">
              <div className="chat-header">
                <div className="panel-heading">
                  <div className="chat-avatar">✳</div>
                  <div>
                    <h2>Chat with your knowledge</h2>
                    <p>Ask grounded questions about your sources</p>
                  </div>
                </div>
                <span className="live-badge"><span /> LIVE</span>
              </div>

              <div className="chat-context">
                <span className="context-icon">⌘</span>
                <span>
                  Searching across <strong>{sources.length} indexed source{sources.length === 1 ? "" : "s"}</strong>
                </span>
              </div>

              <div className="chat-messages" aria-live="polite">
                {messages.length === 0 ? (
                  <WelcomeMessage />
                ) : (
                  messages.map((message) => (
                    <div
                      className={`message-row ${message.role} ${message.failed ? "failed" : ""}`}
                      key={message.id}
                    >
                      {message.role === "assistant" && (
                        <div className="message-avatar">✳</div>
                      )}
                      <div className="message-content">
                        <div className="message-meta">
                          <strong>{message.role === "user" ? "You" : "Semantic AI"}</strong>
                          {message.role === "assistant" && !message.failed && (
                            <span className="grounded-tag">GROUNDED RESPONSE</span>
                          )}
                        </div>
                        <div className="message-text">{message.text}</div>

                        {message.citations && message.citations.length > 0 && (
                          <div className="citations-block">
                            <div className="citations-title">
                              <span>↗</span> SOURCE EVIDENCE
                              <span className="citations-count">{message.citations.length}</span>
                            </div>
                            {message.citations.map((citation) => (
                              <div
                                className="citation-card"
                                key={`${message.id}-${citation.number}-${citation.chunk_id ?? citation.source}`}
                              >
                                <div className="citation-card-top">
                                  <span className="citation-number">[{citation.number}]</span>
                                  <strong>{citation.source}</strong>
                                  {citation.page != null && (
                                    <span className="page-number">Page {citation.page}</span>
                                  )}
                                </div>
                                {citation.text && <p>{citation.text}</p>}
                              </div>
                            ))}
                          </div>
                        )}

                        {message.pii && message.pii.length > 0 && (
                          <div className="pii-note">
                            Privacy checks reported: {message.pii.join(", ")}
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
                {sending && (
                  <div className="message-row assistant">
                    <div className="message-avatar">✳</div>
                    <div className="message-content">
                      <div className="message-meta"><strong>Semantic AI</strong><span className="grounded-tag">RETRIEVING</span></div>
                      <div className="typing-indicator"><span /><span /><span /> Looking through your sources…</div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <form className="chat-composer" onSubmit={(event) => void sendMessage(event)}>
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="Ask a question about your documents…"
                  rows={2}
                  aria-label="Your question"
                  disabled={sending}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void sendMessage();
                    }
                  }}
                />
                <div className="composer-bottom">
                  <span><kbd>↵</kbd> to send <span className="composer-dot">·</span> <kbd>⇧ ↵</kbd> for new line</span>
                  <button className="send-button" type="submit" disabled={!input.trim() || sending}>
                    <span>{sending ? "…" : "↑"}</span> Send
                  </button>
                </div>
              </form>

              <div className="chat-disclaimer">
                <span>✧</span> Responses are AI-generated. Verify important details against the cited source.
              </div>
            </article>
          </section>

          <footer className="page-footer">
            <span>SEMANTIC AI WORKSPACE <span className="footer-dot">·</span> BUILT WITH AZURE AI</span>
            <span><span className={`footer-status ${health}`} /> {health === "online" ? "Backend connected" : health === "checking" ? "Checking backend" : "Backend disconnected"}</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
