"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import ReactMarkdown from "react-markdown";

type Message = { role: "user" | "assistant"; content: string };

const RAW_API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const API_BASE = RAW_API_BASE.trim().replace(/\/+$/, "");

const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function HomePage() {
  // State
  const [sessionId, setSessionId] = useState<string>("");
  const { data: gamesData, mutate: refetchGames, isLoading: loadingGames, error: gamesError } = useSWR<{ games: string[] }>(`${API_BASE}/games`, fetcher);
  const [selectedGame, setSelectedGame] = useState<string | "">("");
  const [includeWeb, setIncludeWeb] = useState<boolean>(false);
  const [model, setModel] = useState<string>("[Anthropic] Claude Sonnet 4");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const historyStripRef = useRef<HTMLDivElement>(null);
  const eventRef = useRef<EventSource | null>(null);
  const historyDragRef = useRef<{ isDown: boolean; startX: number; scrollLeft: number; dragged: boolean }>({ isDown: false, startX: 0, scrollLeft: 0, dragged: false });
  const historyDraggingFlag = useRef<boolean>(false);
  const loadedKeyRef = useRef<string | null>(null);

  const games = gamesData?.games || [];

  // Load/save conversation per session+game to localStorage
  useEffect(() => {
    if (!selectedGame) return;
    const key = `boardrag_conv:${sessionId}:${selectedGame}`;
    const raw = localStorage.getItem(key);
    if (raw) {
      try { setMessages(JSON.parse(raw)); } catch { setMessages([]); }
    } else {
      setMessages([]);
    }
    loadedKeyRef.current = key;
  }, [sessionId, selectedGame]);

  useEffect(() => {
    if (!selectedGame) return;
    const key = `boardrag_conv:${sessionId}:${selectedGame}`;
    // Prevent saving under a new game's key before its messages are loaded
    if (loadedKeyRef.current !== key) return;
    try { localStorage.setItem(key, JSON.stringify(messages)); } catch {}
  }, [sessionId, selectedGame, messages]);

  // Ensure stable browser session id across refreshes
  useEffect(() => {
    try {
      let sid = localStorage.getItem("boardrag_session_id");
      if (!sid) {
        sid = (typeof crypto !== 'undefined' && 'randomUUID' in crypto)
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
        localStorage.setItem("boardrag_session_id", sid);
      }
      setSessionId(sid);
    } catch {
      // Fallback if localStorage unavailable
      setSessionId(`${Date.now()}-${Math.random().toString(16).slice(2)}`);
    }
  }, []);

  // Restore last selected game once games are loaded
  useEffect(() => {
    if (!gamesData?.games || gamesData.games.length === 0) return;
    if (selectedGame) return; // already chosen in this session
    try {
      const last = localStorage.getItem("boardrag_last_game");
      if (last && gamesData.games.includes(last)) {
        setSelectedGame(last);
      }
    } catch {}
  }, [gamesData?.games]);

  // Persist selected game choice
  useEffect(() => {
    if (!selectedGame) return;
    try { localStorage.setItem("boardrag_last_game", selectedGame); } catch {}
  }, [selectedGame]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const { bookmarkLabels, bookmarkUserIndices } = useMemo(() => {
    // Build labels for assistant replies and map each to its preceding user index
    const labels: string[] = [];
    const userIdxs: number[] = [];
    let userCount = 0;
    for (let i = 0; i < messages.length; i++) {
      const m = messages[i];
      if (m.role === "user") {
        userCount += 1;
      } else if (m.role === "assistant") {
        const first = m.content.split("\n").find((l) => l.trim().length > 0) || m.content;
        const trimmed = first.replace(/^[#*\s]+/, "").slice(0, 60);
        labels.push(trimmed);
        userIdxs.push(Math.max(0, userCount - 1));
      }
    }
    return { bookmarkLabels: labels, bookmarkUserIndices: userIdxs };
  }, [messages]);

  // Ensure history strip shows the most recent question
  useEffect(() => {
    const bar = historyStripRef.current;
    if (!bar) return;
    // Scroll to the end so the latest pill is visible
    bar.scrollTo({ left: bar.scrollWidth, behavior: 'smooth' });
  }, [bookmarkLabels, selectedGame]);

  // Draggable history strip (mouse & touch)
  useEffect(() => {
    const el = historyStripRef.current;
    if (!el) return;

    const onDown = (clientX: number) => {
      historyDragRef.current.isDown = true;
      historyDragRef.current.dragged = false;
      historyDragRef.current.startX = clientX + el.scrollLeft;
      historyDragRef.current.scrollLeft = el.scrollLeft;
      el.classList.add('dragging');
      historyDraggingFlag.current = true;
    };
    const onMove = (clientX: number, e?: Event) => {
      const state = historyDragRef.current;
      if (!state.isDown) return;
      if (e) { e.preventDefault(); }
      const walk = state.startX - clientX; // drag direction
      if (Math.abs(walk) > 3) state.dragged = true;
      el.scrollLeft = walk;
    };
    const onUp = () => {
      if (!historyDragRef.current.isDown) return;
      historyDragRef.current.isDown = false;
      setTimeout(() => { historyDraggingFlag.current = false; }, 0);
      el.classList.remove('dragging');
    };

    const md = (e: MouseEvent) => onDown(e.clientX);
    const mm = (e: MouseEvent) => onMove(e.clientX, e);
    const mu = (_e: MouseEvent) => onUp();
    const tlstart = (e: TouchEvent) => { if (e.touches && e.touches[0]) onDown(e.touches[0].clientX); };
    const tlmove = (e: TouchEvent) => { if (e.touches && e.touches[0]) onMove(e.touches[0].clientX, e); };
    const tlend = (_e: TouchEvent) => onUp();

    el.addEventListener('mousedown', md);
    window.addEventListener('mousemove', mm, { passive: false });
    window.addEventListener('mouseup', mu);
    el.addEventListener('touchstart', tlstart, { passive: true });
    el.addEventListener('touchmove', tlmove, { passive: false });
    el.addEventListener('touchend', tlend);

    return () => {
      el.removeEventListener('mousedown', md);
      window.removeEventListener('mousemove', mm as any);
      window.removeEventListener('mouseup', mu as any);
      el.removeEventListener('touchstart', tlstart as any);
      el.removeEventListener('touchmove', tlmove as any);
      el.removeEventListener('touchend', tlend as any);
    };
  }, []);

  const scrollToAssistant = (assistantIndex: number) => {
    const parent = chatScrollRef.current;
    if (!parent) return;
    // Scroll to the corresponding user question (the message just before that assistant reply)
    const target = parent.querySelector(`[data-user-index="${assistantIndex}"]`) as HTMLElement | null;
    if (!target) return;
    const top = target.offsetTop - parent.offsetTop;
    parent.scrollTo({ top, behavior: "smooth" });
  };

  const deleteHistoryAt = (assistantIndex: number) => {
    // Find absolute positions: the user message corresponding to this assistant and the assistant itself
    let seenAssistants = -1;
    let absAssistant = -1;
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === "assistant") {
        seenAssistants += 1;
        if (seenAssistants === assistantIndex) { absAssistant = i; break; }
      }
    }
    if (absAssistant < 0) return;
    // The paired user is the nearest user before this assistant
    let absUser = -1;
    for (let i = absAssistant - 1; i >= 0; i--) {
      if (messages[i].role === "user") { absUser = i; break; }
    }
    if (absUser < 0) return;
    const newMsgs = [...messages.slice(0, absUser), ...messages.slice(absAssistant + 1)];
    setMessages(newMsgs);
    // persist immediately
    try {
      if (selectedGame) {
        const key = `boardrag_conv:${sessionId}:${selectedGame}`;
        localStorage.setItem(key, JSON.stringify(newMsgs));
      }
    } catch {}
  };

  // Build long-press handlers for history pills
  const longPressHandlers = (assistantIndex: number) => {
    let timer: any;
    let triggered = false;
    const start = () => {
      triggered = false;
      timer = setTimeout(() => {
        triggered = true;
        if (window.confirm("Delete this QA pair?")) {
          deleteHistoryAt(assistantIndex);
        }
      }, 600);
    };
    const clear = () => { if (timer) clearTimeout(timer); };
    return {
      onMouseDown: start,
      onMouseUp: (e: any) => { clear(); if (!triggered) scrollToAssistant(assistantIndex); },
      onMouseLeave: clear,
      onTouchStart: start,
      onTouchEnd: (e: any) => { clear(); if (!triggered) scrollToAssistant(assistantIndex); },
    } as any;
  };

  const onSubmit = async () => {
    const q = input.trim();
    if (!q || !selectedGame) return;
    setInput("");
    setIsStreaming(true);
    const nextMessages = [...messages, { role: "user", content: q } as Message];
    setMessages(nextMessages);

    const url = new URL(`${API_BASE}/stream`);
    url.searchParams.set("q", q);
    url.searchParams.set("game", selectedGame);
    url.searchParams.set("include_web", String(includeWeb));
    url.searchParams.set("history", nextMessages
      .slice(-20)
      .filter((m) => m.role !== "assistant")
      .map((m) => `${m.role[0].toUpperCase()}${m.role.slice(1)}: ${m.content}`)
      .join("\n"));
    url.searchParams.set("game_names", selectedGame);

    const es = new EventSource(url.toString());
    eventRef.current = es;
    let acc = "";

    es.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(ev.data);
        if (parsed.type === "token") {
          acc += parsed.data;
          // add assistant message on first token, update afterwards
          setMessages((cur) => {
            const last = cur[cur.length - 1];
            if (!last || last.role !== "assistant") {
              return [...cur, { role: "assistant", content: acc }];
            }
            const updated = [...cur];
            updated[updated.length - 1] = { role: "assistant", content: acc };
            return updated;
          });
        } else if (parsed.type === "done") {
          // could append formatted Sources here based on parsed.meta
          es.close();
          eventRef.current = null;
          setIsStreaming(false);
        } else if (parsed.type === "error") {
          es.close();
          eventRef.current = null;
          setIsStreaming(false);
        }
      } catch {
        // ignore
      }
    };
    es.onerror = () => {
      es.close();
      eventRef.current = null;
      setIsStreaming(false);
    };
  };

  const onStop = () => {
    try { eventRef.current?.close(); } catch {}
    eventRef.current = null;
    setIsStreaming(false);
  };

  const onUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    const resp = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
    if (resp.ok) {
      await refetchGames();
    }
  };

  const [sheetOpen, setSheetOpen] = useState(false);

  return (
    <>
      <div className="container">
        {/* Chat column */}
        <div className="chat surface" style={{ height: "100%" }}>
          <div className="row title" style={{ gap: 12, justifyContent: 'space-between' }}>
            <span>BoardRAG</span>
            <select
              className="select compact"
              value={selectedGame}
              onChange={(e) => {
                if (selectedGame) {
                  const oldKey = `boardrag_conv:${sessionId}:${selectedGame}`;
                  try { localStorage.setItem(oldKey, JSON.stringify(messages)); } catch {}
                }
                setSelectedGame(e.target.value);
              }}
              style={{ minWidth: 180 }}
            >
              <option value="">Select game…</option>
              {games.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </div>

          {/* Horizontal history strip */}
          <div className="history-strip" ref={historyStripRef}>
            {bookmarkLabels.length === 0 ? (
              <div className="muted" style={{ fontSize: 12 }}>No history yet</div>
            ) : (
              bookmarkLabels.map((b, i) => (
                <button
                  key={i}
                  className="history-pill btn"
                  onClick={(e) => {
                    if (historyDraggingFlag.current || historyDragRef.current.dragged) return; // ignore click if just dragged
                    scrollToAssistant(bookmarkUserIndices[i]);
                  }}
                  {...longPressHandlers(i)}
                >
                  {b}
                </button>
              ))
            )}
          </div>

          {/* Game selector row removed; dropdown moved into title */}

          {/* Chat history */}
          <div className="chat-scroll" ref={chatScrollRef}>
            {(() => {
              let assistantCounter = -1;
              let userCounter = -1;
              return messages.map((m, i) => {
                const props: any = { key: i, className: `bubble ${m.role}` };
                if (m.role === "assistant") {
                  assistantCounter += 1;
                  props["data-assistant-index"] = assistantCounter;
                } else if (m.role === "user") {
                  userCounter += 1;
                  props["data-user-index"] = userCounter;
                }
                return (
                  <div {...props}>
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                );
              });
            })()}
            <div ref={chatEndRef} />
          </div>

          {/* Input row */}
          <div className="input-row">
            {isStreaming ? (
              <div className="input indicator" aria-live="polite">
                <span className="spinner" />
                <span style={{ marginLeft: 8 }}>Generating answer…</span>
              </div>
            ) : (
              <input
                className="input"
                placeholder="Your question…"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    onSubmit();
                  }
                }}
              />
            )}

            {isStreaming ? (
              <button className="btn stop" onClick={onStop}>Stop</button>
            ) : (
              <button className="btn primary" onClick={onSubmit} disabled={!selectedGame}>Send</button>
            )}
          </div>
        </div>

        {/* Sidebar (desktop only) */}
        <div className="sidebar-panel">
          <section className="surface pad section">
            <summary>History</summary>
            {bookmarkLabels.length === 0 ? (
              <div className="muted" style={{ fontSize: 13 }}>No bookmarks yet</div>
            ) : (
              <div style={{ marginTop: 8 }}>
                {bookmarkLabels.map((b, i) => (
                  <button key={i} className="bookmark btn" {...longPressHandlers(i)} onClick={() => scrollToAssistant(bookmarkUserIndices[i])}>
                    {b}
                  </button>
                ))}
              </div>
            )}
          </section>

          <section className="surface pad section" style={{ marginTop: 12 }}>
            <summary>Options</summary>
            <div style={{ display: "grid", gap: 10, marginTop: 8 }}>
              <label style={{ display: "grid", gap: 6 }}>
                <span className="muted">Model</span>
                <select className="select" value={model} onChange={(e) => setModel(e.target.value)}>
                  <option>[Anthropic] Claude Sonnet 4</option>
                  <option>[OpenAI] o3</option>
                </select>
              </label>
              <label className="row" style={{ gap: 10 }}>
                <input type="checkbox" checked={includeWeb} onChange={(e) => setIncludeWeb(e.target.checked)} />
                <span>Include Web Search</span>
              </label>
            </div>
          </section>

          <section className="surface pad section" style={{ marginTop: 12 }}>
            <summary>Add New Game</summary>
            <div style={{ marginTop: 8 }}>
              <input className="input" type="file" accept="application/pdf" multiple onChange={(e) => onUpload(e.target.files)} />
            </div>
          </section>
        </div>
      </div>

      {/* Mobile bottom sheet toggle */}
      <button className="sheet-toggle" onClick={() => setSheetOpen((s) => !s)}>Menu</button>
      <div className={`mobile-sheet ${sheetOpen ? 'open' : ''}`}>
        <section className="surface pad section">
          <summary>History</summary>
          {bookmarkLabels.length === 0 ? (
            <div className="muted" style={{ fontSize: 13 }}>No bookmarks yet</div>
          ) : (
            <div style={{ marginTop: 8 }}>
              {bookmarkLabels.map((b, i) => (
                <button key={i} className="bookmark btn" {...longPressHandlers(i)} onClick={() => scrollToAssistant(bookmarkUserIndices[i])}>
                  {b}
                </button>
              ))}
            </div>
          )}
        </section>
        <section className="surface pad section" style={{ marginTop: 12 }}>
          <summary>Options</summary>
          <div style={{ display: "grid", gap: 10, marginTop: 8 }}>
            <label style={{ display: "grid", gap: 6 }}>
              <span className="muted">Model</span>
              <select className="select" value={model} onChange={(e) => setModel(e.target.value)}>
                <option>[Anthropic] Claude Sonnet 4</option>
                <option>[OpenAI] o3</option>
              </select>
            </label>
            <label className="row" style={{ gap: 10 }}>
              <input type="checkbox" checked={includeWeb} onChange={(e) => setIncludeWeb(e.target.checked)} />
              <span>Include Web Search</span>
            </label>
          </div>
        </section>
        <section className="surface pad section" style={{ marginTop: 12 }}>
          <summary>Add New Game</summary>
          <div style={{ marginTop: 8 }}>
            <input className="input" type="file" accept="application/pdf" multiple onChange={(e) => onUpload(e.target.files)} />
          </div>
        </section>
      </div>
    </>
  );
}


