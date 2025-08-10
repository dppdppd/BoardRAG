"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { mutate as swrMutate } from "swr";
import useSWR from "swr";
import ReactMarkdown from "react-markdown";

type Message = { role: "user" | "assistant"; content: string };

import { API_BASE } from "../lib/config";

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
  const initialScrollDoneRef = useRef<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadMsg, setUploadMsg] = useState<string>("");
  const firstTokenSeenRef = useRef<boolean>(false);
  const lastSubmittedUserIndexRef = useRef<number>(-1);
  const [retryableUsers, setRetryableUsers] = useState<number[]>([]);

  const addRetryable = (idx: number | null | undefined) => {
    if (idx == null || idx < 0) return;
    setRetryableUsers((cur) => (cur.includes(idx) ? cur : [...cur, idx]));
  };
  const removeRetryable = (idx: number | null | undefined) => {
    if (idx == null || idx < 0) return;
    setRetryableUsers((cur) => cur.filter((i) => i !== idx));
  };

  const games = gamesData?.games || [];

  // Keep only valid QA pairs.
  // - Drop any user question that never received an assistant reply
  // - Drop any assistant reply that has no preceding user question
  // - When multiple consecutive user questions exist before an assistant,
  //   only keep the most recent user question for that assistant
  const sanitizeConversation = (history: Message[]): Message[] => {
    const sanitized: Message[] = [];
    const pendingUsers: Message[] = [];
    for (const msg of history || []) {
      if (msg.role === "user") {
        pendingUsers.push(msg);
      } else if (msg.role === "assistant") {
        // Only keep assistant if there is at least one pending user
        if (pendingUsers.length > 0) {
          const lastUser = pendingUsers[pendingUsers.length - 1];
          // Drop older pending users; only keep the last one answered by this assistant
          pendingUsers.length = 0;
          sanitized.push(lastUser);
          sanitized.push(msg);
        } else {
          // assistant without a preceding question → drop
        }
      }
    }
    // Any remaining pendingUsers at the end had no answer → drop
    return sanitized;
  };

  // Load/save conversation per session+game to localStorage
  useEffect(() => {
    if (!selectedGame) return;
    const key = `boardrag_conv:${sessionId}:${selectedGame}`;
    const raw = localStorage.getItem(key);
    if (raw) {
      try {
        const parsed = JSON.parse(raw) as Message[];
        const cleaned = sanitizeConversation(Array.isArray(parsed) ? parsed : []);
        setMessages(cleaned);
        // If anything was removed, persist cleaned history back to storage
        if (cleaned.length !== (Array.isArray(parsed) ? parsed.length : 0)) {
          try { localStorage.setItem(key, JSON.stringify(cleaned)); } catch {}
        }
      } catch {
        setMessages([]);
      }
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

  // Do not auto-scroll on every token; we'll control scroll when an answer starts
  useEffect(() => {
    // intentionally empty to avoid continuous auto scroll during streaming
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

  // After loading a session, scroll to the last asked question (last user message)
  useEffect(() => {
    if (!selectedGame || !messages || messages.length === 0) return;
    if (isStreaming) return; // avoid fighting live streaming scroll
    const key = `boardrag_conv:${sessionId}:${selectedGame}`;
    if (loadedKeyRef.current !== key) return; // ensure messages for this key are loaded
    if (initialScrollDoneRef.current === key) return; // already scrolled for this key

    // Find the last user index
    let lastUserIdx = -1;
    let count = -1;
    for (const msg of messages) {
      if (msg.role === 'user') { count += 1; lastUserIdx = count; }
    }
    if (lastUserIdx < 0) return;

    const parent = chatScrollRef.current;
    if (!parent) return;

    let attempts = 0;
    const tryScroll = () => {
      const target = parent.querySelector(`[data-user-index="${lastUserIdx}"]`) as HTMLElement | null;
      if (target) {
        const top = target.offsetTop - parent.offsetTop - 12;
        parent.scrollTo({ top: Math.max(0, top), behavior: 'auto' });
        initialScrollDoneRef.current = key;
      } else if (attempts < 6) {
        attempts += 1;
        setTimeout(tryScroll, 16);
      }
    };
    requestAnimationFrame(tryScroll);
  }, [messages, selectedGame, sessionId, isStreaming]);

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

  const startQuery = (question: string, reuseUserIndex?: number | null) => {
    if (!selectedGame || !question) return;
    setIsStreaming(true);
    let workingMessages = messages;
    if (reuseUserIndex == null) {
      // append new user message
      workingMessages = [...messages, { role: "user", content: question } as Message];
      setMessages(workingMessages);
      const numUsers = workingMessages.filter((m) => m.role === "user").length;
      lastSubmittedUserIndexRef.current = numUsers - 1;
    } else {
      lastSubmittedUserIndexRef.current = reuseUserIndex;
    }
    firstTokenSeenRef.current = false;

    // Immediately scroll so the asked question is at the top, then stop auto-scrolling
    const anchorToQuestion = () => {
      let attempts = 0;
      const tryScroll = () => {
        const parent = chatScrollRef.current;
        if (!parent) return;
        const target = parent.querySelector(`[data-user-index="${lastSubmittedUserIndexRef.current}"]`) as HTMLElement | null;
        if (target) {
          const top = target.offsetTop - parent.offsetTop - 12;
          parent.scrollTo({ top: Math.max(0, top), behavior: "auto" });
          // Prevent later first-token scroll
          firstTokenSeenRef.current = true;
        } else if (attempts < 6) {
          attempts += 1;
          setTimeout(tryScroll, 16);
        }
      };
      requestAnimationFrame(tryScroll);
    };
    anchorToQuestion();

    // Prefer NDJSON endpoint in production; SSE works locally
    const NDJSON = (process.env.NEXT_PUBLIC_USE_NDJSON === '1') || (typeof window !== 'undefined' && window.location.hostname.endsWith('.vercel.app'));
    const url = new URL(`${API_BASE}/${NDJSON ? 'stream-ndjson' : 'stream'}`);
    url.searchParams.set("q", question);
    url.searchParams.set("game", selectedGame);
    url.searchParams.set("include_web", String(includeWeb));
    url.searchParams.set("history", workingMessages
      .slice(-20)
      .filter((m) => m.role !== "assistant")
      .map((m) => `${m.role[0].toUpperCase()}${m.role.slice(1)}: ${m.content}`)
      .join("\n"));
    url.searchParams.set("game_names", selectedGame);
    // Send stable browser session id for server-side blocking
    url.searchParams.set("sid", sessionId || "");
    url.searchParams.set("_", String(Date.now()));

    // Unified SSE event handler (used by EventSource or fetch-stream parser)
    let acc = "";
    const handleSseData = (payload: string) => {
      try {
        const parsed = JSON.parse(payload);
        if (parsed.type === "token") {
          acc += parsed.data;
          setMessages((cur) => {
            const last = cur[cur.length - 1];
            if (!last || last.role !== "assistant") return [...cur, { role: "assistant", content: acc }];
            const updated = [...cur]; updated[updated.length - 1] = { role: "assistant", content: acc }; return updated;
          });
        } else if (parsed.type === "done") {
          try { eventRef.current && (eventRef.current as any).close?.(); } catch {}
          eventRef.current = null;
          setIsStreaming(false);
          removeRetryable(lastSubmittedUserIndexRef.current);
        } else if (parsed.type === "error") {
          try { eventRef.current && (eventRef.current as any).close?.(); } catch {}
          eventRef.current = null;
          setIsStreaming(false);
          if (parsed.error === "blocked") {
            try { sessionStorage.setItem("boardrag_blocked", "1"); } catch {}
          } else {
            addRetryable(lastSubmittedUserIndexRef.current);
          }
        }
      } catch {}
    };

    // Streaming via fetch+ReadableStream (NDJSON/SSE parser)
    const streamWithFetch = async () => {
      try {
        const headers: any = NDJSON ? { Accept: 'application/x-ndjson' } : { Accept: 'text/event-stream' };
        const resp = await fetch(url.toString(), { headers, cache: "no-store" });
        const reader = resp.body?.getReader();
        if (!reader) throw new Error("no reader");
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          if (NDJSON) {
            let idx;
            while ((idx = buffer.indexOf("\n")) !== -1) {
              const line = buffer.slice(0, idx).trim();
              buffer = buffer.slice(idx + 1);
              if (line) handleSseData(line);
            }
          } else {
            let idx;
            while ((idx = buffer.indexOf("\n\n")) !== -1) {
              const raw = buffer.slice(0, idx); buffer = buffer.slice(idx + 2);
              const dataLine = raw.split("\n").find((l) => l.startsWith("data:"));
              if (!dataLine) continue;
              handleSseData(dataLine.slice(5).trim());
            }
          }
        }
      } catch {
        setIsStreaming(false); addRetryable(lastSubmittedUserIndexRef.current);
      }
    };

    // If production (or explicitly toggled), prefer fetch streaming immediately
    const FORCE_FETCH = (process.env.NEXT_PUBLIC_SSE_FETCH === '1') || (typeof window !== 'undefined' && window.location.hostname.endsWith('.vercel.app'));
    if (FORCE_FETCH) {
      streamWithFetch();
      return;
    }

    // Default path: EventSource with rapid fallback to fetch if buffered
    const es = new EventSource(url.toString());
    eventRef.current = es;
    const switchTimer = window.setTimeout(() => {
      streamWithFetch();
    }, 300);

    es.onmessage = (ev) => {
      clearTimeout(switchTimer);
      handleSseData(ev.data);
    };
    es.onerror = () => {
      clearTimeout(switchTimer);
      try { es.close(); } catch {}
      eventRef.current = null;
      // Try fetch-based streaming immediately on error
      streamWithFetch();
    };
  };

  const onSubmit = () => {
    const q = input.trim();
    if (!q || !selectedGame) return;
    // If this browser session was blocked, short-circuit locally without hitting the server
    try { if (sessionStorage.getItem("boardrag_blocked") === "1") { return; } } catch {}
    setInput("");
    startQuery(q, null);
  };

  const onStop = () => {
    try { eventRef.current?.close(); } catch {}
    eventRef.current = null;
    setIsStreaming(false);
    addRetryable(lastSubmittedUserIndexRef.current);
  };

  const onUpload = async (files: FileList | null) => {
    if (!files || files.length === 0 || uploading) return;
    setUploading(true);
    setUploadMsg("");
    try {
      const form = new FormData();
      Array.from(files).forEach((f) => form.append("files", f));
      const resp = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
      if (resp.ok) {
        try {
          const data = await resp.json();
          setUploadMsg(data?.message || "Upload complete.");
        } catch {
          setUploadMsg("Upload complete.");
        }
        await refetchGames();
        // Also refresh PDF assignment list for Admin screen
        try { await swrMutate(`${API_BASE}/pdf-choices`); } catch {}
      } else {
        setUploadMsg(`Upload failed (${resp.status}).`);
      }
    } catch {
      setUploadMsg("Upload failed. Network error.");
    } finally {
      setUploading(false);
      try { if (uploadInputRef.current) uploadInputRef.current.value = ""; } catch {}
    }
  };

  const [sheetOpen, setSheetOpen] = useState(false);
  const forcedOnceRef = useRef<boolean>(false);

  // On first mobile visit, if no saved game and none selected, auto-open menu to prompt game choice
  useEffect(() => {
    if (forcedOnceRef.current) return;
    const isMobile = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(max-width: 900px)').matches;
    if (!isMobile) return;
    if (!games || games.length === 0) return;
    if (selectedGame) return;
    try {
      const last = localStorage.getItem('boardrag_last_game');
      if (!last) {
        setSheetOpen(true);
        forcedOnceRef.current = true;
      }
    } catch {
      setSheetOpen(true);
      forcedOnceRef.current = true;
    }
  }, [games, selectedGame]);

  return (
    <>
      <div className="container">
        {/* Chat column */}
        <div className="chat surface" style={{ height: "100%" }}>
          <div className="row title" style={{ gap: 12, justifyContent: 'space-between' }}>
            <span>BG-GPT{selectedGame ? ` — ${selectedGame}` : ""}</span>
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
                const isUser = m.role === "user";
                const showActions = isUser && retryableUsers.includes(userCounter);
                return (
                  <React.Fragment key={i}>
                    <div {...props}>
                      <ReactMarkdown>{m.content}</ReactMarkdown>
                    </div>
                    {showActions && (
                      <div style={{ maxWidth: "92%", marginLeft: "auto", marginTop: 6 }}>
                        <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                          <button
                            className="btn"
                            title="Retry"
                            aria-label="Retry"
                            onClick={() => { setRetryableUsers((cur) => cur.filter((n) => n !== userCounter)); startQuery(m.content, userCounter); }}
                            style={{ width: 44, height: 44, minHeight: 44, padding: 0, fontSize: 18, color: "#fff", borderColor: "var(--accent)", background: "var(--accent)", display: "inline-grid", placeItems: "center" }}
                          >↻</button>
                          <button
                            className="btn"
                            title="Delete"
                            aria-label="Delete"
                            onClick={() => {
                              // Remove this user message
                              const targetUserIdx = userCounter;
                              let seen = -1;
                              const newList: Message[] = [];
                              for (const msg of messages) {
                                if (msg.role === 'user') {
                                  seen += 1;
                                  if (seen === targetUserIdx) continue; // skip this one
                                }
                                newList.push(msg);
                              }
                              setMessages(newList);
                              setRetryableUsers((cur) => cur.filter((n) => n !== userCounter));
                              try {
                                if (selectedGame) {
                                  const key = `boardrag_conv:${sessionId}:${selectedGame}`;
                                  localStorage.setItem(key, JSON.stringify(newList));
                                }
                              } catch {}
                            }}
                            style={{ width: 44, height: 44, minHeight: 44, padding: 0, fontSize: 18, color: "#fff", borderColor: "var(--accent)", background: "var(--accent)", display: "inline-grid", placeItems: "center" }}
                          >✕</button>
                        </div>
                      </div>
                    )}
                  </React.Fragment>
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

            <div className="actions" style={{ display: "flex", gap: 8, alignItems: "stretch" }}>
              {isStreaming ? (
                <button className="btn stop" onClick={onStop} style={{ height: 44, minHeight: 44, paddingTop: 0, paddingBottom: 0 }}>Stop</button>
              ) : (
                <button className="btn primary" onClick={onSubmit} disabled={!selectedGame} style={{ fontSize: 18 }}>
                  Ask
                </button>
              )}
              {/* Kebab to toggle the bottom sheet; placed to the right of Send */}
              <button
                className="btn menu-toggle"
                onClick={() => setSheetOpen((s) => !s)}
                aria-label="Menu"
                title="Menu"
              >
                ⋮
              </button>
            </div>
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
            <summary>Game</summary>
            <div style={{ display: "grid", gap: 10, marginTop: 8 }}>
              <label style={{ display: "grid", gap: 6 }}>
                <span className="muted">Select game</span>
                <select
                  className="select"
                  value={selectedGame}
                  onChange={(e) => {
                    if (selectedGame) {
                      const oldKey = `boardrag_conv:${sessionId}:${selectedGame}`;
                      try { localStorage.setItem(oldKey, JSON.stringify(messages)); } catch {}
                    }
                    setSelectedGame(e.target.value);
                  }}
                >
                  <option value="">Select game…</option>
                  {games.map((g) => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
              </label>
              <label className="row" style={{ gap: 10 }}>
                <input type="checkbox" checked={includeWeb} onChange={(e) => setIncludeWeb(e.target.checked)} />
                <span>Include Web Search</span>
              </label>
            </div>
          </section>

          <section className="surface pad section" style={{ marginTop: 12 }}>
            <summary>Upload PDFs</summary>
            <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
              <input
                ref={uploadInputRef}
                className="input"
                type="file"
                accept="application/pdf"
                multiple
                disabled={uploading}
                onChange={(e) => onUpload(e.target.files)}
                title={uploading ? "Uploading…" : "Select PDF files"}
              />
              {uploading && (
                <div className="indicator" style={{ gap: 8 }}>
                  <span className="spinner" />
                  <span>Uploading… This may take a moment.</span>
                </div>
              )}
              {!!uploadMsg && !uploading && (
                <div className="muted" style={{ fontSize: 13 }}>{uploadMsg}</div>
              )}
            </div>
          </section>
        </div>
      </div>

      {/* Mobile bottom sheet */}
      <div className={`mobile-sheet ${sheetOpen ? 'open' : ''}`}>
        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="title" style={{ padding: 0, margin: 0, textAlign: 'left' }}>Menu</div>
          <button className="btn" onClick={() => setSheetOpen(false)} aria-label="Close menu" title="Close menu" style={{ width: 44, height: 44, minHeight: 44, padding: 0, display: 'inline-grid', placeItems: 'center' }}>×</button>
        </div>
        <section className="surface pad section" style={{ marginTop: 12 }}>
          <summary>Game</summary>
          <div style={{ display: "grid", gap: 10, marginTop: 8 }}>
            <label style={{ display: "grid", gap: 6 }}>
              <span className="muted">Select game</span>
              <select
                className="select"
                value={selectedGame}
                onChange={(e) => {
                  if (selectedGame) {
                    const oldKey = `boardrag_conv:${sessionId}:${selectedGame}`;
                    try { localStorage.setItem(oldKey, JSON.stringify(messages)); } catch {}
                  }
                  setSelectedGame(e.target.value);
                }}
              >
                <option value="">Select game…</option>
                {games.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </label>
            <label className="row" style={{ gap: 10 }}>
              <input type="checkbox" checked={includeWeb} onChange={(e) => setIncludeWeb(e.target.checked)} />
              <span>Include Web Search</span>
            </label>
          </div>
        </section>
        <section className="surface pad section" style={{ marginTop: 12 }}>
          <summary>Upload PDFs</summary>
          <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
            <input
              ref={uploadInputRef}
              className="input"
              type="file"
              accept="application/pdf"
              multiple
              disabled={uploading}
              onChange={(e) => onUpload(e.target.files)}
              title={uploading ? "Uploading…" : "Select PDF files"}
            />
            {uploading && (
              <div className="indicator" style={{ gap: 8 }}>
                <span className="spinner" />
                <span>Uploading… This may take a moment.</span>
              </div>
            )}
            {!!uploadMsg && !uploading && (
              <div className="muted" style={{ fontSize: 13 }}>{uploadMsg}</div>
            )}
          </div>
        </section>
      </div>
    </>
  );
}


