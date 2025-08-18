"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { mutate as swrMutate } from "swr";
import useSWR from "swr";
import ReactMarkdown from "react-markdown";
import { Document, Page, pdfjs } from "react-pdf";
import InputRow from "./components/InputRow";
import HistoryStrip from "./components/HistoryStrip";
import BottomSheetMenu from "./components/BottomSheetMenu";
// Modal viewer removed; reuse preview panel for all screen sizes
import PdfPreview from "./components/PdfPreview";
import 'react-pdf/dist/esm/Page/TextLayer.css';

type Message = { role: "user" | "assistant"; content: string; pinned?: boolean; style?: "default" | "brief" | "detailed" };

import { API_BASE, isLocalhost } from "../lib/config";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

// Remove common Markdown formatting from a single-line string for use in history labels
const stripMarkdown = (input: string): string => {
  if (!input) return "";
  let s = input;
  // Remove leading/trailing whitespace
  s = s.trim();
  // Strip HTML tags
  s = s.replace(/<[^>]+>/g, "");
  // Images: ![alt](url) -> alt
  s = s.replace(/!\[(.*?)\]\((.*?)\)/g, "$1");
  // Links: [text](url) -> text
  s = s.replace(/\[(.*?)\]\((.*?)\)/g, "$1");
  // Headings at start: ### Title -> Title
  s = s.replace(/^#{1,6}\s*/g, "");
  // Blockquote marker at start: > quote -> quote
  s = s.replace(/^>\s*/g, "");
  // List markers at start: -, *, +, 1. -> (remove)
  s = s.replace(/^\s*(?:[-*+]\s+|\d+\.\s+)/g, "");
  // Bold: **text** or __text__ -> text
  s = s.replace(/(\*\*|__)(.*?)\1/g, "$2");
  // Italic: *text* or _text_ -> text
  s = s.replace(/(\*|_)(.*?)\1/g, "$2");
  // Strikethrough: ~~text~~ -> text
  s = s.replace(/~~(.*?)~~/g, "$1");
  // Inline code: `code` -> code
  s = s.replace(/`([^`]+)`/g, "$1");
  // Remove residual markdown escape backslashes
  s = s.replace(/\\([\\`*_{}\[\]()#+\-.!])/g, "$1");
  // Collapse whitespace
  s = s.replace(/\s+/g, " ").trim();
  return s;
};

export default function HomePage() {
  const DEBUG_LOCAL = typeof window !== 'undefined' && isLocalhost();
  // Configure pdfjs worker URL once
  useEffect(() => {
    try {
      // next public path suitable for both dev and prod
      const ver = (pdfjs as any).version || '4.4.168';
      (pdfjs as any).GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${ver}/build/pdf.worker.min.js`;
    } catch {}
  }, []);
  // Turn bare section citations like [3.5] or [3.5.1] and verbal tags like [EQUIPMENT] into clickable links we can intercept
  const decorateCitations = (input: string): string => {
    if (!input) return input;
    let out = input;
    // Convert inline metadata citations like [6.1: {json}] -> [6.1](section:6.1 "base64(json)")
    // Support codes like 3.5, 3.5.1, F4, F4.a, A10.2b, 1B6b, ranges like 6.41-6.43, uppercase text like HOTELS, and mixed-case text like Combat Basics
    out = out.replace(/\[((?:[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?|\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?|\d+(?:\.[A-Za-z0-9]+)*|[A-Z][A-Z0-9 _@-]*[A-Z0-9]|[A-Z][A-Za-z0-9 _@-]*[A-Za-z0-9])(?:-\d+(?:\.[A-Za-z0-9]+)*)?)\s*:\s*(\{[\s\S]*?\})\]/g, (_m, code, json) => {
      const encCode = encodeURIComponent(String(code));
      // Put the literal JSON into the title so tooltip shows the human-readable metadata
      const safe = String(json).replace(/'/g, "&#39;");
      return `[${code}](section:${encCode} '${safe}')`;
    });
    // Replace [3.5] with markdown link [3.5](section:3.5) only when not already a link
    out = out.replace(/\[(\d+(?:\.\d+)+)\](?!\()/g, "[$1](section:$1)");
    // Replace alpha/alpha-numeric codes like [F4], [F4.a], [A10.2b], [1B6b] when not already a link
    out = out.replace(/\[((?:[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?|\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?))\](?!\()/g, "[$1](section:$1)");
    // Replace verbal-only tags like [EQUIPMENT], [ATTACKING], [The 56 RISK@ Cards]
    // Only when not already a link and contains at least one letter
    out = out.replace(/\[((?=[^\]]*[A-Za-z])[^\]\n]{2,80})\](?!\()/g, (_m, p1) => {
      try {
        const enc = encodeURIComponent(String(p1));
        return `[${p1}](section:${enc})`;
      } catch {
        return `[${p1}]`;
      }
    });
    return out;
  };
  // State
  const [sessionId, setSessionId] = useState<string>("");
  const { data: gamesData, mutate: refetchGames, isLoading: loadingGames, error: gamesError } = useSWR<{ games: string[] }>(`${API_BASE}/games`, fetcher);
  const [selectedGame, setSelectedGame] = useState<string | "">("");
  const [includeWeb, setIncludeWeb] = useState<boolean>(false);
  const [pdfSmoothScroll, setPdfSmoothScroll] = useState<boolean>(() => {
    // Default ON; will refine from session prefs once sessionId is loaded
    try { return localStorage.getItem('boardrag_pdf_smooth') === '1' || localStorage.getItem('boardrag_pdf_smooth') == null; } catch { return true; }
  });
  type PromptStyle = "default" | "brief" | "detailed";
  const [promptStyle, setPromptStyle] = useState<PromptStyle>("default");
  const [model, setModel] = useState<string>("[Anthropic] Claude Sonnet 4");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [debugShowCitationMeta, setDebugShowCitationMeta] = useState<boolean>(() => {
    try { return localStorage.getItem('boardrag_debug_citation_meta') === '1'; } catch { return false; }
  });
  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const historyStripRef = useRef<HTMLDivElement>(null);
  const eventRef = useRef<EventSource | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const switchTimerRef = useRef<number | null>(null);
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
  const [showHoldHint, setShowHoldHint] = useState<boolean>(false);
  // Active stream request id as announced by the server; used to ignore late chunks from older streams
  const streamReqIdRef = useRef<string | null>(null);

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

  // Load prompt style per game when game changes
  useEffect(() => {
    if (!selectedGame) return;
    try {
      const key = `boardrag_style:${selectedGame}`;
      const savedRaw = localStorage.getItem(key);
      const allowed: PromptStyle[] = ["default", "brief", "detailed"];
      const valid = (savedRaw && (allowed as readonly string[]).includes(savedRaw)) ? (savedRaw as PromptStyle) : "default";
      setPromptStyle(valid);
      if (savedRaw && !allowed.includes(savedRaw as any)) {
        try { localStorage.setItem(key, valid); } catch {}
      }
    } catch {
      setPromptStyle("default");
    }
  }, [selectedGame]);

  // Persist prompt style per game when it changes
  useEffect(() => {
    if (!selectedGame) return;
    try {
      const key = `boardrag_style:${selectedGame}`;
      localStorage.setItem(key, promptStyle);
    } catch {}
  }, [promptStyle, selectedGame]);

  // Load PDF smooth scroll setting from session-scoped prefs when sessionId is known
  useEffect(() => {
    if (!sessionId) return;
    try {
      const key = `boardrag_prefs:${sessionId}`;
      const raw = localStorage.getItem(key);
      if (raw) {
        const prefs = JSON.parse(raw);
        if (typeof prefs?.pdfSmoothScroll === 'boolean') {
          setPdfSmoothScroll(prefs.pdfSmoothScroll);
        }
      } else {
        // Create default prefs with smooth scrolling enabled
        const prefs = { pdfSmoothScroll: true };
        localStorage.setItem(key, JSON.stringify(prefs));
      }
    } catch {}
  }, [sessionId]);

  // Persist PDF smooth scroll setting to both global fallback and session-scoped prefs
  useEffect(() => {
    try { localStorage.setItem('boardrag_pdf_smooth', pdfSmoothScroll ? '1' : '0'); } catch {}
    if (!sessionId) return;
    try {
      const key = `boardrag_prefs:${sessionId}`;
      const raw = localStorage.getItem(key);
      const prefs = raw ? (() => { try { return JSON.parse(raw as string) || {}; } catch { return {}; } })() : {};
      (prefs as any).pdfSmoothScroll = pdfSmoothScroll;
      localStorage.setItem(key, JSON.stringify(prefs));
    } catch {}
  }, [pdfSmoothScroll, sessionId]);

  // Do not auto-scroll on every token; we'll control scroll when an answer starts
  useEffect(() => {
    // intentionally empty to avoid continuous auto scroll during streaming
  }, [messages]);

  const { bookmarkLabels, bookmarkUserIndices, bookmarkAssistantIndices } = useMemo(() => {
    // Build labels only for pinned assistant replies and map each to its preceding user index
    const labels: string[] = [];
    const userIdxs: number[] = [];
    const assistantAbsIdxs: number[] = [];
    let userCount = 0;
    let assistantCount = -1;
    for (let i = 0; i < messages.length; i++) {
      const m = messages[i];
      if (m.role === "user") {
        userCount += 1;
      } else if (m.role === "assistant") {
        assistantCount += 1;
        if (m.pinned) {
          const first = m.content.split("\n").find((l) => l.trim().length > 0) || m.content;
          const cleaned = stripMarkdown(first);
          const trimmed = cleaned.slice(0, 30);
          labels.push(trimmed);
          userIdxs.push(Math.max(0, userCount - 1));
          assistantAbsIdxs.push(assistantCount);
        }
      }
    }
    return { bookmarkLabels: labels, bookmarkUserIndices: userIdxs, bookmarkAssistantIndices: assistantAbsIdxs };
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

  // Enable mouse wheel to scroll the horizontal mobile history (map vertical wheel to horizontal scroll)
  useEffect(() => {
    const el = historyStripRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const target = el;
      if (!target) return;
      // Only act when horizontal overflow exists
      if (target.scrollWidth <= target.clientWidth) return;
      const primaryDelta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      if (primaryDelta === 0) return;
      // Prevent page scroll and translate to horizontal scroll
      e.preventDefault();
      target.scrollLeft += primaryDelta;
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => { el.removeEventListener('wheel', onWheel as any); };
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

  // Clear entire conversation history for the current game
  const clearAllHistory = () => {
    if (!selectedGame) return;
    if (!messages || messages.length === 0) return;
    const emptied: Message[] = [];
    setMessages(emptied);
    setRetryableUsers([]);
    try {
      const key = `boardrag_conv:${sessionId}:${selectedGame}`;
      localStorage.setItem(key, JSON.stringify(emptied));
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

  // Long-press handler for clearing all history (no extra confirm; long-press is the confirmation)
  const longPressClearAllHandlers = () => {
    let timer: any;
    let triggered = false;
    const start = () => {
      triggered = false;
      timer = setTimeout(() => {
        triggered = true;
        clearAllHistory();
      }, 600);
    };
    const clear = () => { if (timer) clearTimeout(timer); };
    return {
      onMouseDown: start,
      onMouseUp: (_e: any) => { const wasTriggered = triggered; clear(); if (!wasTriggered) { setShowHoldHint(true); setTimeout(() => setShowHoldHint(false), 1200); } },
      onMouseLeave: clear,
      onTouchStart: start,
      onTouchEnd: (_e: any) => { const wasTriggered = triggered; clear(); if (!wasTriggered) { setShowHoldHint(true); setTimeout(() => setShowHoldHint(false), 1200); } },
    } as any;
  };

  // Long-press delete for an assistant QA pair inside the chat
  const longPressDeleteAssistant = (assistantIndex: number) => {
    let timer: any;
    let triggered = false;
    const start = () => {
      triggered = false;
      timer = setTimeout(() => {
        triggered = true;
        deleteHistoryAt(assistantIndex);
      }, 600);
    };
    const clear = () => { if (timer) clearTimeout(timer); };
    return {
      onMouseDown: start,
      onMouseUp: (_e: any) => { clear(); },
      onMouseLeave: clear,
      onTouchStart: start,
      onTouchEnd: (_e: any) => { clear(); },
    } as any;
  };

  const startQuery = async (question: string, reuseUserIndex?: number | null, styleOverride?: PromptStyle) => {
    if (!selectedGame || !question) return;
    setIsStreaming(true);
    let workingMessages = messages;
    if (reuseUserIndex == null) {
      // append new user message
      const styleForQuery = styleOverride ?? promptStyle;
      workingMessages = [...messages, { role: "user", content: question, style: styleForQuery } as Message];
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

    // Use original SSE endpoint by default; NDJSON only when explicitly enabled
    const NDJSON = (process.env.NEXT_PUBLIC_USE_NDJSON === '1');
    const applyPromptStyle = (q: string, style: PromptStyle): string => {
      // Append single-line ASCII instructions to avoid issues with SSE URLs
      switch (style) {
        case "brief":
          return `${q} Instruction: Answer extremely concisely in 1-3 short sentences or a compact bulleted list.`;
        case "detailed":
          return `${q} Instruction: Provide a thorough, step-by-step explanation with relevant details and examples.`;
        default:
          return q;
      }
    };
    const styleForAugment = (typeof styleOverride !== 'undefined' ? styleOverride : promptStyle);
    const augmentedQuestion = applyPromptStyle(question, styleForAugment);
    // Helper: clear credentials and force login screen
    const kickToLogin = () => {
      try {
        sessionStorage.removeItem("boardrag_role");
        sessionStorage.removeItem("boardrag_token");
        localStorage.removeItem("boardrag_role");
        localStorage.removeItem("boardrag_token");
        localStorage.removeItem("boardrag_saved_pw");
      } catch {}
      try { window.location.reload(); } catch {}
    };
    const url = new URL(`${API_BASE}/${NDJSON ? 'stream-ndjson' : 'stream'}`);
    url.searchParams.set("q", augmentedQuestion);
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
    // Debug flag: request server to not strip citation metadata from inline brackets
    try {
      const dbg = localStorage.getItem('boardrag_debug_citation_meta') === '1';
      url.searchParams.set("debug_cite_meta", dbg ? "1" : "0");
    } catch {}
    // Ensure we have a token; if missing, try to re-unlock with saved password once
    const ensureToken = async (): Promise<string | null> => {
      try {
        let tok = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token");
        if (tok) return tok;
        const savedPw = localStorage.getItem("boardrag_saved_pw");
        if (!savedPw) return null;
        const form = new FormData();
        form.append("password", savedPw);
        const resp = await fetch(`${API_BASE}/auth/unlock`, { method: 'POST', body: form });
        if (!resp.ok) return null;
        const data = await resp.json();
        const t = String(data?.token || "");
        const r = String(data?.role || "user");
        if (!t) return null;
        try { sessionStorage.setItem('boardrag_token', t); } catch {}
        try { sessionStorage.setItem('boardrag_role', r); } catch {}
        try { localStorage.setItem('boardrag_token', t); } catch {}
        try { localStorage.setItem('boardrag_role', r); } catch {}
        return t;
      } catch { return null; }
    };
    // Ensure a valid token before opening stream
    let tokenNow: string | null = null;
    try { tokenNow = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token"); } catch {}
    if (!tokenNow) {
      tokenNow = await ensureToken();
    }
    if (tokenNow) {
      try { url.searchParams.set('token', tokenNow); } catch {}
    } else {
      // No token available → cannot proceed; force a lightweight reset to prompt login
      setIsStreaming(false);
      try {
        fetch(`${API_BASE}/admin/log`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ line: '[client] No auth token available; redirecting to /reset' }) });
      } catch {}
      try { window.location.href = '/reset'; } catch {}
      return;
    }

    // Probe authorization up-front: if token is invalid, immediately kick to login
    try {
      if (tokenNow) {
        const probeUrl = new URL(`${API_BASE}/section-chunks`);
        probeUrl.searchParams.set("limit", "1");
        probeUrl.searchParams.set("token", tokenNow);
        const probe = await fetch(probeUrl.toString(), { cache: "no-store" });
        if (probe.status === 401) { kickToLogin(); return; }
      }
    } catch {}

    // Unified SSE event handler (used by EventSource or fetch-stream parser)
    let acc = "";
    const mergeChunk = (previous: string, next: string): string => {
      // Deduplicate overlaps when transport resends a boundary; look back up to 200 chars
      const MAX_OVERLAP = 200;
      const base = previous ?? "";
      const piece = next ?? "";
      if (!base) return piece;
      if (!piece) return base;
      const start = Math.max(0, base.length - MAX_OVERLAP);
      const window = base.slice(start);
      const max = Math.min(window.length, piece.length);
      for (let k = max; k > 0; k--) {
        if (window.slice(window.length - k) === piece.slice(0, k)) {
          return base + piece.slice(k);
        }
      }
      return base + piece;
    };
    const handleSseData = (payload: string) => {
      try {
        const parsed = JSON.parse(payload);
        if (parsed.type === "token") {
          // Adopt the server-announced req id on first token; ignore mismatches later
          if (parsed.req && typeof parsed.req === 'string') {
            if (streamReqIdRef.current == null) {
              streamReqIdRef.current = parsed.req;
            } else if (streamReqIdRef.current !== parsed.req) {
              return; // late or cross-stream chunk – ignore
            }
          }
          acc = mergeChunk(acc, String(parsed.data));
          setMessages((cur) => {
            const last = cur[cur.length - 1];
            if (!last || last.role !== "assistant") return [...cur, { role: "assistant", content: acc, pinned: false }];
            const updated = [...cur]; updated[updated.length - 1] = { ...last, content: acc }; return updated;
          });
        } else if (parsed.type === "done") {
          if (parsed.req && typeof parsed.req === 'string' && streamReqIdRef.current && parsed.req !== streamReqIdRef.current) {
            return; // not our stream
          }
          try { eventRef.current && (eventRef.current as any).close?.(); } catch {}
          eventRef.current = null;
          setIsStreaming(false);
          removeRetryable(lastSubmittedUserIndexRef.current);
          streamReqIdRef.current = null;
          // Seed section-chunks cache from SSE meta if present
          try {
            const meta = parsed.meta || {};
            const chunks = Array.isArray(meta.chunks) ? meta.chunks : [];
            if (chunks && chunks.length > 0) {
              const gameKey = String(selectedGame || "");
              const groups = new Map<string, any[]>();
              for (const c of chunks) {
                const secNum = String(c.section_number || "").trim();
                const secLbl = String(c.section || "").trim();
                let keyNum = "";
                if (secNum) keyNum = secNum;
                else {
                  const m = secLbl.match(/^\s*([A-Za-z]?\d+(?:\.[A-Za-z0-9]+)*)\b/);
                  if (m) keyNum = m[1];
                }
                const keys: string[] = [];
                if (keyNum) keys.push(keyNum);
                if (secLbl) keys.push(secLbl);
                // Always group full set under a generic key as well
                if (keys.length === 0) keys.push("__all__");
                for (const k of keys) {
                  const ck = `${gameKey}::${k}`;
                  if (!groups.has(ck)) groups.set(ck, []);
                  groups.get(ck)!.push(c);
                }
              }
              // Write to cache
              groups.forEach((arr, ck) => {
                try { sectionCacheRef.current.set(ck, arr as any); } catch {}
              });
            }
          } catch {}
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
        // Prefer Authorization header for fetch path if token present
        try {
          let t: string | null = sessionStorage.getItem("boardrag_token");
          if (!t) t = localStorage.getItem("boardrag_token");
          if (t) headers['Authorization'] = `Bearer ${t}`;
          // Also ensure token is on URL for servers that read query param only
          if (t) try { url.searchParams.set('token', t); } catch {}
        } catch {}
        // Create/replace an AbortController so Stop can cancel the fetch
        const controller = new AbortController();
        abortRef.current = controller;
        let resp = await fetch(url.toString(), { headers, cache: "no-store", signal: controller.signal });
        if (resp.status === 401) {
          // Try to transparently refresh token once via saved password
          try {
            const savedPw = localStorage.getItem('boardrag_saved_pw');
            if (savedPw) {
              const form = new FormData();
              form.append('password', savedPw);
              const unlockResp = await fetch(`${API_BASE}/auth/unlock`, { method: 'POST', body: form });
              if (unlockResp.ok) {
                const data = await unlockResp.json();
                const t = String(data?.token || '');
                const r = String(data?.role || 'user');
                if (t) {
                  try { sessionStorage.setItem('boardrag_token', t); } catch {}
                  try { localStorage.setItem('boardrag_token', t); } catch {}
                  try { sessionStorage.setItem('boardrag_role', r); } catch {}
                  try { localStorage.setItem('boardrag_role', r); } catch {}
                  try { url.searchParams.set('token', t); } catch {}
                  headers['Authorization'] = `Bearer ${t}`;
                  resp = await fetch(url.toString(), { headers, cache: 'no-store', signal: controller.signal });
                }
              }
            }
          } catch {}
        }
        // If unauthorized, force full reset and re-login
        if (!resp.ok && resp.status === 401) {
          try {
            sessionStorage.removeItem("boardrag_role");
            sessionStorage.removeItem("boardrag_token");
            localStorage.removeItem("boardrag_role");
            localStorage.removeItem("boardrag_token");
            localStorage.removeItem("boardrag_saved_pw");
          } catch {}
          try { window.location.reload(); } catch {}
          return;
        }
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
        // Flush any remaining decoder state and process leftover buffer
        buffer += decoder.decode();
        if (NDJSON) {
          let idx;
          while ((idx = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (line) handleSseData(line);
          }
          const tail = buffer.trim();
          if (tail) handleSseData(tail);
        } else {
          let idx;
          while ((idx = buffer.indexOf("\n\n")) !== -1) {
            const raw = buffer.slice(0, idx); buffer = buffer.slice(idx + 2);
            const dataLine = raw.split("\n").find((l) => l.startsWith("data:"));
            if (dataLine) handleSseData(dataLine.slice(5).trim());
          }
          const raw = buffer;
          if (raw && raw.indexOf("data:") !== -1) {
            const dataLine = raw.split("\n").find((l) => l.startsWith("data:"));
            if (dataLine) handleSseData(dataLine.slice(5).trim());
          }
        }
      } catch {
        setIsStreaming(false); addRetryable(lastSubmittedUserIndexRef.current);
      } finally {
        // Clear abort ref after completion/abort
        abortRef.current = null;
      }
    };

    // If production (or explicitly toggled), prefer fetch streaming immediately
    const FORCE_FETCH = (process.env.NEXT_PUBLIC_SSE_FETCH === '1');
    if (FORCE_FETCH) {
      streamWithFetch();
      return;
    }

    // Default path: EventSource only (no transport fallback)
    const es = new EventSource(url.toString());
    eventRef.current = es;

    es.onmessage = (ev) => {
      handleSseData(ev.data);
    };
    es.onerror = async (_ev: any) => {
      try { es.close(); } catch {}
      eventRef.current = null;
      // If unauthorized, clear creds and reload to login
      try {
        if (tokenNow) {
          const probeUrl = new URL(`${API_BASE}/section-chunks`);
          probeUrl.searchParams.set("limit", "1");
          probeUrl.searchParams.set("token", tokenNow);
          const probe = await fetch(probeUrl.toString(), { cache: "no-store" });
          if (probe.status === 401) { kickToLogin(); return; }
        }
      } catch {}
      // Otherwise stop streaming and allow user to retry
      setIsStreaming(false);
      addRetryable(lastSubmittedUserIndexRef.current);
    };
  };
  const togglePin = (assistantIndex: number) => {
    // Find absolute assistant position in messages
    let seenAssistants = -1;
    let absAssistant = -1;
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === "assistant") {
        seenAssistants += 1;
        if (seenAssistants === assistantIndex) { absAssistant = i; break; }
      }
    }
    if (absAssistant < 0) return;
    const target = messages[absAssistant];
    const newMsgs = [...messages];
    newMsgs[absAssistant] = { ...target, pinned: !target.pinned } as Message;
    setMessages(newMsgs);
    try {
      if (selectedGame) {
        const key = `boardrag_conv:${sessionId}:${selectedGame}`;
        localStorage.setItem(key, JSON.stringify(newMsgs));
      }
    } catch {}
  };


  const onSubmit = () => {
    const q = input.trim();
    if (!q || !selectedGame) return;
    // If this browser session was blocked, short-circuit locally without hitting the server
    try { if (sessionStorage.getItem("boardrag_blocked") === "1") { return; } } catch {}
    setInput("");
    startQuery(q, null);
  };

  const onSubmitWithStyle = (style: PromptStyle) => {
    const q = input.trim();
    if (!q || !selectedGame) return;
    try { if (sessionStorage.getItem("boardrag_blocked") === "1") { return; } } catch {}
    setInput("");
    setPromptStyle(style);
    startQuery(q, null, style);
  };

  const onStop = () => {
    // Close EventSource if active
    try { eventRef.current?.close(); } catch {}
    eventRef.current = null;
    // Abort fetch streaming if active
    try { abortRef.current?.abort(); } catch {}
    abortRef.current = null;
    // Clear any pending fallback timer
    if (switchTimerRef.current != null) { try { clearTimeout(switchTimerRef.current); } catch {} switchTimerRef.current = null; }
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
  const postLog = (line: string) => {
    try {
      fetch(`${API_BASE}/admin/log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ line }),
      });
    } catch {}
  };
  // Responsive breakpoints
  const [isDesktop, setIsDesktop] = useState<boolean>(typeof window !== 'undefined' ? window.innerWidth >= 800 : true);
  const [canShowPreview, setCanShowPreview] = useState<boolean>(typeof window !== 'undefined' ? window.innerWidth >= 1550 : false);
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onResize = () => {
      setIsDesktop(window.innerWidth >= 800);
      setCanShowPreview(window.innerWidth >= 1550);
    };
    window.addEventListener('resize', onResize);
    onResize();
    return () => window.removeEventListener('resize', onResize);
  }, []);
  // Modal state for section chunks
  const [sectionModalOpen, setSectionModalOpen] = useState<boolean>(false);
  const [sectionLoading, setSectionLoading] = useState<boolean>(false);
  const [sectionError, setSectionError] = useState<string>("");
  const [sectionTitle, setSectionTitle] = useState<string>("");
  const [sectionChunks, setSectionChunks] = useState<Array<{ uid?: string; text: string; source: string; page?: number; section?: string; section_number?: string }>>([]);
  const [pdfMeta, setPdfMeta] = useState<{ filename?: string; pages?: number } | null>(null);
  // Desktop preview state
  const [previewOpen, setPreviewOpen] = useState<boolean>(false);
  const [previewTitle, setPreviewTitle] = useState<string>("");
  const [previewChunks, setPreviewChunks] = useState<Array<{ uid?: string; text: string; source: string; page?: number; section?: string; section_number?: string }>>([]);
  const [previewPdfMeta, setPreviewPdfMeta] = useState<{ filename?: string; pages?: number } | null>(null);
  const [previewTargetPage, setPreviewTargetPage] = useState<number | null>(null);
  const [anchorNonce, setAnchorNonce] = useState<number>(0);
  // Stable auth token to avoid rebuilding PDF URLs on every render
  const tokenRef = useRef<string | null>(null);
  // Portal root inside chat for mobile modal alignment
  const chatModalRootRef = useRef<HTMLDivElement | null>(null);
  // Local cache for section chunks to avoid refetching
  const sectionCacheRef = useRef<Map<string, Array<{ text: string; source: string; page?: number; section?: string; section_number?: string; rects_norm?: any }>>>(new Map());

  useEffect(() => {
    try {
      tokenRef.current = sessionStorage.getItem('boardrag_token') || localStorage.getItem('boardrag_token');
    } catch {}
  }, []);

  const openSectionModal = async (sec: string, uid?: string, meta?: any) => {
    if (!sec) return;
    setSectionTitle(sec);
    setPreviewTitle(sec);
    setSectionError("");
    setSectionLoading(true);
    // Always use the preview panel (desktop and mobile)
      setPreviewOpen(true);
      setSectionModalOpen(false);
    try {
      // If metadata from citation is provided (file, page, header), preselect file and page
      try {
        if (meta && typeof meta === 'object') {
          const mfile = String((meta.file || meta.filename || '') as string).toLowerCase();
          const mpage = Number(meta.page);
          if (mfile) {
            const norm = (s: string) => (s || '').trim().toLowerCase();
            const samePreview = !!(previewPdfMeta && norm(previewPdfMeta.filename || '') === norm(mfile));
            const sameModal = !!(pdfMeta && norm(pdfMeta.filename || '') === norm(mfile));
            if (!sameModal) setPdfMeta({ filename: mfile, pages: undefined });
            if (!samePreview) setPreviewPdfMeta({ filename: mfile, pages: undefined });
            try { if (selectedGame) localStorage.setItem(`boardrag_last_pdf:${selectedGame}`, mfile); } catch {}
          }
          if (isFinite(mpage) && mpage > 0) {
            try { setPreviewTargetPage(mpage); setAnchorNonce((n) => (n + 1) | 0); } catch {}
          }
        }
      } catch {}
      const preferredPage = (() => { try { const n = Number((meta && (meta as any).page)); return isFinite(n) && n > 0 ? n : null; } catch { return null; } })();
      // If we have complete metadata, avoid any server call: synthesize minimal chunks and proceed
      if (meta && (meta as any).file && preferredPage) {
        const mfile = String((meta as any).file || '').toLowerCase();
        const headerText = String((meta as any).header || sec || '').trim();
        const syntheticChunks = [
          { text: headerText, source: mfile, page: Number(preferredPage) - 1, section: sec, section_number: sec }
        ];
        // Cache under multiple keys
        try {
          const keys: string[] = [sec];
          try {
            const m = sec.match(/^[A-Za-z]?\d+(?:\.[A-Za-z0-9]+)*/);
            if (m && m[0] && m[0] !== sec) keys.push(m[0]);
            const up = sec.toUpperCase(); if (up !== sec) keys.push(up);
          } catch {}
          for (const k of keys) {
            const ck = `${selectedGame || ''}::${k}`;
            sectionCacheRef.current.set(ck, syntheticChunks as any);
          }
        } catch {}
        try { setSectionChunks(syntheticChunks as any); } catch {}
        try { setPreviewChunks(syntheticChunks as any); } catch {}
        // Update viewer metas
        const norm = (s: string) => (s || '').trim().toLowerCase();
        const samePreview = !!(previewPdfMeta && norm(previewPdfMeta.filename || '') === norm(mfile));
        const sameModal = !!(pdfMeta && norm(pdfMeta.filename || '') === norm(mfile));
        if (!sameModal) setPdfMeta({ filename: mfile, pages: undefined });
        if (!samePreview) setPreviewPdfMeta({ filename: mfile, pages: undefined });
        try { setPreviewTargetPage(preferredPage as number); setAnchorNonce((n) => (n + 1) | 0); } catch {}
        // PdfPreview now owns initial anchoring/scrolling
        setSectionLoading(false);
        return;
      }
      // 1) Try cache first (game + section key)
      const mk = (k: string) => `${selectedGame || ''}::${k}`;
      const candidates: string[] = [sec];
      // Also try numeric extraction and label upper-case variants
      try {
        const m = sec.match(/^[A-Za-z]?\d+(?:\.[A-Za-z0-9]+)*/);
        if (m && m[0] && m[0] !== sec) candidates.push(m[0]);
        const up = sec.toUpperCase(); if (up !== sec) candidates.push(up);
      } catch {}
      let cached: any[] | undefined;
      for (const k of candidates) {
        const hit = sectionCacheRef.current.get(mk(k));
        if (hit && hit.length) { cached = hit as any[]; break; }
      }
      if (cached && Array.isArray(cached) && cached.length > 0) {
        try { setSectionChunks(cached as any); } catch {}
        try { setPreviewChunks(cached as any); } catch {}
        // PdfPreview now owns initial anchoring/scrolling
        const inferFromChunk = String(cached[0]?.source || '').toLowerCase();
        const currentKnown = (isDesktop && canShowPreview) ? (previewPdfMeta?.filename || '') : (pdfMeta?.filename || '');
        const fn = (currentKnown || inferFromChunk || '').toLowerCase();
        const norm = (s: string) => (s || '').trim().toLowerCase();
        const samePreview = !!(previewPdfMeta && norm(previewPdfMeta.filename || '') === norm(fn));
        const sameModal = !!(pdfMeta && norm(pdfMeta.filename || '') === norm(fn));
        if (!sameModal) setPdfMeta({ filename: fn, pages: undefined });
        if (!samePreview) setPreviewPdfMeta({ filename: fn, pages: undefined });
        // PdfPreview now owns initial anchoring/scrolling
        setSectionLoading(false);
        return;
      }
      // Ensure token
      let t: string | null = null;
      try { t = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token"); } catch {}
      const url = new URL(`${API_BASE}/section-chunks`);
      url.searchParams.set("section", sec);
      if (uid) url.searchParams.set("id", uid);
      if (selectedGame) url.searchParams.set("game", selectedGame);
      if (t) url.searchParams.set("token", t);
      const resp = await fetch(url.toString(), { headers: t ? { Authorization: `Bearer ${t}` } as any : undefined });
      if (!resp.ok) {
        const txt = await resp.text().catch(() => "error");
        throw new Error(`HTTP ${resp.status}: ${txt}`);
      }
      const data = await resp.json();
      const chunks = Array.isArray(data?.chunks) ? data.chunks : [];
      // 2) Cache the result for this section under multiple keys (numeric and label)
      try {
        const keys: string[] = [sec];
        try {
          const m = sec.match(/^[A-Za-z]?\d+(?:\.[A-Za-z0-9]+)*/);
          if (m && m[0] && m[0] !== sec) keys.push(m[0]);
          const up = sec.toUpperCase(); if (up !== sec) keys.push(up);
        } catch {}
        for (const k of keys) {
          const ck = `${selectedGame || ''}::${k}`;
          sectionCacheRef.current.set(ck, chunks as any);
        }
      } catch {}
      // Update modal and preview chunk lists so both viewers have content
      try { setSectionChunks(chunks as any); } catch {}
      try { setPreviewChunks(chunks as any); } catch {}
      // Stash filename for viewer: prefer existing loaded filename, else infer from chunks
      const inferFromChunk = chunks && chunks.length > 0 ? String(chunks[0].source || '').toLowerCase() : '';
      // Prefer the last known filename if it matches the inferred; otherwise use inferred
      const currentKnown = (isDesktop && canShowPreview) ? (previewPdfMeta?.filename || '') : (pdfMeta?.filename || '');
      const fn = (currentKnown || inferFromChunk || '').toLowerCase();
      const norm = (s: string) => (s || '').trim().toLowerCase();
      const samePreview = !!(previewPdfMeta && norm(previewPdfMeta.filename || '') === norm(fn));
      const sameModal = !!(pdfMeta && norm(pdfMeta.filename || '') === norm(fn));
      // Only update each viewer's meta if its file actually changed to avoid reloading
      // Only change each viewer if it's pointing at a different file; otherwise keep current state to avoid re-mount flicker
      if (!sameModal) setPdfMeta({ filename: fn, pages: undefined });
      if (!samePreview) setPreviewPdfMeta({ filename: fn, pages: undefined });
      // Persist last-used PDF per game
      try {
        if (selectedGame && fn) {
          localStorage.setItem(`boardrag_last_pdf:${selectedGame}`, fn);
        }
      } catch {}
       // Set target page for PdfPreview.tsx to handle
       try {
         const pageSet = new Set<number>();
         chunks.forEach((c: any) => { if (typeof c.page === 'number') pageSet.add(Number(c.page) + 1); });
         const citedPages = Array.from(pageSet).sort((a,b) => a-b);
         const targetPage = (preferredPage && preferredPage > 0) ? preferredPage : (citedPages.length > 0 ? citedPages[0] : 1);
         try { setPreviewTargetPage(targetPage); setAnchorNonce((n) => (n + 1) | 0); } catch {}
       } catch {}
    } catch (e: any) {
      setSectionError(String(e?.message || e || "Failed to load section"));
    } finally {
      setSectionLoading(false);
    }
  };

  // Load last-used PDF filename when game changes (for preview context)
  useEffect(() => {
    if (!selectedGame) {
      // Clear PDF previewer when no game is selected
      setPdfMeta({ filename: undefined, pages: undefined });
      setPreviewPdfMeta({ filename: undefined, pages: undefined });
      return;
    }

    let cancelled = false; // Track if this effect was cancelled by a new game selection

    const loadPdfForGame = async () => {
      try {
        // Clear PDF previewer immediately when game switches
        setPdfMeta({ filename: undefined, pages: undefined });
        setPreviewPdfMeta({ filename: undefined, pages: undefined });

        // Short delay to let any previous PDF components fully unmount
        await new Promise(resolve => setTimeout(resolve, 50));
        
        if (cancelled) return; // Exit early if user switched games again

        // Check for stored PDF preference for this game
        const key = `boardrag_last_pdf:${selectedGame}`;
        const storedPdf = localStorage.getItem(key) || '';
        
        if (storedPdf) {
          if (!cancelled) {
            // Load stored PDF preference
            setPdfMeta({ filename: storedPdf, pages: undefined });
            setPreviewPdfMeta({ filename: storedPdf, pages: undefined });
          }
          return;
        }

        // No stored preference - fetch PDFs for this game to decide what to do
        let token: string | null = null;
        try { 
          token = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token"); 
        } catch {}
        
        const url = new URL(`${API_BASE}/game-pdfs`);
        url.searchParams.set("game", selectedGame);
        if (token) url.searchParams.set("token", token);
        
        const resp = await fetch(url.toString(), { 
          headers: token ? { Authorization: `Bearer ${token}` } as any : undefined 
        });
        
        if (cancelled) return; // Exit early if user switched games again
        
        if (!resp.ok) {
          console.warn(`Failed to fetch PDFs for game ${selectedGame}: ${resp.status}`);
          return;
        }

        const data = await resp.json();
        const pdfs = data.pdfs || [];

        if (!cancelled && pdfs.length === 1) {
          // Exactly one PDF - auto-load it
          const pdfFilename = pdfs[0];
          setPdfMeta({ filename: pdfFilename, pages: undefined });
          setPreviewPdfMeta({ filename: pdfFilename, pages: undefined });
        }
        // Note: Zero PDFs or multiple PDFs - keep previewer empty (already cleared above)
      } catch (error) {
        if (!cancelled) {
          console.warn(`Error loading PDF for game ${selectedGame}:`, error);
        }
      }
    };

    loadPdfForGame();

    // Cleanup function to cancel ongoing operations when game changes
    return () => {
      cancelled = true;
    };
  }, [selectedGame]);

  // If we have a last-used section for the game, open it automatically (desktop preview only)
  useEffect(() => {
    if (!selectedGame) return;
    if (!canShowPreview) return;
    try {
      const sec = localStorage.getItem(`boardrag_last_section:${selectedGame}`);
      if (sec) openSectionModal(sec);
    } catch {}
  }, [selectedGame, canShowPreview]);

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



  const parseCitationMeta = (title?: string | null): any | undefined => {
    if (!title) return undefined;
    const decodeHtml = (s: string) => s
      .replace(/&#39;/g, "'")
      .replace(/&quot;/g, '"')
      .replace(/&amp;/g, '&');
    try {
      // Prefer direct JSON (we now store literal JSON in the title)
      return JSON.parse(decodeHtml(title));
    } catch {}
    try {
      // Fallback: some older messages may still use base64
      // Note: atob may throw on non-base64 strings
      return JSON.parse(atob(title));
    } catch {}
    try {
      return JSON.parse(decodeURIComponent(title));
    } catch {}
    return undefined;
  };

  // Simplified citation opener: first click snaps to page; if already on that page, centers on the section name
  const openCitation = (sec: string, meta: any | undefined) => {
    try {
      setPreviewOpen(true);
      const mfile = (() => { try { return String(meta?.file || meta?.filename || '').toLowerCase(); } catch { return ''; } })();
      const mpage = (() => { try { const n = Number(meta?.page); return isFinite(n) && n > 0 ? n : 1; } catch { return 1; } })();
      const sectionText = (() => { try { return String(meta?.snippet || meta?.header || sec || '').trim(); } catch { return String(sec || ''); } })();
      // Set target PDF and page
      if (mfile) {
        const norm = (s: string) => (s || '').trim().toLowerCase();
        const samePreview = !!(previewPdfMeta && norm(previewPdfMeta.filename || '') === norm(mfile));
        const sameModal = !!(pdfMeta && norm(pdfMeta.filename || '') === norm(mfile));
        if (!sameModal) setPdfMeta({ filename: mfile, pages: undefined });
        if (!samePreview) setPreviewPdfMeta({ filename: mfile, pages: undefined });
        try { if (selectedGame) localStorage.setItem(`boardrag_last_pdf:${selectedGame}`, mfile); } catch {}
      }
      try { 
        setPreviewTargetPage(mpage); 
        setAnchorNonce((n) => (n + 1) | 0);
      } catch {}

      // Synthesize minimal chunks so the unified preview panel can render/highlight the target
      try {
        const syntheticChunks = [
          { text: sectionText || sec, source: mfile, page: Number(mpage) - 1, section: sec, section_number: sec } as any,
        ];
        setPreviewChunks(syntheticChunks as any);
        // Also seed cache under common keys to enable quick reopening
        const keys: string[] = [sec];
        try {
          const m = sec.match(/^[A-Za-z]?\d+(?:\.[A-Za-z0-9]+)*/);
          if (m && m[0] && m[0] !== sec) keys.push(m[0]);
          const up = sec.toUpperCase(); if (up !== sec) keys.push(up);
        } catch {}
        const gameKey = String(selectedGame || '');
        for (const k of keys) {
          const ck = `${gameKey}::${k}`;
          try { sectionCacheRef.current.set(ck, syntheticChunks as any); } catch {}
        }
      } catch {}
      
    } catch {}
  };

  return (
    <>
      <div className={`container${(previewOpen && !canShowPreview) ? ' preview-open' : ''}`}>
        {/* Desktop PDF preview (left of chat) - only render on wide screens */}
        {canShowPreview && (
          <div className={`preview-panel${previewOpen ? ' has-content' : ''}`}>
            {/* Back button in overlay mode (mobile/all) */}
            <button className="btn preview-back" onClick={() => setPreviewOpen(false)} aria-label="Back" title="Back">
              ←
            </button>
            <div className="preview-body">
              {sectionLoading && (
                <div className="indicator" style={{ gap: 8, position: 'absolute', right: 8, top: 8 }}>
                  <span className="spinner" />
                </div>
              )}
              {!!sectionError && !sectionLoading && (
                <div className="muted" style={{ color: 'var(--danger, #b00020)' }}>{sectionError}</div>
              )}
              {((previewPdfMeta && previewPdfMeta.filename) || (previewChunks && previewChunks.length > 0)) && (
                <div className="preview-top pdf-viewer">
                  {(() => {
                    const inferred = (previewChunks && previewChunks[0] && previewChunks[0].source ? String(previewChunks[0].source) : '').toLowerCase();
                    const filename = String((previewPdfMeta && previewPdfMeta.filename) ? previewPdfMeta.filename : inferred).toLowerCase();
                    const pageSet = new Set<number>();
                    previewChunks.forEach((c) => { if (typeof c.page === 'number') pageSet.add(Number(c.page) + 1); });
                    const citedPages = Array.from(pageSet).sort((a,b) => a-b);
                    const totalPages = (previewPdfMeta && previewPdfMeta.pages) ? Number(previewPdfMeta.pages) : undefined;
                    let pages = totalPages ? Array.from({ length: totalPages }, (_v, i) => i + 1) : (citedPages.length > 0 ? citedPages : [1]);
                    const targetPage = citedPages.length > 0 ? citedPages[0] : 1;
                    const pdfUrl = filename ? `${API_BASE}/pdf?filename=${encodeURIComponent(filename)}${(() => { const t = tokenRef.current; return t ? `&token=${encodeURIComponent(t)}` : ''; })()}` : '';
                    if (!pdfUrl) return <div className="muted">Missing PDF filename.</div>;
                    const byPage = (p: number) => previewChunks.filter((c) => (Number(c.page) + 1) === Number(p));
                    const highlight = (el: HTMLElement, texts: string[]) => {
                      try {
                        const layer = el.querySelector('.textLayer');
                        if (!layer) return;
                        const spans = Array.from(layer.querySelectorAll('span')) as HTMLSpanElement[];
                        const needles = texts.map((t) => (t || '').trim()).filter(Boolean);
                        if (needles.length === 0) return;
                        const joined = needles.join('\n');
                        const normalizedNeedles = needles;
                        let i = 0;
                        while (i < spans.length) {
                          let j = i; let acc = '';
                          while (j < spans.length && acc.length < joined.length + 50) {
                            acc += spans[j].textContent || '';
                            const match = normalizedNeedles.some((n) => acc.includes(n));
                            if (match) { for (let k = i; k <= j; k++) spans[k].classList.add('chunk-highlight'); i = j + 1; break; }
                            j += 1;
                          }
                          i += 1;
                        }
                      } catch {}
                    };
                    return (
                      <PdfPreview
                        API_BASE={API_BASE}
                        token={tokenRef.current}
                        title={previewTitle}
                        chunks={previewChunks as any}
                        pdfMeta={previewPdfMeta as any}
                        setPdfMeta={(m) => setPreviewPdfMeta(m as any)}
                        targetPage={previewTargetPage || undefined}
                        adjacentPageWindow={1}
                        anchorNonce={anchorNonce}
                      />
                    );
                  })()}
                </div>
              )}
              {/* Chunk reveal panel removed */}
                  </div>
                        </div>
          )}
        {/* Chat column */}
        <div className={`chat surface${sheetOpen ? ' menu-open' : ''}`} style={{ height: "100%", position: 'relative' }}>
          {/* Root container for mobile modal portal to guarantee alignment with chat */}
          <div id="chat-modal-root" ref={chatModalRootRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} />
          {/* Modal preview mounted into chat modal root so width/position match chat exactly */}
          {(!canShowPreview && chatModalRootRef.current) && (
            createPortal(
              <div className={`modal-preview${previewOpen ? ' open' : ''}`}>
                <div className="preview-body">
                  {sectionLoading && (
                    <div className="indicator" style={{ gap: 8, position: 'absolute', right: 8, top: 8 }}>
                      <span className="spinner" />
                      </div>
                  )}
                  {!!sectionError && !sectionLoading && (
                    <div className="muted" style={{ color: 'var(--danger, #b00020)' }}>{sectionError}</div>
                  )}
                  {((previewPdfMeta && previewPdfMeta.filename) || (previewChunks && previewChunks.length > 0)) && (
                    <div className="preview-top pdf-viewer">
                      {(() => {
                        const inferred = (previewChunks && (previewChunks as any)[0] && (previewChunks as any)[0].source ? String((previewChunks as any)[0].source) : '').toLowerCase();
                        const filename = String((previewPdfMeta && (previewPdfMeta as any).filename) ? (previewPdfMeta as any).filename : inferred).toLowerCase();
                        const pageSet = new Set<number>();
                        (previewChunks as any).forEach((c: any) => { if (typeof c.page === 'number') pageSet.add(Number(c.page) + 1); });
                        const citedPages = Array.from(pageSet).sort((a,b) => a-b);
                        const totalPages = (previewPdfMeta && (previewPdfMeta as any).pages) ? Number((previewPdfMeta as any).pages) : undefined;
                        let pages = totalPages ? Array.from({ length: totalPages }, (_v, i) => i + 1) : (citedPages.length > 0 ? citedPages : [1]);
                        const targetPage = citedPages.length > 0 ? citedPages[0] : 1;
                        const pdfUrl = filename ? `${API_BASE}/pdf?filename=${encodeURIComponent(filename)}${(() => { const t = tokenRef.current; return t ? `&token=${encodeURIComponent(t)}` : ''; })()}` : '';
                        if (!pdfUrl) return <div className="muted">Missing PDF filename.</div>;
                        const byPage = (p: number) => (previewChunks as any).filter((c: any) => (Number(c.page) + 1) === Number(p));
                        const highlight = (el: HTMLElement, texts: string[]) => {
                          try {
                            const layer = el.querySelector('.textLayer');
                            if (!layer) return;
                            const spans = Array.from(layer.querySelectorAll('span')) as HTMLSpanElement[];
                            const needles = texts.map((t) => (t || '').trim()).filter(Boolean);
                            if (needles.length === 0) return;
                            const joined = needles.join('\n');
                            const normalizedNeedles = needles;
                            let i = 0;
                            while (i < spans.length) {
                              let j = i; let acc = '';
                              while (j < spans.length && acc.length < joined.length + 50) {
                                acc += spans[j].textContent || '';
                                const match = normalizedNeedles.some((n) => acc.includes(n));
                                if (match) { for (let k = i; k <= j; k++) spans[k].classList.add('chunk-highlight'); i = j + 1; break; }
                                j += 1;
                              }
                              i += 1;
                            }
                          } catch {}
                        };
                        return (
                          <PdfPreview
                            API_BASE={API_BASE}
                            token={tokenRef.current}
                            title={previewTitle}
                            chunks={previewChunks as any}
                            pdfMeta={previewPdfMeta as any}
                            setPdfMeta={(m) => setPreviewPdfMeta(m as any)}
                            targetPage={previewTargetPage || undefined}
                            adjacentPageWindow={1}
                          />
                        );
                      })()}
                </div>
              )}
                  {/* Chunk reveal panel removed */}
            </div>
                <button className="btn preview-back" onClick={() => setPreviewOpen(false)} aria-label="Back" title="Back">←</button>
              </div>,
              chatModalRootRef.current
            )
        )}
          <div className="row title" style={{ gap: 12, justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="app-title" style={{ whiteSpace: 'nowrap' }}>Board Game Jippity</span>
            <div className="row" style={{ gap: 8, alignItems: 'center' }}>
              <select
                className="select"
                value={selectedGame || ""}
                onChange={(e) => {
                  if (selectedGame) {
                    const oldKey = `boardrag_conv:${sessionId}:${selectedGame}`;
                    try { localStorage.setItem(oldKey, JSON.stringify(messages)); } catch {}
                  }
                  setSelectedGame(e.target.value);
                }}
                style={{ minWidth: 260, maxWidth: 320 }}
                aria-label="Game"
              >
                <option value="">Select game…</option>
                {games.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Horizontal history strip */}
          <HistoryStrip
            labels={bookmarkLabels}
            containerRef={historyStripRef}
            onClickLabel={(userIdx) => scrollToAssistant(userIdx)}
            longPressHandlersFor={(assistantIdx) => longPressHandlers(assistantIdx)}
            bookmarkUserIndices={bookmarkUserIndices}
            bookmarkAssistantIndices={bookmarkAssistantIndices}
            isDragging={() => (historyDraggingFlag.current || historyDragRef.current.dragged)}
          />

          {/* Game selector row removed; dropdown moved into title */}

          {/* Chat history */}
          <div className="chat-scroll" ref={chatScrollRef}>
            {(() => {
              let assistantCounter = -1;
              let userCounter = -1;
              return messages.map((m, i) => {
                const props: any = { key: i, className: `bubble ${m.role}` };
                let acForHandlers: number | null = null;
                let ucForHandlers: number | null = null;
                if (m.role === "assistant") {
                  assistantCounter += 1;
                  props["data-assistant-index"] = assistantCounter;
                  acForHandlers = assistantCounter;
                } else if (m.role === "user") {
                  userCounter += 1;
                  props["data-user-index"] = userCounter;
                  ucForHandlers = userCounter;
                }
                                 const isUser = m.role === "user";
                 const showActions = isUser && ucForHandlers != null && retryableUsers.includes(ucForHandlers);
                 // Add waiting animation class to the last submitted user message when waiting for stream to start
                 const isWaitingForResponse = isUser && isStreaming && ucForHandlers === lastSubmittedUserIndexRef.current && !firstTokenSeenRef.current;
                 if (isWaitingForResponse) {
                   props.className += " waiting-response";
                 }
                 return (
                  <React.Fragment key={i}>
                    {m.role === 'assistant' ? (
                      <div className="assistant-row" style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto', columnGap: 6, alignItems: 'end' }}>
                        <div {...props} style={{ ...(props.style || {}), maxWidth: '100%' }}>
                          <ReactMarkdown
                            {...({ urlTransform: (url: string) => url } as any)}
                            components={{
                              a({ href, title, children, ...props }: { href?: string; title?: string; children?: any }) {
                                if (href && typeof href === 'string' && href.startsWith('section:')) {
                                  const sec = decodeURIComponent(href.slice('section:'.length));
                                  return (
                                    <button
                                      className="btn link"
                                      onClick={(e) => {
                                        e.preventDefault();
                                        const meta = parseCitationMeta(title);
                                        // Always handle citations client-side (no server calls)
                                        openCitation(sec, meta);
                                      }}
                                      title={title}
                                      style={{
                                        padding: 0,
                                        background: 'none',
                                        border: 'none',
                                        color: 'var(--accent)',
                                        textDecoration: 'underline',
                                        cursor: 'pointer',
                                      }}
                                    >
                                      {"["}{children}{"]"}
                                    </button>
                                  );
                                }
                                return <a href={href as string} {...(props as any)}>{children}</a>;
                              },
                            }}
                          >
                            {decorateCitations(m.content)}
                          </ReactMarkdown>
                        </div>
                        <div className="assistant-actions" style={{ display: 'flex', flexDirection: 'column', gap: 6, alignSelf: 'end' }}>
                          <button
                            className="btn"
                            title={m.pinned ? "Remove bookmark" : "Add bookmark"}
                            aria-label={m.pinned ? "Remove bookmark" : "Add bookmark"}
                            aria-pressed={!!m.pinned}
                            onClick={() => acForHandlers != null && togglePin(acForHandlers)}
                            style={{ width: 36, height: 36, minHeight: 36, padding: 0, display: 'inline-grid', placeItems: 'center', color: m.pinned ? '#fff' : 'var(--text)', borderColor: m.pinned ? 'var(--accent)' : 'var(--control-border)', background: m.pinned ? 'var(--accent)' : 'var(--control-bg)' }}
                          >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                              <path d="M6 2h8a2 2 0 0 1 2 2v16l-6-4-6 4V4a2 2 0 0 1 2-2z" fill={m.pinned ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                            </svg>
                          </button>
                          <button
                            className="btn"
                            title="Hold to delete"
                            aria-label="Delete QA (hold)"
                            {...(acForHandlers != null ? longPressDeleteAssistant(acForHandlers) : {})}
                            style={{ width: 36, height: 36, minHeight: 36, padding: 0, display: 'inline-grid', placeItems: 'center', background: 'var(--control-bg)', borderColor: 'var(--control-border)' }}
                          >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                              <path d="M3 6h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                              <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                              <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                              <path d="M10 11v7M14 11v7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                            </svg>
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div {...props}>
                        {/* style indicator removed */}
                        <ReactMarkdown
                          // Allow custom schemes like section: without sanitization interfering
                          {...({ urlTransform: (url: string) => url } as any)}
                          components={{
                            a({ href, title, children, ...props }: { href?: string; title?: string; children?: any }) {
                              if (href && typeof href === 'string' && href.startsWith('section:')) {
                                const sec = decodeURIComponent(href.slice('section:'.length));
                                return (
                                  <button
                                    className="btn link"
                                    onClick={(e) => {
                                      e.preventDefault();
                                      try {
                                        if (selectedGame) localStorage.setItem(`boardrag_last_section:${selectedGame}`, sec);
                                      } catch {}
                                      const meta = parseCitationMeta(title);
                                      try {
                                        const mfile = String(meta?.file || '').toLowerCase();
                                        if (mfile && !mfile.endsWith('.pdf')) { meta.file = undefined; }
                                        if (mfile === '1.pdf' || mfile === '0.pdf') { meta.file = undefined; }
                                      } catch {}
                                      // Always handle citations client-side
                                      openCitation(sec, meta);
                                    }}
                                    style={{
                                      padding: 0,
                                      background: 'none',
                                      border: 'none',
                                      color: 'var(--accent)',
                                      textDecoration: 'underline',
                                      cursor: 'pointer',
                                    }}
                                  >
                                    {"["}{children}{"]"}
                                  </button>
                                );
                              }
                              return <a href={href as string} {...(props as any)}>{children}</a>;
                            },
                          }}
                        >
                          {m.content}
                        </ReactMarkdown>
                      </div>
                    )}
                    {showActions && (
                      <div style={{ maxWidth: "92%", marginLeft: "auto", marginTop: 6 }}>
                        <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                          <button
                            className="btn"
                            title="Retry"
                            aria-label="Retry"
                            onClick={() => { if (ucForHandlers == null) return; setRetryableUsers((cur) => cur.filter((n) => n !== ucForHandlers!)); startQuery(m.content, ucForHandlers!); }}
                            style={{ width: 44, height: 44, minHeight: 44, padding: 0, fontSize: 18, color: "#fff", borderColor: "var(--accent)", background: "var(--accent)", display: "inline-grid", placeItems: "center" }}
                          >↻</button>
                          <button
                            className="btn"
                            title="Delete"
                            aria-label="Delete"
                            onClick={() => {
                              // Remove this user message
                              const targetUserIdx = ucForHandlers;
                              let seen = -1;
                              const newList: Message[] = [];
                              for (const msg of messages) {
                                if (msg.role === 'user') {
                                  seen += 1;
                                  if (targetUserIdx != null && seen === targetUserIdx) continue; // skip this one
                                }
                                newList.push(msg);
                              }
                              setMessages(newList);
                              if (ucForHandlers != null) setRetryableUsers((cur) => cur.filter((n) => n !== ucForHandlers!));
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
          <InputRow
            isStreaming={isStreaming}
            input={input}
            onChangeInput={(v) => setInput(v)}
            onSubmit={onSubmit}
            onSubmitWithStyle={onSubmitWithStyle}
            onStop={onStop}
            toggleSheet={() => setSheetOpen((s) => !s)}
            selectedGame={selectedGame}
            promptStyle={promptStyle as any}
            setPromptStyle={(s) => setPromptStyle(s as any)}
          />

          <BottomSheetMenu
            open={sheetOpen}
            onClose={() => setSheetOpen(false)}
            games={games}
            selectedGame={selectedGame}
            setSelectedGame={(g) => setSelectedGame(g)}
            sessionId={sessionId}
            messages={messages}
            promptStyle={promptStyle}
            setPromptStyle={(s) => setPromptStyle(s as any)}
            includeWeb={includeWeb}
            setIncludeWeb={(v) => setIncludeWeb(v)}
            pdfSmoothScroll={pdfSmoothScroll}
            setPdfSmoothScroll={(v) => setPdfSmoothScroll(v)}
            onUpload={onUpload}
            uploadInputRef={uploadInputRef}
            uploading={uploading}
            uploadMsg={uploadMsg}
          />
        </div>


      </div>
      {/* Modal removed; preview panel reused for all screens */}
    </>
  );
}


