"use client";

import React, { useEffect, useRef, useState, useMemo } from "react";
import useSWR from "swr";

import { API_BASE } from "../../lib/config";
const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function AdminPage() {
  const [role, setRole] = useState<string>("none");
  const [pw, setPw] = useState<string>("");
  const [message, setMessage] = useState<string>("");
  const [busy, setBusy] = useState<null | "rebuild" | "refresh" | "rechunk" | "delete" | "rename">(null);
  const [consoleText, setConsoleText] = useState<string>("");
  const consoleRef = useRef<HTMLPreElement | null>(null);

  const { data: gamesData, mutate: refetchGames } = useSWR<{ games: string[] }>(`${API_BASE}/games`, fetcher);
  const { data: pdfChoicesData, mutate: refetchChoices } = useSWR<{ choices: string[] }>(
    `${API_BASE}/pdf-choices`,
    fetcher,
    { revalidateOnMount: true, revalidateOnFocus: true }
  );
  const { data: storageData, mutate: refetchStorage } = useSWR<{ markdown: string }>(`${API_BASE}/storage`, fetcher);
  const { data: blockedData, mutate: refetchBlocked } = useSWR<{ sessions: { sid: string; since?: string | null }[] }>(`${API_BASE}/admin/blocked`, fetcher, { revalidateOnFocus: true });
  const { data: catalogData, mutate: refetchCatalog } = useSWR<{ entries: { filename: string; file_id?: string; game_name?: string; size_bytes?: number; updated_at?: string }[]; games: string[]; error?: string }>(`${API_BASE}/admin/catalog`, fetcher, { revalidateOnFocus: true });
  const pdfStatusUrl = useMemo(() => {
    try {
      const t = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token");
      const u = new URL(`${API_BASE}/admin/pdf-status`);
      if (t) u.searchParams.set("token", t);
      return u.toString();
    } catch {
      return `${API_BASE}/admin/pdf-status`;
    }
  }, [role]);
  const { data: pdfStatusData, mutate: refetchPdfStatus } = useSWR<{ items: { filename: string; total_pages: number; processed_pages?: number }[] }>(pdfStatusUrl, fetcher, { revalidateOnFocus: true });
  const appendConsole = (line: string) => setConsoleText((cur) => (cur ? cur + "\n" + line : line));

  const [renameSelection, setRenameSelection] = useState<string[]>([]);
  const [newName, setNewName] = useState<string>("");
  const [unblockSelection, setUnblockSelection] = useState<string[]>([]);

  // --- Global Model Selector (applies to all users) -------------------------
  const [modelLabel, setModelLabel] = useState<string>("");
  const [savingModel, setSavingModel] = useState<boolean>(false);

  // Catalog data and sorting — declared before any early returns so hooks are stable
  const catalog = catalogData?.entries || [];
  const SORT_KEYS = ["filename", "game_name", "file_id", "size_bytes", "updated_at"] as const;
  type SortKey = typeof SORT_KEYS[number];
  const [sortBy, setSortBy] = useState<SortKey>(SORT_KEYS[0]);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const toggleSort = (key: SortKey) => {
    if (sortBy === key) {
      setSortDir((dir) => (dir === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(key);
      setSortDir("asc");
    }
  };
  const sortedCatalog = useMemo(() => {
    const entries = [...catalog];
    const cmp = (a: any, b: any) => {
      const dir = sortDir === "asc" ? 1 : -1;
      const av = (a?.[sortBy] ?? null);
      const bv = (b?.[sortBy] ?? null);
      if (sortBy === "size_bytes") {
        const an = typeof av === "number" ? av : -1;
        const bn = typeof bv === "number" ? bv : -1;
        if (an === bn) return 0;
        return an < bn ? -1 * dir : 1 * dir;
      }
      if (sortBy === "updated_at") {
        const an = av ? Date.parse(String(av)) : 0;
        const bn = bv ? Date.parse(String(bv)) : 0;
        if (an === bn) return 0;
        return an < bn ? -1 * dir : 1 * dir;
      }
      const as = String(av || "").toLowerCase();
      const bs = String(bv || "").toLowerCase();
      if (as === bs) return 0;
      return as < bs ? -1 * dir : 1 * dir;
    };
    entries.sort(cmp);
    return entries;
  }, [catalog, sortBy, sortDir]);
  const statusMap = useMemo(() => {
    const map = new Map<string, { total: number; processed: number }>();
    const items = pdfStatusData?.items || [];
    for (const it of items) {
      map.set(it.filename, { total: it.total_pages || 0, processed: it.processed_pages || 0 });
    }
    return map;
  }, [pdfStatusData]);

  useEffect(() => {
    // Fetch current global model
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/admin/model`);
        if (resp.ok) {
          const data = await resp.json();
          const gen = String(data?.generator || "");
          setModelLabel(gen);
        }
      } catch {}
    })();
  }, []);

  useEffect(() => {
    try {
      const fromSession = sessionStorage.getItem("boardrag_role");
      if (fromSession) {
        setRole(fromSession);
        return;
      }
      // Prefer full adoption when both role & token are present in localStorage
      const localRole = localStorage.getItem("boardrag_role");
      const localToken = localStorage.getItem("boardrag_token");
      if (localRole && localToken) {
        try { sessionStorage.setItem("boardrag_role", localRole); } catch {}
        try { sessionStorage.setItem("boardrag_token", localToken); } catch {}
        setRole(localRole);
        return;
      }
      // Backward compatibility: if only role exists, adopt it
      const fromLocal = localStorage.getItem("boardrag_role");
      if (fromLocal) {
        sessionStorage.setItem("boardrag_role", fromLocal);
        setRole(fromLocal);
      }
    } catch {}
  }, []);

  // Ensure "Assign PDF(s)" list is fresh on every visit
  useEffect(() => {
    // Trigger revalidation immediately when the page mounts
    refetchChoices();
  }, [refetchChoices]);

  // Auto-scroll console to bottom (must be declared before any early returns)
  useEffect(() => {
    try {
      if (consoleRef.current) {
        consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
      }
    } catch {}
  }, [consoleText]);

  // Connect a global Admin log stream so uploads and other events spew here too
  useEffect(() => {
    let es: EventSource | null = null;
    try {
      // Attach token to admin log stream as well if present
      try {
        const t = sessionStorage.getItem("boardrag_token");
        const url = new URL(`${API_BASE}/admin/log-stream`);
        if (t) url.searchParams.set("token", t);
        es = new EventSource(url.toString());
      } catch {
        es = new EventSource(`${API_BASE}/admin/log-stream`);
      }
      es.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed?.type === "log" && typeof parsed.line === "string") {
            setConsoleText((cur) => (cur ? cur + "\n" + parsed.line : parsed.line));
          }
        } catch {}
      };
      es.onerror = () => {
        appendConsole("❌ Admin log stream error (disconnected)");
        try { es?.close(); } catch {}
      };
    } catch {}
    return () => { try { es?.close(); } catch {} };
  }, []);

  // Helper to inject breadcrumbs into admin console even if server log stream connection is momentarily lagging
  const logClient = async (line: string) => {
    try {
      // Local echo for instant feedback
      setConsoleText((cur) => (cur ? cur + "\n" + line : line));
      const headers: any = { "Content-Type": "application/json" };
      try {
        let t: string | null = sessionStorage.getItem("boardrag_token");
        if (!t) t = localStorage.getItem("boardrag_token");
        if (t) headers["Authorization"] = `Bearer ${t}`;
      } catch {}
      const resp = await fetch(`${API_BASE}/admin/log`, {
        method: "POST",
        headers,
        body: JSON.stringify({ line }),
      });
      if (!resp.ok && resp.status === 401) {
        try {
          sessionStorage.removeItem("boardrag_role");
          sessionStorage.removeItem("boardrag_token");
          localStorage.removeItem("boardrag_role");
          localStorage.removeItem("boardrag_token");
          localStorage.removeItem("boardrag_saved_pw");
        } catch {}
        try { window.location.reload(); } catch {}
      }
    } catch {}
  };

  const unlock = async () => {
    setMessage("");
    const form = new FormData();
    form.append("password", pw);
    const resp = await fetch(`${API_BASE}/auth/unlock`, { method: "POST", body: form });
    if (resp.ok) {
      const data = await resp.json();
      setRole(data.role);
      try {
        sessionStorage.setItem("boardrag_role", data.role);
        if (data.token) sessionStorage.setItem("boardrag_token", data.token);
      } catch {}
      try {
        // Persist for cross-restart persistence
        localStorage.setItem("boardrag_role", data.role);
        if (data.token) localStorage.setItem("boardrag_token", data.token);
      } catch {}
      appendConsole("🔓 Admin unlocked");
    } else {
      setMessage("Invalid password");
      appendConsole("❌ Invalid admin password");
    }
  };

  if (role !== "admin") {
    return (
      <div style={{ padding: 16, maxWidth: 640 }}>
        <h2>Admin</h2>
        <div style={{ display: "flex", gap: 8 }}>
          <input type="password" placeholder="Enter admin code" value={pw} onChange={(e) => setPw(e.target.value)} />
          <button onClick={unlock}>Unlock</button>
        </div>
        {message && <div style={{ color: "crimson", marginTop: 8 }}>{message}</div>}
      </div>
    );
  }

  const rebuild = async () => {
    setBusy("rebuild");
    setConsoleText("");
    appendConsole("🔄 Rebuild requested…");
    logClient("[client] 🔄 Rebuild requested…");
    try {
      const es = new EventSource(`${API_BASE}/admin/rebuild-stream`);
      es.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed.type === "log") {
            setConsoleText((cur) => (cur ? cur + "\n" + parsed.line : parsed.line));
          } else if (parsed.type === "done") {
            setConsoleText((cur) => (cur ? cur + "\n" + (parsed.message || "Done.") : (parsed.message || "Done.")));
            es.close();
            setBusy(null);
            Promise.all([refetchGames(), refetchChoices(), refetchStorage()]).catch(() => {});
          }
        } catch {}
      };
      es.onerror = () => { try { es.close(); } catch {}; appendConsole("❌ Rebuild stream error"); setBusy(null); };
    } catch (e) {
      setConsoleText("❌ Rebuild failed. See server logs.");
      setBusy(null);
    }
  };

  const refresh = async () => {
    setBusy("refresh");
    setConsoleText("");
    appendConsole("🔄 Refresh requested…");
    logClient("[client] 🔄 Refresh requested…");
    try {
      const es = new EventSource(`${API_BASE}/admin/refresh-stream`);
      es.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed.type === "log") {
            setConsoleText((cur) => (cur ? cur + "\n" + parsed.line : parsed.line));
          } else if (parsed.type === "done") {
            setConsoleText((cur) => (cur ? cur + "\n" + (parsed.message || "Done.") : (parsed.message || "Done.")));
            es.close();
            setBusy(null);
            Promise.all([refetchGames(), refetchChoices(), refetchStorage()]).catch(() => {});
          }
        } catch {}
      };
      es.onerror = () => { try { es.close(); } catch {}; appendConsole("❌ Refresh stream error"); setBusy(null); };
    } catch (e) {
      setConsoleText("❌ Refresh failed. See server logs.");
      setBusy(null);
    }
  };

  const rechunk = async () => {
    setBusy("rechunk");
    setConsoleText("");
    appendConsole("🔄 Rechunk requested…");
    logClient("[client] 🔄 Rechunk requested…");
    try {
      const es = new EventSource(`${API_BASE}/admin/rechunk-stream`);
      es.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed.type === "log") {
            setConsoleText((cur) => (cur ? cur + "\n" + parsed.line : parsed.line));
          } else if (parsed.type === "done") {
            setConsoleText((cur) => (cur ? cur + "\n" + (parsed.message || "Done.") : (parsed.message || "Done.")));
            es.close();
            setBusy(null);
            Promise.all([refetchGames(), refetchChoices(), refetchStorage()]).catch(() => {});
          }
        } catch {}
      };
      es.onerror = () => { try { es.close(); } catch {}; appendConsole("❌ Rechunk stream error"); setBusy(null); };
    } catch (e) {
      setConsoleText("❌ Rechunk failed. See server logs.");
      setBusy(null);
    }
  };

  const deletePdfsReq = async () => {
    appendConsole(`🗑️ Delete PDF requested: ${renameSelection.join(", ") || "<none>"}`);
    const resp = await fetch(`${API_BASE}/admin/delete-pdfs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(renameSelection),
    });
    const data = await resp.json();
    setMessage(data.message || "");
    if (data?.message) appendConsole(data.message);
    setRenameSelection([]);
    await Promise.all([refetchGames(), refetchChoices(), refetchStorage()]);
  };

  const renameReq = async () => {
    appendConsole(`✏️ Rename requested: [${renameSelection.join(", ")}] → "${newName}"`);
    const resp = await fetch(`${API_BASE}/admin/rename`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entries: renameSelection, new_name: newName }),
    });
    const data = await resp.json();
    setMessage(data.message || "");
    if (data?.message) appendConsole(data.message);
    setRenameSelection([]);
    await Promise.all([refetchGames(), refetchChoices(), refetchStorage(), refetchCatalog()]);
  };

  const rechunkSelected = async (entries: string[]) => {
    if (!entries || entries.length === 0) return;
    setBusy("rechunk");
    setConsoleText("");
    const label = entries.length === 1 ? entries[0] : `${entries.length} PDFs`;
    appendConsole(`🧩 Rechunk selected requested: ${label}`);
    logClient(`[client] 🧩 Rechunk selected requested…`);
    try {
      const url = new URL(`${API_BASE}/admin/rechunk-selected-stream`);
      url.searchParams.set("entries", JSON.stringify(entries));
      const es = new EventSource(url.toString());
      es.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed.type === "log") {
            setConsoleText((cur) => (cur ? cur + "\n" + parsed.line : parsed.line));
          } else if (parsed.type === "done") {
            setConsoleText((cur) => (cur ? cur + "\n" + (parsed.message || "Done.") : (parsed.message || "Done.")));
            es.close();
            setBusy(null);
            Promise.all([refetchGames(), refetchChoices(), refetchStorage()]).catch(() => {});
          }
        } catch {}
      };
      es.onerror = () => { try { es.close(); } catch {}; appendConsole("❌ Rechunk selected stream error"); setBusy(null); };
    } catch (e) {
      setConsoleText("❌ Rechunk selected failed. See server logs.");
      setBusy(null);
    }
  };

  const games = gamesData?.games || [];
  const pdfChoices = pdfChoicesData?.choices || [];
  const blockedSessions = blockedData?.sessions || [];

  const saveModel = async () => {
    setSavingModel(true);
    try {
      const resp = await fetch(`${API_BASE}/admin/model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selection: modelLabel }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.detail || "Failed to set model");
      setMessage(`Model set to ${data.generator}`);
      appendConsole(`🧠 Model set → ${data.provider}/${data.generator}`);
    } catch (e: any) {
      setMessage(e?.message || "Failed to set model");
    } finally {
      setSavingModel(false);
    }
  };

  const unblockSelected = async () => {
    if (unblockSelection.length === 0) return;
    try {
      const resp = await fetch(`${API_BASE}/admin/blocked/unblock`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sids: unblockSelection }),
      });
      const data = await resp.json();
      if (data?.message) appendConsole(data.message);
      setMessage(data?.message || "");
      setUnblockSelection([]);
      await refetchBlocked();
    } catch (e) {
      appendConsole("❌ Failed to unblock sessions");
    }
  };

  return (
    <div style={{ display: "grid", gap: 8, padding: 8, gridTemplateColumns: "1fr", height: "100vh", overflowY: "auto", alignContent: "start", fontSize: 14 }}>
      {message && <div style={{ color: "#444", background: "#f3f3f3", padding: 6, borderRadius: 4 }}>{message}</div>}

      {/* Library controls removed (obsolete) */}

      <div className="admin-tool alt">
        <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Catalog (DB-less)</h3>
        {/* Fixed header outside scroll panel */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "32px 1.2fr 0.9fr 1fr 0.6fr 0.8fr 0.8fr",
            alignItems: "center",
            padding: 2,
            border: "1px solid #eee",
            borderRadius: 4,
            borderBottomLeftRadius: 0,
            borderBottomRightRadius: 0,
            background: "#fafafa",
            fontSize: 11,
            fontWeight: 600,
          }}
        >
          <div />
          <button className="btn link" onClick={() => toggleSort("filename")} style={{ textAlign: "left", padding: 0 }}>
            Filename{sortBy === "filename" ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
          </button>
          <button className="btn link" onClick={() => toggleSort("game_name")} style={{ textAlign: "left", padding: 0 }}>
            Game{sortBy === "game_name" ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
          </button>
          <button className="btn link" onClick={() => toggleSort("file_id")} style={{ textAlign: "left", padding: 0 }}>
            File ID{sortBy === "file_id" ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
          </button>
          <button className="btn link" onClick={() => toggleSort("size_bytes")} style={{ textAlign: "left", padding: 0 }}>
            Size{sortBy === "size_bytes" ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
          </button>
          <button className="btn link" onClick={() => toggleSort("updated_at")} style={{ textAlign: "left", padding: 0 }}>
            Updated{sortBy === "updated_at" ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
          </button>
          <div style={{ textAlign: "left" }}>Processed</div>
        </div>
        <div style={{ height: "35vh", overflow: "auto", border: "1px solid #eee", borderTop: "none", borderRadius: 4, borderTopLeftRadius: 0, borderTopRightRadius: 0, fontSize: 11 }}>
          <div style={{ display: "grid", gridTemplateColumns: "32px 1.2fr 0.9fr 1fr 0.6fr 0.8fr 0.8fr" }}>
            {sortedCatalog.map((e) => {
              const selected = renameSelection.includes(e.filename);
              const st = statusMap.get(e.filename);
              const processed = st ? `${st.processed} / ${st.total}` : "—";
              return (
                <React.Fragment key={e.filename}>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={(ev) => {
                        const fn = e.filename;
                        setRenameSelection((cur) => ev.target.checked ? Array.from(new Set([...(cur||[]), fn])) : (cur||[]).filter((v) => v !== fn));
                      }}
                    />
                  </div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{e.filename}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{e.game_name || "—"}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace" }} title={e.file_id || ""}>{(e.file_id || "").slice(0, 24) || "—"}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{typeof e.size_bytes === "number" ? `${Math.round(e.size_bytes/1024/1024)} MB` : "—"}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{e.updated_at ? new Date(e.updated_at).toLocaleString() : "—"}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{processed}</div>
                </React.Fragment>
              );
            })}
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, marginTop: 8, alignItems: "center" }}>
          <input
            placeholder="New game name for selected"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            style={{ fontSize: 13, padding: 6, flex: 1 }}
          />
          <button
            onClick={renameReq}
            disabled={renameSelection.length === 0 || !newName.trim()}
            style={{ padding: "6px 10px" }}
          >Assign</button>
          <button
            onClick={async () => {
              if (renameSelection.length === 0) return;
              appendConsole(`Clearing DB for: ${renameSelection.join(", ")}`);
              try {
                const headers: any = { "Content-Type": "application/json" };
                try {
                  const t = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token");
                  if (t) headers["Authorization"] = `Bearer ${t}`;
                } catch {}
                const resp = await fetch(`${API_BASE}/admin/clear-selected`, {
                  method: "POST",
                  headers,
                  body: JSON.stringify(renameSelection),
                });
                const data = await resp.json().catch(() => ({} as any));
                if (!resp.ok) throw new Error(data?.detail || `HTTP ${resp.status}`);
                appendConsole(data?.message || "Cleared.");
                // Also refresh Admin log stream is already open and pdf-status
                await Promise.all([refetchPdfStatus()]);
              } catch (e: any) {
                appendConsole(`Clear failed: ${e?.message || e || "error"}`);
              }
            }}
            disabled={renameSelection.length === 0}
            style={{ padding: "6px 10px" }}
          >Clear Selected (DB)</button>
          <button
            onClick={async () => {
              if (renameSelection.length === 0) return;
              const label = renameSelection.length === 1 ? renameSelection[0] : `${renameSelection.length} PDFs`;
              appendConsole(`🧩 Process selected requested: ${label}`);
              logClient(`[client] 🧩 Process selected requested…`);
              try {
                const url = new URL(`${API_BASE}/admin/process-selected-stream`);
                url.searchParams.set("entries", JSON.stringify(renameSelection));
                try {
                  const t = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token");
                  if (t) url.searchParams.set("token", t);
                } catch {}
                const es = new EventSource(url.toString());
                es.onmessage = (ev) => {
                  try {
                    const parsed = JSON.parse(ev.data);
                    if (parsed.type === "log") {
                      setConsoleText((cur) => (cur ? cur + "\n" + parsed.line : parsed.line));
                    } else if (parsed.type === "done") {
                      setConsoleText((cur) => (cur ? cur + "\n" + (parsed.message || "Done.") : (parsed.message || "Done.")));
                      es.close();
                      Promise.all([refetchGames(), refetchChoices(), refetchStorage(), refetchCatalog(), refetchPdfStatus()]).catch(() => {});
                    }
                  } catch {}
                };
                es.onerror = () => { try { es.close(); } catch {}; appendConsole("❌ Process selected stream error"); };
              } catch (e) {
                setConsoleText("❌ Process selected failed. See server logs.");
              }
            }}
            disabled={renameSelection.length === 0}
            style={{ padding: "6px 10px" }}
          >Process Selected</button>
          <button
            onClick={async () => {
              appendConsole(`🗑️ Delete PDF requested: ${renameSelection.join(", ") || "<none>"}`);
              const resp = await fetch(`${API_BASE}/admin/delete-pdfs`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(renameSelection),
              });
              const data = await resp.json();
              setMessage(data.message || "");
              if (data?.message) appendConsole(data.message);
              setRenameSelection([]);
              await Promise.all([refetchGames(), refetchChoices(), refetchStorage(), refetchCatalog()]);
            }}
            disabled={renameSelection.length === 0}
            style={{ padding: "6px 10px" }}
          >Delete Selected</button>
          <button onClick={async () => {
            try {
              appendConsole("Refreshing catalog …");
              const resp = await fetch(`${API_BASE}/admin/catalog/refresh`, { method: "POST" });
              let bodyText = "";
              try { bodyText = await resp.text(); } catch {}
              let js: any = null;
              try { js = bodyText ? JSON.parse(bodyText) : null; } catch {}
              if (!resp.ok) {
                const detail = (js && (js.message || js.error)) || bodyText || `HTTP ${resp.status}`;
                throw new Error(detail);
              }
              appendConsole("Catalog refreshed");
              await Promise.all([refetchCatalog(), refetchPdfStatus()]);
            } catch (e: any) {
              appendConsole(`Catalog refresh failed: ${e?.message || e || "unknown error"}`);
            }
          }} style={{ padding: "6px 10px", marginLeft: "auto" }}>🔄 Refresh Catalog</button>
          <button onClick={async () => { appendConsole("Refreshing processed status …"); await refetchPdfStatus(); appendConsole("Processed status refreshed"); }} style={{ padding: "6px 10px", marginLeft: 6 }}>Refresh Processed</button>
        </div>
      </div>

      <div className="admin-tool alt">
        <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Blocked Sessions</h3>
        <div style={{ display: "grid", gap: 6 }}>
          {blockedSessions.length === 0 ? (
            <div className="muted" style={{ fontSize: 13 }}>No blocked sessions</div>
          ) : (
            <>
              <select
                multiple
                size={Math.min(8, Math.max(3, blockedSessions.length))}
                value={unblockSelection}
                onChange={(e) => setUnblockSelection(Array.from(e.target.selectedOptions).map((o) => o.value))}
                style={{ width: "100%", fontSize: 12, padding: 4 }}
              >
                {blockedSessions.map((s) => (
                  <option key={s.sid} value={s.sid} title={s.sid}>
                    {s.sid} {s.since ? `(since ${new Date(s.since).toLocaleString()})` : ""}
                  </option>
                ))}
              </select>
              <div>
                <button onClick={unblockSelected} disabled={unblockSelection.length === 0} style={{ padding: "6px 10px" }}>Unblock Selected</button>
                <button onClick={() => refetchBlocked()} style={{ padding: "6px 10px", marginLeft: 6 }}>Refresh</button>
              </div>
            </>
          )}
        </div>
      </div>

      

      <div className="admin-tool">
        <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Global Model</h3>
        <label style={{ display: "grid", gap: 4 }}>
          <span>Model (applies to all users)</span>
          <select
            className="select"
            value={modelLabel}
            onChange={(e) => setModelLabel(e.target.value)}
            style={{ fontSize: 13, padding: 6 }}
          >
            <option value="claude-3-5-haiku-latest">claude-3-5-haiku-latest</option>
            <option value="claude-sonnet-4-20250514">claude-sonnet-4-20250514</option>
            <option value="o3">o3</option>
            <option value="gpt-4o-mini">gpt-4o-mini</option>
            <option value="gpt-5-mini">gpt-5-mini</option>
          </select>
        </label>
        <div style={{ marginTop: 6 }}>
          <button onClick={saveModel} disabled={savingModel} style={{ padding: "6px 10px" }}>
            {savingModel ? "Saving…" : "Save"}
          </button>
        </div>
      </div>

      {/* Removed legacy Assign/Reprocess/Delete control; use Catalog section above */}

      <div className="admin-tool" style={{ gridColumn: "1 / -1" }}>
        <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Technical Info</h3>
        <pre style={{ whiteSpace: "pre-wrap", background: "#f7f7f7", padding: 8, borderRadius: 4, fontSize: 12 }}>{storageData?.markdown || ""}</pre>
        <button style={{ padding: "6px 10px" }} onClick={() => { appendConsole("📦 Refresh storage stats"); refetchStorage().then(() => appendConsole("✅ Storage stats refreshed")).catch(() => appendConsole("❌ Storage refresh failed")); }}>🔄 Refresh Storage Stats</button>
      </div>

      {/* Shared console output pinned at bottom with fixed height */}
      <div className="admin-tool alt" style={{ gridColumn: "1 / -1" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Console</h3>
          <div>
            <button onClick={() => setConsoleText("")} style={{ padding: "6px 10px" }}>🧹 Clear</button>
          </div>
        </div>
        <div className="surface" style={{ padding: 6 }}>
          <pre
            ref={consoleRef}
            style={{ whiteSpace: "pre-wrap", height: 180, overflow: "auto", margin: 0, fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace", fontSize: 12 }}
          >
{consoleText || ""}
          </pre>
        </div>
      </div>
    </div>
  );
}



