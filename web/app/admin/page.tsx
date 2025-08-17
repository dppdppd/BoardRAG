"use client";

import React, { useEffect, useRef, useState } from "react";
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
  const appendConsole = (line: string) => setConsoleText((cur) => (cur ? cur + "\n" + line : line));

  const [renameSelection, setRenameSelection] = useState<string[]>([]);
  const [newName, setNewName] = useState<string>("");
  const [unblockSelection, setUnblockSelection] = useState<string[]>([]);

  // --- Global Model Selector (applies to all users) -------------------------
  const [modelLabel, setModelLabel] = useState<string>("");
  const [savingModel, setSavingModel] = useState<boolean>(false);

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
  const catalog = catalogData?.entries || [];

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

      <div className="admin-tool">
        <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Library</h3>
        <div style={{ display: "flex", gap: 6 }}>
          <button onClick={rebuild} disabled={busy!==null} style={{ padding: "6px 10px" }}>
            {busy === "rebuild" ? (<span className="indicator" style={{ gap: 8 }}><span className="spinner" /> Rebuilding…</span>) : "🔄 Rebuild Library"}
          </button>
          <button onClick={refresh} disabled={busy!==null} style={{ padding: "6px 10px" }}>
            {busy === "refresh" ? (<span className="indicator" style={{ gap: 8 }}><span className="spinner" /> Processing…</span>) : "🔄 Process New PDFs"}
          </button>
          <button onClick={rechunk} disabled={busy!==null} style={{ padding: "6px 10px" }}>
            {busy === "rechunk" ? (<span className="indicator" style={{ gap: 8 }}><span className="spinner" /> Rechunking…</span>) : "🧩 Rechunk Library (preserve names)"}
          </button>
        </div>
      </div>

      <div className="admin-tool alt">
        <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Catalog (DB-less)</h3>
        <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 6 }}>
          <button onClick={async () => {
            try {
              appendConsole("📚 Refreshing catalog …");
              const resp = await fetch(`${API_BASE}/admin/catalog/refresh`, { method: "POST" });
              let bodyText = "";
              try { bodyText = await resp.text(); } catch {}
              let js: any = null;
              try { js = bodyText ? JSON.parse(bodyText) : null; } catch {}
              if (!resp.ok) {
                const detail = (js && (js.message || js.error)) || bodyText || `HTTP ${resp.status}`;
                throw new Error(detail);
              }
              appendConsole("✅ Catalog refreshed");
              await refetchCatalog();
            } catch (e: any) {
              appendConsole(`❌ Catalog refresh failed: ${e?.message || e || "unknown error"}`);
            }
          }} style={{ padding: "6px 10px" }}>🔄 Refresh Catalog</button>
          <div className="muted" style={{ fontSize: 12 }}>
            {catalog.length === 0 ? "No entries found (place PDFs in /data)" : `${catalog.length} PDF(s)`}
          </div>
        </div>
        <div style={{ maxHeight: 260, overflow: "auto", border: "1px solid #eee", borderRadius: 4 }}>
          <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left" }}>
                <th style={{ padding: 6, borderBottom: "1px solid #eee" }} />
                <th style={{ padding: 6, borderBottom: "1px solid #eee" }}>Filename</th>
                <th style={{ padding: 6, borderBottom: "1px solid #eee" }}>Game</th>
                <th style={{ padding: 6, borderBottom: "1px solid #eee" }}>File ID</th>
                <th style={{ padding: 6, borderBottom: "1px solid #eee" }}>Size</th>
                <th style={{ padding: 6, borderBottom: "1px solid #eee" }}>Updated</th>
              </tr>
            </thead>
            <tbody>
              {catalog.map((e) => {
                const selected = renameSelection.includes(e.filename);
                return (
                  <tr key={e.filename}>
                    <td style={{ padding: 6, borderBottom: "1px solid #f2f2f2" }}>
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={(ev) => {
                          const fn = e.filename;
                          setRenameSelection((cur) => ev.target.checked ? Array.from(new Set([...(cur||[]), fn])) : (cur||[]).filter((v) => v !== fn));
                        }}
                      />
                    </td>
                    <td style={{ padding: 6, borderBottom: "1px solid #f2f2f2" }}>{e.filename}</td>
                    <td style={{ padding: 6, borderBottom: "1px solid #f2f2f2" }}>{e.game_name || "—"}</td>
                    <td style={{ padding: 6, borderBottom: "1px solid #f2f2f2", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace" }} title={e.file_id || ""}>{(e.file_id || "").slice(0, 24) || "—"}</td>
                    <td style={{ padding: 6, borderBottom: "1px solid #f2f2f2" }}>{typeof e.size_bytes === "number" ? `${Math.round(e.size_bytes/1024/1024)} MB` : "—"}</td>
                    <td style={{ padding: 6, borderBottom: "1px solid #f2f2f2" }}>{e.updated_at ? new Date(e.updated_at).toLocaleString() : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
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



