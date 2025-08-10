"use client";

import React, { useEffect, useRef, useState } from "react";
import useSWR from "swr";

import { API_BASE } from "../../lib/config";
const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function AdminPage() {
  const [role, setRole] = useState<string>("none");
  const [pw, setPw] = useState<string>("");
  const [message, setMessage] = useState<string>("");
  const [busy, setBusy] = useState<null | "rebuild" | "refresh" | "delete" | "rename">(null);
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
  const appendConsole = (line: string) => setConsoleText((cur) => (cur ? cur + "\n" + line : line));

  const [deleteSelection, setDeleteSelection] = useState<string[]>([]);
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
      // Backward compatibility: if an older tab stored localStorage, respect it once
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
      es = new EventSource(`${API_BASE}/admin/log-stream`);
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
      await fetch(`${API_BASE}/admin/log`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ line }),
      });
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
      try { sessionStorage.setItem("boardrag_role", data.role); } catch {}
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

  const deleteGamesReq = async () => {
    appendConsole(`🗑️ Delete requested: ${deleteSelection.join(", ") || "<none>"}`);
    const resp = await fetch(`${API_BASE}/admin/delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(deleteSelection),
    });
    const data = await resp.json();
    setMessage(data.message || "");
    if (data?.message) appendConsole(data.message);
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
    await Promise.all([refetchGames(), refetchChoices(), refetchStorage()]);
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

      <div className="admin-tool">
        <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Library</h3>
        <div style={{ display: "flex", gap: 6 }}>
          <button onClick={rebuild} disabled={busy!==null} style={{ padding: "6px 10px" }}>
            {busy === "rebuild" ? (<span className="indicator" style={{ gap: 8 }}><span className="spinner" /> Rebuilding…</span>) : "🔄 Rebuild Library"}
          </button>
          <button onClick={refresh} disabled={busy!==null} style={{ padding: "6px 10px" }}>
            {busy === "refresh" ? (<span className="indicator" style={{ gap: 8 }}><span className="spinner" /> Processing…</span>) : "🔄 Process New PDFs"}
          </button>
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

      <div className="admin-tool alt">
        <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Delete Games</h3>
        <select multiple size={5} value={deleteSelection} onChange={(e) => setDeleteSelection(Array.from(e.target.selectedOptions).map((o) => o.value))} style={{ width: "100%", fontSize: 13, padding: 4 }}>
          {games.map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
        <div style={{ marginTop: 6 }}>
          <button onClick={deleteGamesReq} disabled={deleteSelection.length === 0} style={{ padding: "6px 10px" }}>Delete</button>
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

      <div className="admin-tool alt">
        <h3 style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.2 }}>Assign PDF(s)</h3>
        <label style={{ display: "grid", gap: 4 }}>
          <span>PDF entries</span>
          <select multiple size={6} value={renameSelection} onChange={(e) => setRenameSelection(Array.from(e.target.selectedOptions).map((o) => o.value))} style={{ width: "100%", fontSize: 13, padding: 4 }}>
            {pdfChoices.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
        <label style={{ display: "grid", gap: 4, marginTop: 6 }}>
          <span>New game assignment</span>
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Campaign for North Africa, The" style={{ fontSize: 13, padding: 6 }} />
        </label>
        <div style={{ marginTop: 6 }}>
          <button onClick={renameReq} disabled={renameSelection.length === 0 || !newName.trim()} style={{ padding: "6px 10px" }}>Rename</button>
        </div>
      </div>

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



