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
  const appendConsole = (line: string) => setConsoleText((cur) => (cur ? cur + "\n" + line : line));

  const [deleteSelection, setDeleteSelection] = useState<string[]>([]);
  const [renameSelection, setRenameSelection] = useState<string[]>([]);
  const [newName, setNewName] = useState<string>("");

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
    try {
      const es = new EventSource(`${API_BASE}/admin/rebuild-stream`);
      es.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed.type === "log") {
            setConsoleText((cur) => (cur ? cur + "\n" + parsed.line : parsed.line));
          } else if (parsed.type === "done") {
            setMessage(parsed.message || "Done.");
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
    try {
      const es = new EventSource(`${API_BASE}/admin/refresh-stream`);
      es.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed.type === "log") {
            setConsoleText((cur) => (cur ? cur + "\n" + parsed.line : parsed.line));
          } else if (parsed.type === "done") {
            setMessage(parsed.message || "Done.");
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

  return (
    <div style={{ display: "grid", gap: 16, padding: 16, gridTemplateColumns: "1fr 1fr" }}>
      <div style={{ gridColumn: "1 / -1" }}>
        <h2>Admin</h2>
        {message && <div style={{ color: "#444", background: "#f3f3f3", padding: 8, borderRadius: 6 }}>{message}</div>}
      </div>

      <div>
        <h3>Library</h3>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={rebuild} disabled={busy!==null}>
            {busy === "rebuild" ? (<span className="indicator" style={{ gap: 8 }}><span className="spinner" /> Rebuilding…</span>) : "🔄 Rebuild Library"}
          </button>
          <button onClick={refresh} disabled={busy!==null}>
            {busy === "refresh" ? (<span className="indicator" style={{ gap: 8 }}><span className="spinner" /> Processing…</span>) : "🔄 Process New PDFs"}
          </button>
        </div>
        <div style={{ marginTop: 12 }}>
          <h4>Delete game(s)</h4>
          <select multiple size={6} value={deleteSelection} onChange={(e) => setDeleteSelection(Array.from(e.target.selectedOptions).map((o) => o.value))} style={{ width: "100%" }}>
            {games.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
          <div style={{ marginTop: 8 }}>
            <button onClick={deleteGamesReq} disabled={deleteSelection.length === 0}>Delete</button>
          </div>
        </div>
      </div>

      <div>
        <h3>Assign PDF(s)</h3>
        <label style={{ display: "grid", gap: 6 }}>
          <span>PDF entries</span>
          <select multiple size={8} value={renameSelection} onChange={(e) => setRenameSelection(Array.from(e.target.selectedOptions).map((o) => o.value))} style={{ width: "100%" }}>
            {pdfChoices.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
        <label style={{ display: "grid", gap: 6, marginTop: 8 }}>
          <span>New game assignment</span>
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Campaign for North Africa, The" />
        </label>
        <div style={{ marginTop: 8 }}>
          <button onClick={renameReq} disabled={renameSelection.length === 0 || !newName.trim()}>Rename</button>
        </div>
      </div>

      <div style={{ gridColumn: "1 / -1" }}>
        <h3>Technical Info</h3>
        <pre style={{ whiteSpace: "pre-wrap", background: "#f7f7f7", padding: 12, borderRadius: 6 }}>{storageData?.markdown || ""}</pre>
        <button onClick={() => { appendConsole("📦 Refresh storage stats"); refetchStorage().then(() => appendConsole("✅ Storage stats refreshed")).catch(() => appendConsole("❌ Storage refresh failed")); }}>🔄 Refresh Storage Stats</button>
      </div>

      {/* Shared console output pinned at bottom with fixed height */}
      <div style={{ gridColumn: "1 / -1" }}>
        <h3>Console</h3>
        <div className="surface" style={{ padding: 8 }}>
          <pre
            ref={consoleRef}
            style={{ whiteSpace: "pre-wrap", height: 240, overflow: "auto", margin: 0, fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace", fontSize: 13 }}
          >
{consoleText || ""}
          </pre>
        </div>
      </div>
    </div>
  );
}


