"use client";

import React, { useEffect, useState } from "react";
import useSWR from "swr";

const RAW_API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const API_BASE = RAW_API_BASE.trim().replace(/\/+$/, "");
const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function AdminPage() {
  const [role, setRole] = useState<string>("none");
  const [pw, setPw] = useState<string>("");
  const [message, setMessage] = useState<string>("");

  const { data: gamesData, mutate: refetchGames } = useSWR<{ games: string[] }>(`${API_BASE}/games`, fetcher);
  const { data: pdfChoicesData, mutate: refetchChoices } = useSWR<{ choices: string[] }>(`${API_BASE}/pdf-choices`, fetcher);
  const { data: storageData, mutate: refetchStorage } = useSWR<{ markdown: string }>(`${API_BASE}/storage`, fetcher);

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

  const unlock = async () => {
    setMessage("");
    const form = new FormData();
    form.append("password", pw);
    const resp = await fetch(`${API_BASE}/auth/unlock`, { method: "POST", body: form });
    if (resp.ok) {
      const data = await resp.json();
      setRole(data.role);
      try { sessionStorage.setItem("boardrag_role", data.role); } catch {}
    } else {
      setMessage("Invalid password");
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
    const resp = await fetch(`${API_BASE}/admin/rebuild`, { method: "POST" });
    const data = await resp.json();
    setMessage(data.message || "");
    await Promise.all([refetchGames(), refetchChoices(), refetchStorage()]);
  };

  const refresh = async () => {
    const resp = await fetch(`${API_BASE}/admin/refresh`, { method: "POST" });
    const data = await resp.json();
    setMessage(data.message || "");
    await Promise.all([refetchGames(), refetchChoices(), refetchStorage()]);
  };

  const deleteGamesReq = async () => {
    const resp = await fetch(`${API_BASE}/admin/delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(deleteSelection),
    });
    const data = await resp.json();
    setMessage(data.message || "");
    await Promise.all([refetchGames(), refetchChoices(), refetchStorage()]);
  };

  const renameReq = async () => {
    const resp = await fetch(`${API_BASE}/admin/rename`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entries: renameSelection, new_name: newName }),
    });
    const data = await resp.json();
    setMessage(data.message || "");
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
          <button onClick={rebuild}>🔄 Rebuild Library</button>
          <button onClick={refresh}>🔄 Process New PDFs</button>
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
          <span>New game name</span>
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g., Up Front" />
        </label>
        <div style={{ marginTop: 8 }}>
          <button onClick={renameReq} disabled={renameSelection.length === 0 || !newName.trim()}>Rename</button>
        </div>
      </div>

      <div style={{ gridColumn: "1 / -1" }}>
        <h3>Technical Info</h3>
        <pre style={{ whiteSpace: "pre-wrap", background: "#f7f7f7", padding: 12, borderRadius: 6 }}>{storageData?.markdown || ""}</pre>
        <button onClick={() => refetchStorage()}>🔄 Refresh Storage Stats</button>
      </div>
    </div>
  );
}


