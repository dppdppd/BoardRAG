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

  const gamesUrl = useMemo(() => {
    try {
      const t = (typeof window !== 'undefined') ? (sessionStorage.getItem('boardrag_token') || localStorage.getItem('boardrag_token')) : null;
      const u = new URL(`${API_BASE}/games`);
      if (t) u.searchParams.set('token', t);
      return u.toString();
    } catch { return `${API_BASE}/games`; }
  }, [role]);
  const { data: gamesData, mutate: refetchGames } = useSWR<{ games: string[] }>(gamesUrl, fetcher);
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
  type PdfStatusItem = { filename: string; total_pages: number; pages_exported?: number; analyzed_present?: number; evals_present?: number; chunks_present?: number; complete?: boolean };
  const { data: pdfStatusData, mutate: refetchPdfStatus } = useSWR<{ items: PdfStatusItem[] }>(pdfStatusUrl, fetcher, { revalidateOnFocus: true });
  const appendConsole = (line: string) => setConsoleText((cur) => (cur ? cur + "\n" + line : line));

  // Global model (provider + generator) – reflect current backend config
  const modelUrl = useMemo(() => {
    try {
      const t = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token");
      const u = new URL(`${API_BASE}/admin/model`);
      if (t) u.searchParams.set("token", t);
      return u.toString();
    } catch {
      return `${API_BASE}/admin/model`;
    }
  }, [role]);
  const { data: modelData, mutate: refetchModel } = useSWR<{ provider: string; generator: string }>(modelUrl, fetcher, { revalidateOnFocus: true });

  const [renameSelection, setRenameSelection] = useState<string[]>([]);
  const [newName, setNewName] = useState<string>("");
  const [unblockSelection, setUnblockSelection] = useState<string[]>([]);
  const [action, setAction] = useState<string>("split-missing");

  // Simple DATA_PATH browser
  type FsEntry = { name: string; is_dir: boolean; size?: number | null; mtime?: string; ext?: string };
  type FsListResp = { base: string; cwd: string; parent?: string | null; entries: FsEntry[] };
  const [fsOpen, setFsOpen] = useState<boolean>(false);
  const [fsCwd, setFsCwd] = useState<string>("");
  const [fsParent, setFsParent] = useState<string | null>(null);
  const [fsEntries, setFsEntries] = useState<FsEntry[]>([]);
  const [fsLoading, setFsLoading] = useState<boolean>(false);
  const formatSize = (n?: number | null) => {
    if (!n && n !== 0) return "—";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
    return `${Math.round(n / 1024 / 1024)} MB`;
  };
  const fetchFs = async (path?: string) => {
    setFsLoading(true);
    try {
      const headers: any = {};
      try {
        const t = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token");
        if (t) headers["Authorization"] = `Bearer ${t}`;
      } catch {}
      const u = new URL(`${API_BASE}/admin/fs-list`);
      if (path) u.searchParams.set("path", path);
      const resp = await fetch(u.toString(), { headers });
      const data: FsListResp = await resp.json();
      setFsCwd(data?.cwd || "");
      setFsParent((data?.parent as any) ?? null);
      setFsEntries(Array.isArray(data?.entries) ? data.entries : []);
    } catch {}
    setFsLoading(false);
  };
  const openFs = async () => {
    setFsOpen(true);
    await fetchFs("");
  };

  // Global model fixed to Sonnet 4; no selector

  // Catalog data and sorting — declared before any early returns so hooks are stable
  const catalog = catalogData?.entries || [];
  const SORT_KEYS = ["filename", "game_name", "size_bytes", "updated_at"] as const;
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
    const map = new Map<string, PdfStatusItem>();
    const items = pdfStatusData?.items || [];
    for (const it of items) {
      map.set(it.filename, it);
    }
    return map;
  }, [pdfStatusData]);

  // Jobs snapshot + stream
  const jobsUrl = useMemo(() => {
    try {
      const t = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token");
      const u = new URL(`${API_BASE}/admin/jobs`);
      if (t) u.searchParams.set("token", t);
      return u.toString();
    } catch { return `${API_BASE}/admin/jobs`; }
  }, [role]);
  const { data: jobsData, mutate: refetchJobs } = useSWR<{ jobs: any[] }>(jobsUrl, fetcher, { revalidateOnFocus: true });
  const [jobs, setJobs] = useState<any[]>([]);
  useEffect(() => { setJobs(jobsData?.jobs || []); }, [jobsData]);
  useEffect(() => {
    let es: EventSource | null = null;
    try {
      const t = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token");
      const u = new URL(`${API_BASE}/admin/jobs-stream`);
      if (t) u.searchParams.set("token", t);
      es = new EventSource(u.toString());
      es.onmessage = (ev) => {
        try {
          const payload = JSON.parse(ev.data);
          if (payload?.type === "snapshot") {
            setJobs(Array.isArray(payload.jobs) ? payload.jobs : []);
          } else if (payload?.type === "start" && payload.job) {
            setJobs((cur) => {
              const map = new Map((cur || []).map((j: any) => [j.id, j]));
              map.set(payload.job.id, payload.job);
              return Array.from(map.values());
            });
          } else if (payload?.type === "progress") {
            setJobs((cur) => (cur || []).map((j: any) => j.id === payload.job_id ? { ...j, updated_at: new Date().toISOString(), } : j));
          } else if (payload?.type === "done") {
            setJobs((cur) => (cur || []).map((j: any) => j.id === payload.job_id ? { ...j, status: "done", updated_at: new Date().toISOString() } : j));
          }
        } catch {}
      };
      es.onerror = () => { try { es?.close(); } catch {}; };
    } catch {}
    return () => { try { es?.close(); } catch {} };
  }, [role]);
  // No model fetch; model is fixed

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
  // Model is fixed; no save handler

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

  const startAction = async () => {
    if (renameSelection.length === 0) return;
    const token = (() => {
      try { return sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token"); } catch { return null; }
    })();
    const buildAndRun = (path: string, mode: "all" | "missing", start?: string) => {
      const url = new URL(`${API_BASE}${path}`);
      url.searchParams.set("entries", JSON.stringify(renameSelection));
      url.searchParams.set("mode", mode);
      if (start) url.searchParams.set("start", start);
      if (token) url.searchParams.set("token", token);
      const es = new EventSource(url.toString());
      es.onmessage = (ev) => {
        try {
          const p = JSON.parse(ev.data);
          if (p.type === "log") setConsoleText((c)=> (c? c+"\n"+p.line : p.line));
          if (p.type === "done") {
            setConsoleText((c)=> (c? c+"\n"+ (p.message||"Done.") : (p.message||"Done.")));
            es.close();
            Promise.all([refetchPdfStatus()]).catch(()=>{});
          }
        } catch {}
      };
      es.onerror = () => { try { es.close(); } catch {}; appendConsole("❌ Stream error"); };
    };
    switch (action) {
      case "split-all":
        // Run full pipeline starting from split
        buildAndRun("/admin/pipeline-stream", "all", "split");
        break;
      case "split-missing":
        buildAndRun("/admin/pipeline-stream", "missing", "split");
        break;
      case "eval-all":
        // Start from eval, then compute, then populate
        buildAndRun("/admin/pipeline-stream", "all", "eval");
        break;
      case "eval-missing":
        buildAndRun("/admin/pipeline-stream", "missing", "eval");
        break;
      case "compute-all":
        // Start from compute → populate
        buildAndRun("/admin/pipeline-stream", "all", "compute");
        break;
      case "compute-missing":
        buildAndRun("/admin/pipeline-stream", "missing", "compute");
        break;
      case "populate-all":
        // Populate only; this is effectively pipeline start=populate
        buildAndRun("/admin/pipeline-stream", "all", "populate");
        break;
      case "populate-missing":
        buildAndRun("/admin/pipeline-stream", "missing", "populate");
        break;
      case "pipeline-all":
        buildAndRun("/admin/pipeline-stream", "all");
        break;
      case "pipeline-missing":
        buildAndRun("/admin/pipeline-stream", "missing");
        break;
    }
  };

  const clearJobs = async () => {
    try {
      appendConsole("🧹 Clearing jobs …");
      const headers: any = { "Content-Type": "application/json" };
      try {
        const t = sessionStorage.getItem("boardrag_token") || localStorage.getItem("boardrag_token");
        if (t) headers["Authorization"] = `Bearer ${t}`;
      } catch {}
      const resp = await fetch(`${API_BASE}/admin/jobs/clear`, { method: "POST", headers });
      if (resp.status === 401) {
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
      const data = await resp.json().catch(() => ({} as any));
      if (!resp.ok) throw new Error(data?.message || data?.detail || `HTTP ${resp.status}`);
      if (data?.message) setMessage(data.message);
      appendConsole(data?.message || "✅ Jobs cleared");
      try { await refetchJobs(); } catch {}
    } catch (e: any) {
      appendConsole(`❌ Clear jobs failed: ${e?.message || e || "error"}`);
    }
  };

  return (
    <div style={{ display: "grid", gap: 8, padding: 8, gridTemplateColumns: "1fr", height: "100vh", overflowY: "auto", alignContent: "start", fontSize: 16 }}>
      {message && <div style={{ color: "#444", background: "#f3f3f3", padding: 6, borderRadius: 4 }}>{message}</div>}

      {/* Library controls removed (obsolete) */}

      <div className="admin-tool alt">
        {jobs && jobs.length > 0 && (
          <div style={{ margin: "6px 0", padding: 6, border: "1px dashed #ddd", borderRadius: 4 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
              <div style={{ fontSize: 14, fontWeight: 600 }}>Active jobs</div>
              <button onClick={clearJobs} className="btn" style={{ padding: "4px 8px", fontSize: 13 }}>Clear</button>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 0.6fr 1fr 1fr", gap: 4 }}>
              <div style={{ fontSize: 13, fontWeight: 600 }}>Step</div>
              <div style={{ fontSize: 13, fontWeight: 600 }}>Mode</div>
              <div style={{ fontSize: 13, fontWeight: 600 }}>Entries</div>
              <div style={{ fontSize: 13, fontWeight: 600 }}>Status</div>
              {jobs.map((j: any) => (
                <React.Fragment key={j.id}>
                  <div style={{ fontSize: 13 }}>{j.step}</div>
                  <div style={{ fontSize: 13 }}>{j.mode}</div>
                  <div style={{ fontSize: 13, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{Array.isArray(j.entries) ? j.entries.join(", ") : ""}</div>
                  <div style={{ fontSize: 13 }}>{j.status || "running"}</div>
                </React.Fragment>
              ))}
            </div>
          </div>
        )}
        {/* Fixed header outside scroll panel */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "32px 1.2fr 1fr 0.6fr 0.7fr 0.7fr 0.5fr 0.5fr 0.5fr 0.5fr",
            alignItems: "center",
            padding: 2,
            border: "1px solid #eee",
            borderRadius: 4,
            borderBottomLeftRadius: 0,
            borderBottomRightRadius: 0,
            background: "#fafafa",
            fontSize: 13,
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
          <button className="btn link" onClick={() => toggleSort("size_bytes")} style={{ textAlign: "left", padding: 0 }}>
            Size{sortBy === "size_bytes" ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
          </button>
          <button className="btn link" onClick={() => toggleSort("updated_at")} style={{ textAlign: "left", padding: 0 }}>
            Updated{sortBy === "updated_at" ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
          </button>
          <div style={{ textAlign: "left" }}>Pages</div>
          <div style={{ textAlign: "left" }}>LLM Raw</div>
          <div style={{ textAlign: "left" }}>Local</div>
          <div style={{ textAlign: "left" }}>Chunks</div>
          <div style={{ textAlign: "left" }}>Complete</div>
        </div>
        <div style={{ height: "35vh", overflow: "auto", border: "1px solid #eee", borderTop: "none", borderRadius: 4, borderTopLeftRadius: 0, borderTopRightRadius: 0, fontSize: 13 }}>
          <div style={{ display: "grid", gridTemplateColumns: "32px 1.2fr 1fr 0.6fr 0.7fr 0.7fr 0.5fr 0.5fr 0.5fr 0.5fr" }}>
            {sortedCatalog.map((e) => {
              const selected = renameSelection.includes(e.filename);
              const st = statusMap.get(e.filename);
              const pagesXY = st ? `${st.pages_exported ?? 0} / ${st.total_pages ?? 0}` : "—";
              const analyzedXY = st ? `${st.analyzed_present ?? 0} / ${st.total_pages ?? 0}` : "—";
              const evalsXY = st ? `${st.evals_present ?? 0} / ${st.total_pages ?? 0}` : "—";
              const chunksXY = st ? `${st.chunks_present ?? 0} / ${st.total_pages ?? 0}` : "—";
              const complete = st?.complete ? "✔" : "";
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
                  
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{typeof e.size_bytes === "number" ? `${Math.round(e.size_bytes/1024/1024)} MB` : "—"}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{e.updated_at ? new Date(e.updated_at).toLocaleString() : "—"}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{pagesXY}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{analyzedXY}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{evalsXY}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{chunksXY}</div>
                  <div style={{ padding: 2, borderBottom: "1px solid #f2f2f2" }}>{complete}</div>
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
            style={{ fontSize: 15, padding: 8, flex: 1 }}
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
          {/* New pipeline controls: dropdown + Take Action */}
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            <select
              value={action}
              onChange={(e) => setAction(e.target.value)}
              style={{ padding: 8, fontSize: 15 }}
            >
              <option value="split-all">Split (all)</option>
              <option value="split-missing">Split (missing)</option>
              <option value="eval-all">LLM Eval (all)</option>
              <option value="eval-missing">LLM Eval (missing)</option>
              <option value="compute-missing">Compute local (missing)</option>
              <option value="compute-all">Compute local (all)</option>
              <option value="populate-all">Populate DB (all)</option>
              <option value="populate-missing">Populate DB (missing)</option>
              <option value="pipeline-missing">Run pipeline (missing)</option>
              <option value="pipeline-all">Run pipeline (all)</option>
            </select>
            <button
              onClick={startAction}
              disabled={renameSelection.length === 0}
              style={{ padding: "6px 10px" }}
            >Take Action</button>
          </div>
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
        <h3 style={{ margin: "4px 0", fontSize: 16, lineHeight: 1.2 }}>Blocked Sessions</h3>
        <div style={{ display: "grid", gap: 6 }}>
          {blockedSessions.length === 0 ? (
            <div className="muted" style={{ fontSize: 15 }}>No blocked sessions</div>
          ) : (
            <>
              <select
                multiple
                size={Math.min(8, Math.max(3, blockedSessions.length))}
                value={unblockSelection}
                onChange={(e) => setUnblockSelection(Array.from(e.target.selectedOptions).map((o) => o.value))}
                style={{ width: "100%", fontSize: 14, padding: 6 }}
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
        <h3 style={{ margin: "4px 0", fontSize: 16, lineHeight: 1.2 }}>Global Model</h3>
        <div className="muted" style={{ fontSize: 15 }}>
          {(() => {
            const prov = (modelData?.provider || '').toLowerCase();
            const gen = modelData?.generator || '';
            if (!gen) return "Loading…";
            const label = prov === 'anthropic' ? `[Anthropic] ${gen}` : prov === 'openai' ? `[OpenAI] ${gen}` : gen;
            return `Active: ${label}`;
          })()}
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
          <button className="btn" onClick={() => refetchModel()} style={{ padding: '6px 10px' }}>Refresh</button>
        </div>
      </div>

      {/* Removed legacy Assign/Reprocess/Delete control; use Catalog section above */}

      <div className="admin-tool" style={{ gridColumn: "1 / -1" }}>
        <h3 style={{ margin: "4px 0", fontSize: 16, lineHeight: 1.2 }}>Technical Info</h3>
        <pre style={{ whiteSpace: "pre-wrap", background: "#f7f7f7", padding: 10, borderRadius: 4, fontSize: 14 }}>{storageData?.markdown || ""}</pre>
        <div style={{ display: "flex", gap: 6 }}>
          <button style={{ padding: "6px 10px" }} onClick={() => { appendConsole("📦 Refresh storage stats"); refetchStorage().then(() => appendConsole("✅ Storage stats refreshed")).catch(() => appendConsole("❌ Storage refresh failed")); }}>🔄 Refresh Storage Stats</button>
          <button onClick={openFs} style={{ padding: "6px 10px" }}>Browse DATA</button>
        </div>
      </div>

      {/* Simple FS browser modal */}
      {fsOpen && (
        <div className="modal" role="dialog" style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000 }}>
          <div style={{ background: "#fff", padding: 12, borderRadius: 6, width: "80vw", height: "70vh", display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <strong>DATA Browser</strong>
              <span style={{ color: "#666" }}>/ {fsCwd || ""}</span>
              <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                <button onClick={() => fetchFs(fsParent || "")} disabled={!fsParent || fsLoading} style={{ padding: "6px 10px" }}>Up</button>
                <button onClick={() => fetchFs(fsCwd)} disabled={fsLoading} style={{ padding: "6px 10px" }}>Refresh</button>
                <button onClick={() => setFsOpen(false)} style={{ padding: "6px 10px" }}>Close</button>
              </div>
            </div>
            <div style={{ border: "1px solid #eee", borderRadius: 4, overflow: "hidden", flex: 1 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 120px 180px", gap: 0, background: "#fafafa", fontSize: 14, fontWeight: 600, padding: 8 }}>
                <div>Name</div>
                <div>Size</div>
                <div>Modified</div>
              </div>
              <div style={{ height: "100%", overflow: "auto" }}>
                {fsLoading ? (
                  <div style={{ padding: 10, fontSize: 15 }}>Loading…</div>
                ) : (
                  fsEntries.map((e) => (
                    <div key={e.name} style={{ display: "grid", gridTemplateColumns: "1fr 120px 180px", gap: 0, padding: 6, borderTop: "1px solid #f3f3f3", cursor: e.is_dir ? "pointer" : "default" }} onClick={() => { if (e.is_dir) fetchFs((fsCwd ? fsCwd + "/" : "") + e.name); }}>
                      <div>{e.is_dir ? "📁 " : "📄 "}{e.name}</div>
                      <div>{e.is_dir ? "—" : formatSize(e.size)}</div>
                      <div>{e.mtime ? new Date(e.mtime).toLocaleString() : ""}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Shared console output pinned at bottom with fixed height */}
      <div className="admin-tool alt" style={{ gridColumn: "1 / -1" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <h3 style={{ margin: "4px 0", fontSize: 16, lineHeight: 1.2 }}>Console</h3>
          <div>
            <button onClick={() => setConsoleText("")} style={{ padding: "6px 10px" }}>🧹 Clear</button>
          </div>
        </div>
        <div className="surface" style={{ padding: 6 }}>
          <pre
            ref={consoleRef}
            style={{ whiteSpace: "pre-wrap", height: 220, overflow: "auto", margin: 0, fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace", fontSize: 14 }}
          >
{consoleText || ""}
          </pre>
        </div>
      </div>
    </div>
  );
}





