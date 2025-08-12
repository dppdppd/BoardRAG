"use client";

import React, { useEffect, useState } from "react";
import { API_BASE } from "../lib/config";

type Props = { children: React.ReactNode };

// API_BASE is normalized to include protocol and no trailing slash

export default function AuthGate({ children }: Props) {
  const [role, setRole] = useState<string | null>(null); // null = unknown (loading), "none" = locked
  const [pw, setPw] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        // Prefer sessionStorage role+token
        const fromSession = sessionStorage.getItem("boardrag_role");
        const token = sessionStorage.getItem("boardrag_token");
        if (fromSession && token) {
          setRole(fromSession);
          return;
        }

        // If both role and token exist in localStorage, adopt them for this tab
        const localRole = localStorage.getItem("boardrag_role");
        const localToken = localStorage.getItem("boardrag_token");
        if (localRole && localToken) {
          try { sessionStorage.setItem("boardrag_role", localRole); } catch {}
          try { sessionStorage.setItem("boardrag_token", localToken); } catch {}
          setRole(localRole);
          return;
        }

        // If saved password present, auto-unlock to refresh token
        const savedPw = localStorage.getItem("boardrag_saved_pw");
        if (savedPw) {
          const form = new FormData();
          form.append("password", savedPw);
          try {
            const resp = await fetch(`${API_BASE}/auth/unlock`, { method: "POST", body: form });
            if (resp.ok) {
              const data = await resp.json();
              const r = data.role || "user";
              try { sessionStorage.setItem("boardrag_role", r); } catch {}
              try {
                if (data.token) {
                  sessionStorage.setItem("boardrag_token", data.token);
                  localStorage.setItem("boardrag_token", data.token);
                }
                localStorage.setItem("boardrag_role", r);
              } catch {}
              setRole(r);
              return;
            }
          } catch {}
        }
      } catch {}
      setRole("none");
    })();
  }, []);

  const unlock = async () => {
    setError("");
    try {
      const form = new FormData();
      form.append("password", pw);
      const resp = await fetch(`${API_BASE}/auth/unlock`, { method: "POST", body: form });
      if (!resp.ok) {
        setError("Invalid password");
        setRole("none");
        return;
      }
      const data = await resp.json();
      sessionStorage.setItem("boardrag_role", data.role || "user");
      if (data.token) {
        try { sessionStorage.setItem("boardrag_token", data.token); } catch {}
      }
      // Always persist to localStorage for cross-restart persistence
      try {
        localStorage.setItem("boardrag_role", data.role || "user");
        if (data.token) localStorage.setItem("boardrag_token", data.token);
        localStorage.setItem("boardrag_saved_pw", pw);
      } catch {}
      setRole(data.role || "user");
    } catch (e) {
      setError("Network error");
    }
  };

  if (role === null || role === "none") {
    return (
      <div style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        background: "var(--bg)",
        color: "var(--text)",
        padding: 16,
      }}>
        <div className="surface" style={{ width: "min(92vw, 520px)", padding: 16 }}>
          <div className="title" style={{ padding: 0, marginBottom: 6 }}>BoardRAG</div>
          <div className="subtitle" style={{ marginTop: 0 }}>Enter access code to continue.</div>
          <div className="row" style={{ gap: 8, marginTop: 10 }}>
            <input
              className="input"
              autoFocus
              type="password"
              placeholder="Password"
              value={pw}
              onChange={(e) => setPw(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") unlock(); }}
              style={{ flex: 1 }}
            />
            <button className="btn primary" onClick={unlock}>Unlock</button>
          </div>
          {error && <div style={{ color: "crimson", marginTop: 8 }}>{error}</div>}
        </div>
      </div>
    );
  }

  return <>{children}</>;
}


