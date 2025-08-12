"use client";

import React, { useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export default function ResetPage() {
  const router = useRouter();
  const params = useSearchParams();

  useEffect(() => {
    try {
      // Always clear session-only data (role, etc.)
      sessionStorage.clear();
      // Remove auth-related items from localStorage (preserve conversations by default)
      ["boardrag_role", "boardrag_token", "boardrag_saved_pw"].forEach((k) => {
        try { localStorage.removeItem(k); } catch {}
      });

      // If all=1, clear everything app-related (auth + conversations + prefs + session id)
      const clearAll = params.get("all") === "1";
      if (clearAll) {
        try {
          const toDelete: string[] = [];
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (!key) continue;
            // Delete all known app keys
            if (key === "boardrag_session_id") toDelete.push(key);
            if (key === "boardrag_last_game") toDelete.push(key);
            if (key.startsWith("boardrag_conv:")) toDelete.push(key);
            if (key.startsWith("boardrag_style:")) toDelete.push(key);
          }
          toDelete.forEach((k) => localStorage.removeItem(k));
        } catch {}
      }
    } catch {}

    // Navigate back to home
    router.replace("/");
  }, [router, params]);

  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", background: "var(--bg)", color: "var(--text)", padding: 16 }}>
      <div className="surface" style={{ padding: 16 }}>
        <div className="title" style={{ padding: 0, marginBottom: 6 }}>Resetting…</div>
        <div className="subtitle">Clearing session and redirecting.</div>
      </div>
    </div>
  );
}


