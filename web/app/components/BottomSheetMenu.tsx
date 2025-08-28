"use client";

import React, { useState, useEffect } from "react";
import { isLocalhost } from "../../lib/config";
import { API_BASE } from "../../lib/config";

type Props = {
  open: boolean;
  onClose: () => void;
  games: string[];
  selectedGame: string;
  setSelectedGame: (g: string) => void;
  sessionId: string;
  messages: Array<{ role: string; content: string; pinned?: boolean }>;
  includeWeb: boolean;
  setIncludeWeb: (v: boolean) => void;
  pdfSmoothScroll: boolean;
  setPdfSmoothScroll: (v: boolean) => void;
  onUpload: (files: FileList | null) => Promise<void> | void;
  uploadInputRef: React.RefObject<HTMLInputElement>;
  uploading: boolean;
  uploadMsg: string;
  isAdmin: boolean;
  onSignOut?: () => void;
};

export default function BottomSheetMenu({ open, onClose, games, selectedGame, setSelectedGame, sessionId, messages, includeWeb, setIncludeWeb, pdfSmoothScroll, setPdfSmoothScroll, onUpload, uploadInputRef, uploading, uploadMsg, isAdmin, onSignOut }: Props) {
  return (
    <>
      {open && (
        <div className="menu-backdrop" onClick={onClose} style={{ position: 'absolute', inset: 0, zIndex: 12 }} />
      )}
      <div className={`mobile-sheet ${open ? 'open' : ''}`}>
        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="title" style={{ padding: 0, margin: 0, textAlign: 'left' }}>Menu</div>
          <button className="btn" onClick={onClose} aria-label="Close menu" title="Close menu" style={{ width: 44, height: 44, minHeight: 44, padding: 0, display: 'inline-grid', placeItems: 'center', lineHeight: 1, fontSize: 28 }}>×</button>
        </div>
        <section className="surface pad section" style={{ marginTop: 12 }}>
          <summary>Settings</summary>
          <div style={{ display: "grid", gap: 10, marginTop: 8 }}>
            {/* Style selection moved to Ask button long-press; removed from menu */}
            <label className="row" style={{ gap: 10 }}>
              <input type="checkbox" checked={includeWeb} onChange={(e) => setIncludeWeb(e.target.checked)} />
              <span>Include Web Search</span>
            </label>
            <label className="row" style={{ gap: 10 }}>
              <input type="checkbox" checked={pdfSmoothScroll} onChange={(e) => setPdfSmoothScroll(e.target.checked)} />
              <span>Smooth Scrolling PDF</span>
            </label>
          </div>
        </section>
        {isAdmin && (
          <section className="surface pad section" style={{ marginTop: 12 }}>
            <summary>Debug Bbox</summary>
            <DebugBboxControl selectedGame={selectedGame} />
          </section>
        )}
        {isAdmin && (
          <section className="surface pad section" style={{ marginTop: 12 }}>
            <summary>Page Citations (Admin)</summary>
            <AdminPageCitations selectedGame={selectedGame} />
          </section>
        )}
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
        <section className="surface pad section" style={{ marginTop: 12 }}>
          <summary>Account</summary>
          <div style={{ marginTop: 8 }}>
            <button className="btn" onClick={() => onSignOut && onSignOut()} style={{ padding: '6px 10px' }}>Sign out</button>
          </div>
        </section>
      </div>
    </>
  );
}

// Helper function to trigger debug bbox update
function triggerDebugBboxUpdate() {
  // Dispatch a custom event that PdfPreview can listen to
  const event = new CustomEvent('debugBboxUpdate', {
    detail: { timestamp: Date.now() }
  });
  window.dispatchEvent(event);
}

// Debug bbox control component for admins
function DebugBboxControl({ selectedGame }: { selectedGame: string }) {
  const [debugInput, setDebugInput] = useState<string>("46.6,16.5,3.7,1.2,9");
  const [isActive, setIsActive] = useState<boolean>(false);

  // Check if there's an active debug bbox
  useEffect(() => {
    try {
      const stored = sessionStorage.getItem('debug_bbox');
      if (stored) {
        const data = JSON.parse(stored);
        if (data.active) {
          setDebugInput(`${data.x || 0},${data.y || 0},${data.w || 0},${data.h || 0},${data.page || 1}`);
          setIsActive(true);
        }
      }
    } catch (error) {
      console.error('[DEBUG BBOX] Failed to load debug bbox:', error);
    }
  }, []);

  const applyDebugBbox = () => {
    try {
      const parts = debugInput.split(',').map(s => s.trim());
      if (parts.length !== 5) {
        alert('Please enter exactly 5 values: X,Y,W,H,Page');
        return;
      }

      const [x, y, w, h, page] = parts.map(p => parseFloat(p));
      if (parts.some(p => isNaN(parseFloat(p)))) {
        alert('All values must be numbers');
        return;
      }

      const debugData = {
        x, y, w, h,
        page: Math.max(1, Math.round(page)),
        filename: selectedGame,
        active: true,
        timestamp: Date.now()
      };

      sessionStorage.setItem('debug_bbox', JSON.stringify(debugData));
      setIsActive(true);
      console.log('[DEBUG BBOX] Applied debug bbox:', debugData);
      
      // Trigger highlight update without page refresh
      triggerDebugBboxUpdate();
      
    } catch (error) {
      console.error('[DEBUG BBOX] Error applying debug bbox:', error);
      alert('Error applying debug bbox. Check console for details.');
    }
  };

  const clearDebugBbox = () => {
    try {
      sessionStorage.removeItem('debug_bbox');
      setIsActive(false);
      console.log('[DEBUG BBOX] Cleared debug bbox');
      
      // Remove highlight without page refresh
      triggerDebugBboxUpdate();
      
    } catch (error) {
      console.error('[DEBUG BBOX] Error clearing debug bbox:', error);
    }
  };

  // Removed calibration points debug utility

  return (
    <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
      <div style={{ fontSize: 13, color: '#666' }}>
        Test bbox coordinates (X,Y,W,H,Page) as percentages
      </div>
      
      <input
        className="input"
        type="text"
        value={debugInput}
        onChange={(e) => setDebugInput(e.target.value)}
        placeholder="46.6,16.5,3.7,1.2,9"
        style={{ fontFamily: 'monospace', fontSize: 13 }}
      />
      
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <button 
          className="btn primary"
          onClick={applyDebugBbox}
          disabled={!selectedGame}
          style={{ fontSize: 13 }}
        >
          🎯 Apply Debug Bbox
        </button>
        
        <button 
          className="btn"
          onClick={clearDebugBbox}
          style={{ fontSize: 13 }}
        >
          🗑️ Clear
        </button>
        
        {/* Calibration points debug removed */}

        {isActive && (
          <div style={{ 
            padding: '2px 6px', 
            backgroundColor: '#d4edda', 
            color: '#155724', 
            borderRadius: 3, 
            fontSize: 11,
            fontWeight: 600
          }}>
            ✅ Active
          </div>
        )}
      </div>

      <div style={{ fontSize: 11, color: '#666', padding: 6, backgroundColor: '#f8f9fa', borderRadius: 4 }}>
        <div><strong>Example "13.3 WIRE":</strong></div>
        <div>• Actual: 46.6,16.5,3.7,1.2,9</div>
        <div>• LLM: 43.8,2.1,18.5,1.8,9</div>
        <div style={{ marginTop: 4 }}>X=vertical%, Y=horizontal%, W=width%, H=height%, Page=number</div>
      </div>
    </div>
  );
}


function AdminPageCitations({ selectedGame }: { selectedGame: string }) {
  const [pageInput, setPageInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [results, setResults] = useState<any[]>([]);
  const [pdfList, setPdfList] = useState<string[]>([]);
  const [selectedPdf, setSelectedPdf] = useState<string>("");

  const apiBase = API_BASE;

  const onFetch = async () => {
    setError("");
    setResults([]);
    const p = parseInt(pageInput || "", 10);
    if (!selectedGame) { setError("Select a game first"); return; }
    const pdfName = (selectedPdf || "").trim();
    if (!pdfName) { setError("Select a PDF"); return; }
    if (!Number.isFinite(p) || p <= 0) { setError("Enter a valid page number"); return; }
    try {
      setLoading(true);
      const fn = pdfName;
      const base = String(apiBase || "").replace(/\/+$/, "");
      const url = `${base}/admin/citations-by-page?filename=${encodeURIComponent(fn)}&page=${encodeURIComponent(String(p))}`;
      const headers: any = {};
      try {
        const t = sessionStorage.getItem('boardrag_token') || localStorage.getItem('boardrag_token');
        if (t) headers['Authorization'] = `Bearer ${t}`;
      } catch {}
      const resp = await fetch(url, { headers });
      if (!resp.ok) {
        const msg = await resp.text().catch(() => resp.statusText);
        throw new Error(msg || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setResults(Array.isArray(data?.citations) ? data.citations : []);
    } catch (e: any) {
      setError(String(e?.message || e || 'error'));
    } finally {
      setLoading(false);
    }
  };

  // Load PDFs for the current game
  useEffect(() => {
    setPdfList([]);
    setSelectedPdf("");
    if (!selectedGame) return;
    (async () => {
      try {
        const base = String(apiBase || "").replace(/\/+$/, "");
        const url = `${base}/game-pdfs?game=${encodeURIComponent(selectedGame)}`;
        const headers: any = {};
        try { const t = sessionStorage.getItem('boardrag_token') || localStorage.getItem('boardrag_token'); if (t) headers['Authorization'] = `Bearer ${t}`; } catch {}
        const resp = await fetch(url, { headers });
        if (!resp.ok) {
          // fallback to last seen
          let files: string[] = [];
          try { const lastPdf = localStorage.getItem(`boardrag_last_pdf:${selectedGame}`); if (lastPdf) files = [lastPdf]; } catch {}
          setPdfList(files);
          if (files && files.length > 0) setSelectedPdf(files[0]);
          return;
        }
        const data = await resp.json();
        const files: string[] = Array.isArray(data?.pdfs) ? data.pdfs : [];
        setPdfList(files);
        if (files && files.length > 0) setSelectedPdf(files[0]);
      } catch {}
    })();
  }, [selectedGame]);

  return (
    <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
      <div className="row" style={{ gap: 8, alignItems: 'center' }}>
        <select
          className="select compact"
          value={selectedPdf}
          onChange={(e) => setSelectedPdf(e.target.value)}
          style={{ minWidth: 200 }}
        >
          <option value="" disabled>Select PDF…</option>
          {pdfList.map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
        <input
          className="input"
          type="number"
          min={1}
          placeholder="Page #"
          value={pageInput}
          onChange={(e) => setPageInput(e.target.value)}
          style={{ width: 120 }}
        />
        <button className="btn" onClick={onFetch} disabled={loading || !selectedGame} style={{ padding: '6px 10px' }}>
          {loading ? 'Loading…' : 'List citations'}
        </button>
      </div>
      {error && (
        <div className="muted" style={{ color: '#b00020', fontSize: 12 }}>{error}</div>
      )}
      {(!loading && (!results || results.length === 0) && !error) && (
        <div className="bubble assistant" style={{ fontSize: 13 }}>
          Enter a page number and click List citations.
        </div>
      )}
      {!loading && error && (
        <div className="bubble assistant" style={{ fontSize: 13, color: '#b00020' }}>
          {error}
        </div>
      )}
      {!loading && results && (
        <div className="bubble assistant" style={{ fontSize: 13 }}>
          {results.length > 0 ? (
            <div>
              {results.map((r, i) => (
                <div key={i} style={{ marginBottom: 6 }}>
                  <div>
                    {/* Render a clickable section link matching normal answer bubbles */}
                    • <a
                      href={`section:${encodeURIComponent(String(r.code || r.section || ''))}`}
                      className="btn link"
                      onClick={(e) => {
                        e.preventDefault();
                        try {
                          const fn = selectedGame && selectedGame.endsWith('.pdf') ? selectedGame : String(r.file || '');
                          const meta = { file: fn || r.file, page: r.page, header_anchor_bbox_pct: r.header_anchor_bbox_pct, header: r.section };
                          const op = (window as any).__openSectionModal;
                          if (typeof op === 'function') op(String(r.code || r.section || ''), meta);
                        } catch {}
                      }}
                      style={{ padding: 0, background: 'none', border: 'none', color: 'var(--accent)', textDecoration: 'underline', cursor: 'pointer' }}
                    >
                      [{String(r.code || r.section || '—')}]
                    </a>
                    {r.section && r.code ? <span style={{ color: '#666' }}> ({r.section})</span> : null}
                  </div>
                  {Array.isArray(r.header_anchor_bbox_pct) && r.header_anchor_bbox_pct.length >= 4 && (
                    <div style={{ color: '#666', marginLeft: 16 }}>
                      anchor: [{r.header_anchor_bbox_pct.map((n: any) => Number(n).toFixed(2)).join(', ')}]
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div>No citations found for this page.</div>
          )}
        </div>
      )}
    </div>
  );
}

