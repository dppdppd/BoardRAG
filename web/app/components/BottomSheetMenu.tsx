"use client";

import React, { useState, useEffect } from "react";

type Props = {
  open: boolean;
  onClose: () => void;
  games: string[];
  selectedGame: string;
  setSelectedGame: (g: string) => void;
  sessionId: string;
  messages: Array<{ role: string; content: string; pinned?: boolean }>;
  promptStyle: string;
  setPromptStyle: (s: string) => void;
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

export default function BottomSheetMenu({ open, onClose, games, selectedGame, setSelectedGame, sessionId, messages, promptStyle, setPromptStyle, includeWeb, setIncludeWeb, pdfSmoothScroll, setPdfSmoothScroll, onUpload, uploadInputRef, uploading, uploadMsg, isAdmin, onSignOut }: Props) {
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


