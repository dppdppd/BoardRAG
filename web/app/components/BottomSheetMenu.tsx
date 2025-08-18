"use client";

import React from "react";

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
};

export default function BottomSheetMenu({ open, onClose, games, selectedGame, setSelectedGame, sessionId, messages, promptStyle, setPromptStyle, includeWeb, setIncludeWeb, pdfSmoothScroll, setPdfSmoothScroll, onUpload, uploadInputRef, uploading, uploadMsg }: Props) {
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
      </div>
    </>
  );
}


