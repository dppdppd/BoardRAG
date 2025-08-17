"use client";

import React from "react";

type Chunk = { uid?: string; text: string; source?: string; page?: number };

type Props = {
  loading: boolean;
  chunks: Chunk[] | null;
  expanded: boolean;
  setExpanded: (v: boolean) => void;
};

export default function PreviewChunksPanel({ loading, chunks, expanded, setExpanded }: Props) {
  if (loading || !chunks || chunks.length === 0) return null;
  return (
    <>
      <button 
        className="btn chunks-toggle" 
        onClick={() => setExpanded(!expanded)}
        style={{ 
          width: 24,
          height: 24,
          minHeight: 24,
          padding: 0,
          fontSize: 12,
          display: 'inline-grid',
          placeItems: 'center',
          background: 'var(--muted)',
          color: 'var(--text)',
          borderRadius: '3px',
          opacity: 0.6,
          border: '1px solid var(--border)'
        }}
        title={expanded ? 'Hide chunks' : 'Show chunks'}
      >
        {expanded ? '−' : '+'}
      </button>
      <div className={`preview-bottom ${expanded ? 'expanded' : ''}`}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--muted)' }}>Chunks ({chunks.length})</div>
        </div>
        <div className="chunk-list">
          {chunks.map((c, i) => (
            <div key={c.uid || i} className="chunk-item surface">
              <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
                {c.source}{typeof c.page === 'number' ? ` · p.${c.page}` : ''}
              </div>
              <div className="chunk-text">{c.text}</div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}


