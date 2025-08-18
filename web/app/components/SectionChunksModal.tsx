"use client";

import React, { useEffect, useRef, useState } from "react";
import { Document, Page } from "react-pdf";

type Chunk = { uid?: string; text: string; source?: string; page?: number; section?: string; section_number?: string };

type Props = {
  open: boolean;
  onClose: () => void;
  loading: boolean;
  error: string | null;
  chunks: Chunk[] | null;
  API_BASE: string;
  token: string | null;
  sectionTitle?: string;
  setPdfMeta: (meta: { filename?: string; pages?: number }) => void;
  pdfSmoothScroll?: boolean;
};

export default function SectionChunksModal({ open, onClose, loading, error, chunks, API_BASE, token, sectionTitle, setPdfMeta, pdfSmoothScroll = true }: Props) {
  const [numPages, setNumPages] = useState<number | undefined>(undefined);
  const scrolledRef = useRef<boolean>(false);
  useEffect(() => { if (open) { scrolledRef.current = false; } }, [open]);
  if (!open) return null;
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="row" style={{ justifyContent: 'flex-end', alignItems: 'center' }}>
          <button className="btn" onClick={onClose} aria-label="Close" title="Close" style={{ width: 36, height: 36, minHeight: 36, padding: 0, display: 'inline-grid', placeItems: 'center' }}>×</button>
        </div>
        <div className="modal-body">
          {loading && (
            <div className="indicator" style={{ gap: 8 }}>
              <span className="spinner" />
              <span>Loading…</span>
            </div>
          )}
          {!!error && !loading && (
            <div className="muted" style={{ color: 'var(--danger, #b00020)' }}>{error}</div>
          )}
          {!loading && (!chunks || chunks.length === 0) && (
            <div className="muted">No chunks found for this section.</div>
          )}
          {!loading && chunks && chunks.length > 0 && (
            <div className="pdf-viewer" style={{ padding: 0 }}>
              {(() => {
                const first = chunks[0];
                const filename = (first?.source || '').toLowerCase();
                const pageSet = new Set<number>();
                chunks.forEach((c) => { if (typeof c.page === 'number') pageSet.add(Number(c.page) + 1); });
                const citedPages = Array.from(pageSet).sort((a,b) => a-b);
                const pdfUrl = filename ? `${API_BASE}/pdf?filename=${encodeURIComponent(filename)}${token ? `&token=${encodeURIComponent(token)}` : ''}` : '';
                if (!pdfUrl) return <div className="muted">Missing PDF filename.</div>;
                const byPage = (p: number) => chunks.filter((c) => (Number(c.page) + 1) === Number(p));
                const totalPages = numPages;
                const pages = totalPages ? Array.from({ length: totalPages }, (_v, i) => i + 1) : (citedPages.length > 0 ? citedPages : [1]);
                const targetPage = citedPages.length > 0 ? citedPages[0] : 1;
                const highlightHeading = (el: HTMLElement, pageChunks: Chunk[]): number | null => {
                  try {
                    const layer = (el.querySelector('.react-pdf__Page__textContent') || el.querySelector('.textLayer') || el) as HTMLElement | null;
                    if (!layer) return;
                    const spans = Array.from(layer.querySelectorAll('span')) as HTMLSpanElement[];
                    if (spans.length === 0) return;
                    const norm = (s: string) => s.replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
                    const candidates: string[] = [];
                    for (const c of pageChunks) {
                      const sec = (c.section || '').trim();
                      if (sec) candidates.push(sec.replace(/:$/, ''));
                      // Numeric heading at start of chunk text
                      const mNum = (c.text || '').match(/^(\d+(?:\.\d+)+)\s+(.{2,80})/m);
                      if (mNum) candidates.push(`${mNum[1]} ${mNum[2].split(':',1)[0]}`.trim());
                      // Alphanumeric heading at start of chunk text (e.g., F1.b Title)
                      const mAlpha = (c.text || '').match(/^([A-Za-z]\d+(?:\.[A-Za-z0-9]+)*)\s+(.{2,80})/m);
                      if (mAlpha) candidates.push(`${mAlpha[1]} ${mAlpha[2].split(':',1)[0]}`.trim());
                    }
                    const unique = Array.from(new Set(candidates.filter(Boolean).map(norm)));
                    const textAll = spans.map(s => norm(s.textContent || '')).join('');
                    let startSpan = -1, endSpan = -1;
                    for (const cand of unique) {
                      const idx = textAll.indexOf(cand);
                      if (idx >= 0) {
                        let acc = '';
                        for (let si = 0; si < spans.length; si++) {
                          const prev = acc.length;
                          acc += norm(spans[si].textContent || '');
                          if (startSpan < 0 && acc.length > idx) startSpan = si;
                          if (startSpan >= 0 && acc.length >= idx + cand.length) { endSpan = si; break; }
                        }
                        break;
                      }
                    }
                    if (startSpan < 0) {
                      for (let si = 0; si < spans.length; si++) {
                        const t = norm(spans[si].textContent || '');
                        if (/^(?:[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*|\d+(?:\.\d+)+)\b/.test(t)) { startSpan = si; endSpan = si; break; }
                      }
                    }
                    if (startSpan >= 0) {
                      try {
                        const pageRect = el.getBoundingClientRect();
                        const sr = (spans[startSpan] as any).getBoundingClientRect?.();
                        if (sr) {
                          // Return CENTER Y of the heading relative to the page container
                          const headingCenterY = Math.max(0, (sr.top - pageRect.top) + (sr.height ? sr.height / 2 : ((sr.bottom - sr.top) / 2)));
                          return headingCenterY;
                        }
                      } catch {}
                    }
                  } catch {}
                  return null;
                };
                return (
                  <div style={{ width: '100%' }}>
                    <Document
                      key={'modal-doc'}
                      file={pdfUrl}
                      onLoadSuccess={(info: { numPages: number }) => { setPdfMeta({ filename, pages: info.numPages }); setNumPages(info.numPages); }}
                      loading={null}
                    >
                      {pages.map((p) => (
                        <div key={p} className="pdf-page" data-page-number={p} data-target-section={sectionTitle}>
                          {(() => {
                            const chunksForPage = byPage(p);
                            const norm = (s: string) => s.replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
                            const textRenderer = ({ str }: { str: string }) => str;
                            return (
                              <Page
                                pageNumber={p}
                                width={(document?.querySelector?.('.modal .modal-body') as HTMLElement)?.clientWidth || 900}
                                renderTextLayer
                                renderAnnotationLayer={false}
                                onRenderSuccess={() => {
                                  try {
                                    const containerAll = document?.querySelector?.(`.modal .pdf-page[data-page-number='${p}']`) as HTMLElement | null;
                                    const elAll = containerAll || (document?.querySelector?.('.pdf-page:last-child') as HTMLElement);
                                    if (elAll) {
                                      try { elAll.querySelectorAll('.pdf-dark-overlay').forEach((n) => n.remove()); } catch {}
                                      const headingCenterY = highlightHeading(elAll, chunksForPage);
                                      // Scroll modal container to the target page once
                                      try {
                                        if (p === targetPage && !scrolledRef.current) {
                                          const sc = document.querySelector('.modal .modal-body') as HTMLElement | null;
                                          if (sc) {
                                            scrolledRef.current = true;
                                            const baseTop = (elAll as HTMLElement).offsetTop;
                                            const desired = (typeof headingCenterY === 'number')
                                              ? (baseTop + headingCenterY - (sc.clientHeight / 2))
                                              : (baseTop - 8);
                                            const maxTop = Math.max(0, sc.scrollHeight - sc.clientHeight);
                                            const topTarget = Math.max(0, Math.min(desired, maxTop));
                                            sc.scrollTo({ top: topTarget, behavior: (pdfSmoothScroll ? 'smooth' : 'auto') as ScrollBehavior });
                                          }
                                        }
                                      } catch {}
                                    }
                                  } catch {}
                                }}
                                customTextRenderer={textRenderer}
                              />
                            );
                          })()}
                          <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>{filename}</div>
                        </div>
                      ))}
                    </Document>
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


