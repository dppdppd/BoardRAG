"use client";

import React, { useEffect, useRef, useState } from "react";
import { Document, Page } from "react-pdf";

type Chunk = { text: string; source?: string; page?: number; rects_norm?: any };

type PdfMeta = { filename?: string; pages?: number };

type Props = {
  API_BASE: string;
  token: string | null;
  title: string;
  chunks: Chunk[];
  pdfMeta: PdfMeta | undefined;
  setPdfMeta: (m: PdfMeta) => void;
  targetPage?: number | null;
  adjacentPageWindow?: number; // how many pages on each side of target to render
};

export default function PdfPreview({ API_BASE, token, title, chunks, pdfMeta, setPdfMeta, targetPage, adjacentPageWindow = 1 }: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [centerPage, setCenterPage] = useState<number | null>(null);
  const inferred = (chunks && chunks[0] && chunks[0].source ? String(chunks[0].source) : '').toLowerCase();
  const filename = String((pdfMeta && pdfMeta.filename) ? pdfMeta.filename : inferred).toLowerCase();
  const pageSet = new Set<number>();
  chunks.forEach((c) => { if (typeof c.page === 'number') pageSet.add(Number(c.page) + 1); });
  const citedPages = Array.from(pageSet).sort((a,b) => a-b);
  const totalPages = (pdfMeta && pdfMeta.pages) ? Number(pdfMeta.pages) : undefined;
  const computedTarget = (typeof targetPage === 'number' && targetPage > 0)
    ? targetPage
    : (citedPages.length > 0 ? citedPages[0] : 1);
  let pages: number[] = [];
  if (typeof targetPage === 'number') {
    if (typeof totalPages === 'number' && totalPages > 0) {
      const start = Math.max(1, targetPage - adjacentPageWindow);
      const end = Math.min(totalPages, targetPage + adjacentPageWindow);
      for (let p = start; p <= end; p++) pages.push(p);
    } else {
      // Unknown total yet → render only the target; once numPages arrives, window will expand automatically
      pages = [targetPage];
    }
  } else {
    // Fallback: previous behavior
    pages = totalPages ? Array.from({ length: totalPages }, (_v, i) => i + 1) : (citedPages.length > 0 ? citedPages : [1]);
  }
  const targetPageResolved = computedTarget;
  // Prefer scroll-detected center page when available; otherwise use prop-derived target
  const effectiveCenter = typeof centerPage === 'number' && centerPage > 0 ? centerPage : targetPageResolved;
  const pdfUrl = filename ? `${API_BASE}/pdf?filename=${encodeURIComponent(filename)}${token ? `&token=${encodeURIComponent(token)}` : ''}` : '';
  if (!pdfUrl) return <div className="muted">Missing PDF filename.</div>;
  const byPage = (p: number) => chunks.filter((c) => (Number(c.page) + 1) === Number(p));
  const containerWidth = (
    (document?.querySelector?.('.modal-preview .preview-top') as HTMLElement)?.clientWidth ||
    (document?.querySelector?.('.preview-panel .preview-top') as HTMLElement)?.clientWidth ||
    (undefined as any)
  );
  const placeholderAspect = 1.35; // approximate page height/width ratio for placeholders
  const shouldRenderReal = (p: number): boolean => {
    if (typeof effectiveCenter === 'number' && typeof totalPages === 'number') {
      return p >= Math.max(1, effectiveCenter - adjacentPageWindow) && p <= Math.min(totalPages, effectiveCenter + adjacentPageWindow);
    }
    // When no center/total available, render only the conservative set
    return pages.includes(p);
  };
  const allPages: number[] = (typeof totalPages === 'number' && totalPages > 0)
    ? Array.from({ length: totalPages }, (_v, i) => i + 1)
    : pages;

  // Keep centerPage in sync with prop target changes
  useEffect(() => {
    if (typeof targetPage === 'number' && targetPage > 0) {
      setCenterPage(targetPage);
    }
  }, [targetPage]);

  // After we know total pages, ensure we have an initial center
  useEffect(() => {
    if (!centerPage && typeof targetPageResolved === 'number' && targetPageResolved > 0) {
      setCenterPage(targetPageResolved);
    }
  }, [centerPage, targetPageResolved]);

  // Detect manual scroll position and update center page dynamically
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    // Find nearest scroll container (the preview-top element)
    const findScrollContainer = (el: HTMLElement | null): HTMLElement | null => {
      let cur: HTMLElement | null = el;
      while (cur) {
        if (cur.classList && cur.classList.contains('preview-top')) return cur;
        cur = cur.parentElement as HTMLElement | null;
      }
      return null;
    };
    const scrollContainer = findScrollContainer(root);
    if (!scrollContainer) return;
    let rafId: number | null = null;
    const onScrollOrResize = () => {
      if (rafId != null) return;
      rafId = requestAnimationFrame(() => {
        rafId = null;
        try {
          // Compute which page is closest to the vertical center of the container
          const containerRect = scrollContainer.getBoundingClientRect();
          const containerCenterY = containerRect.top + containerRect.height / 2;
          const nodes = Array.from(root.querySelectorAll('.pdf-page')) as HTMLElement[];
          let bestPage = typeof effectiveCenter === 'number' ? effectiveCenter : 1;
          let bestDist = Number.POSITIVE_INFINITY;
          for (const n of nodes) {
            const r = n.getBoundingClientRect();
            const mid = r.top + r.height / 2;
            const d = Math.abs(mid - containerCenterY);
            const pn = Number(n.getAttribute('data-page-number') || '0');
            if (pn > 0 && d < bestDist) { bestDist = d; bestPage = pn; }
          }
          if (bestPage && bestPage !== centerPage) setCenterPage(bestPage);
        } catch {}
      });
    };
    scrollContainer.addEventListener('scroll', onScrollOrResize, { passive: true });
    window.addEventListener('resize', onScrollOrResize);
    // Initial compute
    onScrollOrResize();
    return () => {
      scrollContainer.removeEventListener('scroll', onScrollOrResize as any);
      window.removeEventListener('resize', onScrollOrResize as any);
      if (rafId != null) cancelAnimationFrame(rafId);
    };
  }, [effectiveCenter]);

  const highlight = (el: HTMLElement, texts: string[]) => {
    try {
      const layer = el.querySelector('.textLayer');
      if (!layer) return;
      const spans = Array.from(layer.querySelectorAll('span')) as HTMLSpanElement[];
      const needles = texts.map((t) => (t || '').trim()).filter(Boolean);
      if (needles.length === 0) return;
      const joined = needles.join('\n');
      const normalizedNeedles = needles;
      let i = 0;
      while (i < spans.length) {
        let j = i; let acc = '';
        while (j < spans.length && acc.length < joined.length + 50) {
          acc += spans[j].textContent || '';
          const match = normalizedNeedles.some((n) => acc.includes(n));
          if (match) { for (let k = i; k <= j; k++) spans[k].classList.add('chunk-highlight'); i = j + 1; break; }
          j += 1;
        }
        i += 1;
      }
    } catch {}
  };

  return (
    <div ref={rootRef} className="pdf-preview-root">
    <Document key={'preview-doc'} file={pdfUrl} onLoadSuccess={(info: { numPages: number }) => setPdfMeta({ filename, pages: info.numPages })} loading={null}>
      {allPages.map((p) => {
        const renderReal = shouldRenderReal(p);
        if (!renderReal) {
          const phHeight = typeof containerWidth === 'number' && isFinite(containerWidth)
            ? Math.round(containerWidth * placeholderAspect)
            : 900;
          return (
            <div key={p} className="pdf-page placeholder" data-page-number={p} data-target-section={title} style={{ height: phHeight }}>
              <div className="muted" style={{ fontSize: 12 }}>{`Page ${p}`}</div>
            </div>
          );
        }
        return (
          <div key={p} className="pdf-page" data-page-number={p} data-target-section={title}>
            <Page
              pageNumber={p}
              width={containerWidth}
              renderTextLayer
              renderAnnotationLayer={false}
              onRenderSuccess={() => {
                try {
                  const container =
                    (document?.querySelector?.(`.modal-preview .pdf-page[data-page-number='${p}']`) as HTMLElement) ||
                    (document?.querySelector?.(`.preview-panel .pdf-page[data-page-number='${p}']`) as HTMLElement) ||
                    undefined;
                  const el =
                    container ||
                    (document?.querySelector?.('.modal-preview .pdf-page:last-child') as HTMLElement) ||
                    (document?.querySelector?.('.preview-panel .pdf-page:last-child') as HTMLElement);
                  if (!el) return;
                  const chunksForPage = byPage(p);
                  const texts = chunksForPage.map((c) => c.text);
                  highlight(el, texts);
                  // Draw rectangle overlays from normalized rects
                  try {
                    const pageViewport = el.getBoundingClientRect();
                    const existing = el.querySelectorAll('.rect-overlay'); existing.forEach((n) => n.remove());
                    const rects: Array<[number, number, number, number]> = [];
                    chunksForPage.forEach((c) => {
                      let rn: any = (c as any).rects_norm;
                      try { if (typeof rn === 'string') rn = JSON.parse(rn); } catch {}
                      if (Array.isArray(rn)) rn.forEach((r: any) => rects.push(r as any));
                    });
                    rects.forEach(([x0, y0, x1, y1]) => {
                      const div = document.createElement('div');
                      div.className = 'rect-overlay';
                      const w = el.clientWidth || pageViewport.width;
                      const h = el.clientHeight || pageViewport.height;
                      const left = x0 * w;
                      const top = y0 * h;
                      const width = Math.max(1, (x1 - x0) * w);
                      const height = Math.max(1, (y1 - y0) * h);
                      div.style.left = `${left}px`;
                      div.style.top = `${top}px`;
                      div.style.width = `${width}px`;
                      div.style.height = `${height}px`;
                      el.appendChild(div);
                    });
                  } catch {}
                  // Do not auto-scroll on mount; external controller handles scroll/spotlight
                } catch {}
              }}
              customTextRenderer={({ str }: { str: string }) => str}
            />
            <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>{filename}</div>
          </div>
        );
      })}
    </Document>
    </div>
  );
}


