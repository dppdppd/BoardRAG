"use client";

import React from "react";
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
};

export default function PdfPreview({ API_BASE, token, title, chunks, pdfMeta, setPdfMeta }: Props) {
  const inferred = (chunks && chunks[0] && chunks[0].source ? String(chunks[0].source) : '').toLowerCase();
  const filename = String((pdfMeta && pdfMeta.filename) ? pdfMeta.filename : inferred).toLowerCase();
  const pageSet = new Set<number>();
  chunks.forEach((c) => { if (typeof c.page === 'number') pageSet.add(Number(c.page) + 1); });
  const citedPages = Array.from(pageSet).sort((a,b) => a-b);
  const totalPages = (pdfMeta && pdfMeta.pages) ? Number(pdfMeta.pages) : undefined;
  let pages = totalPages ? Array.from({ length: totalPages }, (_v, i) => i + 1) : (citedPages.length > 0 ? citedPages : [1]);
  const targetPage = citedPages.length > 0 ? citedPages[0] : 1;
  const pdfUrl = filename ? `${API_BASE}/pdf?filename=${encodeURIComponent(filename)}${token ? `&token=${encodeURIComponent(token)}` : ''}` : '';
  if (!pdfUrl) return <div className="muted">Missing PDF filename.</div>;
  const byPage = (p: number) => chunks.filter((c) => (Number(c.page) + 1) === Number(p));

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
    <Document key={'preview-doc'} file={pdfUrl} onLoadSuccess={(info: { numPages: number }) => setPdfMeta({ filename, pages: info.numPages })} loading={null}>
      {pages.map((p) => (
        <div key={p} className="pdf-page" data-page-number={p} data-target-section={title}>
          <Page
            pageNumber={p}
            width={
              (document?.querySelector?.('.modal-preview .preview-top') as HTMLElement)?.clientWidth ||
              (document?.querySelector?.('.preview-panel .preview-top') as HTMLElement)?.clientWidth ||
              (undefined as any)
            }
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
      ))}
    </Document>
  );
}


