"use client";

import { useCallback } from "react";

export function usePdfHeadingSpotlight() {
  const placeSpotlightRing = useCallback((pageEl: HTMLElement, targetSection: string, attempts: number = 0) => {
    try {
      const m = (targetSection || '').match(/\d+(?:\.\d+)+/);
      const needle = m ? String(m[0]).toLowerCase() : '';
      const layer = (pageEl.querySelector('.react-pdf__Page__textContent') || pageEl.querySelector('.textLayer')) as HTMLElement | null;
      if (!layer) { if (attempts < 25) setTimeout(() => placeSpotlightRing(pageEl, targetSection, attempts + 1), 80); return; }
      const spans = Array.from(layer.querySelectorAll('span')) as HTMLSpanElement[];
      if (spans.length === 0) { if (attempts < 25) setTimeout(() => placeSpotlightRing(pageEl, targetSection, attempts + 1), 80); return; }
      const norm = (s: string) => s.replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
      let idx = -1;
      for (let i = 0; i < spans.length; i++) {
        const t0 = norm(spans[i].textContent || '');
        const t1 = i + 1 < spans.length ? norm(spans[i+1].textContent || '') : '';
        const t2 = i + 2 < spans.length ? norm(spans[i+2].textContent || '') : '';
        const joined = (t0 + t1 + t2).slice(0, Math.max(needle.length + 4, 8));
        if (needle && joined.startsWith(needle)) { idx = i; break; }
        if (!needle && /^(\d+(?:\.\d+)+)\b/.test(t0)) { idx = i; break; }
      }
      if (idx < 0) { if (attempts < 25) setTimeout(() => placeSpotlightRing(pageEl, targetSection, attempts + 1), 80); return; }
      const span = spans[idx];
      const pageRect = (pageEl as HTMLElement).getBoundingClientRect();
      const r = span.getBoundingClientRect();
      const cx = (r.left + r.right) / 2 - pageRect.left;
      const cy = (r.top + r.bottom) / 2 - pageRect.top;
      try { pageEl.querySelectorAll('.spotlight-ring').forEach((n) => n.remove()); } catch {}
      const ring = document.createElement('div');
      ring.className = 'spotlight-ring';
      const diameter = Math.max(40, pageRect.width * 0.2);
      ring.style.width = `${diameter}px`;
      ring.style.height = `${diameter}px`;
      ring.style.left = `${Math.max(0, cx - diameter / 2)}px`;
      ring.style.top = `${Math.max(0, cy - diameter / 2)}px`;
      pageEl.appendChild(ring);
    } catch {}
  }, []);

  const scrollToTargetPage = useCallback((container: HTMLElement, pageEl: HTMLElement, smooth: boolean) => {
    try {
      container.scrollTo({ top: pageEl.offsetTop - 8, behavior: (smooth ? 'smooth' : 'auto') as ScrollBehavior });
    } catch {}
  }, []);

  const centerOnSpotlight = useCallback((container: HTMLElement, pageEl: HTMLElement, smooth: boolean) => {
    try {
      const ring = pageEl.querySelector('.spotlight-ring') as HTMLElement | null;
      if (!ring) {
        // Fallback to page top
        container.scrollTo({ top: pageEl.offsetTop - 8, behavior: (smooth ? 'smooth' : 'auto') as ScrollBehavior });
        return;
      }
      const pageRect = pageEl.getBoundingClientRect();
      const r = ring.getBoundingClientRect();
      const cy = (r.top + r.bottom) / 2 - pageRect.top;
      const desired = pageEl.offsetTop + cy - (container.clientHeight / 2);
      const maxTop = Math.max(0, container.scrollHeight - container.clientHeight);
      const topTarget = Math.max(0, Math.min(desired, maxTop));
      container.scrollTo({ top: topTarget, behavior: (smooth ? 'smooth' : 'auto') as ScrollBehavior });
    } catch {}
  }, []);

  return { placeSpotlightRing, scrollToTargetPage, centerOnSpotlight };
}


