"use client";

import React, { useEffect, useRef, useState } from "react";
import { isLocalhost } from "../../lib/config";
import { Document, Page } from "react-pdf";

type Chunk = { 
  text: string; 
  source?: string; 
  page?: number; 
  rects_norm?: any; 
  text_spans?: Array<{[key: string]: number}>; 
  header_anchor_bbox_pct?: any;
};

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
  anchorNonce?: number; // increments whenever a new anchor action happens
  anchorRectPct?: [number, number, number, number] | null; // optional fallback anchor bbox from citation
};

export default function PdfPreview({ API_BASE, token, title, chunks, pdfMeta, setPdfMeta, targetPage, adjacentPageWindow = 1, anchorNonce, anchorRectPct }: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [centerPage, setCenterPage] = useState<number | null>(null);
  const anchoringRef = useRef<boolean>(false);
  const currentTargetRef = useRef<number | null>(null);
  const [measuredWidth, setMeasuredWidth] = useState<number | null>(null);
  const anchorPhaseRef = useRef<number>(0); // 0: none, 1: on page, 2: on citation (done)

  const [renderWindowDown, setRenderWindowDown] = useState<number>(0);
  const [measuredPageHeight, setMeasuredPageHeight] = useState<number | null>(null);
  const [dynamicSpan, setDynamicSpan] = useState<number>(2);
  const [currentScrollCenter, setCurrentScrollCenter] = useState<number | null>(null);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isInitialGameLoad, setIsInitialGameLoad] = useState<boolean>(true); // Start as true for initial load
  const [previousAnchorCenter, setPreviousAnchorCenter] = useState<number | null>(null);
  const prevTargetPageRef = useRef<number | null>(null);
  // Default to US Letter aspect ratio ~ 11/8.5 ≈ 1.294 to minimize jump before we can measure
  const [pageAspectRatio, setPageAspectRatio] = useState<number>(1.294);
  // Suppress manual scroll tracking during programmatic anchor scrolls
  const programmaticScrollRef = useRef<boolean>(false);
  const programmaticTimerRef = useRef<NodeJS.Timeout | null>(null);
  const inferred = (chunks && chunks[0] && chunks[0].source ? String(chunks[0].source) : '');
  const filename = String((pdfMeta && pdfMeta.filename) ? pdfMeta.filename : inferred);
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
  // Use current scroll center for manual scrolling, fall back to citation target
  const effectiveCenter = currentScrollCenter ?? targetPageResolved;
  const pdfUrl = filename ? `${API_BASE}/pdf?filename=${encodeURIComponent(filename)}${token ? `&token=${encodeURIComponent(token)}` : ''}` : '';
  if (!pdfUrl || !filename) return <div className="muted">Missing PDF filename.</div>;
  const byPage = (p: number) => chunks.filter((c) => (Number(c.page) + 1) === Number(p));
  // Measure available width and expand render window near viewport during manual scroll
  useEffect(() => {
    const findScrollContainer = (el: HTMLElement | null): HTMLElement | null => {
      let cur: HTMLElement | null = el;
      while (cur) {
        if (cur.classList && cur.classList.contains('preview-top')) return cur;
        cur = cur.parentElement as HTMLElement | null;
      }
      return null;
    };
    const update = () => {
      const root = rootRef.current;
      if (!root) return;
      const sc = findScrollContainer(root);
      if (sc && sc.clientWidth) setMeasuredWidth(sc.clientWidth);
      if (sc && measuredPageHeight) {
        const approxPages = Math.max(2, Math.ceil(sc.clientHeight / Math.max(1, measuredPageHeight)) + 1);
        setDynamicSpan((prev) => Math.max(prev, approxPages));
      }
    };
    const onScroll = () => {
      const root = rootRef.current; if (!root) return;
      const sc = findScrollContainer(root); if (!sc) return;
      
      // Update dynamic span based on viewport
      if (measuredPageHeight) {
        const approxPages = Math.max(2, Math.ceil(sc.clientHeight / Math.max(1, measuredPageHeight)) + 1);
        setDynamicSpan((prev) => Math.max(prev, approxPages));
      }
      
      // Track which page is currently in the center of the viewport during manual scrolling,
      // but ignore events triggered by our own programmatic anchor scrolls
      if (totalPages && !programmaticScrollRef.current) {
        // Throttle scroll center updates for better performance
        if (scrollTimeoutRef.current) {
          clearTimeout(scrollTimeoutRef.current);
        }
        
        scrollTimeoutRef.current = setTimeout(() => {
          const scrollTop = sc.scrollTop;
          const viewportCenter = scrollTop + (sc.clientHeight / 2);
          
          // Optimize: only consider pages actually rendered in the DOM
          const pageEls = Array.from(root.querySelectorAll('.pdf-page')) as HTMLElement[];
          let bestPage = 1;
          let bestDistance = Infinity;
          for (const el of pageEls) {
            const pageAttr = el.getAttribute('data-page-number');
            const pageNum = pageAttr ? Number(pageAttr) : NaN;
            if (!Number.isFinite(pageNum)) continue;
            const pageHeight = el.offsetHeight || measuredPageHeight || 800;
            let pageTopWithin = 0;
            try {
              pageTopWithin = offsetTopWithin(el, sc);
            } catch {
              pageTopWithin = el.offsetTop;
            }
            const pageCenter = pageTopWithin + (pageHeight / 2);
            const distance = Math.abs(pageCenter - viewportCenter);
            if (distance < bestDistance) {
              bestDistance = distance;
              bestPage = pageNum;
            }
          }
          setCurrentScrollCenter(bestPage);
        }, 100); // 100ms throttle
      }
    };
    update();
    window.addEventListener('resize', update);
    document.addEventListener('scroll', onScroll, true);
    return () => { 
      window.removeEventListener('resize', update); 
      document.removeEventListener('scroll', onScroll, true);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
        scrollTimeoutRef.current = null;
      }
      if (programmaticTimerRef.current) {
        clearTimeout(programmaticTimerRef.current);
        programmaticTimerRef.current = null;
      }
    };
  }, [measuredPageHeight]);
  const containerWidth = measuredWidth ?? 738; // stable default to reduce reflow
  // Back to full list of pages; we will render placeholders for non-window pages
  const allPages: number[] = (typeof totalPages === 'number' && totalPages > 0)
    ? Array.from({ length: totalPages }, (_v, i) => i + 1)
    : [Number(targetPageResolved) || 1];
  const pageContainerHeight = measuredPageHeight ?? Math.round((containerWidth || 738) * pageAspectRatio);
  // Inset used when interpreting normalized bbox percentages
  const CAL_MARGIN_PCT_V = 2.0; // top/bottom inset as % of rendered page height
  const CAL_MARGIN_PCT_H = 2.0; // left/right inset as % of rendered page width

  const mapCalibratedToCanvas = (
    w: number,
    h: number,
    canvasLeft: number,
    canvasTop: number,
    x: number,
    y: number,
    bw: number,
    bh: number,
  ) => {
    const availableW = Math.max(1, w * (1 - 2 * (CAL_MARGIN_PCT_H / 100)));
    const availableH = Math.max(1, h * (1 - 2 * (CAL_MARGIN_PCT_V / 100)));
    const originLeft = canvasLeft + (CAL_MARGIN_PCT_H / 100) * w;
    const originTop = canvasTop + (CAL_MARGIN_PCT_V / 100) * h;
    const left = originLeft + (Math.max(0, y) / 100) * availableW; // y is horizontal
    const top = originTop + (Math.max(0, x) / 100) * availableH; // x is vertical
    const width = Math.max(1, (Math.max(0, bw) / 100) * availableW);
    const height = Math.max(1, (Math.max(0, bh) / 100) * availableH);
    return { left, top, width, height } as const;
  };
  const shouldRenderReal = (p: number): boolean => {
    if (typeof effectiveCenter === 'number' && typeof totalPages === 'number') {
      const isAnchoring = anchoringRef.current === true;
      const isManualScrolling = currentScrollCenter !== null; // allow manual rendering even during anchoring
      
      if (isManualScrolling) {
        // During manual scrolling, use a generous window to ensure smooth reading
        const scrollWindow = Math.max(dynamicSpan, 3); // At least 3 pages in each direction
        return p >= Math.max(1, effectiveCenter - scrollWindow) && p <= Math.min(totalPages, effectiveCenter + scrollWindow);
      } else {
        // During citation anchoring, render both previous location and destination to prevent jarring transitions
        const lower = adjacentPageWindow;
        const upper = Math.max(adjacentPageWindow, renderWindowDown);
        
        // Check if page is in destination area
        const inDestination = p >= Math.max(1, effectiveCenter - lower) && p <= Math.min(totalPages, effectiveCenter + upper);
        
        // Check if page is in previous area (during transition)
        let inPrevious = false;
        if (isAnchoring && previousAnchorCenter !== null) {
          const prevWindow = Math.max(dynamicSpan, 3); // Use a generous window for previous area
          inPrevious = p >= Math.max(1, previousAnchorCenter - prevWindow) && p <= Math.min(totalPages, previousAnchorCenter + prevWindow);
        }
        
        return inDestination || inPrevious;
      }
    }
    return p === (Number(targetPageResolved) || 1);
  };

  // Disable text layer entirely to avoid AbortException spam and heavy rAF work

  // No center tracking; keep render window centered on the requested target to avoid bounce

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

  // Ensure we scroll to the target page once it's available (works for both desktop and mobile modal)
  // Helper utilities
  const getScrollableAncestor = (node: HTMLElement | null): HTMLElement | null => {
    let cur: HTMLElement | null = node?.parentElement || null;
    while (cur) {
      try { if (cur.scrollHeight > cur.clientHeight + 1) return cur; } catch {}
      cur = cur.parentElement as HTMLElement | null;
    }
    return null;
  };
  const offsetTopWithin = (node: HTMLElement, ancestor: HTMLElement): number => {
    let y = 0; let cur: HTMLElement | null = node;
    while (cur && cur !== ancestor) { y += cur.offsetTop || 0; cur = cur.offsetParent as HTMLElement | null; }
    return y;
  };
  const getTargetPageEl = (pageNum: number): HTMLElement | null => {
    return (document?.querySelector?.(`.modal-preview .pdf-page[data-page-number='${pageNum}']`) as HTMLElement)
      || (document?.querySelector?.(`.preview-panel .pdf-page[data-page-number='${pageNum}']`) as HTMLElement)
      || null;
  };
  // No fallback-based citation detection. Anchoring relies solely on chunk-provided coordinates.

  const debugLog = (...args: any[]) => { try { if (typeof window !== 'undefined') console.debug('[PdfPreview]', ...args); } catch {} };

  const performAnchorOnce = (retryCount = 0) => {
    debugLog('performAnchorOnce enter', { retryCount, targetPageResolved, anchorPhase: anchorPhaseRef.current, isInitialGameLoad });
    if (anchoringRef.current && retryCount === 0) {
      debugLog('performAnchorOnce: already anchoring; bail');
      return false; // Prevent multiple simultaneous anchoring attempts
    }
    
    const target = Number(targetPageResolved);

    if (!(target > 0)) return false;
    
    if (retryCount === 0) anchoringRef.current = true; // Mark as anchoring
    const pageEl = getTargetPageEl(target);
    if (!pageEl) {
      debugLog('performAnchorOnce: pageEl not ready');
      return false;
    }

    const sc = getScrollableAncestor(pageEl) || (document?.querySelector?.('.modal-preview .preview-top') as HTMLElement) || (document?.querySelector?.('.preview-panel .preview-top') as HTMLElement) || null;
    if (!sc) return false;
    const pageTop = offsetTopWithin(pageEl, sc);
    const viewTop = sc.scrollTop;
    
    // Check for debug bbox first
    let rect: { top: number; left: number; width: number; height: number } | null = null;
    try {
      const debugBboxRaw = sessionStorage.getItem('debug_bbox');
      if (debugBboxRaw) {
        const debugBbox = JSON.parse(debugBboxRaw);
        if (debugBbox.active && debugBbox.page === target) {
          const canvasEl = pageEl.querySelector('canvas') as HTMLCanvasElement;
          const pageBox = pageEl.getBoundingClientRect();
          const canvasBox = canvasEl?.getBoundingClientRect();
          
          const w = canvasEl?.clientWidth || pageEl.clientWidth || pageBox.width;
          const h = canvasEl?.clientHeight || pageEl.clientHeight || pageBox.height;
          const canvasLeft = canvasBox ? (canvasBox.left - pageBox.left) : 0;
          const canvasTop = canvasBox ? (canvasBox.top - pageBox.top) : 0;
          const mapped = mapCalibratedToCanvas(w, h, canvasLeft, canvasTop, debugBbox.x, debugBbox.y, debugBbox.w, debugBbox.h);
          rect = { top: mapped.top, left: mapped.left, width: mapped.width, height: mapped.height };
          console.log(`[DEBUG BBOX] Applied debug bbox on page ${target}:`, { 
            debugBbox, 
            mapped,
            canvasSize: { w, h },
            canvasOffset: { canvasLeft, canvasTop }
          });
        }
      }
    } catch (error) {
      console.error('[DEBUG BBOX] Error processing debug bbox:', error);
    }

    // If no debug bbox, prefer header anchor from chunk metadata
    if (!rect) {
      try {
      const chunksForPage = byPage(target);
      console.log(`[calculateBboxAnchor] page=${target}, chunks:`, chunksForPage.length);
      
      for (const c of chunksForPage) {
        const targetSection = (c as any).section;
        console.log(`[calculateBboxAnchor] looking for section:`, targetSection);
        
        const anchorsRaw: any = (c as any).header_anchor_bbox_pct || (c as any).header_anchors_pct || (c as any).header_anchors || (c as any).anchors;
        let anchors: Record<string, number[]> | null = null;
        try { anchors = typeof anchorsRaw === 'string' ? JSON.parse(anchorsRaw) : anchorsRaw; } catch { anchors = null; }
        console.log(`[calculateBboxAnchor] found anchors data:`, anchors);
        
        if (anchors && typeof anchors === 'object') {
          const names = Object.keys(anchors);
          console.log(`[calculateBboxAnchor] anchor keys:`, names);
          
          // Try to find anchor for the specific section first
          let selectedAnchor = null;
          let selectedKey = null;
          
          if (targetSection && anchors[targetSection]) {
            selectedAnchor = anchors[targetSection];
            selectedKey = targetSection;
            console.log(`[calculateBboxAnchor] using specific section anchor:`, selectedKey);
          } else if (names.length > 0) {
            selectedAnchor = anchors[names[0]];
            selectedKey = names[0];
            console.log(`[calculateBboxAnchor] using first available anchor:`, selectedKey);
          }
          
          if (Array.isArray(selectedAnchor) && selectedAnchor.length >= 4) {
            const canvasEl = pageEl.querySelector('canvas') as HTMLCanvasElement;
            const pageBox = pageEl.getBoundingClientRect();
            const canvasBox = canvasEl?.getBoundingClientRect();
            
            // Use canvas dimensions (actual PDF content) instead of page container
            const w = canvasEl?.clientWidth || pageEl.clientWidth || pageBox.width;
            const h = canvasEl?.clientHeight || pageEl.clientHeight || pageBox.height;
            
            // Calculate position relative to canvas, not page container
            const canvasLeft = canvasBox ? (canvasBox.left - pageBox.left) : 0;
            const canvasTop = canvasBox ? (canvasBox.top - pageBox.top) : 0;
            console.log(`[calculateBboxAnchor] element dimensions:`, {
              pageElDims: { w: pageEl.clientWidth, h: pageEl.clientHeight },
              canvasDims: { w: canvasEl?.clientWidth, h: canvasEl?.clientHeight },
              boundingBox: { w: pageBox.width, h: pageBox.height },
              finalUsed: { w, h }
            });
            const x = Number(selectedAnchor[0]) || 0; 
            const y = Number(selectedAnchor[1]) || 0; 
            const bw = Number(selectedAnchor[2]) || 0; 
            const bh = Number(selectedAnchor[3]) || 0;
            const mapped = mapCalibratedToCanvas(w, h, canvasLeft, canvasTop, x, y, bw, bh);
            rect = { top: mapped.top, left: mapped.left, width: mapped.width, height: mapped.height } as any;
            console.log(`[calculateBboxAnchor] SUCCESS: Found bbox anchor`, { 
              selectedKey, x, y, bw, bh, 
              mapped,
              canvasSize: { w, h },
              pageRect: pageBox,
              calculations: {
                insetH: `${CAL_MARGIN_PCT_H}%`,
                insetV: `${CAL_MARGIN_PCT_V}%`
              },
              canvasOffset: { canvasLeft, canvasTop }
            });
            debugLog('performAnchorOnce: using header_anchors_pct', { selectedKey, mapped });
            break;
          }
        }
      }
    } catch (e) {
      console.error('[calculateBboxAnchor] error:', e);
    }
    } // Close the if (!rect) block
    if (!rect) {
      debugLog('performAnchorOnce: no anchor rect found in chunk data; skipping phase 1 centering');
    }
    
    // Determine scroll behavior: instant for game switches, smooth for regular citations
    const scrollBehavior = isInitialGameLoad ? 'auto' : 'smooth';
    const phase = Number(anchorPhaseRef.current || 0);
    console.log(`[performAnchorOnce] phase begin: ${phase}, hasRect: ${!!rect}, pageTop: ${pageTop}, viewTop: ${viewTop}`);
    debugLog('performAnchorOnce: phase begin', { phase, hasRect: !!rect, pageTop, viewTop });

    // Phase 0: if we have a bbox, scroll it near the top (~20%) and add a padded, fading highlight
    if (phase === 0) {
      let topTo = Math.max(0, pageTop);
      const VIEW_TOP_OFFSET = Math.round(sc.clientHeight * 0.2);
      if (rect) {
        topTo = Math.max(0, pageTop + rect.top - VIEW_TOP_OFFSET);
      }
      programmaticScrollRef.current = true;
      if (programmaticTimerRef.current) { clearTimeout(programmaticTimerRef.current); programmaticTimerRef.current = null; }
      sc.scrollTo({ top: topTo, behavior: scrollBehavior });
      const delay = scrollBehavior === 'auto' ? 0 : 300;
      programmaticTimerRef.current = setTimeout(() => { programmaticScrollRef.current = false; }, delay);
      // Add highlight overlay if bbox present; strong then fade over 5s
      if (rect) {
        try {
          const existing = pageEl.querySelectorAll('.bbox-highlight, .bbox-highlight-label');
          existing.forEach(el => el.remove());
          const pad = Math.max(8, Math.round(rect.height * 0.15));
          const canvasEl = pageEl.querySelector('canvas') as HTMLCanvasElement;
          const pageBox = pageEl.getBoundingClientRect();
          const canvasW = canvasEl?.clientWidth || pageEl.clientWidth || pageBox.width;
          const canvasH = canvasEl?.clientHeight || pageEl.clientHeight || pageBox.height;
          const padded = {
            left: Math.max(0, rect.left - pad),
            top: Math.max(0, rect.top - pad),
            width: Math.min(canvasW, rect.width + pad * 2),
            height: Math.min(canvasH, rect.height + pad * 2),
          };
          const highlightDiv = document.createElement('div');
          highlightDiv.className = 'bbox-highlight';
          highlightDiv.style.position = 'absolute';
          highlightDiv.style.left = `${padded.left}px`;
          highlightDiv.style.top = `${padded.top}px`;
          highlightDiv.style.width = `${padded.width}px`;
          highlightDiv.style.height = `${padded.height}px`;
          highlightDiv.style.backgroundColor = 'rgba(255, 255, 0, 0.5)';
          highlightDiv.style.boxShadow = '0 0 0 2px rgba(255, 200, 0, 0.95)';
          highlightDiv.style.transition = 'opacity 5s ease-out';
          highlightDiv.style.opacity = '1';
          highlightDiv.style.pointerEvents = 'none';
          highlightDiv.style.zIndex = '1000';
          pageEl.appendChild(highlightDiv);
          requestAnimationFrame(() => { try { highlightDiv.style.opacity = '0'; } catch {} });
          setTimeout(() => { try { highlightDiv.remove(); } catch {} }, 5200);
        } catch {}
      }
      anchorPhaseRef.current = 2; // ready; next click restarts via nonce
      console.log(`[performAnchorOnce] phase 0 → scroll to ${rect ? 'bbox near top' : 'page top'}; next phase=2, scrolled to: ${topTo}`);
      debugLog('performAnchorOnce: phase 0 scroll complete');
      if (retryCount === 0) {
        anchoringRef.current = false;
        setIsInitialGameLoad(false);
        setTimeout(() => setPreviousAnchorCenter(null), 500);
      }
      return true;
    }

    // Phase 1 unused – positioning and highlight handled in phase 0

    // Phase 2: check if we're still near the anchored position
    if (phase === 2 && rect) {
      const rectCenterY = pageTop + rect.top + rect.height / 2;
      const expectedScrollTop = Math.max(0, rectCenterY - sc.clientHeight / 2);
      const currentScrollTop = sc.scrollTop;
      const scrollTolerance = 50; // pixels
      
      if (Math.abs(currentScrollTop - expectedScrollTop) > scrollTolerance) {
        // User scrolled away, reset to phase 0
        anchorPhaseRef.current = 0;
        console.log(`[performAnchorOnce] phase 2 → user scrolled away (${Math.abs(currentScrollTop - expectedScrollTop)}px), reset to phase 0`);
        // Recursively call with phase 0
        return performAnchorOnce(retryCount);
      } else {
        // Still near anchored position, stay at phase 2
        console.log(`[performAnchorOnce] phase 2 → staying centered on bbox, no action needed`);
        debugLog('performAnchorOnce: phase 2 → stay centered on bbox');
        if (retryCount === 0) {
          anchoringRef.current = false;
          setIsInitialGameLoad(false);
        }
        return true;
      }
    }
    
    // Phase 2 without rect or other phases: provide a gentle bump and reset
    const currentTop = sc.scrollTop;
    const maxTop = Math.max(0, sc.scrollHeight - sc.clientHeight);
    const bumpUp = Math.max(0, Math.min(maxTop, currentTop - 8));
    programmaticScrollRef.current = true;
    if (programmaticTimerRef.current) { clearTimeout(programmaticTimerRef.current); programmaticTimerRef.current = null; }
    sc.scrollTo({ top: bumpUp, behavior: 'smooth' });
    setTimeout(() => {
      const settleTop = Math.max(0, currentTop);
      sc.scrollTo({ top: settleTop, behavior: 'smooth' });
      programmaticTimerRef.current = setTimeout(() => { programmaticScrollRef.current = false; }, 350);
    }, 120);
    anchorPhaseRef.current = 0; // reset cycle for next citation
    console.log(`[performAnchorOnce] phase ${phase} → gentle bump, reset to phase 0`);
    debugLog('performAnchorOnce: phase 2+ → gentle bump, reset to phase 0');
    if (retryCount === 0) {
      anchoringRef.current = false;
      setIsInitialGameLoad(false);
      setTimeout(() => setPreviousAnchorCenter(null), 500);
    }
    return true;
  };



  // Track when filename changes to detect game switches
  const prevFilenameRef = useRef<string>(filename);
  useEffect(() => {
    if (prevFilenameRef.current !== filename && prevFilenameRef.current !== '' && filename !== '') {
      // Filename changed - this is likely a game switch, use instant scrolling
      setIsInitialGameLoad(true);
    }
    prevFilenameRef.current = filename;
  }, [filename]);

  // Track anchor nonce changes to detect citation clicks (vs filename changes)
  const prevAnchorNonceRef = useRef<number>(anchorNonce || 0);
  useEffect(() => {
    const currentNonce = anchorNonce || 0;
    if (prevAnchorNonceRef.current !== currentNonce && prevAnchorNonceRef.current !== 0) {
      // Citation click: ensure we re-run anchoring on this page
      setIsInitialGameLoad(false);
      anchorPhaseRef.current = 0; // force phase 0 so onRenderSuccess will trigger
      anchoringRef.current = false; // allow performAnchorOnce to proceed
    }
    prevAnchorNonceRef.current = currentNonce;
  }, [anchorNonce]);

  // Reset and attempt anchoring on change
  useEffect(() => {
    anchoringRef.current = false; // Reset to allow new anchoring
    currentTargetRef.current = Number(targetPageResolved) || null;
    setRenderWindowDown(0);
    
    // Capture current scroll center before anchoring (for smooth transitions)
    // Only do this for citation clicks, not game switches
    if (!isInitialGameLoad) {
      if (currentScrollCenter !== null) {
        setPreviousAnchorCenter(currentScrollCenter);
      } else if (prevTargetPageRef.current !== null) {
        // Use the previous target page if we don't have a current scroll center
        setPreviousAnchorCenter(prevTargetPageRef.current);
      } else if (typeof targetPageResolved === 'number') {
        // Final fallback to current target
        setPreviousAnchorCenter(targetPageResolved);
      }
    } else {
      // For game switches, clear any previous anchor center
      setPreviousAnchorCenter(null);
    }
    
    // Update the previous target reference
    prevTargetPageRef.current = Number(targetPageResolved) || null;
    
    // Reset scroll center tracking to let citation take control
    setCurrentScrollCenter(null);
    // Clear any pending scroll center updates
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
      scrollTimeoutRef.current = null;
    }
    // If total pages are known, attempt immediately; otherwise wait for render callback
    if (typeof totalPages === 'number' && isFinite(totalPages) && totalPages > 0) {
      // Only reset phase if we're switching to a different page
      if (prevTargetPageRef.current !== Number(targetPageResolved)) {
        anchorPhaseRef.current = 0;
        console.log(`[useEffect] Switching pages, reset phase to 0`);
      } else {
        console.log(`[useEffect] Same page, keeping current phase: ${anchorPhaseRef.current}`);
      }
      const did = performAnchorOnce();
      // Expand downward only to avoid shifting target content
      setTimeout(() => { setRenderWindowDown(adjacentPageWindow); }, 120);
    }
  }, [anchorNonce, filename, targetPageResolved, totalPages]);

  // Listen for debug bbox updates and refresh highlights
  useEffect(() => {
    const handleDebugBboxUpdate = () => {
      console.log('[PdfPreview] Debug bbox update triggered');
      // Re-trigger anchoring to apply/remove debug bbox highlights
      if (targetPageResolved) {
        anchoringRef.current = false;
        anchorPhaseRef.current = 0;
        const success = performAnchorOnce();
        console.log('[PdfPreview] Debug bbox update result:', success);
      }
    };

    window.addEventListener('debugBboxUpdate', handleDebugBboxUpdate);
    return () => {
      window.removeEventListener('debugBboxUpdate', handleDebugBboxUpdate);
    };
  }, [targetPageResolved]);

  return (
    <div ref={rootRef} className="pdf-preview-root">
    <Document 
      key={`preview-doc-${filename || 'empty'}`} 
      file={pdfUrl} 
      onLoadSuccess={(info: { numPages: number }) => setPdfMeta({ filename, pages: info.numPages })} 
      loading={null}
    >
      {allPages.map((p) => {
        const renderReal = shouldRenderReal(p);
        const renderText = false;
        if (!renderReal) {
          return (
            <div key={p} className="pdf-page placeholder" data-page-number={p} data-target-section={title} style={{ height: pageContainerHeight }}>
              <div className="muted" style={{ fontSize: 12 }}>{`Page ${p}`}</div>
            </div>
          );
        }
        return (
          <div key={p} className="pdf-page" data-page-number={p} data-target-section={title} style={{ minHeight: pageContainerHeight }}>
            <Page
              pageNumber={p}
              width={containerWidth}
              renderTextLayer={renderText}
              renderAnnotationLayer={false}
              onRenderSuccess={() => {
                try {
                  debugLog('onRenderSuccess', { page: p, renderText });
                  const container =
                    (document?.querySelector?.(`.modal-preview .pdf-page[data-page-number='${p}']`) as HTMLElement) ||
                    (document?.querySelector?.(`.preview-panel .pdf-page[data-page-number='${p}']`) as HTMLElement) ||
                    undefined;
                  const el =
                    container ||
                    (document?.querySelector?.('.modal-preview .pdf-page:last-child') as HTMLElement) ||
                    (document?.querySelector?.('.preview-panel .pdf-page:last-child') as HTMLElement);
                  if (!el) return;
                  // Text-layer based highlighting disabled with text layer off
                  const chunksForPage = byPage(p);
                  // Draw rectangle overlays from normalized rects
                  try {
                    const pageViewport = el.getBoundingClientRect();
                    // Capture a reasonable height to use for placeholders
                    if (!measuredPageHeight && pageViewport.height) {
                      const h = Math.round(pageViewport.height);
                      setMeasuredPageHeight(h);
                      const w = pageViewport.width || containerWidth || 738;
                      if (w > 0) setPageAspectRatio(h / w);
                    }
                    const existing = el.querySelectorAll('.rect-overlay'); existing.forEach((n) => n.remove());
                    const rects: Array<[number, number, number, number]> = [];
                    chunksForPage.forEach((c: Chunk) => {
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
                  // If this is the target page, run anchor logic for phase 0 only
                  const target = currentTargetRef.current;
                  if (typeof target === 'number' && target === p) {
                    if ((anchorPhaseRef.current || 0) === 0) {
                      setTimeout(() => performAnchorOnce(), 60);
                    }
                  }
                } catch {}
              }}
            />
            <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>{filename}</div>
          </div>
        );
      })}
    </Document>
    </div>
  );
}


