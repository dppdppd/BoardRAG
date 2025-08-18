"use client";

import React, { useEffect, useRef, useState } from "react";
import { isLocalhost } from "../../lib/config";
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
  anchorNonce?: number; // increments whenever a new anchor action happens
};

export default function PdfPreview({ API_BASE, token, title, chunks, pdfMeta, setPdfMeta, targetPage, adjacentPageWindow = 1, anchorNonce }: Props) {
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
          
          // Try to find the actual page element closest to viewport center
          let bestPage = 1;
          let bestDistance = Infinity;
          
          for (let pageNum = 1; pageNum <= totalPages; pageNum++) {
            const pageEl = root.querySelector(`.pdf-page[data-page-number='${pageNum}']`) as HTMLElement;
            if (pageEl) {
              const pageHeight = pageEl.offsetHeight || measuredPageHeight || 800;
              let pageTopWithin = 0;
              try {
                pageTopWithin = offsetTopWithin(pageEl, sc);
              } catch {
                pageTopWithin = pageEl.offsetTop;
              }
              const pageCenter = pageTopWithin + (pageHeight / 2);
              const distance = Math.abs(pageCenter - viewportCenter);
              
              if (distance < bestDistance) {
                bestDistance = distance;
                bestPage = pageNum;
              }
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
  const getCitationRectInPage = (pageEl: HTMLElement): { top: number; left: number; width: number; height: number } | null => {
    
    // 1) If a heading overlay/highlight exists, use it
    const heading = pageEl.querySelector('.heading-overlay, .heading-highlight') as HTMLElement | null;
    if (heading) {

      const r = heading.getBoundingClientRect(); const pr = pageEl.getBoundingClientRect();
      return { top: r.top - pr.top, left: r.left - pr.left, width: r.width, height: r.height };
    }
    // 2) Try to resolve by explicit section number (e.g., "28.3") across logical lines
    try {
      // First try title, then fall back to chunk section data
      let sectionNum: string | null = null;
      const numMatch = (title || '').match(/^\s*(\d+(?:\.\d+)+)/);
      if (numMatch) {
        sectionNum = String(numMatch[1]).trim();
      } else if (chunks && chunks.length > 0) {
        // Try to get section number from chunk data
        const chunk = chunks[0] as any;
        const candidateSection = chunk.section_number || chunk.section || chunk.text;
        if (candidateSection && typeof candidateSection === 'string') {
          const chunkMatch = candidateSection.match(/^\s*(\d+(?:\.\d+)+)/);
          if (chunkMatch) {
            sectionNum = String(chunkMatch[1]).trim();
          } else if (/^\d+(\.\d+)+$/.test(candidateSection.trim())) {
            sectionNum = candidateSection.trim();
          }
        }
      }
      
      
      // More comprehensive span detection
      const spanNodes = Array.from(pageEl.querySelectorAll('.textLayer span, .react-pdf__Page__textContent span, .react-pdf__Page__textLayer span')) as HTMLSpanElement[];

      
      if (spanNodes.length === 0) {
        // Text layer might not be ready
        return null;
      }
      
      if (spanNodes.length > 0 && sectionNum) {
        const pr = pageEl.getBoundingClientRect();
        // Sort spans by visual position
        const sorted = spanNodes.map((el) => ({ el, r: el.getBoundingClientRect() }))
          .sort((a, b) => (a.r.top - b.r.top) || (a.r.left - b.r.left));
        // Group into lines by near-equal top (within 3px)
        const lines: { text: string; rect: { top: number; left: number; width: number; height: number } }[] = [];
        let i = 0;
        while (i < sorted.length) {
          const lineTop = sorted[i].r.top;
          let j = i;
          let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity, text = '';
          while (j < sorted.length && Math.abs(sorted[j].r.top - lineTop) < 3) {
            left = Math.min(left, sorted[j].r.left);
            right = Math.max(right, sorted[j].r.right);
            top = Math.min(top, sorted[j].r.top);
            bottom = Math.max(bottom, sorted[j].r.bottom);
            text += (sorted[j].el.textContent || '');
            j += 1;
          }
          lines.push({
            text: text.replace(/\s+/g, ' ').trim(),
            rect: { top: top - pr.top, left: left - pr.left, width: Math.max(1, right - left), height: Math.max(1, bottom - top) },
          });
          i = j;
        }
        

        
        // Find the exact span containing "4.3" 
        
        // Look for spans containing "4.3" - try multiple patterns
        let matchingSpan = null;
        
        // Pattern 1: Exact "4.3" with word boundaries
        const exactRegex = new RegExp(`\\b${sectionNum.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`);
        matchingSpan = spanNodes.find(span => {
          const text = span.textContent || '';
          return exactRegex.test(text);
        });
        
        // Pattern 2: Contains "4.3" but not "4.31", "4.32", etc.
        if (!matchingSpan) {
          const noExtensionRegex = new RegExp(`${sectionNum.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?!\\d)`);
          matchingSpan = spanNodes.find(span => {
            const text = span.textContent || '';
            return noExtensionRegex.test(text);
          });
        }
        
        // Pattern 3: Simple contains check as last resort
        if (!matchingSpan) {
          matchingSpan = spanNodes.find(span => {
            const text = span.textContent || '';
            return text.includes(sectionNum);
          });
        }
        
        if (matchingSpan) {
          const spanRect = matchingSpan.getBoundingClientRect();
          const pr = pageEl.getBoundingClientRect();
          return {
            top: spanRect.top - pr.top,
            left: spanRect.left - pr.left,
            width: Math.max(1, spanRect.width),
            height: Math.max(1, spanRect.height)
          };
        }
        

      }
    } catch (e) {
      // Silent error handling
    }
    // 4) If we can't find the section header, return null to trigger page-level scroll
    return null;
  };

  const debugLog = (...args: any[]) => { try { if (typeof window !== 'undefined' && isLocalhost()) console.debug('[PdfPreview]', ...args); } catch {} };

  const performAnchorOnce = (retryCount = 0) => {
    if (anchoringRef.current && retryCount === 0) {
      return false; // Prevent multiple simultaneous anchoring attempts
    }
    
    const target = Number(targetPageResolved);

    if (!(target > 0)) return false;
    
    if (retryCount === 0) anchoringRef.current = true; // Mark as anchoring
    const pageEl = getTargetPageEl(target);
    if (!pageEl) {
      return false;
    }

    // Check if text layer is ready FIRST - don't scroll until it's ready
    const spanNodes = Array.from(pageEl.querySelectorAll('.textLayer span, .react-pdf__Page__textContent span, .react-pdf__Page__textLayer span'));
    
    if (spanNodes.length === 0 && retryCount < 2) {
      // Text layer not ready, wait a bit and retry - don't scroll yet
      setTimeout(() => performAnchorOnce(retryCount + 1), 300);
      return false;
    }
    
    const sc = getScrollableAncestor(pageEl) || (document?.querySelector?.('.modal-preview .preview-top') as HTMLElement) || (document?.querySelector?.('.preview-panel .preview-top') as HTMLElement) || null;
    if (!sc) return false;
    const pageTop = offsetTopWithin(pageEl, sc);
    const viewTop = sc.scrollTop;
    
    const rect = getCitationRectInPage(pageEl);
    
    // Determine scroll behavior: instant for game switches, smooth for regular citations
    const scrollBehavior = isInitialGameLoad ? 'auto' : 'smooth';
    
    if (rect) {
      const rectCenterY = pageTop + rect.top + rect.height / 2;
      const currentCenterY = viewTop + sc.clientHeight / 2;
      const nearCitation = Math.abs(rectCenterY - currentCenterY) < 24;
      
      if (nearCitation && !isInitialGameLoad) {
        // Already centered on citation → provide a gentle visual acknowledgment bump (but only for regular citations)
        const currentTop = sc.scrollTop;
        const maxTop = Math.max(0, sc.scrollHeight - sc.clientHeight);
        const bumpUp = Math.max(0, Math.min(maxTop, currentTop - 8));
        // Mark as programmatic during our controlled scrolls to avoid jitter
        programmaticScrollRef.current = true;
        if (programmaticTimerRef.current) { clearTimeout(programmaticTimerRef.current); programmaticTimerRef.current = null; }
        sc.scrollTo({ top: bumpUp, behavior: 'smooth' });
        setTimeout(() => {
          const settleTop = Math.max(0, rectCenterY - sc.clientHeight / 2);
          sc.scrollTo({ top: settleTop, behavior: 'smooth' });
          // Give the browser a moment to settle, then re-enable manual tracking
          programmaticTimerRef.current = setTimeout(() => { programmaticScrollRef.current = false; }, 350);
        }, 120);
      } else {
        // Scroll to center the citation
        const top = Math.max(0, rectCenterY - sc.clientHeight / 2);
        programmaticScrollRef.current = true;
        if (programmaticTimerRef.current) { clearTimeout(programmaticTimerRef.current); programmaticTimerRef.current = null; }
        sc.scrollTo({ top, behavior: scrollBehavior });
        // For instant jumps, clear immediately; for smooth, clear after a short delay
        const delay = scrollBehavior === 'auto' ? 0 : 300;
        programmaticTimerRef.current = setTimeout(() => { programmaticScrollRef.current = false; }, delay);
      }
      if (retryCount === 0) {
        anchoringRef.current = false;
        setIsInitialGameLoad(false); // Reset after first scroll
        // Clear previous anchor center after successful scroll
        setTimeout(() => setPreviousAnchorCenter(null), 500); // Small delay to let scroll animation finish
      }
      return true;
    } else {
      // No citation found - go to page top as fallback
      const topTo = Math.max(0, pageTop);
      programmaticScrollRef.current = true;
      if (programmaticTimerRef.current) { clearTimeout(programmaticTimerRef.current); programmaticTimerRef.current = null; }
      sc.scrollTo({ top: topTo, behavior: scrollBehavior });
      const delay = scrollBehavior === 'auto' ? 0 : 300;
      programmaticTimerRef.current = setTimeout(() => { programmaticScrollRef.current = false; }, delay);
      if (retryCount === 0) {
        anchoringRef.current = false;
        setIsInitialGameLoad(false); // Reset after first scroll
        // Clear previous anchor center after successful scroll
        setTimeout(() => setPreviousAnchorCenter(null), 500); // Small delay to let scroll animation finish
      }
      return true;
    }
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
      // Anchor nonce changed - this is a citation click, use smooth scrolling
      setIsInitialGameLoad(false);
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
      anchorPhaseRef.current = 0;
      const did = performAnchorOnce();
      // Expand downward only to avoid shifting target content
      setTimeout(() => { setRenderWindowDown(adjacentPageWindow); }, 120);
    }
  }, [anchorNonce, filename, targetPageResolved, totalPages]);

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
                    // Capture a reasonable height to use for placeholders
                    if (!measuredPageHeight && pageViewport.height) {
                      const h = Math.round(pageViewport.height);
                      setMeasuredPageHeight(h);
                      const w = pageViewport.width || containerWidth || 738;
                      if (w > 0) setPageAspectRatio(h / w);
                    }
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
                  // If this is the target page, run anchor logic after text layer is ready
                  const target = currentTargetRef.current;
                  if (typeof target === 'number' && target === p) {
                    // Wait a moment for text layer to be fully ready
                    setTimeout(() => performAnchorOnce(), 100);
                  }
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


