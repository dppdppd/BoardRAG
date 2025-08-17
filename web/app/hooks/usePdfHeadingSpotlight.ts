"use client";

import { useCallback } from "react";

export function usePdfHeadingSpotlight() {
  const placeSpotlightRing = useCallback((_pageEl: HTMLElement, _targetSection: string, _attempts: number = 0) => {
    // Spotlight disabled (no-op)
    return;
  }, []);

  const placeSpotlightAtSubstring = useCallback((_pageEl: HTMLElement, _substring: string, _attempts: number = 0) => {
    // Spotlight disabled (no-op)
    return;
  }, []);

  const scrollToTargetPage = useCallback((container: HTMLElement, pageEl: HTMLElement, smooth: boolean) => {
    try {
      container.scrollTo({ top: pageEl.offsetTop - 8, behavior: (smooth ? 'smooth' : 'auto') as ScrollBehavior });
    } catch {}
  }, []);

  const centerOnSpotlight = useCallback((_container: HTMLElement, _pageEl: HTMLElement, _smooth: boolean) => {
    // Spotlight disabled (no-op)
    return;
  }, []);

  return { placeSpotlightRing, placeSpotlightAtSubstring, scrollToTargetPage, centerOnSpotlight };
}


