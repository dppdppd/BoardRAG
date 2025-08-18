export function getApiBase(): string {
  const raw = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").trim();
  const trimmed = raw.replace(/\/+$/, "");
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  // If the value lacks a protocol (common misconfig), choose http for localhost, https otherwise
  const host = trimmed.replace(/^\/+/, "");
  const isLocal = /^(localhost|127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?$/i.test(host);
  return `${isLocal ? "http" : "https"}://${host}`;
}

export const API_BASE = getApiBase();

export function isLocalhost(): boolean {
  if (typeof window === 'undefined') return false;
  const h = window.location.hostname || '';
  return h === 'localhost' || h === '127.0.0.1' || h === '0.0.0.0';
}


