export function getApiBase(): string {
  const raw = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").trim();
  const trimmed = raw.replace(/\/+$/, "");
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  // If the value lacks a protocol (common misconfig), default to https
  return `https://${trimmed.replace(/^\/+/, "")}`;
}

export const API_BASE = getApiBase();


