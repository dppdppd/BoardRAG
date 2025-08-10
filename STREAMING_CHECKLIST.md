## Streaming (SSE) Troubleshooting Checklist

Use this list to systematically restore progressive streaming in production (Vercel → Railway). Mark each item as you verify/complete it.

### Preflight
- [ ] Point frontend directly at Railway via `NEXT_PUBLIC_API_BASE` (no Next.js API proxy)
- [ ] Verify local streaming in DevTools (`/stream` Response updates progressively)
- [ ] Verify deployed streaming in DevTools (not a single buffered payload)

### Backend (Railway) – things we can try now
- [ ] Increase initial SSE padding to 16KB and keep 1s heartbeats
- [ ] Split token writes into smaller chunks (flush every ~50–100 chars)
- [ ] Add a second endpoint `/stream-ndjson` that emits one JSON per line (NDJSON)
- [ ] Add server toggle to choose SSE vs NDJSON via query param (e.g., `mode=ndjson`)

### Frontend – things we can try now
- [ ] Force fetch streaming (ReadableStream) for chat in production (skip EventSource)
- [ ] Set fetch fallback delay to 0ms on production hostname
- [ ] Add NDJSON client parser (line-based) and switch when `mode=ndjson`
- [ ] Add explicit `cache: "no-store"` and `Accept: text/event-stream` to fetch requests
- [ ] Auto-reconnect/backoff if the stream drops mid-answer

### Vercel config – things we can try now
- [ ] Add `vercel.json` rewrites so `/stream` and `/stream-ndjson` hit Railway directly
- [ ] Add `vercel.json` headers for these paths:
  - `Cache-Control: no-cache, no-transform, private`
  - `Content-Encoding: identity`
  - `Connection: keep-alive`
  - (optional) `Accept-Encoding: identity`

### Diagnostics – things we can do now
- [ ] `curl -N` directly against Railway `/stream` (verify progressive output)
- [ ] `curl -N` against Vercel `/stream` after rewrites (verify progressive output)
- [ ] Capture response headers for `/stream` on Vercel (ensure `no-transform`, `identity`)

```json
{
  "rewrites": [
    { "source": "/stream", "destination": "https://YOUR-railway-app.up.railway.app/stream" },
    { "source": "/stream-ndjson", "destination": "https://YOUR-railway-app.up.railway.app/stream-ndjson" }
  ],
  "headers": [
    {
      "source": "/(stream|stream-ndjson)",
      "headers": [
        { "key": "Cache-Control", "value": "no-cache, no-transform, private" },
        { "key": "Content-Encoding", "value": "identity" },
        { "key": "X-Accel-Buffering", "value": "no" },
        { "key": "Connection", "value": "keep-alive" }
      ]
    }
  ]
}
```

### Diagnostics
- [ ] DevTools → Network → `/stream` Response grows over time; heartbeats appear (`: ping`)
- [ ] `curl -N https://YOUR-railway-app.up.railway.app/stream?…` prints tokens progressively
- [ ] If Vercel still buffers, `curl -N https://YOUR-vercel-site.vercel.app/stream?…` also streams (after rewrites)

### Fallbacks – things we can try now
- [ ] Default chat to fetch streaming on production if SSE remains buffered
- [ ] Implement WebSocket streaming path (frontend connects to Railway WS)
- [ ] Host frontend+backend on one origin (Railway) or place both behind Nginx with `proxy_buffering off;`

Notes
- SSE is sensitive to intermediaries (CDNs, proxies, compression). Disabling transforms and keeping writes small/frequent helps flush.
- NDJSON over chunked fetch often survives where SSE is buffered.


