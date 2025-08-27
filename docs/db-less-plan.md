## BoardRAG – DB‑less Architecture Plan (Claude Sonnet 4)

### Overview
Move from a local vector DB to a fully DB‑less flow powered by Anthropic Claude Sonnet 4 with PDF Files API and citations. The app uploads PDFs once, keeps a local catalog mapping each PDF to its `file_id` and commercial game name, and answers queries directly against the original PDFs. Answers include citation pages and spotlight coordinates for referenced headers.

### Goals
- Replace ChromaDB retrieval with direct LLM calls using the PDFs as context.
- Maintain a lightweight local catalog of available PDFs and their Anthropic `file_id`s.
- Support multi‑PDF games (e.g., base + expansions) without a DB.
- Return precise spotlight anchors for cited headers using `(x, y)` page coordinates.
- Keep detailed logs for observability; make behavior deterministic and robust.

### Key Components and Files
- `api/main.py`
  - On startup, scans `data/` and populates the catalog (uploads any missing PDFs to Anthropic Files API).
  - Logs catalog status to the admin log stream.
- `src/catalog.py`
  - Catalog file: `data/catalog/games_catalog.json` with entries: `{ game_name, pages, size_bytes, updated_at }` keyed by PDF filename.
  - `ensure_catalog_up_to_date(log)` scans `data/`, records metadata and `game_name` (defaults to filename stem), and persists the catalog. Main PDF uploads and top-level `file_id` are not used.
- `src/query.py`
  - DB‑less branch uses the catalog’s `(path, file_id)` pairs to call Anthropic Messages API with citations enabled.
  - Asks the model to return `{ answer, spans: [{page, header}] }` per file; merges multi‑PDF results.
  - Computes spotlight anchors via `find_header_anchor(pdf_path, page, header)` and returns `(x, y)`.
- `src/pdf_utils.py`
  - `find_header_anchor()` locates a header’s starting point `(x, y)` on a page for spotlighting.
- `src/llm_outline_helpers.py`
  - `upload_pdf_to_anthropic_files()` handles Files API upload and returns `file_id`.
  - `anthropic_pdf_messages_with_file()` sends Messages API requests referencing `file_id` with citations enabled.

### Startup Behavior
1. Server starts (`uvicorn` → `api/main:app`).
2. `api/main.py` startup hook:
   - Prints routes for debugging.
   - Calls `ensure_catalog_up_to_date()`:
     - Scans `data/*.pdf`.
     - For PDFs not in catalog, uploads to Anthropic Files API → `file_id`.
     - Queries Sonnet 4 once per new file to extract `{ "game_name": "…" }` and stores it.
     - Persists `.cache/games_catalog.json`.
   - Emits admin log lines for progress and completion.

### Catalog Format
Stored at `.cache/games_catalog.json`:
```json
{
  "Up Front Rulebook bw.pdf": {
    "file_id": "file_abc123",
    "game_name": "Up Front",
    "pages": null,
    "size_bytes": 1234567,
    "updated_at": "2025-08-17T12:34:56Z"
  }
}
```

### Query Flow (DB‑less)
1. Client calls `/stream` or `/stream-ndjson` with `q` and optional `game`.
2. Server resolves the relevant PDFs directly from the catalog entries by `game_name` or filename prefix.
   - If `game` is provided: select only catalog entries matching the game name (case-insensitive) or filename prefix.
   - If absent: include all cataloged PDFs.
   - Cap included PDFs to a small number (default 2) to respect input size/time.
3. For each selected PDF’s pages (page-level file_ids are managed elsewhere):
   - Build a single request to Anthropic Messages API (citations enabled) including:
     - `system`: role and constraints.
     - `user`: instruction to answer based only on the attached PDF and to return JSON with `answer` and `spans`.
     - `document` block referencing the `file_id`.
4. Parse the last JSON object from the LLM output. Aggregate across PDFs:
   - First non‑empty `answer` wins; concatenate unique `spans`.
   - For each span `{page, header}`, compute `(x, y)` via `find_header_anchor(path, page, header)`.
5. Return streaming tokens (for `/stream`) and a final metadata payload that includes `spans` with `{ file, page, header, x, y }` for UI spotlighting.

### LLM Response Contract for DB‑less
Model is instructed to return strictly:
```json
{
  "answer": "…",
  "spans": [
    { "page": 12, "header": "3.4 Movement" },
    { "page": 13, "header": "3.5 Combat: Ranged" }
  ]
}
```
Notes:
- Page numbers are 1‑based.
- `header` must match the exact header line text used in the justification (ideally “code + title”).
- If no specific headers were used, the model returns an empty array for `spans`.

### Spotlighting/Highlighting Strategy
- We do not store rectangles. The UI expects the anchor of the header line only.
- `find_header_anchor(pdf_path, page, header)` uses PyMuPDF to get the bounding box for the first matching line and returns the start `(x, y)` coordinate.
- The API response includes `spans: [{ file, page, header, x, y }]` which the UI uses to spotlight.

### Multi‑PDF Games
- The catalog can contain multiple PDFs for one game (base, expansions, scenarios). The query flow:
  - Selects up to N PDFs (default 2) that match the game name.
  - Calls the model once per file, merges answers and spans.
  - Optionally, future optimization can attach multiple `document` blocks in a single request if input limits allow.

### Environment Variables
- `ANTHROPIC_API_KEY` (required): Auth for Files/Messages API.
- `OUTLINE_LLM_MODEL` (default `claude-sonnet-4-20250514`): Model id for PDF tasks.
- `DATA_PATH` (default `data`): Directory scanned on startup for PDFs.
- `STREAM_ECHO_STDOUT` (default `1`): Echo stream tokens to stdout for local debugging.

### Operational Notes
- The API will always attempt a catalog scan on startup and log progress to the admin log stream.
- If `ANTHROPIC_API_KEY` is missing, uploads are skipped and a warning is logged; queries will fail if no `file_id`s exist.
- The `/games` route reads available games from the DB (legacy) but will be naturally superseded by the catalog usage on the query path. A later cleanup can shift `/games` to list from `catalog.py` (`list_games_from_catalog()`).
- Legacy DB endpoints remain for compatibility but are not used in DB‑less mode.

### Error Handling and Robustness
- Startup cataloging logs per‑file successes/failures; failures do not abort startup.
- JSON parsing from LLM responses is brace‑balanced (grab the last valid JSON object) to avoid partial parse errors.
- PDF text extraction and header anchoring degrade gracefully; spans without anchors omit `(x, y)`.
- Requests timeouts are set conservatively to avoid hangs.

### Testing and Telemetry
- Keep prior validation tools for outline/enrichment as optional, but they’re not required for runtime since we’re DB‑less.
- Admin log stream exposes server‑side progress and errors.
- For local dev, enable stdout echoes for quick inspection of streaming.

### Future Enhancements
- Batch multiple `file_id` documents in a single Messages API call when the combined input fits context limits.
- Expand `list_games` to read from the catalog directly and include multi‑PDF grouping.
- Persist page counts and cover thumbnails (optional) in the catalog for a richer UI.
- Add retry/backoff for Files API uploads and name extraction calls.

### Quick Start
1. Place PDFs in `data/`.
2. Set `ANTHROPIC_API_KEY` in the environment.
3. Launch the API: `launch_api.cmd` or `launch_local.cmd`.
4. On startup, the server scans PDFs and persists metadata and `game_name`. It does not upload main PDFs or cache top-level `file_id`s; only per-page IDs are used during extraction.
5. Ask a question via `/stream` with an optional `game` filter. The response includes spotlight anchors.


