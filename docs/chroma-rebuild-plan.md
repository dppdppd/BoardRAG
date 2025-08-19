## BoardRAG – Chroma Vector DB Rebuild (Page‑centric, PDF‑first, no fallbacks)

### Overview
Reintroduce a minimal, modular vector retrieval layer centered on per‑page chunks. Each page of a PDF is exported as a single‑page PDF (not images). For unprocessed pages, we call Sonnet‑4 with page 1 and optionally page 2 (to capture spillover) and store a structured, strict‑JSON result. Query‑time routing uses a per‑page `visual_importance` score: if 4–5, send the page PDF(s) with the user prompt; otherwise, send the chunk text.

This plan avoids fallbacks and emphasizes determinism, idempotency, and inspectability.

### Goals
- **Per‑page processing**: Export single‑page PDFs and persist page‑level chunks.
- **Strict structured extraction**: Sonnet‑4 returns summary, sections, full text (with spillover), visual descriptions, and a 1–5 `visual_importance` score.
- **Deterministic storage**: Stable IDs, hashes, schema `version`, no silent fallbacks.
- **Targeted retrieval**: Route to PDF or text context based on `visual_importance`.
- **Operational clarity**: Scripts for each step; admin UI to process selected PDFs and show progress.

### Data directories and IDs
- `data/` – original PDFs live here (as today).
- `data/pages/<pdf_basename>/` – derived single‑page PDFs.
  - Files named `p0001.pdf`, `p0002.pdf`, …
- Stable chunk `id`: `<pdf_basename>#p<index>`, 0‑based page index.
- Hashes:
  - `pdf_sha256`: hash of the source full PDF.
  - `page_pdf_sha256`: hash of the single‑page PDF for the page.

### Processing pipeline (single PDF)
1) Ensure `data/pages/<pdf_basename>/` exists.
2) For each page `n`:
   - Ensure `p{n+1:04}.pdf` exists; create via PyMuPDF/PyPDF if missing.
   - Check Chroma for chunk id `<pdf_basename>#p<n>`.
   - If missing or stale (schema `version` bump or page hash changed):
     - Load text of pages `n` and `n+1` (for spillover) using `pypdf`.
     - Call Sonnet‑4 with two files:
       - Required: `p{n+1:04}.pdf` (this is the primary page to analyze)
       - Optional: `p{n+2:04}.pdf` (spillover only)
       - Include the extracted text of both pages as additional text inputs to increase robustness/determinism.
     - Ask for strict JSON with keys:
       - `summary` (short overview of sections on page 1; include any spillover from page 2)
       - `sections` (list of section labels/titles present on page 1)
       - `full_text` (complete text of page‑1 sections including any content that spills into page 2; no truncation)
       - `visuals` (array of objects describing any charts, images, diagrams, tables on page 1, with brief description and relevance)
       - `visual_importance` (integer 1–5; how critical the visuals are for understanding page 1)
     - Upsert to Chroma: embed `full_text` and store full metadata (see schema below).

Notes:
- We use PDFs only (no PNGs). PDF inputs preserve both text and images.
- Spillover page is provided solely to avoid truncation of page‑1 sections; we still cite page‑1 for attribution.

### LLM strict JSON schema (Sonnet‑4)
Request a single JSON object with exactly these keys and types:

```json
{
  "summary": "string",
  "sections": ["string", "string"],
  "full_text": "string",
  "visuals": [
    { "type": "chart|image|diagram|table|other", "description": "string", "relevance": "string" }
  ],
  "visual_importance": 1
}
```

Prompt guidance (enforced):
- Analyze page 1 fully; include spillover content from page 2 only for sections that begin on page 1.
- Do not invent content. If unknown, return an empty array/empty string.
- Return exactly one JSON object; no prose or code fences.

### Vector store schema (Chroma)
- Collection: `boardrag_pages`
- Document: `full_text` (embedded)
- `id`: `<pdf_basename>#p<index>`
- `metadata` fields:
  - `source` (pdf basename)
  - `page` (int, 0‑based)
  - `next_page` (int or null) – included when spillover was consulted
  - `summary` (string)
  - `sections` (array of strings)
  - `visuals` (array as above)
  - `visual_importance` (int 1–5)
  - `pdf_sha256` (string)
  - `page_pdf_sha256` (string)
  - `created_at` (iso8601)
  - `version` (string; bump to force reprocess)

### Retrieval and chat routing
1) Perform similarity search over `full_text` (filter by selected game/pdf if provided).
2) For each candidate chunk:
   - If `visual_importance` ∈ {4, 5}:
     - Send the user question plus `p{page+1:04}.pdf` and `p{page+2:04}.pdf` (if exists) as Files to Sonnet‑4.
     - In the system/user instructions, require citations to reference the provided page number `page+1` explicitly.
   - Else (`visual_importance` ∈ {1, 2, 3}):
     - Send the user question plus the chunk’s `full_text` as context.
3) Keep existing citation behavior; only change is that we instruct the model to cite the page number we pass in (page 1 of the provided files corresponds to original page `page+1`).

### Admin API and UI
- Backend
  - `GET /admin/pdf-status`: returns per‑PDF status: total pages, processed pages, unprocessed pages.
  - `POST /admin/process-selected` (SSE): processes selected PDFs, streams logs.
- Frontend (`web/app/admin/page.tsx`)
  - Add a column: **Processed** (`X / Y pages`).
  - Add a button: **Process selected**; connects to SSE endpoint and shows progress in the console panel.

### CLI tools (deterministic, no fallbacks)
- `scripts/process_pdf.py <pdf>` – process one PDF end‑to‑end.
- `scripts/process_all.py` – iterate over `data/` and process all PDFs.
- `scripts/find_chunk.py` –
  - by page: `--pdf <name> --page <n>`
  - by text: `--query "…" [--pdf <name>]`
- `scripts/debug_page_export.py <pdf> --page <n>` – verify single‑page PDF export and hashes.

All scripts exit non‑zero on failure and print precise, step‑labeled logs.

### Modules and file boundaries (≤ 500 LOC each)
- `src/pdf_pages.py`
  - Export single‑page PDFs, compute `pdf_sha256`, `page_pdf_sha256`, page count.
- `src/embeddings.py`
  - Deterministic embeddings; model from `config.EMBEDDER_MODEL`.
- `src/vector_store.py`
  - Chroma init (`CHROMA_PATH`), upsert, by‑page fetch, similarity search, filters.
- `src/llm_page_extract.py`
  - Build Sonnet‑4 prompt, attach page PDFs and page texts, return strict JSON.
- `src/chunk_schema.py`
  - Typed dicts/dataclasses for chunk metadata; schema `version` constant.
- `src/query_routing.py` (or adapt `src/query.py` with a small adapter)
  - Retrieval, decision by `visual_importance`, context assembly, citation instruction injection.

### Determinism and idempotency
- Stable IDs based on basename and page index.
- Hash checks (`pdf_sha256`, `page_pdf_sha256`) to avoid stale chunks.
- Schema `version` bump forces reprocess.
- No alternate providers, no best‑effort modes, no silent retries.

### Example stored metadata (abbreviated)
```json
{
  "id": "catan_base#p12",
  "page": 12,
  "next_page": 13,
  "source": "catan_base.pdf",
  "summary": "Setup overview and initial placement rules.",
  "sections": ["1.1 Setup", "1.2 Initial Placement"],
  "visuals": [
    { "type": "diagram", "description": "Board layout example with ports.", "relevance": "Clarifies placement rules" }
  ],
  "visual_importance": 4,
  "pdf_sha256": "…",
  "page_pdf_sha256": "…",
  "version": "v1"
}
```

### Implementation notes
- Reintroduce `chromadb`/`langchain-chroma` as isolated dependencies used only by `src/vector_store.py` and scripts.
- Keep changes to `src/query.py` minimal by delegating to a routing helper.
- Maintain strict JSON parsing with clear errors; log the raw model output on violations.

### Testing plan
- Unit tests per module where feasible (export, hashing, schema validation).
- CLI tests in CI for `process_pdf`, `find_chunk` on small fixture PDFs.
- Admin SSE endpoint smoke test.

### Constraints
- No fallbacks. Fail fast with actionable errors.
- No PNG/image rasterization; PDFs only for both processing and query‑time attachments.
- Keep modules small, cohesive, and auditable.


