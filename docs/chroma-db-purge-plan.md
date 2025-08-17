## BoardRAG – Local DB/Chroma Purge Plan

### Overview
Purge all local DB and Chroma vector-store code and assets. Leave a clean, DB‑less project that uses the catalog and LLM Files API only.

### Scope (what to remove)
- **Legacy DB code paths and imports**:
  - `src/query.py` (any `chromadb`, `langchain_chroma`, `Chroma`, `get_embedding_function`, `CHROMA_PATH`, `get_chromadb_settings`, `suppress_chromadb_telemetry`). Remove legacy DB-backed branches in both normal and streaming flows.
  - `src/retrieval/game_names.py` (remove legacy DB-backed store/get fallback; keep catalog-only path).
  - `src/services/library_service.py` (remove `rebuild_library`, `rechunk_library`, and any `add_to_chroma` calls).
  - `src/handlers/library.py` (remove handlers for rebuild/reset/rechunk DB endpoints).
  - `api/main.py` (remove any DB reset/rebuild endpoints or startup code referencing Chroma; keep catalog bootstrap only).
  - `src/config.py` (remove Chroma helpers: `disable_chromadb_telemetry`, `get_chromadb_settings`, `suppress_chromadb_telemetry`, `CHROMA_PATH`).
  - `src/embedding_function.py` (delete if unused after cleanup).
- **Standalone utilities to delete**:
  - `src/populate_database.py`, root `populate_database.py`.
  - `src/visualize_db_argilla.py`, root `visualize_db_argilla.py` (if present).
- **Files/directories**:
  - `chroma/` directory and contents.
- **Docs/config/deps**:
  - `README.md` sections referencing vector DB and population scripts.
  - `requirements.txt`/`pyproject.toml` entries: `chromadb`, `langchain-chroma` (and any other vector-store libs).
  - `docker-compose.yml` Chroma volumes/vars (if any).
  - `.dockerignore` entry for `/chroma` (optional to keep; harmless).

### Step-by-step plan
1) **Remove legacy DB logic and imports**
   - Edit `src/query.py` to be DB‑less only; delete all Chroma/DB branches and related imports.
   - Edit `src/retrieval/game_names.py` to keep only catalog-based storage and reads.
   - Edit `src/services/library_service.py` to remove DB-centric flows; keep file save + catalog updates only.
   - Edit `src/handlers/library.py` to remove endpoints for rebuild/reset/rechunk.
   - Edit `api/main.py` to remove DB endpoints and startup DB logic; keep catalog bootstrap.
   - Edit `src/config.py` to remove Chroma helpers/consts; ensure `DB_LESS` defaults to true.
   - Delete `src/embedding_function.py` if no longer referenced.

2) **Delete DB population and visualization utilities**
   - Delete `src/populate_database.py`, root `populate_database.py`.
   - Delete `src/visualize_db_argilla.py`, root `visualize_db_argilla.py` (if present).

3) **Frontend/Admin cleanup**
   - In `web/app/admin/page.tsx` and `web/app/page.tsx`, remove UI/actions that trigger DB resets/rebuilds or show DB status.
   - Keep admin UX focused on PDF management (upload/refresh catalog).

4) **Dependency cleanup**
   - Remove vector-store/local-DB packages from `requirements.txt` (and `pyproject.toml` if applicable): `chromadb`, `langchain-chroma`, etc.
   - Ensure only DB‑less dependencies remain (Anthropic SDK/wrappers, PDF libs, etc.).

5) **Ops and config cleanup**
   - Delete the `chroma/` directory from the workspace.
   - In `docker-compose.yml`, remove Chroma volumes/env vars (e.g., `CHROMA_PATH`, `ALLOW_RESET`).
   - Optionally remove `/chroma` from `.dockerignore`.

6) **Documentation update**
   - Update `README.md` to remove the “Vector Database” and `populate_database.py` instructions.
   - Document DB‑less flow: PDFs in `data/`, catalog auto‑sync on API startup, queries via Anthropic Files API with citations.
   - Keep `docs/db-less-plan.md`; remove or update any other docs that reference Chroma.

7) **Tests and examples**
   - Ensure tests use DB‑less `query.py` paths only; remove tests depending on a local vector DB or populate scripts.
   - Update CLI examples to drop `populate_database.py` and DB reset instructions.

8) **Safety guardrails**
   - Add a static check (pre-commit/CI) that fails on these strings: `chromadb`, `langchain_chroma`, `Chroma(`, `CHROMA_PATH`, `get_chromadb_settings`, `suppress_chromadb_telemetry`, `add_to_chroma`.
   - Default `DB_LESS=true`; optionally fail fast on startup if Chroma is imported anywhere.

9) **One-time cleanup commands**
```bash
# Remove local DB dir
rm -rf chroma

# Sanity search (repeat until zero results)
rg -n "chromadb|langchain_chroma|Chroma\(|CHROMA_PATH|get_chromadb_settings|suppress_chromadb_telemetry|add_to_chroma" .
```

10) **Runtime verification**
- **API startup**: Catalog sync runs without touching any DB; logs reflect DB‑less mode.
- **Query flow**: Sample queries return answers with citations and no DB connections.
- **Admin UI**: No DB actions are present; upload/refresh catalog works.

### Deliverables checklist
- **Code**:
  - `src/query.py` DB‑less only; streaming variant DB‑less.
  - `src/retrieval/game_names.py` catalog-only.
  - `src/services/library_service.py` without DB functions.
  - `src/handlers/library.py` without DB admin endpoints.
  - `api/main.py` without any DB ops; uses catalog bootstrap.
  - Removed: `src/populate_database.py`, `populate_database.py`, `src/visualize_db_argilla.py`, `visualize_db_argilla.py`, `src/embedding_function.py`.
  - `src/config.py` without Chroma helpers/consts; `DB_LESS` defaults to true.
- **Dependencies/config**:
  - `requirements.txt`/`pyproject.toml` without vector-store deps.
  - `docker-compose.yml` without Chroma volumes/vars.
  - Deleted `chroma/` folder.
- **Docs**:
  - `README.md` updated to DB‑less only.
  - Any remaining docs updated or removed if they reference Chroma/local DB.


