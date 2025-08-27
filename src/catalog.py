from __future__ import annotations

import os
import json
from datetime import datetime
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from . import config as cfg  # type: ignore
from .llm_outline_helpers import upload_pdf_to_anthropic_files, anthropic_pdf_messages_with_file


# Persist the catalog under the persistent data volume, e.g., /data on Railway
# Use a dedicated subdirectory to avoid cluttering the PDF directory
CATALOG_DIR = Path(getattr(cfg, "DATA_PATH", "data")) / "catalog"
CATALOG_PATH = CATALOG_DIR / "games_catalog.json"
CATALOG_LOCK = threading.RLock()


def _now_iso() -> str:
    try:
        return datetime.utcnow().isoformat() + "Z"
    except Exception:
        return ""


def load_catalog() -> Dict[str, dict]:
    try:
        if CATALOG_PATH.exists():
            # Read without lock to allow concurrent readers; writes are atomic
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_catalog(cat: Dict[str, dict]) -> None:
    try:
        with CATALOG_LOCK:
            CATALOG_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = CATALOG_PATH.with_suffix(CATALOG_PATH.suffix + ".tmp")
            data = json.dumps(cat, ensure_ascii=False, indent=2)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(data)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            # Atomic replace to avoid readers seeing a partially-written file
            try:
                os.replace(tmp_path, CATALOG_PATH)
            except Exception:
                # Fallback: attempt rename via Path API
                try:
                    Path(tmp_path).replace(CATALOG_PATH)
                except Exception:
                    pass
    except Exception:
        pass


def _extract_game_name_from_pdf(api_key: str, model: str, file_id: str) -> str:
    # Main PDF is no longer uploaded or inspected for name extraction.
    # Fallback to empty so callers can default to filename stem.
    return ""


def ensure_catalog_up_to_date(log: Optional[callable] = None) -> Dict[str, dict]:
    """Scan data/ for PDFs and maintain catalog metadata without uploading main PDFs.

    Only per-page file_ids are supported elsewhere; here we record game_name (from filename stem)
    and basic metadata. We never attempt to upload the full PDF or store a top-level file_id.
    Returns the updated catalog.
    """
    def _log(msg: str) -> None:
        try:
            if log:
                log(msg)
        except Exception:
            pass

    data_dir = Path(getattr(cfg, "DATA_PATH", "data"))
    if log:
        try:
            log(f"🗂 DATA_PATH = {data_dir.resolve()}")
        except Exception:
            pass
    pdfs = sorted([p for p in data_dir.glob("*.pdf")])
    try:
        if log:
            log(f"🔎 Found {len(pdfs)} PDF(s) to consider")
    except Exception:
        pass
    cat = load_catalog()
    try:
        from os import path as _ospath
        cat_file = CATALOG_PATH
        if log:
            if cat:
                log(f"📒 Loaded catalog with {len(cat)} entrie(s)")
            else:
                if cat_file.exists():
                    try:
                        sz = cat_file.stat().st_size
                    except Exception:
                        sz = -1
                    log(f"⚠️ Catalog file present but empty/unreadable (size={sz} bytes); proceeding to rebuild entries")
                else:
                    log("ℹ️ No existing catalog file; starting fresh")
    except Exception:
        pass
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    model = os.getenv("OUTLINE_LLM_MODEL", "claude-sonnet-4-20250514")

    for idx, p in enumerate(pdfs, start=1):
        key = p.name
        try:
            stat = p.stat()
            size = stat.st_size
        except Exception:
            size = 0
        entry = cat.get(key) or {}
        # Preserve any user-assigned name if present; otherwise default to filename stem
        existing_name = str((entry or {}).get("game_name") or "").strip()
        final_name = existing_name or p.stem
        # Update in-place to avoid dropping other fields; never set top-level file_id
        entry = entry or {}
        entry.pop("file_id", None)
        entry["game_name"] = final_name
        entry.setdefault("pages", entry.get("pages") if isinstance(entry.get("pages"), dict) else None)
        entry["size_bytes"] = size
        entry["updated_at"] = _now_iso()
        cat[key] = entry
        try:
            if log:
                log(f"✅ Cataloged {key} → '{final_name}'")
        except Exception:
            pass
        save_catalog(cat)
    # Do not purge catalog entries whose PDFs are missing; keep catalog authoritative
    try:
        if log:
            log("📚 Catalog scan complete")
    except Exception:
        pass
    return cat


def ensure_catalog_for_files(filenames: List[str], log: Optional[callable] = None) -> Dict[str, dict]:
    """Update catalog entries only for the specified PDF filenames without uploading main PDFs.

    - Only processes the provided `filenames` (basenames under DATA_PATH)
    - Does NOT upload or store a top-level `file_id`
    - Sets/keeps `game_name` (defaults to filename stem if missing)
    - Does NOT scan or modify entries for other PDFs
    """
    def _log(msg: str) -> None:
        try:
            if log:
                log(msg)
        except Exception:
            pass

    data_dir = Path(getattr(cfg, "DATA_PATH", "data"))
    pdfs: List[Path] = []
    for name in (filenames or []):
        try:
            fn = Path(name).name
            p = (data_dir / fn)
            if p.exists() and p.is_file() and p.suffix.lower() == ".pdf":
                pdfs.append(p)
        except Exception:
            continue
    try:
        if log:
            log(f"🔎 Considering {len(pdfs)} uploaded PDF(s) for catalog update")
    except Exception:
        pass

    cat = load_catalog()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    model = os.getenv("OUTLINE_LLM_MODEL", "claude-sonnet-4-20250514")

    for idx, p in enumerate(pdfs, start=1):
        key = p.name
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        entry = cat.get(key) or {}
        # Preserve any user-assigned name if present; otherwise default to filename stem
        existing_name = str((entry or {}).get("game_name") or "").strip()
        final_name = existing_name or p.stem
        # Update in-place; never set top-level file_id
        entry = entry or {}
        entry.pop("file_id", None)
        entry["game_name"] = final_name
        entry.setdefault("pages", entry.get("pages") if isinstance(entry.get("pages"), dict) else None)
        entry["size_bytes"] = size
        entry["updated_at"] = _now_iso()
        cat[key] = entry
        try:
            if log:
                log(f"✅ Cataloged {key} → '{final_name}'")
        except Exception:
            pass
        save_catalog(cat)

    try:
        if log:
            log("📚 Catalog update for uploaded files complete")
    except Exception:
        pass
    return cat


def backfill_catalog_from_data(log: Optional[callable] = None) -> Dict[str, dict]:
    """Ensure every top-level PDF in DATA_PATH has a catalog entry.

    Does not upload or set top-level file_id. Creates minimal entries with
    game_name defaulting to filename stem and size_bytes. Preserves existing fields.
    """
    def _log(msg: str) -> None:
        try:
            if log:
                log(msg)
        except Exception:
            pass

    data_dir = Path(getattr(cfg, "DATA_PATH", "data"))
    pdfs = []
    try:
        pdfs = sorted([p for p in data_dir.glob("*.pdf") if p.is_file()])
    except Exception:
        pdfs = []

    cat = load_catalog()
    created = 0
    for p in pdfs:
        key = p.name
        if key in cat:
            continue
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        entry: dict = {
            "game_name": p.stem,
            "pages": None,
            "size_bytes": size,
            "updated_at": _now_iso(),
        }
        cat[key] = entry
        created += 1
    if created:
        try:
            _log(f"➕ Backfilled {created} missing PDF entrie(s) from data/")
        except Exception:
            pass
        save_catalog(cat)
    return cat

def set_page_file_id(pdf_filename: str, page_1based: int, file_id: str, page_pdf_sha256: Optional[str] = None) -> None:
    """Persist a per-page file_id under the parent PDF entry in the catalog.

    Creates or updates `pages[<1-based>] = { file_id, page_pdf_sha256, updated_at }`.
    """
    if not pdf_filename or not file_id:
        return
    try:
        page_key = str(int(page_1based))
    except Exception:
        page_key = str(page_1based)
    cat = load_catalog()
    entry = cat.get(pdf_filename) or {}
    # Ensure a default game_name exists for entries created via per-page uploads
    try:
        has_name = bool(str((entry or {}).get("game_name") or "").strip())
    except Exception:
        has_name = False
    if not has_name:
        try:
            entry["game_name"] = Path(pdf_filename).stem
        except Exception:
            entry["game_name"] = str(pdf_filename)
    pages = entry.get("pages")
    if not isinstance(pages, dict):
        pages = {}
    pages[page_key] = {
        "file_id": str(file_id),
        "page_pdf_sha256": str(page_pdf_sha256 or ""),
        "updated_at": _now_iso(),
    }
    entry["pages"] = pages
    entry["updated_at"] = _now_iso()
    cat[pdf_filename] = entry
    save_catalog(cat)


def get_page_file_id(pdf_filename: str, page_1based: int) -> Optional[str]:
    """Return persisted per-page file_id for a given PDF filename and 1-based page, if available."""
    if not pdf_filename:
        return None
    try:
        page_key = str(int(page_1based))
    except Exception:
        page_key = str(page_1based)
    cat = load_catalog()
    entry = cat.get(pdf_filename) or {}
    pages = entry.get("pages")
    if not isinstance(pages, dict):
        return None
    info = pages.get(page_key)
    if not isinstance(info, dict):
        return None
    fid = str(info.get("file_id") or "").strip()
    return fid or None


def list_games_from_catalog() -> List[str]:
    cat = load_catalog()
    names: List[str] = []
    seen: set[str] = set()
    for fname, meta in cat.items():
        # Fallback to filename stem when game_name is missing
        fallback_name = Path(fname).stem
        n = str(meta.get("game_name") or "").strip() or fallback_name
        key = n.lower()
        if key not in seen:
            names.append(n)
            seen.add(key)
    names.sort(key=lambda s: s.lower())
    return names


 


# ---------------------------------------------------------------------------
# Catalog validator – minimal check of file_id existence in Anthropic
# ---------------------------------------------------------------------------

def validate_catalog(log: Optional[callable] = None) -> Dict[str, object]:
    """Validate that each catalog entry has a working file_id (no re-upload).

    Returns a report with per-entry status and a summary.
    """
    def _log(msg: str) -> None:
        try:
            if log:
                log(msg)
        except Exception:
            pass

    cat = load_catalog()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    entries = []
    ok = 0
    invalid = 0
    missing = 0
    skipped = 0

    if not cat:
        return {"entries": [], "summary": {"total": 0, "ok": 0, "invalid": 0, "missing": 0, "skipped": 0}}

    # Lazy import to avoid hard dependency at module import time
    try:
        from .llm_outline_helpers import validate_anthropic_file  # type: ignore
    except Exception:
        def validate_anthropic_file(_api_key: str, _fid: str) -> bool:  # type: ignore
            return False

    _log(f"🔍 Validating {len(cat)} catalog entrie(s)…")
    for i, (fname, meta) in enumerate(sorted(cat.items(), key=lambda kv: kv[0].lower()), start=1):
        fid = str((meta or {}).get("file_id") or "").strip()
        gname = str((meta or {}).get("game_name") or "").strip()
        # We no longer maintain top-level file_id; mark as skipped when absent
        if not fid:
            status = "skipped_no_main_file_id"
            skipped += 1
            entries.append({"filename": fname, "game_name": gname, "file_id": None, "status": status})
            _log(f"⏭  [{i}] Skipped (no main file_id by design): {fname}")
            continue
        if not api_key:
            status = "skipped_no_api_key"
            skipped += 1
            entries.append({"filename": fname, "game_name": gname, "file_id": fid, "status": status})
            _log(f"⏭  [{i}] Skipped (no API key): {fname}")
            continue
        ok_flag = False
        try:
            ok_flag = bool(validate_anthropic_file(api_key, fid))
        except Exception:
            ok_flag = False
        if ok_flag:
            status = "ok"
            ok += 1
            _log(f"✅ [{i}] OK: {fname}")
        else:
            status = "invalid"
            invalid += 1
            _log(f"❌ [{i}] Invalid file_id: {fname} ({fid})")
        entries.append({"filename": fname, "game_name": gname, "file_id": fid, "status": status})

    summary = {"total": len(cat), "ok": ok, "invalid": invalid, "missing": missing, "skipped": skipped}
    _log(f"📊 Validation summary: {summary}")
    return {"entries": entries, "summary": summary}


# ---------------------------------------------------------------------------
# Admin helpers for DB-less assign/choices
# ---------------------------------------------------------------------------

def set_game_name_for_filenames(filenames: List[str], new_name: str) -> int:
    """Set game_name for the given PDF filenames in the catalog.

    Creates catalog entries when missing. Returns the number of entries written.
    """
    if not new_name or not filenames:
        return 0
    cat = load_catalog()
    updated = 0
    for fn in filenames:
        key = Path(fn).name
        try:
            entry = cat.get(key) or {}
            entry["game_name"] = new_name
            entry.setdefault("file_id", str(entry.get("file_id") or ""))
            entry.setdefault("pages", entry.get("pages") if isinstance(entry.get("pages"), dict) else None)
            entry["updated_at"] = _now_iso()
            cat[key] = entry
            updated += 1
        except Exception:
            pass
    if updated:
        save_catalog(cat)
    return updated


def get_pdf_filenames_for_game(game_name: str) -> List[str]:
    """Return list of PDF filenames associated with a specific game."""
    if not game_name:
        return []
    
    cat = load_catalog()
    if not cat:
        return []
    
    key = str(game_name).strip().lower()
    filenames: List[str] = []
    
    for fname, meta in cat.items():
        stored_game_name = str(meta.get("game_name") or "").strip()
        if not stored_game_name:
            continue
        
        # Match by game name or filename prefix
        if stored_game_name.lower() == key or fname.lower().startswith(key):
            filenames.append(fname)
    
    return sorted(filenames)


def get_pdf_choices_from_catalog() -> List[str]:
    """Return entries like 'Game Name - filename.pdf' sourced from catalog."""
    cat = load_catalog()
    if not cat:
        return []
    choices: List[str] = []
    for fname, meta in sorted(cat.items(), key=lambda kv: kv[0].lower()):
        title = str((meta or {}).get("game_name") or Path(fname).name)
        choices.append(f"{title} - {Path(fname).name}")
    return choices

