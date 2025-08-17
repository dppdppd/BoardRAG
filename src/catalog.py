from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from . import config as cfg  # type: ignore
from .llm_outline_helpers import upload_pdf_to_anthropic_files, anthropic_pdf_messages_with_file


CATALOG_PATH = Path(".cache/games_catalog.json")


def _now_iso() -> str:
    try:
        return datetime.utcnow().isoformat() + "Z"
    except Exception:
        return ""


def load_catalog() -> Dict[str, dict]:
    try:
        if CATALOG_PATH.exists():
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_catalog(cat: Dict[str, dict]) -> None:
    try:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_PATH.write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _extract_game_name_from_pdf(api_key: str, model: str, file_id: str) -> str:
    system = (
        "You extract the official commercial game/module name from a boardgame rulebook PDF. "
        "Return JSON ONLY: {\"game_name\": string}."
    )
    user = (
        "Identify the official commercial game name as printed on the cover/title/intro pages.\n"
        "Return JSON ONLY: {\\\"game_name\\\": \\\"<short title>\\\"}"
    )
    try:
        raw = anthropic_pdf_messages_with_file(api_key, model, system, user, file_id)
        # Pull last JSON object
        import re as _re
        mlist = list(_re.finditer(r"\{[\s\S]*?\}", raw))
        for m in reversed(mlist):
            try:
                obj = json.loads(m.group(0))
                name = str(obj.get("game_name") or "").strip()
                if name:
                    return name
            except Exception:
                continue
    except Exception:
        pass
    return ""


def ensure_catalog_up_to_date(log: Optional[callable] = None) -> Dict[str, dict]:
    """Scan data/ for PDFs, upload any missing via Files API, and store {file_id, game_name}.

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
        if entry.get("file_id"):
            # Already in catalog: do nothing
            try:
                if log:
                    log(f"⏭  [{idx}/{len(pdfs)}] Skip (already cataloged): {key}")
            except Exception:
                pass
            continue
        if not api_key:
            _log(f"⚠️ Skipping upload for {key}: ANTHROPIC_API_KEY missing")
            continue
        try:
            _log(f"📤 [{idx}/{len(pdfs)}] Uploading {key} …")
            fid = upload_pdf_to_anthropic_files(api_key, str(p.resolve()), retries=1, backoff_s=0.0)
            _log(f"🔗 Received file_id={fid} for {key}")
            name = _extract_game_name_from_pdf(api_key, model, fid) or p.stem
            if name:
                try:
                    _log(f"🏷  Extracted game_name='{name}'")
                except Exception:
                    pass
            cat[key] = {
                "file_id": fid,
                "game_name": name,
                "pages": None,
                "size_bytes": size,
                "updated_at": _now_iso(),
            }
            _log(f"✅ Cataloged {key} → '{name}' ({fid})")
            save_catalog(cat)
        except Exception as e:
            _log(f"❌ Upload failed for {key}: {e}")
            continue
    try:
        if log:
            log("📚 Catalog scan complete")
    except Exception:
        pass
    return cat


def list_games_from_catalog() -> List[str]:
    cat = load_catalog()
    names: List[str] = []
    seen: set[str] = set()
    for _fname, meta in cat.items():
        n = str(meta.get("game_name") or "").strip()
        if not n:
            continue
        if n.lower() not in seen:
            names.append(n)
            seen.add(n.lower())
    names.sort(key=lambda s: s.lower())
    return names


def resolve_file_ids_for_game(game: Optional[str]) -> List[Tuple[str, str]]:
    """Return list of (abs_path, file_id) for a given game name; empty game returns all."""
    cat = load_catalog()
    if not game:
        out: List[Tuple[str, str]] = []
        base = Path(getattr(cfg, "DATA_PATH", "data")).resolve()
        for fname, meta in cat.items():
            fid = str(meta.get("file_id") or "").strip()
            if not fid:
                continue
            out.append((str((base / fname).resolve()), fid))
        return out
    key = str(game).strip().lower()
    out: List[Tuple[str, str]] = []
    base = Path(getattr(cfg, "DATA_PATH", "data")).resolve()
    for fname, meta in cat.items():
        name = str(meta.get("game_name") or "").strip()
        fid = str(meta.get("file_id") or "").strip()
        if not fid or not name:
            continue
        if name.strip().lower() == key or fname.strip().lower().startswith(key):
            out.append((str((base / fname).resolve()), fid))
    return out


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
        if not fid:
            status = "missing_file_id"
            missing += 1
            entries.append({"filename": fname, "game_name": gname, "file_id": None, "status": status})
            _log(f"⚠️  [{i}] Missing file_id: {fname}")
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
    """Update catalog entries' game_name for the given filenames.

    Returns the number of entries updated. Does not create new entries.
    """
    if not new_name or not filenames:
        return 0
    cat = load_catalog()
    updated = 0
    for fn in filenames:
        key = Path(fn).name
        if key in cat:
            try:
                entry = cat.get(key) or {}
                entry["game_name"] = new_name
                entry["updated_at"] = _now_iso()
                cat[key] = entry
                updated += 1
            except Exception:
                pass
    if updated:
        save_catalog(cat)
    return updated


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

