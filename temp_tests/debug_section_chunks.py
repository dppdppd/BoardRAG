#!/usr/bin/env python3
"""
debug_section_chunks.py — mimic clicking a [section] link and show the relevant chunks

Usage examples (run from repo root, inside venv):
  venv\Scripts\python debug_section_chunks.py --section 3.1 --game "Advanced Squad Leader Starter Kit #4" --limit 20
  venv\Scripts\python debug_section_chunks.py --id <uid-from-ui>
  venv\Scripts\python debug_section_chunks.py --section 3.1 --game RISK --json

This script mirrors the server's /section-chunks logic to help debug results
without going through the API.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from langchain_chroma import Chroma

# Local imports from project
# Robust imports when run as a script or module
try:
    from src.config import CHROMA_PATH, get_chromadb_settings, suppress_chromadb_telemetry  # type: ignore
    from src.embedding_function import get_embedding_function  # type: ignore
    from src.query import get_stored_game_names  # type: ignore
except Exception:
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config import CHROMA_PATH, get_chromadb_settings, suppress_chromadb_telemetry  # type: ignore
    from src.embedding_function import get_embedding_function  # type: ignore
    from src.query import get_stored_game_names  # type: ignore


def _make_uid(text: str, meta: dict) -> str:
    try:
        payload = f"{meta.get('source') or ''}|{meta.get('page') or ''}|{meta.get('section') or ''}|{meta.get('section_number') or ''}|{(text or '')[:160]}".encode(
            "utf-8", errors="ignore"
        )
        h = hashlib.sha1(payload).digest()
        return base64.urlsafe_b64encode(h).decode("ascii").rstrip("=")
    except Exception:
        return ""


def _clean_chunk_text(s: str) -> str:
    try:
        # Remove long unicode ellipsis runs and dot leaders; trim page-number dot leaders
        s2 = re.sub(r"[…]{2,}", "", s)
        s2 = re.sub(r"\.{5,}", "", s2)
        cleaned_lines: List[str] = []
        for ln in s2.splitlines():
            ln2 = re.sub(r"\.{2,}\s*\d+\s*$", "", ln)
            cleaned_lines.append(ln2.rstrip())
        return "\n".join(cleaned_lines)
    except Exception:
        return s


def _normalize_section_number(raw: str) -> str:
    try:
        raw = (raw or "").strip()
        if not raw:
            return raw
        if "." in raw:
            return raw
        return f"{raw}.0"
    except Exception:
        return raw


def _final_sort_key(it: dict) -> Tuple[Any, Any, Tuple[int, ...]]:
    src = str(it.get("source") or "").lower()
    pg = it.get("page")
    try:
        pgk = int(pg) if isinstance(pg, (int, float, str)) and str(pg).isdigit() else 999999
    except Exception:
        pgk = 999999
    try:
        sec_path = tuple(int(p) for p in str(it.get("section_number") or "").split(".") if p.isdigit())
    except Exception:
        sec_path = tuple()
    return (src, pgk, sec_path)


def resolve_target_files(game: Optional[str]) -> Optional[Set[str]]:
    if not game:
        return None
    try:
        name_map = get_stored_game_names()
        key = (game or "").strip().lower()
        # Match by mapped game name first
        files = {Path(fname).name.lower() for fname, gname in name_map.items() if (gname or "").strip().lower() == key}
        if not files:
            # Fallback: match by filename-style key (e.g., "catan_base")
            files = {
                Path(fname).name.lower()
                for fname in name_map.keys()
                if Path(fname).name.replace(".pdf", "").replace(" ", "_").lower() == key
            }
        return files if files else set()
    except Exception:
        return set()


def fetch_section_chunks(section: Optional[str], game: Optional[str], limit: int, uid: Optional[str]) -> Dict[str, Any]:
    sec = (section or "").strip()
    if not sec and not uid:
        raise SystemExit("error: missing --section or --id")

    # Connect to DB
    embedding_function = get_embedding_function()
    with suppress_chromadb_telemetry():
        persistent_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=get_chromadb_settings())
        db = Chroma(client=persistent_client, embedding_function=embedding_function)

    # Resolve PDFs for selected game (if provided)
    target_files = resolve_target_files(game)

    try:
        raw = db.get()
        documents: List[str] = raw.get("documents", [])
        metadatas: List[dict] = raw.get("metadatas", [])
    except Exception as e:
        raise SystemExit(f"db read error: {e}")

    out: List[Dict[str, Any]] = []

    def _matches_section(sec_input: str, meta: dict) -> Tuple[bool, int]:
        """Return (is_match, rank) lower is better rank."""
        sec_num = str(meta.get("section_number") or "").strip()
        sec_label = str(meta.get("section") or "").strip()
        exact_num = bool(sec_num) and (sec_num == sec_input)
        child_num = bool(sec_num) and sec_num.startswith(sec_input + ".")
        m = re.match(r"^\s*(\d+(?:\.\d+)*)\b", sec_label)
        label_num = m.group(1) if m else ""
        exact_label = bool(label_num) and (label_num == sec_input)
        child_label = bool(label_num) and label_num.startswith(sec_input + ".")
        if not (exact_num or child_num or exact_label or child_label):
            return False, 999
        cross_ref = bool(re.search(r"&\s*\d", sec_label))
        base_rank = 0 if exact_num else 1 if child_num else 2 if exact_label else 3
        penalty = 2 if cross_ref and not exact_num else 0
        return True, base_rank + penalty

    candidates: List[Tuple[int, str, str, dict]] = []
    for text, meta in zip(documents, metadatas):
        if not isinstance(meta, dict):
            continue
        # Filter by game PDFs if provided
        if target_files is not None and len(target_files) > 0:
            pdf_fn = str(meta.get("pdf_filename") or "").lower()
            if pdf_fn not in target_files:
                continue
        u = _make_uid(str(text or ""), meta)
        if uid and u == uid:
            # Exact match by uid
            source_path = str(meta.get("source") or "")
            return {
                "section": sec,
                "game": game,
                "chunks": [
                    {
                        "uid": u,
                        "text": _clean_chunk_text(str(text or "")),
                        "source": Path(source_path).name if source_path else "unknown",
                        "page": meta.get("page"),
                        "section": str(meta.get("section") or ""),
                        "section_number": str(meta.get("section_number") or ""),
                        "rects_norm": meta.get("rects_norm"),
                    }
                ],
            }
        if sec:
            ok, rank = _matches_section(sec, meta)
            if ok:
                candidates.append((rank, u, text, meta))

    if uid and not out:
        return {"section": sec, "game": game, "chunks": []}

    if sec:
        # Sort by rank, then by page number then by text length (shorter first)
        def _key(item):
            r, u, t, m = item
            pg = m.get("page")
            try:
                pgk = int(pg) if isinstance(pg, (int, float, str)) and str(pg).isdigit() else 999999
            except Exception:
                pgk = 999999
            return (r, pgk, len(str(t or "")))

        candidates.sort(key=_key)
        limit_n = max(1, min(50, int(limit) if isinstance(limit, int) else 12))
        for r, u, text, meta in candidates[:limit_n]:
            sec_num = str(meta.get("section_number") or "").strip()
            sec_label = str(meta.get("section") or "").strip()
            source_path = str(meta.get("source") or "")
            out.append(
                {
                    "uid": u,
                    "text": _clean_chunk_text(str(text or "")),
                    "source": Path(source_path).name if source_path else "unknown",
                    "page": meta.get("page"),
                    "section": sec_label,
                    "section_number": sec_num,
                    "rects_norm": meta.get("rects_norm"),
                }
            )

    out.sort(key=_final_sort_key)
    return {"section": sec, "game": game, "chunks": out}


def main() -> None:
    p = argparse.ArgumentParser(description="Debug section chunks (like clicking [3.1] in UI)")
    p.add_argument("--section", type=str, help="Section number or tag, e.g. 3.1")
    p.add_argument("--game", type=str, help="Game name as shown in UI (or filename key)")
    p.add_argument("--id", type=str, help="Exact chunk UID to fetch")
    p.add_argument("--limit", type=int, default=12, help="Max chunks to return")
    p.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = p.parse_args()

    res = fetch_section_chunks(args.section, args.game, args.limit, args.id)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    chunks = res.get("chunks", [])
    print(f"Section: {res.get('section') or '-'}  Game: {res.get('game') or '-'}  (chunks={len(chunks)})\n")
    for i, c in enumerate(chunks, 1):
        sec = c.get("section") or ""
        sec_num = c.get("section_number") or ""
        page = c.get("page")
        src = c.get("source")
        uid = c.get("uid")
        print(f"{i:02d}. {src} · p.{page} · {sec_num} {sec}")
        print(f"    uid={uid}")
        txt = str(c.get("text") or "").strip()
        if txt:
            print("    " + "\n    ".join(txt.splitlines()))
        print()


if __name__ == "__main__":
    main()


