from __future__ import annotations

"""
LLM-powered PDF outline extraction for BoardRAG.

Given a PDF path, extract a semantic outline via an LLM:
- sections: [{code, title, first_page, section_kind}]
- alias_map: { alias_phrase -> code }
- glossary_codes: [codes]

This module favors compact prompts by sending candidate header lines with page
numbers instead of full pages. Downstream code can still use the outline to
segment pages deterministically.
"""

import os
import re
import json
from typing import Any, Dict, List, Tuple
import pathlib


from src.llm_outline_helpers import load_pdf_pages as _load_pdf_pages  # type: ignore


from src.llm_outline_helpers import strip_code_fences as _strip_code_fences  # type: ignore


from src.llm_outline_helpers import find_json_objects as _find_json_objects  # type: ignore


from src.llm_outline_helpers import gather_candidates_regex as _gather_candidates  # type: ignore


from src.llm_outline_helpers import make_llm as _make_llm  # type: ignore
from src.llm_outline_helpers import anthropic_pdf_messages, upload_pdf_to_anthropic_files, anthropic_pdf_messages_with_file  # type: ignore


_SYS = (
    "You extract a semantic outline from boardgame rulebooks. "
    "Given page texts, return STRICT JSON only with keys: "
    "game_name (string), sections (array), alias_map (object), glossary_codes (array), objects (array)."
)

_USR_TEMPLATE = (
    "Return JSON ONLY.\n\n"
    "Rules for game_name:\n"
    "- Provide the official game/module name extracted from the cover/title/intro lines. Keep it short and exact (e.g., 'High Frontier 4 All').\n\n"
    "Rules for sections:\n"
    "- Extract code, title, and first_page (smallest page where code appears). If a code is not visible, omit it (do not invent), but still include a section with code=''.\n"
    "- code must be a visible section code: numeric dotted (e.g., 3.5, 3.5.1), letter-first (e.g., F4, F4.a, A10.2b), or digit-first (e.g., 1B6, 1B6b).\n"
    "- title is the label text following the code on the same line, without trailing ':' or '.'.\n"
    "- section_kind ∈ {{setup, operation, movement, hazard, scoring, endgame, glossary, component, politics, economy, reference, example, note, other}}.\n"
    "- Choose the most specific; use 'glossary' for term definitions.\n\n"
    "Rules for alias_map:\n"
    "- Map common phrases to canonical codes (e.g., 'sunspot cycle phase' -> C8).\n"
    "- Only include codes that exist in sections.\n\n"
    "Rules for glossary_codes:\n"
    "- List any codes that are glossary entries (if obvious).\n\n"
    "Rules for objects (fine-grained references):\n"
    "- Extract notable items within sections: definitions, examples, tables, figures, charts, boxes, notes.\n"
    "- For each object, provide: parent_code, kind, title (short label), first_page, and anchor_code.\n"
    "- anchor_code must include ONLY letters/digits/periods and start with parent_code, e.g., '1B6.ex1', 'F4.def1', 'C8.fig1'.\n"
    "- Optional: snippet (<=320 chars) quoting the relevant lines.\n\n"
    "Input pages (format === Page N ===):\n{pages}\n\n"
    "Output schema (STRICT JSON object only):\n"
    "{\"game_name\": string, \"sections\": [{\"code\": string, \"title\": string, \"first_page\": int, \"section_kind\": string}], \"alias_map\": {string: string}, \"glossary_codes\": [string], \"objects\": [{\"parent_code\": string, \"kind\": string, \"title\": string, \"first_page\": int, \"anchor_code\": string, \"snippet\": string}]}\n"
    "STRICT JSON RULES: single JSON object, no prose/markdown, no trailing commas, properly closed arrays/objects, exact keys."
)


_USR_CANDIDATES_TEMPLATE = (
    "Return JSON ONLY.\n\n"
    "You are given candidate header lines extracted from the PDF, each prefixed with 'p <page>:' where <page> is 1-based.\n"
    "Rules for game_name:\n"
    "- Provide the official game/module name extracted from the cover/title/intro lines. Keep it short and exact.\n\n"
    "Rules for sections:\n"
    "- Extract ALL sections using ONLY visible codes that appear at the START of the candidate lines.\n"
    "- For each section include: code, title (substring after the code on the same line; if a colon ':' exists, use the substring BEFORE the first ':'), first_page (smallest page for that code), and section_kind.\n"
    "- Code patterns include (examples): 3.5, 3.5.1, 5.6.1, F4, F4.a, A10.2b, 1B6, 1B6b.\n"
    "- Include sublevels (e.g., 5.6.1) not just top-level.\n"
    "- If a code is missing on a line, skip it (do not invent).\n\n"
    "Rules for alias_map:\n"
    "- Map common phrases to canonical codes that exist in sections. Keep small.\n\n"
    "Rules for glossary_codes:\n"
    "- List any codes that are glossary entries (if obvious).\n\n"
    "Rules for objects:\n"
    "- Optional. Extract notable items within sections (definition/example/table/figure/chart/box/note).\n"
    "- Provide: parent_code, kind, title (short), first_page, anchor_code.\n\n"
    "Input candidate lines (format: 'p N: ...'):\n{candidates}\n\n"
    "Output schema (STRICT JSON object only):\n"
    "{\"game_name\": string, \"sections\": [{\"code\": string, \"title\": string, \"first_page\": int, \"section_kind\": string}], \"alias_map\": {string: string}, \"glossary_codes\": [string], \"objects\": [{\"parent_code\": string, \"kind\": string, \"title\": string, \"first_page\": int, \"anchor_code\": string, \"snippet\": string}]}\n"
    "STRICT JSON RULES: single JSON object, no prose/markdown, no trailing commas, properly closed arrays/objects, exact keys."
)

_USR_WINDOW_TEMPLATE = (
    "Return JSON ONLY.\n\n"
    "You are given the text for two consecutive pages: page N and page N+1.\n"
    "Use BOTH pages to avoid cutting headers or content at page breaks.\n\n"
    "Extract ALL of the following:\n"
    "- candidates: [{page:int, line:string}] where 'line' is a full line that STARTS with a visible section code\n"
    "- sections: [{code, title, first_page, section_kind}] where title is the substring after the code; if a colon ':' is present, use the substring BEFORE the first ':'; first_page is the smallest page number the code appears on within these two pages\n"
    "- alias_map: { alias_phrase: code } for codes present in 'sections'\n"
    "- glossary_codes: [codes] if obvious\n"
    "- objects: [{parent_code, kind, title, first_page, anchor_code, snippet}]\n\n"
    "Rules for codes: numeric dotted (e.g., 3.5, 3.5.1, 5.6.1), letter-first (F4, F4.a, A10.2b), digit-first (1B6, 1B6b). Include sublevels.\n"
    "Do NOT invent codes; only use visible codes at start of line.\n\n"
    "Input pages:\n{pages}\n\n"
    "Output schema (STRICT JSON object only):\n"
    "{\"candidates\": [{\"page\": int, \"line\": string}], \"sections\": [{\"code\": string, \"title\": string, \"first_page\": int, \"section_kind\": string}], \"alias_map\": {string: string}, \"glossary_codes\": [string], \"objects\": [{\"parent_code\": string, \"kind\": string, \"title\": string, \"first_page\": int, \"anchor_code\": string, \"snippet\": string}]}\n"
    "STRICT JSON RULES: single JSON object, no prose/markdown, no trailing commas, properly closed arrays/objects, exact keys."
)


def extract_pdf_outline(pdf_path: str, debug_dir: str | None = None) -> Dict[str, Any]:
    """Extract a semantic outline via LLM from a PDF path.

    Returns a dict with keys: sections, alias_map, glossary_codes.
    """
    pages = _load_pdf_pages(pdf_path)
    from langchain.schema import HumanMessage, SystemMessage  # type: ignore
    # Helper: first-pages fallback (LLM-only, no regex)
    def _llm_first_pages_outline(all_pages: List[str]) -> Dict[str, Any]:
        MAX_PAGES = 8
        MAX_CHARS_PER_PAGE = 1800
        selected = all_pages[:MAX_PAGES]
        trimmed = []
        for i, t in enumerate(selected, start=1):
            s_page = (t or "").strip()
            if len(s_page) > MAX_CHARS_PER_PAGE:
                s_page = s_page[:MAX_CHARS_PER_PAGE]
            trimmed.append(f"=== Page {i} ===\n{s_page}")
        pages_block = "\n\n".join(trimmed)
        try:
            llm = _make_llm()
            usr = _USR_TEMPLATE.replace("{pages}", pages_block)
            out = llm.invoke([SystemMessage(content=_SYS), HumanMessage(content=usr)])
            s = str(getattr(out, "content", "") or "").strip()
            if debug_dir:
                try:
                    p = pathlib.Path(debug_dir)
                    p.mkdir(parents=True, exist_ok=True)
                    (p / "system.txt").write_text(_SYS, encoding="utf-8")
                    (p / "user.txt").write_text(usr, encoding="utf-8")
                    (p / "raw.txt").write_text(s, encoding="utf-8")
                except Exception:
                    pass
        except Exception as e:
            try:
                print(f"⚠️  LLM outline error (fallback): {e}")
            except Exception:
                pass
            return {"sections": [], "alias_map": {}, "glossary_codes": []}
        s_clean = _strip_code_fences(s)
        candidates_json: List[str] = _find_json_objects(s_clean)
        if not candidates_json:
            return {"sections": [], "alias_map": {}, "glossary_codes": []}
        data: Dict[str, Any] | None = None
        for js in candidates_json[::-1]:
            try:
                tmp = json.loads(js)
            except Exception:
                continue
            if isinstance(tmp, dict) and ("sections" in tmp or "alias_map" in tmp or "game_name" in tmp):
                data = tmp
                break
        if data is None or not isinstance(data, dict):
            return {"sections": [], "alias_map": {}, "glossary_codes": []}
        if debug_dir:
            try:
                p = pathlib.Path(debug_dir)
                (p / "parsed.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        # Normalize
        secs = []
        seen = set()
        for it in (data.get("sections") or []):
            try:
                code = str(it.get("code") or "").strip()
                title = str(it.get("title") or "").strip()
                first_page = int(it.get("first_page") or 0)
                kind = str(it.get("section_kind") or "").strip().lower()
                if not code or first_page <= 0:
                    continue
                if code in seen:
                    continue
                seen.add(code)
                secs.append({"code": code, "title": title, "first_page": first_page, "section_kind": kind})
            except Exception:
                continue
        data["sections"] = sorted(secs, key=lambda x: x.get("first_page", 1e9))
        if not isinstance(data.get("alias_map"), dict):
            data["alias_map"] = {}
        if not isinstance(data.get("glossary_codes"), list):
            data["glossary_codes"] = []
        objs_out = []
        for it in (data.get("objects") or []):
            try:
                parent = str(it.get("parent_code") or "").strip()
                kind = str(it.get("kind") or "").strip().lower()
                title = str(it.get("title") or "").strip()
                first_page = int(it.get("first_page") or 0)
                anchor = str(it.get("anchor_code") or "").strip()
                snippet = str(it.get("snippet") or "").strip()
                if not parent or not kind or first_page <= 0:
                    continue
                anchor = re.sub(r"[^A-Za-z0-9.]", "", anchor or "")
                if not anchor or not anchor.startswith(parent):
                    prefix = {
                        "definition": "def",
                        "example": "ex",
                        "table": "tbl",
                        "figure": "fig",
                        "chart": "chart",
                        "box": "box",
                        "note": "note",
                    }.get(kind, "obj")
                    anchor = f"{parent}.{prefix}1"
                objs_out.append({
                    "parent_code": parent,
                    "kind": kind,
                    "title": title,
                    "first_page": first_page,
                    "anchor_code": anchor,
                    "snippet": snippet,
                })
            except Exception:
                continue
        data["objects"] = objs_out
        data["game_name"] = str(data.get("game_name") or "").strip()
        return data

    # Candidate-based batching across entire document (LLM-first gather)
    def _llm_gather_candidates(all_pages: List[str]) -> List[Tuple[int, str]]:
        from langchain.schema import HumanMessage, SystemMessage  # type: ignore
        # Build page-preserving segments so no line is cut off mid-way
        SEGMENT_CHARS = 1600
        segments: List[Tuple[int, int, str]] = []  # (page, part_idx, text)
        part_counts: Dict[int, int] = {}
        for i, text in enumerate(all_pages, start=1):
            s = (text or "").strip()
            if not s:
                continue
            buf_lines: List[str] = []
            buf_len = 0
            part_idx = 1
            for ln in s.splitlines():
                ln2 = (ln or "")
                # flush if adding this line would exceed segment size
                if buf_len and (buf_len + len(ln2) + 1 > SEGMENT_CHARS):
                    segments.append((i, part_idx, "\n".join(buf_lines)))
                    part_idx += 1
                    buf_lines = []
                    buf_len = 0
                buf_lines.append(ln2)
                buf_len += len(ln2) + 1
            if buf_lines:
                segments.append((i, part_idx, "\n".join(buf_lines)))

        # Pack segments into batches by character budget
        batches_pages: List[List[Tuple[int, int, str]]] = []
        cur: List[Tuple[int, int, str]] = []
        MAX_BATCH_CHARS = 24000
        cur_chars = 0
        for tup in segments:
            page_text = tup[2]
            if cur and (cur_chars + len(page_text) + 32 > MAX_BATCH_CHARS):
                batches_pages.append(cur)
                cur = []
                cur_chars = 0
            cur.append(tup)
            cur_chars += len(page_text) + 32
        if cur:
            batches_pages.append(cur)

        # Prompt template for candidate discovery
        usr_find = (
            "Return JSON ONLY. You are given page texts and must find lines that are LIKELY section headers.\n"
            "Output: {candidates: [{page:int, line:string}]}\n"
            "Rules:\n"
            "- Consider ONLY lines that START with a visible section code: numeric dotted (e.g., 3.5, 5.6.1), letter-first (F4, F4.a, A10.2b), digit-first (1B6, 1B6b).\n"
            "- Return the full original line as 'line' (do NOT truncate).\n"
            "- Ignore non-header lines.\n\n"
            "Input pages (format === Page N ===):\n{pages}\n\n"
            "Output schema: {\"candidates\": [{\"page\": int, \"line\": string}]}"
        )

        out_cands: List[Tuple[int, str]] = []
        try:
            llm = _make_llm()
        except Exception as e:
            try:
                print(f"⚠️  LLM candidate gather init error: {e}")
            except Exception:
                pass
            return []

        for bi, pg_block in enumerate(batches_pages, start=1):
            pages_block = "\n\n".join([(
                f"=== Page {p} part {k} ===\n{t}" if k > 1 else f"=== Page {p} ===\n{t}"
            ) for (p, k, t) in pg_block])
            usr = usr_find.replace("{pages}", pages_block)
            try:
                out = llm.invoke([SystemMessage(content=_SYS), HumanMessage(content=usr)])
                s = str(getattr(out, "content", "") or "").strip()
            except Exception as e:
                try:
                    print(f"⚠️  LLM candidate gather batch {bi} error: {e}")
                except Exception:
                    pass
                continue
            if debug_dir:
                try:
                    p = pathlib.Path(debug_dir) / f"find_{bi:02d}"
                    p.mkdir(parents=True, exist_ok=True)
                    (p / "system.txt").write_text(_SYS, encoding="utf-8")
                    (p / "user.txt").write_text(usr, encoding="utf-8")
                    (p / "raw.txt").write_text(s, encoding="utf-8")
                except Exception:
                    pass
            s_clean = _strip_code_fences(s)
            objs = _find_json_objects(s_clean)
            data_b = None
            for js in objs[::-1]:
                try:
                    tmp = json.loads(js)
                except Exception:
                    continue
                if isinstance(tmp, dict) and "candidates" in tmp:
                    data_b = tmp
                    break
            if data_b is None:
                continue
            if debug_dir:
                try:
                    p = pathlib.Path(debug_dir) / f"find_{bi:02d}"
                    (p / "parsed.json").write_text(json.dumps(data_b, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
            for it in (data_b.get("candidates") or []):
                try:
                    pg = int(it.get("page"))
                    line = str(it.get("line") or "").strip()
                    if not pg or not line:
                        continue
                    out_cands.append((pg, line))
                except Exception:
                    continue
        # Deduplicate
        seen = set()
        dedup: List[Tuple[int, str]] = []
        for tup in out_cands:
            if tup not in seen:
                seen.add(tup)
                dedup.append(tup)
        return dedup

    # If Anthropic PDF direct mode is enabled, send the whole PDF in one pass
    if os.getenv("OUTLINE_PDF_DIRECT", "0") in ("1", "true", "True"):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        model = os.getenv("OUTLINE_LLM_MODEL", "claude-sonnet-4-20250514")
        if not api_key:
            # Fallback to normal flow if API key missing
            pass
        else:
            try:
                # Use Files API (Option 3) for repeated queries
                file_id = upload_pdf_to_anthropic_files(api_key, pdf_path)
                usr = _USR_TEMPLATE.replace("{pages}", "(PDF attached via file_id)")
                raw = anthropic_pdf_messages_with_file(api_key, model, _SYS, usr, file_id)
                s_clean = _strip_code_fences(raw)
                objs = _find_json_objects(s_clean)
                data: Dict[str, Any] | None = None
                for js in objs[::-1]:
                    try:
                        tmp = json.loads(js)
                    except Exception:
                        continue
                    if isinstance(tmp, dict) and ("sections" in tmp or "alias_map" in tmp or "game_name" in tmp):
                        data = tmp
                        break
                if isinstance(data, dict):
                    if debug_dir:
                        try:
                            p = pathlib.Path(debug_dir)
                            p.mkdir(parents=True, exist_ok=True)
                            (p / "system.txt").write_text(_SYS, encoding="utf-8")
                            (p / "user.txt").write_text(usr, encoding="utf-8")
                            (p / "raw.txt").write_text(raw, encoding="utf-8")
                            (p / "parsed.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                    # Normalize minimal like fallback path
                    secs = []
                    seen = set()
                    for it in (data.get("sections") or []):
                        try:
                            code = str(it.get("code") or "").strip()
                            title = str(it.get("title") or "").strip()
                            first_page = int(it.get("first_page") or 0)
                            kind = str(it.get("section_kind") or "").strip().lower()
                            if not code or first_page <= 0:
                                continue
                            if code in seen:
                                continue
                            seen.add(code)
                            secs.append({"code": code, "title": title, "first_page": first_page, "section_kind": kind})
                        except Exception:
                            continue
                    data["sections"] = sorted(secs, key=lambda x: x.get("first_page", 1e9))
                    if not isinstance(data.get("alias_map"), dict):
                        data["alias_map"] = {}
                    if not isinstance(data.get("glossary_codes"), list):
                        data["glossary_codes"] = []
                    if not isinstance(data.get("objects"), list):
                        data["objects"] = []
                    data["game_name"] = str(data.get("game_name") or "").strip()
                    return data
            except Exception as e:
                try:
                    print(f"⚠️  Anthropic PDF direct mode failed: {e}")
                except Exception:
                    pass
    # First try LLM-based candidate gathering across the full document
    try:
        cands = _llm_gather_candidates(pages)
    except Exception:
        cands = []
    # If LLM gather failed entirely, fall back to LLM-first pages-only outline (still no regex)
    if not cands:
        return _llm_first_pages_outline(pages)
    try:
        llm = _make_llm()
    except Exception as e:
        try:
            print(f"⚠️  LLM outline init error: {e}")
        except Exception:
            pass
        return {"sections": [], "alias_map": {}, "glossary_codes": []}

    # Prepare candidate lines as 'p N: text'
    cand_lines = [f"p {p}: {s}" for (p, s) in cands]
    batches: List[List[str]] = []
    cur: List[str] = []
    max_lines = 800
    max_chars = 24000
    cur_chars = 0
    for line in cand_lines:
        if (len(cur) >= max_lines) or (cur_chars + len(line) + 1 > max_chars):
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(line)
        cur_chars += len(line) + 1
    if cur:
        batches.append(cur)

    merged_sections: Dict[str, Dict[str, Any]] = {}
    merged_alias: Dict[str, str] = {}
    merged_objects: List[Dict[str, Any]] = []
    game_name: str = ""

    for bi, chunk in enumerate(batches, start=1):
        candidates_block = "\n".join(chunk)
        usr = _USR_CANDIDATES_TEMPLATE.replace("{candidates}", candidates_block)
        try:
            out = llm.invoke([SystemMessage(content=_SYS), HumanMessage(content=usr)])
            s = str(getattr(out, "content", "") or "").strip()
        except Exception as e:
            try:
                print(f"⚠️  LLM outline batch {bi} error: {e}")
            except Exception:
                pass
            continue
        if debug_dir:
            try:
                p = pathlib.Path(debug_dir) / f"batch_{bi:02d}"
                p.mkdir(parents=True, exist_ok=True)
                (p / "system.txt").write_text(_SYS, encoding="utf-8")
                (p / "user.txt").write_text(usr, encoding="utf-8")
                (p / "raw.txt").write_text(s, encoding="utf-8")
            except Exception:
                pass
        # Parse JSON from batch
        s_clean = _strip_code_fences(s)
        objs = _find_json_objects(s_clean)
        data_b: Dict[str, Any] | None = None
        for js in objs[::-1]:
            try:
                tmp = json.loads(js)
            except Exception:
                continue
            if isinstance(tmp, dict) and ("sections" in tmp or "alias_map" in tmp or "game_name" in tmp):
                data_b = tmp
                break
        if data_b is None:
            continue
        if debug_dir:
            try:
                p = pathlib.Path(debug_dir) / f"batch_{bi:02d}"
                (p / "parsed.json").write_text(json.dumps(data_b, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        # Merge sections
        for it in (data_b.get("sections") or []):
            try:
                code = str(it.get("code") or "").strip()
                title = str(it.get("title") or "").strip()
                first_page = int(it.get("first_page") or 0)
                kind = str(it.get("section_kind") or "").strip().lower()
                if not code or first_page <= 0:
                    continue
                prev = merged_sections.get(code)
                if prev is None or first_page < int(prev.get("first_page") or 10**9):
                    merged_sections[code] = {"code": code, "title": title, "first_page": first_page, "section_kind": kind}
            except Exception:
                continue
        # Merge alias
        amap = data_b.get("alias_map") or {}
        if isinstance(amap, dict):
            for k, v in amap.items():
                if isinstance(k, str) and isinstance(v, str):
                    merged_alias[k] = v
        # Merge objects
        for it in (data_b.get("objects") or []):
            if isinstance(it, dict):
                merged_objects.append(it)
        # Game name
        if not game_name:
            game_name = str(data_b.get("game_name") or "").strip()

    # Finalize merged output
    final_sections = sorted(merged_sections.values(), key=lambda x: x.get("first_page", 1e9))
    # Filter alias map to existing codes
    final_alias = {k: v for k, v in merged_alias.items() if v in {s["code"] for s in final_sections}}
    # Normalize objects
    objs_out: List[Dict[str, Any]] = []
    for it in merged_objects:
        try:
            parent = str(it.get("parent_code") or "").strip()
            kind = str(it.get("kind") or "").strip().lower()
            title = str(it.get("title") or "").strip()
            first_page = int(it.get("first_page") or 0)
            anchor = str(it.get("anchor_code") or "").strip()
            snippet = str(it.get("snippet") or "").strip()
            if not parent or not kind or first_page <= 0:
                continue
            anchor = re.sub(r"[^A-Za-z0-9.]", "", anchor or "")
            if not anchor or not anchor.startswith(parent):
                prefix = {
                    "definition": "def",
                    "example": "ex",
                    "table": "tbl",
                    "figure": "fig",
                    "chart": "chart",
                    "box": "box",
                    "note": "note",
                }.get(kind, "obj")
                anchor = f"{parent}.{prefix}1"
            if parent in {s["code"] for s in final_sections}:
                objs_out.append({
                    "parent_code": parent,
                    "kind": kind,
                    "title": title,
                    "first_page": first_page,
                    "anchor_code": anchor,
                    "snippet": snippet,
                })
        except Exception:
            continue
    out_data = {
        "game_name": game_name,
        "sections": final_sections,
        "alias_map": final_alias,
        "glossary_codes": list({}),
        "objects": objs_out,
    }
    # Refinement pass: ask LLM to add any sections present in candidates but missing in current sections
    try:
        if cand_lines:
            existing_codes_csv = ", ".join([s["code"] for s in final_sections])
            refine_usr = (
                "Return JSON ONLY.\n\n"
                "You are given candidate header lines (format 'p N: ...') and a list of codes already extracted.\n"
                "Find any ADDITIONAL sections present in the candidates but NOT in the provided code list.\n"
                "For each new section include: code, title (text after code, use substring BEFORE first ':' if present), first_page (smallest page), section_kind.\n\n"
                f"Existing codes (do NOT return these): {existing_codes_csv}\n\n"
                f"Candidates:\n{chr(10).join(cand_lines)}\n\n"
                "Output schema: {\"sections\": [{\"code\": string, \"title\": string, \"first_page\": int, \"section_kind\": string}]}"
            )
            out = llm.invoke([SystemMessage(content=_SYS), HumanMessage(content=refine_usr)])
            s = str(getattr(out, "content", "") or "").strip()
            if debug_dir:
                try:
                    p = pathlib.Path(debug_dir) / "refine"
                    p.mkdir(parents=True, exist_ok=True)
                    (p / "system.txt").write_text(_SYS, encoding="utf-8")
                    (p / "user.txt").write_text(refine_usr, encoding="utf-8")
                    (p / "raw.txt").write_text(s, encoding="utf-8")
                except Exception:
                    pass
            s_clean = _strip_code_fences(s)
            objs = _find_json_objects(s_clean)
            data_r: Dict[str, Any] | None = None
            for js in objs[::-1]:
                try:
                    tmp = json.loads(js)
                except Exception:
                    continue
                if isinstance(tmp, dict) and ("sections" in tmp):
                    data_r = tmp
                    break
            if isinstance(data_r, dict):
                add = []
                seen_codes = {s["code"] for s in final_sections}
                for it in (data_r.get("sections") or []):
                    try:
                        code = str(it.get("code") or "").strip()
                        title = str(it.get("title") or "").strip()
                        first_page = int(it.get("first_page") or 0)
                        kind = str(it.get("section_kind") or "").strip().lower()
                        if not code or first_page <= 0 or code in seen_codes:
                            continue
                        add.append({"code": code, "title": title, "first_page": first_page, "section_kind": kind})
                        seen_codes.add(code)
                    except Exception:
                        continue
                if add:
                    final_sections.extend(add)
                    final_sections = sorted(final_sections, key=lambda x: x.get("first_page", 1e9))
                    out_data["sections"] = final_sections
                    if debug_dir:
                        try:
                            p = pathlib.Path(debug_dir)
                            (p / "parsed.json").write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
                        except Exception:
                            pass
    except Exception:
        pass
    if debug_dir:
        try:
            p = pathlib.Path(debug_dir)
            (p / "parsed.json").write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return out_data


