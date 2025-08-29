"""
Python script that queries the RAG model with a given query and returns the response.

The script connects to the database, searches for the query in the vector
store, builds the prompt, and generates the response using OpenAI's chat
completion API.

The response includes the generated text, sources, original query, and context.

The script can be run from the command line with the query_text argument.

Example:
    python query_rag.py --query_text "What is the capital of France?"

The script can also include sources and context in the response using the include_sources and include_context arguments.

Example:
    python query_rag.py --query_text "What is the capital of France?" --include_sources --include_context
"""

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document
# config imports extended
from .config import (
    validate_config,
    ENABLE_WEB_SEARCH,
    WEB_SEARCH_RESULTS,
    SEARCH_PROVIDER,
    SERPAPI_API_KEY,
    BRAVE_API_KEY,
    ENABLE_SEARCH_REWRITE,
)
from templates.load_jinja_template import load_jinja2_prompt
from . import config as cfg
from .llm_outline_helpers import (
    upload_pdf_to_anthropic_files,
    anthropic_pdf_messages_with_file,
    find_json_objects,
)
from .pdf_utils import find_header_anchor

# Optional import for web search; only loaded when enabled to avoid extra dependency at runtime
if ENABLE_WEB_SEARCH:
    try:
        from duckduckgo_search import ddg
    except ImportError:  # Fallback if dependency missing
        ddg = None

from . import config as cfg

# ---------------------------------------------------------------------------
# Verbose logging toggle (set VERBOSE=true to enable extra prints)
# ---------------------------------------------------------------------------
VERBOSE_LOGGING = os.getenv("VERBOSE", "False").lower() in {"1", "true", "yes"}
# Performance knobs (env-tunable)
RAG_MAX_DOCS = int(os.getenv("RAG_MAX_DOCS", "12"))  # how many top docs to include
RAG_CONTEXT_CHAR_LIMIT = int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", "8000"))  # cap context size
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))  # cap model output for quicker replies
REQUEST_TIMEOUT_S = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
STREAM_TIMEOUT_S = float(os.getenv("LLM_STREAM_TIMEOUT", "90"))


def perform_web_search(query: str, k: int = 5) -> List[Document]:
    from .retrieval.web_search import perform_web_search as _impl
    return _impl(query, k)
def _openai_requires_default_temperature(model_name: str) -> bool:
    from .retrieval.llm_utils import openai_requires_default_temperature as _impl
    return _impl(model_name)



def _basic_plural_variants(token: str) -> List[str]:
    """Return simple singular/plural variants for a token.

    This is a lightweight heuristic to improve recall without external NLP deps.
    The original token is always included first; duplicates are removed by caller.
    """
    try:
        t = token.strip()
        if not t or any(ch for ch in t if not ch.isalpha()):
            return [t]
        lower = t.lower()
        variants = [t]
        # Singularize basic forms
        if lower.endswith("ies") and len(lower) > 3:
            singular = lower[:-3] + "y"
            variants.append(singular)
        elif lower.endswith("es") and len(lower) > 2:
            # handle boxes → box, classes → class
            variants.append(lower[:-2])
        elif lower.endswith("s") and len(lower) > 1:
            variants.append(lower[:-1])
        # Pluralize basic forms
        if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
            variants.append(lower[:-1] + "ies")
        elif lower.endswith(("s", "x", "z", "ch", "sh")):
            variants.append(lower + "es")
        else:
            variants.append(lower + "s")
        # Deduplicate while preserving order
        seen: set[str] = set()
        out: List[str] = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out
    except Exception:
        return [token]


def generate_query_variants(query_text: str, game_names: Optional[List[str]] = None) -> List[str]:
    from .retrieval.prompt_utils import generate_query_variants as _impl
    return _impl(query_text, game_names)


def normalize_game_title(title: str) -> str:
    from .retrieval.game_names import normalize_game_title as _impl
    return _impl(title)


def extract_game_name_from_filename(filename: str, debug: bool = False) -> str:
    """
    Use LLM API to extract the proper game name from a PDF filename.
    Also reads the first few pages of the PDF for better accuracy.

    Args:
        filename (str): PDF filename like "up front rulebook bw.pdf"
        debug (bool): Whether to print the prompt sent to the LLM

    Returns:
        str: Cleaned game name like "Up Front"
    """
    import time
    import random

    # Try to extract text from first few pages for better context
    pdf_context = ""
    try:
        # Construct full path to PDF
        from .config import DATA_PATH

        pdf_path = Path(DATA_PATH) / filename

        if pdf_path.exists():
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            # Get text from first 10 pages (or fewer if PDF is shorter)
            import os

            # Allow overriding via env var NAME_EXTRACTION_PAGES (default 20)
            pages_limit = int(os.getenv("NAME_EXTRACTION_PAGES", 10))
            pages_to_read = min(pages_limit, len(documents))

            page_texts = []

            for i in range(pages_to_read):
                page_text = documents[i].page_content.strip()
                if page_text:
                    # Take first 500 chars per page to keep context manageable
                    page_texts.append(page_text[:500])

            if page_texts:
                pdf_context = "\n\n".join(page_texts)
                print(
                    f"📖 Extracted {len(pdf_context)} chars from first {pages_to_read} pages of {filename}"
                )
            else:
                print(f"⚠️ No readable text found in first pages of {filename}")
        else:
            print(f"⚠️ PDF file not found: {pdf_path}")
    except Exception as e:
        print(f"⚠️ Could not read PDF content from {filename}: {e}")
        pdf_context = ""

    # Retry configuration
    max_retries = 3
    base_delay = 1

    last_raw_response = None  # For debugging fallback

    for attempt in range(max_retries):
        try:
            # Use configured provider for filename extraction (strict, no normalization)
            _prov = (cfg.LLM_PROVIDER or "").lower()
            if _prov == "anthropic":
                model = ChatAnthropic(model=cfg.GENERATOR_MODEL, temperature=0)
            elif _prov == "ollama":
                from langchain_community.llms.ollama import Ollama

                model = Ollama(model=cfg.GENERATOR_MODEL, base_url=cfg.OLLAMA_URL)
            elif _prov == "openrouter":
                from langchain_openai import ChatOpenAI as _OpenRouterChat
                base_url = getattr(cfg, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                api_key = getattr(cfg, "OPENROUTER_API_KEY", None)
                model = _OpenRouterChat(model=cfg.GENERATOR_MODEL, temperature=0, timeout=REQUEST_TIMEOUT_S, base_url=base_url, api_key=api_key)
            else:  # openai
                if _openai_requires_default_temperature(cfg.GENERATOR_MODEL):
                    model = ChatOpenAI(model=cfg.GENERATOR_MODEL, temperature=1, timeout=REQUEST_TIMEOUT_S)
                else:
                    model = ChatOpenAI(model=cfg.GENERATOR_MODEL, temperature=0, timeout=REQUEST_TIMEOUT_S)

            # Build prompt with optional PDF content
            context_section = ""
            if pdf_context:
                context_section = f"""

CONTENT FROM FIRST PAGES:
{pdf_context}

"""

            prompt = f""" Extract the proper board game name from this filename: "{filename}"{context_section}

Guidelines:
- Return ONLY the official published game name with no preamble or formatting.
- Remove file-related words: "rules", "manual", "rulebook", "complete", "rework", "bw", "color", "v1", "v2"
- If an acronym, find a possible name from the PDF content
- Use proper capitalization for official game titles
- If you see the actual game title in the PDF content, prefer that over filename guessing

Filename: {filename}
Official game name:"""

            if debug:
                print("\n===== PROMPT SENT TO LLM =====\n")
                print(prompt)
                print("\n==============================\n")

            response = model.invoke(prompt)
            last_raw_response = getattr(response, "content", str(response))

            if debug:
                print("\n===== RAW LLM RESPONSE =====\n")
                print(last_raw_response)
                print("\n============================\n")

            game_name = last_raw_response.strip().strip("\"'")

            # If response contains multiple lines, just take the first line
            if "\n" in game_name:
                game_name = game_name.split("\n")[0].strip()

            # Basic validation
            if game_name and len(game_name) <= 50:
                # Normalize the title by moving leading articles to the end
                normalized_name = normalize_game_title(game_name)
                print(
                    f"Successfully extracted game name: '{normalized_name}' from '{filename}'"
                )
                return normalized_name
            else:
                raise ValueError("Invalid response")

        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's an overload error that we should retry
            if any(
                keyword in error_msg
                for keyword in ["overload", "rate limit", "529", "503"]
            ):
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    # Exponential backoff with jitter
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"API overloaded (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    print(
                        f"API overloaded after {max_retries} attempts, falling back to literal filename"
                    )
            else:
                print(f"LLM extraction failed for {filename}: {e}")

            # If we get here, all retries failed or it's not a retryable error
            break

    # Fallback: just use the literal filename
    fallback_name = (
        filename.replace(".pdf", "").replace("-", " ").replace("_", " ").title()
    )
    # Normalize the fallback name as well
    normalized_fallback = normalize_game_title(fallback_name)
    
    if debug and last_raw_response is not None:
        print("\n❌ Using fallback – last raw LLM response was:\n")
        print(last_raw_response)
        print()

    print(f"Using fallback name: '{normalized_fallback}' for '{filename}'")
    return normalized_fallback


def improve_fallback_name(filename: str) -> str:
    from .retrieval.game_names import improve_fallback_name as _impl
    return _impl(filename)


def get_available_games() -> List[str]:
    # Vector-only mode: use stored game names
    from .retrieval.game_names import get_available_games as _impl
    return _impl()


def store_game_name(filename: str, game_name: str):
    from .retrieval.game_names import store_game_name as _impl
    return _impl(filename, game_name)


def get_stored_game_names() -> Dict[str, str]:
    from .retrieval.game_names import get_stored_game_names as _impl
    return _impl()


def extract_and_store_game_name(filename: str) -> str:
    from .retrieval.game_names import extract_and_store_game_name as _impl
    return _impl(filename)


def rewrite_search_query(raw_query: str) -> str:
    from .retrieval.web_search import rewrite_search_query as _impl
    return _impl(raw_query)


## query_rag was deprecated in favor of stream_query_rag and has been removed.


# ---------------------------------------------------------------------------
# Streaming variant – yields tokens incrementally for UI streaming
# ---------------------------------------------------------------------------


def dedupe_then_sort_results(results: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    """Canonical dedupe/sort policy shared by app and tools.

    - Group by (metadata.source, section_id_2) if available; else by (source, first primary section id)
    - Prefer chunks whose page equals section_pages[section_id] (when grouping by code)
    - Break ties by lower score (lower is better)
    - Sort ascending by score
    """
    deduped: List[Tuple[Any, float]] = []
    seen: Dict[Tuple[str, str], Tuple[Any, float, bool]] = {}
    for doc, score in results:
        meta = getattr(doc, 'metadata', {}) or {}
        src = str(meta.get('source') or '')

        # Prefer new section_id_2 for grouping (no fallback in grouping key when present)
        sec_id2_map = {}
        try:
            import json as _json
            raw = meta.get('section_id2_by_code')
            if isinstance(raw, str) and raw:
                sec_id2_map = _json.loads(raw)
            elif isinstance(raw, dict):
                sec_id2_map = raw
        except Exception:
            sec_id2_map = {}

        # Determine grouping key
        group_id = ''
        # If we have a primary code, use its id2
        try:
            primary_id2 = str(meta.get('primary_section_id2') or '').strip()
            if primary_id2:
                group_id = primary_id2
        except Exception:
            group_id = ''

        # Fallback only if id2 is entirely unavailable: use first primary section id
        if not group_id:
            sec_ids: List[str] = []
            try:
                import json as _json
                prim_raw = meta.get('primary_sections')
                if isinstance(prim_raw, str) and prim_raw:
                    sec_ids = _json.loads(prim_raw) or []
                elif isinstance(prim_raw, list):
                    sec_ids = prim_raw
            except Exception:
                pass
            section_id = str(sec_ids[0]) if sec_ids else ''
            group_id = section_id

        key = (src, group_id)
        if not group_id:
            deduped.append((doc, score))
            continue

        # Preference by page match when grouping by canonical code (attempt via section_pages)
        sp_raw = meta.get('section_pages')
        section_page_map: Dict[str, Any] = {}
        try:
            import json as _json
            if isinstance(sp_raw, str) and sp_raw:
                section_page_map = _json.loads(sp_raw)
            elif isinstance(sp_raw, dict):
                section_page_map = sp_raw
        except Exception:
            section_page_map = {}

        # Try to resolve desired page using the first primary section id if available
        desired_page = 0
        try:
            import json as _json
            prim_raw = meta.get('primary_sections')
            prim_list: List[str] = []
            if isinstance(prim_raw, str) and prim_raw:
                prim_list = _json.loads(prim_raw) or []
            elif isinstance(prim_raw, list):
                prim_list = prim_raw
            if prim_list:
                h = str(prim_list[0])
                desired_page = int((section_page_map or {}).get(h) or 0)
        except Exception:
            desired_page = 0

        try:
            doc_page = int(meta.get('page_1based') or (int(meta.get('page') or 0) + 1))
        except Exception:
            try:
                doc_page = int(meta.get('page') or 0) + 1
            except Exception:
                doc_page = 0
        ok = bool(desired_page and doc_page == desired_page)

        prev = seen.get(key)
        if not prev:
            seen[key] = (doc, score, ok)
        else:
            _, prev_score, prev_ok = prev
            if ok and not prev_ok:
                seen[key] = (doc, score, ok)
            elif ok == prev_ok and score < prev_score:
                seen[key] = (doc, score, ok)

    if seen:
        deduped.extend((d, s) for (d, s, _ok) in seen.values())
    try:
        deduped.sort(key=lambda pair: float(pair[1]))
    except Exception:
        pass
    return deduped


def stream_query_rag(
    query_text: str,
    selected_game: Optional[str] = None,
    chat_history: Optional[str] = None,
    game_names: Optional[List[str]] = None,
    enable_web: Optional[bool] = None,
):
    """Same as query_rag but returns a (token_generator, metadata) tuple.

    The *token_generator* yields strings (token chunks). *metadata* is a dict
    identical to the non-streaming response (minus *response_text*).
    """

    print("\n" + "="*80)
    print("🔍 STREAM_QUERY_RAG DEBUG START")
    print("="*80)
    print(f"📝 Query text: '{query_text}'")
    print(f"🎮 Selected game: {selected_game}")
    print(f"🎮 Game names: {game_names}")
    print(f"💬 Chat history length: {len(chat_history) if chat_history else 0}")
    print(f"🌐 Web search enabled: {enable_web}")

    # ---- Vector-DB mode (exclusive) ----
    use_vector = True
    if use_vector:
        try:
            from .vector_store import count_processed_pages  # type: ignore
        except Exception as e:
            def _err_gen():
                yield f"Vector DB unavailable: {e}"
            meta = {"sources": [], "context": "", "prompt": "", "original_query": query_text, "spans": []}
            return _err_gen(), meta

        # Resolve one or more PDF filenames for the selected game (handle str or list)
        pdf_filenames: list[str] = []
        if selected_game:
            try:
                def _norm_game(g):
                    if isinstance(g, list):
                        return (g[0] if g else "").strip()
                    if isinstance(g, str):
                        return g.strip()
                    return str(g or "").strip()
                norm_game = _norm_game(selected_game)
                if norm_game:
                    from .catalog import get_pdf_filenames_for_game  # type: ignore
                    pdf_filenames = list(get_pdf_filenames_for_game(norm_game) or [])
            except Exception:
                pdf_filenames = []
        # Keep retrieval lightweight and predictable per request
        # Increase k to reduce sensitivity to small rank shifts during reranking
        k_results = 12
        results: list[tuple[Any, float]] = []
        # Section-only retrieval
        try:
            from .vector_store import search_section_chunks  # type: ignore
        except Exception:
            search_section_chunks = None  # type: ignore
        if pdf_filenames:
            for fn in pdf_filenames:
                if search_section_chunks is not None:
                    results.extend(search_section_chunks(query_text, pdf=fn, k=k_results))
        else:
            if selected_game:
                results = []
            else:
                if search_section_chunks is not None:
                    results = search_section_chunks(query_text, pdf=None, k=k_results)
                else:
                    results = []
        # Apply canonical dedupe-then-sort policy
        results = dedupe_then_sort_results(results)

        # Decision: for each chunk, if visual_importance >= 4 attach PDFs; else use text
        from pathlib import Path as _P
        from . import config as _cfg2  # type: ignore
        data_dir = _P(getattr(_cfg2, "DATA_PATH", "data"))

        # Precompute the exact instruction text (the same text block we send to the LLM)
        # Strip retrieval-only prefixes and summaries from stored chunk text
        def _strip_embed_prefixes(text: str) -> str:
            try:
                import re as _re
                lines = (text or "").splitlines()
                filtered: list[str] = []
                for ln in lines:
                    s = ln.lstrip()
                    if s.startswith("Search hints:"):
                        continue
                    if s.startswith("Headers:"):
                        continue
                    ls = s.lower()
                    if ls.startswith("summary:"):
                        continue
                    if ls.startswith("page summary:"):
                        continue
                    if ls.startswith("overview:"):
                        continue
                    if ls.startswith("key points:"):
                        continue
                    filtered.append(ln)
                return "\n".join(filtered).strip()
            except Exception:
                return text

        context_blocks: list[str] = []
        top_n = results[:6]
        for doc, score in top_n:
            meta = getattr(doc, 'metadata', {}) or {}
            source = str(meta.get('source') or '')
            base = _P(source).name
            ctx_raw = str(getattr(doc, 'page_content', '') or '')
            ctx = _strip_embed_prefixes(ctx_raw)
            code = str(meta.get('section_code') or '')
            fp = meta.get('first_page')
            suffix = f" {code}" if code else ""
            if isinstance(fp, int) and fp > 0:
                # Add machine-parseable citation line for this context block
                if code:
                    context_blocks.append(f"[Context from {base} p{fp}{suffix}]\nCitation: [{code}]\n{ctx}")
                else:
                    context_blocks.append(f"[Context from {base} p{fp}]\n{ctx}")
            else:
                if code:
                    context_blocks.append(f"[Context from {base}{suffix}]\nCitation: [{code}]\n{ctx}")
                else:
                    context_blocks.append(f"[Context from {base}]\n{ctx}")

        # We no longer include an explicit Allowed citations list in the prompt.
        # Anything included in the context is allowed; each context block will carry its own citation string.

        # Build structured allowed citations list for client-side resolution
        allowed_struct: list[dict[str, Any]] = []
        try:
            seen_pairs: set[tuple[str, str]] = set()
            for doc, _score in top_n:
                meta = getattr(doc, 'metadata', {}) or {}
                src = str(meta.get('source') or '')
                base_name = _P(src).name
                import json as _json
                # Fast-path for section-chunk metadata: emit a single entry per section
                try:
                    is_section = str(meta.get('chunk_kind') or '') == 'section'
                except Exception:
                    is_section = False
                if is_section:
                    try:
                        code = str(meta.get('section_code') or '').strip()
                        # Prefer explicit title; fall back to detected section_start; finally code
                        label = str(meta.get('section_title') or meta.get('section_start') or '').strip() or code
                        fp_val = meta.get('first_page')
                        page_val = int(fp_val) if isinstance(fp_val, int) and fp_val > 0 else None
                        key = (base_name, label or code)
                        if key not in seen_pairs:
                            entry: dict[str, Any] = {
                                "file": base_name,
                                "section": label,
                                "page": page_val,
                                "code": code,
                            }
                            # Attach anchor if present on section metadata under header_anchors_pct keyed by code
                            try:
                                import json as _json2
                                ha_raw = meta.get('header_anchors_pct')
                                anchors_obj = None
                                if isinstance(ha_raw, str) and ha_raw:
                                    anchors_obj = _json2.loads(ha_raw)
                                elif isinstance(ha_raw, dict):
                                    anchors_obj = ha_raw
                                if isinstance(anchors_obj, dict):
                                    arr = anchors_obj.get(code) or anchors_obj.get(label)
                                    if isinstance(arr, list) and len(arr) >= 4:
                                        entry["header_anchor_bbox_pct"] = [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
                            except Exception:
                                pass
                            allowed_struct.append(entry)
                            seen_pairs.add(key)
                    except Exception:
                        pass
                    # Do not attempt page-chunk-specific maps for section chunks
                    continue
                try:
                    sp = _json.loads(meta.get('section_pages') or '{}') or {}
                except Exception:
                    sp = {}
                try:
                    sid_map = _json.loads(meta.get('section_ids') or '{}') or {}
                except Exception:
                    sid_map = {}
                try:
                    anchors = _json.loads(meta.get('header_anchors_pct') or '{}') or {}
                except Exception:
                    anchors = {}
                try:
                    text_spans_data = _json.loads(meta.get('text_spans') or '{}') or {}
                except Exception:
                    text_spans_data = {}
                try:
                    prim = _json.loads(meta.get('primary_sections') or '[]') or []
                except Exception:
                    prim = []
                try:
                    p1_chunk = int(meta.get('page_1based') or (int(meta.get('page') or 0) + 1))
                except Exception:
                    p1_chunk = int(meta.get('page') or 0) + 1
                labels: set[str] = set()
                try:
                    labels.update([str(k) for k in (sp.keys() if isinstance(sp, dict) else [])])
                except Exception:
                    pass
                try:
                    labels.update([str(x) for x in (prim or [])])
                except Exception:
                    pass
                for hdr in sorted(labels):
                    if not hdr:
                        continue
                    key = (base_name, hdr)
                    if key in seen_pairs:
                        continue
                    page_val = None
                    try:
                        page_val = int(sp.get(hdr)) if isinstance(sp, dict) and hdr in sp else None
                    except Exception:
                        page_val = None
                    if not page_val and hdr in prim:
                        page_val = p1_chunk
                    entry: dict[str, Any] = {
                        "file": base_name,
                        "section": hdr,
                        "page": int(page_val) if isinstance(page_val, int) else None,
                        "code": str(sid_map.get(hdr) or ""),
                    }
                    try:
                        # Prefer anchors keyed by canonical section code; fall back to header for backward compat
                        code_key = str(sid_map.get(hdr) or "").strip()
                        arr = None
                        if code_key:
                            arr = anchors.get(code_key)
                        if arr is None:
                            arr = anchors.get(hdr)
                        if isinstance(arr, list) and len(arr) >= 4:
                            entry["header_anchor_bbox_pct"] = [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
                    except Exception:
                        pass
                    # Add text spans if available
                    try:
                        spans = text_spans_data.get(hdr)
                        if spans:
                            entry["text_spans"] = spans
                    except Exception:
                        pass
                    allowed_struct.append(entry)
                    seen_pairs.add(key)
        except Exception:
            allowed_struct = []
        instruction = (
            f"Question: {query_text}\n\n"
        )
        if context_blocks:
            instruction += "\n\n" + "\n\n".join(context_blocks)

        # Note: Do not sanitize or post-process model output; enforce formatting via prompt only.

        # No server-side injection anymore; client will use allowed_struct to resolve pages
        def _inject_pages(text: str) -> str:
            return text

        # Define system prompt once (used for all providers) so it can be returned via metadata
        system_prompt = (
            "You are an expert assistant for boardgame rulebooks. Provide concise answers with inline citations. "
            "Do not use any preambles or phrases that reference the rulebook or the context such as 'Based on the rulebook,' 'According to', 'Based on the provided material,' etc. "
            "Answer the user's question based ONLY on the attached material. "
            # "Answer concisely in a few short paragraphs. "
            "Every paragraph must be a single brief claim (1–2 short sentences maximum) that ends with exactly one inline citation of the form [<section>], placed immediately after the paragraph with no trailing text. "
            "Do not combine multiple sections into a single bracketed citation. "
            "Do not enclose a citation with parentheses or extra text (e.g., '(see [13.1])'). "
            "Use the exact citation string shown as 'Citation: [..]' in the context block for any material you used from that block. Do not invent or modify citations."
        )

        # Collect streaming validation/repair log (if enabled)
        repairs_log: list[dict[str, Any]] = []
        sv_acc: dict[str, str] = {"original": "", "repaired": ""}

        def _gen():
            # If provider is not Anthropic, run a single completion and stream slices
            try:
                from . import config as _cfg2  # type: ignore
            except Exception:
                class _Cfg2:
                    LLM_PROVIDER = "anthropic"
                    GENERATOR_MODEL = "claude-sonnet-4-20250514"
                    OLLAMA_URL = "http://localhost:11434"
                _cfg2 = _Cfg2()  # type: ignore
            provider = str(getattr(_cfg2, "LLM_PROVIDER", "anthropic")).lower()
            if provider != "anthropic":
                try:
                    if provider == "openai":
                        if _openai_requires_default_temperature(getattr(_cfg2, "GENERATOR_MODEL", "")):
                            model = ChatOpenAI(model=_cfg2.GENERATOR_MODEL, temperature=1, timeout=REQUEST_TIMEOUT_S)
                        else:
                            model = ChatOpenAI(model=_cfg2.GENERATOR_MODEL, temperature=0, timeout=REQUEST_TIMEOUT_S)
                    elif provider == "openrouter":
                        # Use OpenRouter base URL and key to access meta-llama/llama-3.3-70b-instruct
                        from langchain_openai import ChatOpenAI as _OpenRouterChat
                        base_url = getattr(_cfg2, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                        api_key = getattr(_cfg2, "OPENROUTER_API_KEY", None)
                        # Do not pass temperature if model rejects it (treat like OpenAI guard if needed)
                        model_name = getattr(_cfg2, "GENERATOR_MODEL", "meta-llama/llama-3.3-70b-instruct")
                        params = {"model": model_name, "timeout": REQUEST_TIMEOUT_S, "base_url": base_url, "api_key": api_key}
                        try:
                            # Many OpenRouter models accept temperature; default to 0 for determinism
                            params["temperature"] = 0
                        except Exception:
                            pass
                        model = _OpenRouterChat(**params)
                    else:
                        from langchain_community.llms.ollama import Ollama  # type: ignore
                        model = Ollama(model=_cfg2.GENERATOR_MODEL, base_url=getattr(_cfg2, "OLLAMA_URL", "http://localhost:11434"))
                    resp = model.invoke(instruction)
                    text = str(getattr(resp, "content", resp) or "")
                    text = _inject_pages(text)
                    def _raw_chunks():
                        slice_size = 64
                        for i in range(0, len(text), slice_size):
                            piece = text[i:i+slice_size]
                            if piece:
                                yield piece
                    for _out in _stream_with_validation(_raw_chunks(), allowed_struct, repairs_log, sv_acc, instruction):
                        yield _out
                except Exception as e:
                    yield f"Model error: {e}"
                return
            # Build a streaming Anthropic request, preferring Files API page file_ids for visual pages
            attach_paths: list[_P] = []
            attach_meta: list[tuple[str, int]] = []  # (pdf filename base, 1-based page)
            seen_attach = set()
            seen_meta = set()
            # Section-only: no page attachments

            if not context_blocks:
                yield "No relevant chunks found."
                return

            # Stream tokens using Files API page file_ids when available; otherwise fall back to base64 page attachments
            api_key = _cfg2.ANTHROPIC_API_KEY
            model = _cfg2.GENERATOR_MODEL
            # Resolve page file_ids from catalog
            page_file_ids: list[str] = []
            try:
                from .catalog import get_page_file_id as _get_page_fid  # type: ignore
            except Exception:
                def _get_page_fid(_fname: str, _p: int):  # type: ignore
                    return None
            for base_name, p1 in attach_meta:
                try:
                    fid = _get_page_fid(base_name, int(p1))
                except Exception:
                    fid = None
                if fid:
                    page_file_ids.append(fid)
            if page_file_ids:
                try:
                    from .llm_outline_helpers import anthropic_pdf_messages_with_file_stream as _file_stream  # type: ignore
                except Exception:
                    def _file_stream(*args, **kwargs):
                        return []  # type: ignore
                # Stream only once to avoid duplicate full answers from multiple page file_ids
                for fid in page_file_ids[:1]:
                    carry = ""
                    import re as _re
                    _incomplete = _re.compile(r"\[[^\]]*:\s*\{[^\]]*$", _re.DOTALL)
                    def _raw_chunks():
                        nonlocal carry
                        for delta in _file_stream(api_key, model, system_prompt, instruction, fid):
                            txt = str(delta or "")
                            if not txt:
                                continue
                            combined = carry + txt
                            replaced = _inject_pages(combined)
                            m = _incomplete.search(replaced)
                            if m:
                                cut = m.start()
                                out = replaced[:cut]
                                carry = replaced[cut:]
                            else:
                                out = replaced
                                carry = ""
                            if out:
                                yield out
                        if carry:
                            yield carry
                    for _out in _stream_with_validation(_raw_chunks(), allowed_struct, repairs_log, sv_acc, instruction):
                        yield _out
            else:
                # No page file_ids available; attempt to upload page PDFs now and persist
                new_fids: list[str] = []
                try:
                    from .llm_outline_helpers import upload_pdf_to_anthropic_files as _upload_page  # type: ignore
                except Exception:
                    def _upload_page(*args, **kwargs):  # type: ignore
                        return None
                try:
                    from .catalog import set_page_file_id as _set_page_fid, get_page_file_id as _get_page_fid  # type: ignore
                except Exception:
                    def _set_page_fid(*args, **kwargs):  # type: ignore
                        return None
                    def _get_page_fid(*args, **kwargs):  # type: ignore
                        return None
                try:
                    from .pdf_pages import compute_sha256 as _sha256  # type: ignore
                except Exception:
                    def _sha256(_p):  # type: ignore
                        return ""

                def _resolve_parent_pdf_name_from_page(pth: _P) -> str:
                    try:
                        # New layout: .../data/<PDF_STEM>/1_pdf_pages/pNNNN.pdf → <PDF_STEM>.pdf
                        cand1 = data_dir / (pth.parent.parent.name + ".pdf")
                        if cand1.exists():
                            return cand1.name
                        # Legacy layout: .../data/pages/<PDF_STEM>/pNNNN.pdf → <PDF_STEM>.pdf
                        cand2 = data_dir / (pth.parent.name + ".pdf")
                        if cand2.exists():
                            return cand2.name
                    except Exception:
                        pass
                    # Default best-effort
                    try:
                        return pth.parent.parent.name + ".pdf"
                    except Exception:
                        return pth.parent.name + ".pdf"

                for p in attach_paths:
                    try:
                        base_pdf_name = _resolve_parent_pdf_name_from_page(p)
                        # Support slugged filenames like <slug>_pNNNN.pdf and legacy pNNNN.pdf
                        try:
                            from .pdf_pages import parse_page_1based_from_name  # type: ignore
                            p1 = int(parse_page_1based_from_name(p.name))
                        except Exception:
                            # Fallback to best-effort legacy parse
                            stem = p.stem
                            p1 = int(stem[1:]) if stem.startswith("p") else 1
                        fid = _get_page_fid(base_pdf_name, p1)
                        if not fid:
                            fid = _upload_page(api_key, str(p))
                            if fid:
                                _set_page_fid(base_pdf_name, p1, fid, _sha256(p))
                        if fid:
                            new_fids.append(str(fid))
                    except Exception:
                        continue

                if new_fids:
                    try:
                        from .llm_outline_helpers import anthropic_pdf_messages_with_file_stream as _file_stream  # type: ignore
                    except Exception:
                        def _file_stream(*args, **kwargs):
                            return []  # type: ignore
                    # Stream only the first uploaded page file to avoid duplicate answers
                    for fid in new_fids[:1]:
                        carry = ""
                        import re as _re
                        _incomplete = _re.compile(r"\[[^\]]*:\s*\{[^\]]*$", _re.DOTALL)
                        def _raw_chunks():
                            nonlocal carry
                            for delta in _file_stream(api_key, model, system_prompt, instruction, fid):
                                txt = str(delta or "")
                                if not txt:
                                    continue
                                combined = carry + txt
                                replaced = _inject_pages(combined)
                                m = _incomplete.search(replaced)
                                if m:
                                    cut = m.start()
                                    out = replaced[:cut]
                                    carry = replaced[cut:]
                                else:
                                    out = replaced
                                    carry = ""
                                if out:
                                    yield out
                            if carry:
                                yield carry
                        for _out in _stream_with_validation(_raw_chunks(), allowed_struct, repairs_log, sv_acc, instruction):
                            yield _out
                else:
                    # Final fallback: stream with base64 page attachments
                    try:
                        from .llm_outline_helpers import anthropic_pdf_messages_with_pages_stream as _pages_stream  # type: ignore
                    except Exception:
                        def _pages_stream(*args, **kwargs):
                            return []  # type: ignore
                    carry = ""
                    import re as _re
                    _incomplete = _re.compile(r"\[[^\]]*:\s*\{[^\]]*$", _re.DOTALL)
                    def _raw_chunks():
                        nonlocal carry
                        for delta in _pages_stream(api_key, model, system_prompt, instruction, [str(p) for p in attach_paths]):
                            if not delta:
                                continue
                            combined = carry + str(delta)
                            replaced = _inject_pages(combined)
                            m = _incomplete.search(replaced)
                            if m:
                                cut = m.start()
                                out = replaced[:cut]
                                carry = replaced[cut:]
                            else:
                                out = replaced
                                carry = ""
                            if out:
                                yield out
                        if carry:
                            yield carry
                    for _out in _stream_with_validation(_raw_chunks(), allowed_struct, repairs_log, sv_acc, instruction):
                        yield _out

        # Build metadata for developer inspection
        try:
            # Construct a compact sources list from retrieved results
            srcs = []
            top_n = results[:6]
            for doc, _s in top_n:
                try:
                    m = getattr(doc, 'metadata', {}) or {}
                    srcs.append({"filepath": m.get("source", "")})
                except Exception:
                    continue
            # Include the exact text context we appended to the instruction
            ctx_text = "\n\n".join(context_blocks) if context_blocks else ""
            # Include retrieved chunks payload for inspection
            included_chunks_payload = []
            for doc, _score in top_n:
                try:
                    meta_doc = getattr(doc, 'metadata', {}) or {}
                    is_section = str(meta_doc.get('chunk_kind') or '') == 'section'
                    from pathlib import Path as _Path
                    src_name = _Path((meta_doc.get("source") or "")).name or (meta_doc.get("source") or "")
                    # Extract labels from primary/continuation sections on this chunk
                    try:
                        import json as _json
                        prim = _json.loads(meta_doc.get("primary_sections") or '[]') or []
                    except Exception:
                        prim = []
                    try:
                        import json as _json
                        cont = _json.loads(meta_doc.get("continuation_sections") or '[]') or []
                    except Exception:
                        cont = []
                    # Include ALL section headers that resolve to this chunk's page
                    labels = [str(x).strip() for x in list(dict.fromkeys(list(prim) + list(cont))) if str(x).strip()]
                    try:
                        import json as _json
                        sp_map = _json.loads(meta_doc.get('section_pages') or '{}') or {}
                    except Exception:
                        sp_map = {}
                    try:
                        p0 = int(meta_doc.get('page') or 0)
                    except Exception:
                        p0 = 0
                    p1_num = p0 + 1
                    try:
                        for hdr, pg in (sp_map.items() if isinstance(sp_map, dict) else []):
                            try:
                                if int(pg) == int(p1_num):
                                    s = str(hdr or '').strip()
                                    if s and s not in labels:
                                        labels.append(s)
                            except Exception:
                                continue
                    except Exception:
                        pass
                    # text_spans is stored per-header map in metadata; pass it forward
                    text_spans_map = None
                    try:
                        import json as _json
                        ts_raw = meta_doc.get('text_spans')
                        if isinstance(ts_raw, str) and ts_raw:
                            text_spans_map = _json.loads(ts_raw)
                        elif isinstance(ts_raw, dict):
                            text_spans_map = ts_raw
                    except Exception:
                        text_spans_map = None
                    # header_anchors_pct per-header map
                    header_anchors_pct = None
                    try:
                        import json as _json
                        ha_raw = meta_doc.get('header_anchors_pct')
                        if isinstance(ha_raw, str) and ha_raw:
                            header_anchors_pct = _json.loads(ha_raw)
                        elif isinstance(ha_raw, dict):
                            header_anchors_pct = ha_raw
                    except Exception:
                        header_anchors_pct = None
                    payload = {
                        "id": str(meta_doc.get("id") or ""),
                        "kind": (meta_doc.get("chunk_kind") or ("section" if is_section else "page")),
                        "text": str(getattr(doc, 'page_content', '') or ''),
                        "source": src_name,
                        "page": meta_doc.get("page"),
                        "first_page": meta_doc.get("first_page"),
                        "pages": meta_doc.get("pages"),
                        "section": (meta_doc.get("section") or "").strip(),
                        "section_number": (meta_doc.get("section_number") or "").strip(),
                        "section_code": (meta_doc.get("section_code") or "").strip(),
                        "section_id2": (meta_doc.get("section_id2") or "").strip(),
                        "labels": labels,
                    }
                    if text_spans_map is not None:
                        payload["text_spans"] = text_spans_map
                    if header_anchors_pct is not None:
                        payload["header_anchors_pct"] = header_anchors_pct
                    included_chunks_payload.append(payload)
                except Exception:
                    continue
            meta = {"sources": srcs, "context": ctx_text, "prompt": instruction, "original_query": query_text, "chunks": included_chunks_payload, "spans": [], "citations": allowed_struct, "system": system_prompt, "stream_validation": {"enabled": bool(getattr(cfg, "ENABLE_STREAM_VALIDATION", False)), "repairs": repairs_log, "original_full": sv_acc.get("original", ""), "repaired_full": sv_acc.get("repaired", "")}}
        except Exception:
            meta = {"sources": [], "context": "", "prompt": query_text, "original_query": query_text, "spans": [], "citations": [], "system": system_prompt, "stream_validation": {"enabled": bool(getattr(cfg, "ENABLE_STREAM_VALIDATION", False)), "repairs": repairs_log, "original_full": sv_acc.get("original", ""), "repaired_full": sv_acc.get("repaired", "")}}
        print("="*80)
        print("🔍 STREAM_QUERY_RAG DEBUG END (VECTOR)")
        print("="*80 + "\n")
        return _gen(), meta


def _stream_with_validation(raw_iter, allowed_struct, repairs_log: Optional[list] = None, sv_acc: Optional[dict] = None, instruction_text: Optional[str] = None):
    """Detached streaming validator/repair wrapper.

    Buffers to sentence boundaries, validates citation format and membership,
    and performs cheap repairs without stalling output.
    """
    try:
        from . import config as _cfg  # type: ignore
    except Exception:
        class _Cfg:
            ENABLE_STREAM_VALIDATION = False
            STREAM_REPAIR_MAX_CONSECUTIVE_INVALID = 3
        _cfg = _Cfg()  # type: ignore

    if not getattr(_cfg, "ENABLE_STREAM_VALIDATION", False):
        for ch in raw_iter:
            yield ch
        return

    # Build allowed section/code set
    allowed_sections: set[str] = set()
    try:
        for item in (allowed_struct or []):
            code = str(item.get("code") or "").strip()
            sec = str(item.get("section") or "").strip()
            if code:
                allowed_sections.add(code)
            if sec:
                allowed_sections.add(sec)
    except Exception:
        allowed_sections = set()

    # Also parse Allowed citations from the instruction text we sent to the model
    try:
        if instruction_text:
            import re as _re
            # Match lines like: - file=..., sections=["...", "..."]
            for line in instruction_text.splitlines():
                m = _re.search(r"sections=\[(.*?)\]", line)
                if not m:
                    continue
                inner = m.group(1)
                for part in inner.split(","):
                    s = part.strip().strip('"').strip("'")
                    if s:
                        allowed_sections.add(s)
    except Exception:
        pass

    import re as _re
    sentence_end = _re.compile(r"([.!?])\s+")
    citation_re = _re.compile(r"\[(?P<section>[^\[\]]+)\]$")
    forbidden = _re.compile(r"\((?:see\s*)?\[[^\]]+\]\)", _re.IGNORECASE)

    buffer = ""
    consecutive_invalid = 0

    def _emit_safe_fallback() -> str:
        try:
            one = next(iter(allowed_sections)) if allowed_sections else ""
        except Exception:
            one = ""
        if one:
            return f"Answer constrained by available sections. [{one}]"
        return ""

    for chunk in raw_iter:
        t = str(chunk)
        buffer += t
        try:
            if isinstance(sv_acc, dict):
                sv_acc["original"] = sv_acc.get("original", "") + t
        except Exception:
            pass
        parts = sentence_end.split(buffer)
        rebuilt: list[str] = []
        i = 0
        while i + 2 < len(parts):
            rebuilt.append(parts[i] + parts[i+1])
            i += 3
        tail = "".join(parts[i:]) if i < len(parts) else ""
        for sent in rebuilt:
            s = sent.strip()
            if not s:
                continue
            is_forbidden = bool(forbidden.search(s))
            m = citation_re.search(s)
            has_one_cite = bool(m)
            sec = (m.group("section").strip() if m else "")
            in_allowed = sec in allowed_sections
            if (not is_forbidden) and has_one_cite and in_allowed:
                consecutive_invalid = 0
                try:
                    if isinstance(sv_acc, dict):
                        sv_acc["repaired"] = sv_acc.get("repaired", "") + sent
                except Exception:
                    pass
                yield sent
                continue
            repaired = _try_llm_repair_if_enabled(s, allowed_sections)
            if repaired:
                try:
                    if isinstance(repairs_log, list):
                        repairs_log.append({"original": s, "repaired": repaired})
                except Exception:
                    pass
                try:
                    if isinstance(sv_acc, dict):
                        sv_acc["repaired"] = sv_acc.get("repaired", "") + repaired + " "
                except Exception:
                    pass
                consecutive_invalid = 0
                yield repaired + " "
                continue
            consecutive_invalid += 1
            if consecutive_invalid >= getattr(_cfg, "STREAM_REPAIR_MAX_CONSECUTIVE_INVALID", 3):
                filler = _emit_safe_fallback()
                if filler:
                    yield filler + " "
                consecutive_invalid = 0
        buffer = tail

    s = buffer.strip()
    if s:
        m = citation_re.search(s)
        is_forbidden = bool(forbidden.search(s))
        if m and (m.group("section").strip() in allowed_sections) and not is_forbidden:
            try:
                if isinstance(sv_acc, dict):
                    sv_acc["repaired"] = sv_acc.get("repaired", "") + s
            except Exception:
                pass
            yield s
        else:
            repaired = _try_llm_repair_if_enabled(s, allowed_sections)
            if repaired:
                try:
                    if isinstance(repairs_log, list):
                        repairs_log.append({"original": s, "repaired": repaired})
                except Exception:
                    pass
                try:
                    if isinstance(sv_acc, dict):
                        sv_acc["repaired"] = sv_acc.get("repaired", "") + repaired
                except Exception:
                    pass
                yield repaired


def _try_micro_repair(sentence: str, allowed_sections: set[str]) -> str:
    """Cheap repair without external calls. Returns fixed sentence or ''."""
    try:
        import re as _re
        cites = _re.findall(r"\[([^\]]+)\]", sentence)
        candidate = cites[-1].strip() if cites else ""
        if candidate in allowed_sections:
            base = _re.sub(r"\s*\[[^\]]+\]\s*", "", sentence).strip()
            return f"{base} [{candidate}]"
        cand = next(iter(allowed_sections)) if allowed_sections else ""
        if cand:
            base = _re.sub(r"\s*\[[^\]]+\]\s*", "", sentence).strip()
            return f"{base} [{cand}]"
        return ""
    except Exception:
        return ""


def _try_llm_repair_if_enabled(sentence: str, allowed_sections: set[str]) -> str:
    """If STREAM_REPAIR_USE_LLM is enabled, run a tiny constrained repair; else micro-repair.

    Returns repaired sentence or ''. Never blocks beyond configured timeout.
    """
    try:
        from . import config as _cfg  # type: ignore
    except Exception:
        class _Cfg:
            STREAM_REPAIR_USE_LLM = False
            STREAM_REPAIR_TIMEOUT_MS = 100
        _cfg = _Cfg()  # type: ignore

    # Fast path: if disabled, use micro-repair
    if not getattr(_cfg, "STREAM_REPAIR_USE_LLM", False):
        return _try_micro_repair(sentence, allowed_sections)

    # If no allowed sections, do not attempt
    if not allowed_sections:
        return ""

    # Compose strict instruction
    allow_list = ", ".join(sorted(f"[{s}]" for s in allowed_sections))
    prompt = (
        "Repair this one paragraph. Rules: "
        "- Keep only content that can end with EXACTLY ONE allowed citation. "
        "- Allowed citations (choose one): " + allow_list + ". "
        "- Delete any extra citations and trailing text after the citation. "
        "- If repair not possible, return an empty string."
    )

    try:
        # Use the same provider model but with a tiny budget
        from . import config as _cfg2  # type: ignore
        model_name = getattr(_cfg2, "GENERATOR_MODEL", "")
        timeout_s = max(0.05, float(getattr(_cfg2, "STREAM_REPAIR_TIMEOUT_MS", 100)) / 1000.0)
        # Prefer OpenAI small models if provider is openai; otherwise reuse primary with short timeout
        if getattr(_cfg2, "LLM_PROVIDER", "anthropic").lower() == "openai":
            try:
                from langchain_openai import ChatOpenAI
                small = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=timeout_s)
                resp = small.invoke(prompt + "\n\nParagraph:\n" + sentence)
                out = str(getattr(resp, "content", resp) or "").strip()
            except Exception:
                out = ""
        else:
            try:
                from langchain_anthropic import ChatAnthropic
                small = ChatAnthropic(model=model_name, temperature=0, timeout=timeout_s)
                resp = small.invoke(prompt + "\n\nParagraph:\n" + sentence)
                out = str(getattr(resp, "content", resp) or "").strip()
            except Exception:
                out = ""

        # Validate output quickly
        if not out:
            return ""
        import re as _re
        # Must end with [section]
        m = _re.search(r"\[([^\]]+)\]$", out)
        if not m:
            return ""
        sec = m.group(1).strip()
        if sec not in allowed_sections:
            return ""
        # Ensure only one citation and clean internal citations
        base = _re.sub(r"\s*\[[^\]]+\]\s*", "", out).strip()
        return f"{base} [{sec}]"
    except Exception:
        return _try_micro_repair(sentence, allowed_sections)


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG model with a given query."
    )
    parser.add_argument(
        "--query_text", type=str, help="The query to be passed to the RAG model."
    )
    parser.add_argument(
        "--game", type=str, help="Game to filter results by (e.g., monopoly, catan)"
    )
    parser.add_argument(
        "--include_sources",
        action=argparse.BooleanOptionalAction,
        help="Include sources in the response.",
    )
    parser.add_argument(
        "--include_context",
        action=argparse.BooleanOptionalAction,
        help="Include context in the response.",
    )
    parser.add_argument(
        "--game_names",
        nargs="*",
        help="Optional game names to include in web search",
    )
    args = parser.parse_args()
    query_text = args.query_text
    selected_game = args.game
    include_sources = args.include_sources
    include_context = args.include_context

    # Use streaming path and aggregate tokens for CLI output
    token_gen, meta = stream_query_rag(
        query_text=query_text,
        selected_game=selected_game,
        chat_history=None,
        game_names=args.game_names,
        enable_web=False,
    )

    aggregated = ""
    for chunk in token_gen:
        aggregated += str(chunk)

    # Extract game name from first source
    sources = meta.get("sources", []) if isinstance(meta, dict) else []
    if sources:
        first_src = sources[0]
        if isinstance(first_src, dict):
            src_path = first_src.get("filepath", "")
        else:
            src_path = first_src
        game_name = os.path.basename(src_path).split()[0].capitalize() if src_path else "Game"
    else:
        game_name = "Game"

    response_text = f"🤖 {game_name}: {aggregated}"

    if include_sources:
        response_text += f"\n\n\n 📜Sources: {sources}"

    if include_context:
        response_text += f"\n\n\n 🌄Context: {meta.get('context', '')}"

    print(response_text)


if __name__ == "__main__":
    main()
