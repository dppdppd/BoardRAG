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
from typing import Dict, List, Optional, Union, Tuple

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
            # Use configured provider for filename extraction
            if cfg.LLM_PROVIDER.lower() == "anthropic":
                model = ChatAnthropic(model=cfg.GENERATOR_MODEL, temperature=0)
            elif cfg.LLM_PROVIDER.lower() == "ollama":
                from langchain_community.llms.ollama import Ollama

                model = Ollama(model=cfg.GENERATOR_MODEL, base_url=cfg.OLLAMA_URL)
            else:  # openai
                # o3 model only supports temperature=1, other OpenAI models use 0 for determinism
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
    # Prefer catalog when DB-less
    try:
        from . import config as _cfg  # type: ignore
        if bool(getattr(_cfg, "DB_LESS", True)):
            try:
                from .catalog import list_games_from_catalog  # type: ignore
                return list_games_from_catalog()
            except Exception:
                pass
    except Exception:
        pass
    from .retrieval.game_names import get_available_games as _impl
    return _impl()


def store_game_name(filename: str, game_name: str):
    # In DB-less mode, store in catalog; otherwise delegate to legacy DB
    try:
        from . import config as _cfg  # type: ignore
        if bool(getattr(_cfg, "DB_LESS", True)):
            try:
                from .catalog import load_catalog, save_catalog  # type: ignore
                from pathlib import Path as _P
                cat = load_catalog()
                key = _P(filename).name
                entry = cat.get(key) or {}
                entry["game_name"] = game_name
                cat[key] = entry
                save_catalog(cat)
                return None
            except Exception:
                pass
    except Exception:
        pass
    from .retrieval.game_names import store_game_name as _impl
    return _impl(filename, game_name)


def get_stored_game_names() -> Dict[str, str]:
    from .retrieval.game_names import get_stored_game_names as _impl
    return _impl()


def extract_and_store_game_name(filename: str) -> str:
    # When DB-less, try to update catalog directly for name extraction
    try:
        from . import config as _cfg  # type: ignore
        if bool(getattr(_cfg, "DB_LESS", True)):
            from .retrieval.game_names import extract_game_name_from_filename  # type: ignore
            name = extract_game_name_from_filename(filename)
            try:
                from .catalog import load_catalog, save_catalog  # type: ignore
                from pathlib import Path as _P
                cat = load_catalog()
                key = _P(filename).name
                entry = cat.get(key) or {}
                entry["game_name"] = name
                cat[key] = entry
                save_catalog(cat)
            except Exception:
                pass
            return name
    except Exception:
        pass
    from .retrieval.game_names import extract_and_store_game_name as _impl
    return _impl(filename)


def rewrite_search_query(raw_query: str) -> str:
    from .retrieval.web_search import rewrite_search_query as _impl
    return _impl(raw_query)


def query_rag(
    query_text: str,
    selected_game: Optional[str] = None,
    chat_history: Optional[str] = None,
    game_names: Optional[List[str]] = None,
    enable_web: Optional[bool] = None,
) -> Dict[str, Union[str, Dict]]:
    """
    Queries the RAG model with the given query and returns the response.

    Args:
        query_text (str): The user's latest question.
        selected_game (Optional[str]): Game to filter results by (e.g., 'monopoly', 'catan').
        chat_history (Optional[str]): Conversation history formatted as a string, where each prior turn is included to provide conversational context. If None, the question is treated as standalone.

    Returns:
        Dict: The response from the RAG model.
    """
    # Deprecated: non-streaming path removed. Do not use.
    raise NotImplementedError("query_rag (non-streaming) has been removed. Use stream_query_rag instead.")
    
    # DB-less mode: use Anthropic Sonnet 4 Files API with citations, no vector DB
    # Prefer cfg.DB_LESS defaulting to True when env is absent
    try:
        from . import config as _cfg  # type: ignore
        _db_less_enabled = bool(getattr(_cfg, "DB_LESS", True))
    except Exception:
        _db_less_enabled = os.getenv("DB_LESS", "1").lower() in {"1", "true", "yes"}
    if _db_less_enabled:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {
                "response_text": "ANTHROPIC_API_KEY missing for DB-less mode.",
                "sources": [],
                "original_query": query_text,
                "context": "",
                "chunks": [],
                "prompt": "",
            }
        # Resolve files from catalog (preferred)
        from pathlib import Path as _P
        try:
            from .catalog import resolve_file_ids_for_game  # type: ignore
            pairs = resolve_file_ids_for_game(selected_game)
        except Exception:
            pairs = []
        # Fallback to scanning data/ if catalog unavailable
        if not pairs:
            data_dir = _P(getattr(cfg, "DATA_PATH", "data"))
            pdfs = sorted([p for p in data_dir.glob("*.pdf")])
            if selected_game:
                key = str(selected_game).strip().lower()
                pdfs = [p for p in pdfs if key in p.name.lower()]
            # Upload and cache file_ids if needed
            # Persist cache under data/catalog alongside catalog.json so Railway keeps it
            from .config import DATA_PATH as _DATA_PATH  # type: ignore
            cache_path = _P(_DATA_PATH) / "catalog" / "anthropic_files.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import json as _json
                cache = _json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
            except Exception:
                cache = {}
            for p in pdfs:
                fp = str(p.resolve())
                fid = cache.get(fp)
                if not fid:
                    try:
                        fid = upload_pdf_to_anthropic_files(api_key, fp)
                        cache[fp] = fid
                    except Exception:
                        continue
                pairs.append((fp, fid))
            try:
                cache_path.write_text(_json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        if not pairs:
            return {
                "response_text": "No PDFs available to query.",
                "sources": [],
                "original_query": query_text,
                "context": "",
                "chunks": [],
                "prompt": "",
            }
        # Use at most 2 PDFs per request to stay within size limits
        file_ids = pairs[:2]
        if not file_ids:
            return {
                "response_text": "Failed to upload PDFs to Files API.",
                "sources": [],
                "original_query": query_text,
                "context": "",
                "chunks": [],
                "prompt": "",
            }
        # Build user instruction: answer plus spans for spotlight
        instruction = (
            "Answer the user's question based ONLY on the attached PDF(s).\n"
            "Give the answer a brief title that is a succinct form of the question/ A few words, no more than 5.\n"
            "Include INLINE citations in square brackets immediately after claims, using the exact section code and title from the rule headers, e.g., [6.2 FIREPOWER], [6.4 FIRE STRENGTH].\n"
            "After the prose, return STRICT JSON as the last block with keys: answer (string), spans (array).\n"
            "Each span: {page:int, header:string}.\n"
            "- header must be the exact header line text used in the justification (code + title).\n"
            "- Page numbers are 1-based.\n"
            "- If no specific headers were used, return spans: [].\n\n"
            f"Question: {query_text}"
        )
        system_prompt = (
            "You are an expert assistant for boardgame rulebooks. "
            "Follow the user's inline citation format and provide concise, accurate answers."
        )
        # Send one message per file_id, then merge answers (first non-empty preferred)
        import os as _os
        model_name = _os.getenv("OUTLINE_LLM_MODEL", "claude-sonnet-4-20250514")
        aggregated_answer = None
        aggregated_spans: List[Dict[str, Union[int, str]]] = []
        sources = []
        for fp, fid in file_ids:
            try:
                raw = anthropic_pdf_messages_with_file(api_key, model_name, system_prompt, instruction, fid)
                # Try parse JSON
                import json as _json
                js_obj = None
                # Extract last JSON object
                import re as _re
                cands = [_m.group(0) for _m in _re.finditer(r"\{[\s\S]*?\}", raw)]
                for _js in cands[::-1]:
                    try:
                        js_obj = _json.loads(_js)
                        break
                    except Exception:
                        continue
                if isinstance(js_obj, dict):
                    ans = str(js_obj.get("answer") or "").strip()
                    spans = js_obj.get("spans") or []
                    if ans and aggregated_answer is None:
                        aggregated_answer = ans
                    # Attach anchors
                    for sp in spans:
                        try:
                            pg = int(sp.get("page"))
                            hdr = str(sp.get("header") or "").strip()
                            pt = find_header_anchor(fp, pg, hdr)
                            x, y = (pt or (None, None))
                            aggregated_spans.append({
                                "file": _P(fp).name,
                                "page": pg,
                                "header": hdr,
                                "x": x,
                                "y": y,
                            })
                        except Exception:
                            continue
                    sources.append({"filepath": fp})
            except Exception:
                continue
        if not aggregated_answer:
            aggregated_answer = "No answer found in the provided PDFs."
        return {
            "response_text": aggregated_answer,
            "sources": sources,
            "original_query": query_text,
            "context": "",
            "chunks": [],
            "prompt": instruction,
            "spans": aggregated_spans,
        }

    # Legacy DB-backed mode has been removed; only DB-less is supported.
    return {
        "response_text": "DB-backed mode has been removed. Please enable DB-less mode.",
        "sources": [],
        "original_query": query_text,
        "context": "",
        "chunks": [],
        "prompt": "",
    }

    # Prepare search query – include chat history to give the retriever more context for follow-up questions
    search_query = (
        f"{chat_history}\n\n{query_text}" if chat_history else query_text
    )

    print("🔍 Searching in the database…")
    print(f"  Search query: '{search_query[:1000]}'")  # truncate long prints
    if selected_game:
        print(f"  Filtering by game: '{selected_game}'")



    # Get results from database (more if filtering is needed)
    # Fewer retrievals yield faster prompts; keep small but sufficient
    k_results = 40 if selected_game else 20
            # Try server-side metadata filtering first (if supported)
    metadata_filter = None
    target_files = None  # set of lowercase PDF filenames used for filtering, when available
    if selected_game:
        # Debug: normalize selected_game input
        def _normalize_game_input(game_input):
            """Convert game input to normalized string."""
            import os  # Import needed for basename function
            if isinstance(game_input, list):
                # Handle list of filenames - find the game they map to
                if game_input:
                    # Look up what game these filenames belong to
                    stored_map = get_stored_game_names()
                    for fname, gname in stored_map.items():
                        # Check if any of the input filenames match this stored file
                        fname_base = os.path.basename(fname).replace(".pdf", "").lower()
                        if any(fname_base == item.strip().lower() for item in game_input):
                            normalized = gname  # Use the mapped game name
                            break
                    else:
                        # Fallback: use first item if no mapping found
                        normalized = game_input[0].strip()
                else:
                    normalized = ""
            elif isinstance(game_input, str):
                normalized = game_input.strip()
            else:
                normalized = str(game_input).strip()
            print(f"🎮 Game input normalization: {game_input!r} → {normalized!r}")
            return normalized.lower()
        
        game_key = _normalize_game_input(selected_game)
        stored_map = get_stored_game_names()
        
        # Debug: show what's in the mapping
        print(f"📚 Current game mappings ({len(stored_map)} entries):")
        for fname, gname in sorted(stored_map.items()):
            print(f"  '{fname}' → '{gname}'")
        
        import os
        # Try matching by game name first, then by filename if that fails
        target_files = {os.path.basename(fname).lower() for fname, gname in stored_map.items() if gname.lower() == game_key}
        
        if not target_files:
            # Fallback: try matching by filename (for UI compatibility)
            target_files = {os.path.basename(fname).lower() for fname in stored_map.keys() if os.path.basename(fname).replace(".pdf", "").replace(" ", "_").lower() == game_key}
        
        print(f"🔍 Looking for game: '{game_key}'")
        print(f"📂 Found {len(target_files)} matching PDFs: {sorted(target_files)}")
        
        if not target_files:
            available_games = sorted(set(gname.lower() for gname in stored_map.values()))
            raise ValueError(f"No PDFs are mapped to the game '{selected_game}'. Available games: {available_games}")
        metadata_filter = {"pdf_filename": {"$in": list(target_files)}}

    # Numeric section targeting: if query asks to quote a specific section number,
    # pull exact section chunks for the selected game and prioritize them.
    requested_section = None
    try:
        m_sec = re.search(r"\bquote\s+((?:\d+(?:\.\d+)*))\b", query_text, re.IGNORECASE)
        if m_sec:
            requested_section = m_sec.group(1)
            print(f"🔎 Requested explicit section: {requested_section}")
    except Exception:
        requested_section = None

    prioritized_results = []
    if requested_section:
        try:
            all_docs = db.get()
            docs = all_docs.get("documents", [])
            metas = all_docs.get("metadatas", [])
            ids = all_docs.get("ids", [])
            for doc_text, meta, _id in zip(docs, metas, ids):
                if not isinstance(meta, dict):
                    continue
                if target_files:
                    pdf_fn = (meta.get("pdf_filename") or "").lower()
                    if pdf_fn not in target_files:
                        continue
                sec_num = (meta.get("section_number") or "").strip()
                sec_label = (meta.get("section") or "").strip()
                if sec_num and (sec_num == requested_section or sec_num.startswith(requested_section + ".")):
                    from langchain.schema.document import Document as _Doc
                    prioritized_results.append((_Doc(page_content=doc_text, metadata=meta), 0.0))
                    continue
                # Fallback: label prefix match
                if sec_label.startswith(requested_section):
                    from langchain.schema.document import Document as _Doc
                    prioritized_results.append((_Doc(page_content=doc_text, metadata=meta), 0.0))
        except Exception as e:
            print(f"⚠️ Section prioritization failed: {e}")
    # Multi-query retrieval with plural/synonym expansion
    vq_extra = []
    try:
        if re.search(r"\bTicket to Ride\b", query_text, re.IGNORECASE):
            vq_extra.append("how many train cards does each player start with")
            vq_extra.append("setup each player starts with train cards")
    except Exception:
        pass
    variant_queries = (generate_query_variants(query_text, game_names) or []) + vq_extra
    if not variant_queries:
        variant_queries = [query_text]
    # Build variant search strings including chat history when present
    search_variants = [
        (f"{chat_history}\n\n{v}" if chat_history else v) for v in variant_queries
    ]
    # Allocate per-variant k to avoid over-fetching
    per_variant_k = max(12, k_results // max(1, len(search_variants)))
    all_results = []
    for vq in search_variants:
        try:
            batch = (
                db.similarity_search_with_score(vq, k=per_variant_k, filter=metadata_filter)
                if metadata_filter
                else db.similarity_search_with_score(vq, k=per_variant_k)
            )
        except TypeError:
            batch = (
                db.similarity_search_with_score(vq, k=per_variant_k, filter=metadata_filter)
                if metadata_filter
                else db.similarity_search_with_score(vq, k=per_variant_k)
            )
        # Extend combined list; we'll dedupe below
        all_results.extend(batch)

    # Merge similarity hits with simple dedupe
    results = []
    seen_keys = set()
    def _key_for(d):
        m = getattr(d, "metadata", {}) or {}
        return (m.get("source"), m.get("page"), (m.get("section") or "").strip())

    # Prepend prioritized section hits (if any), then similarity hits
    for d, s in (prioritized_results or []):
        k = _key_for(d)
        if k not in seen_keys:
            results.append((d, s))
            seen_keys.add(k)

    for d, s in all_results:
        k = _key_for(d)
        if k not in seen_keys:
            results.append((d, s))
            seen_keys.add(k)

    # Keep only the top-N results for prompt building
    results = results[: RAG_MAX_DOCS]

    # Heuristic re-rank: for phase-related queries, prefer chunks with numeric section headers
    try:
        if re.search(r"\bphase(s)?\b", query_text, re.IGNORECASE):
            def _is_numeric_section(doc):
                sec = (getattr(doc, 'metadata', {}) or {}).get('section') or ''
                return bool(re.match(r"^\s*\d+(?:\.\d+)*\b", str(sec)))
            results.sort(key=lambda pair: (not _is_numeric_section(pair[0]), pair[1]))
    except Exception:
        pass
    # Heuristic re-rank for specific query intents
    try:
        if re.search(r"train\s*cards?", query_text, re.IGNORECASE):
            def _pref_train_cards(doc):
                text = (getattr(doc, 'page_content', '') or '').lower()
                hints = 0
                for pat in ("each player", "start", "starting", "initial", "setup", "begin"):
                    if pat in text:
                        hints += 1
                for pat in ("train car", "train cards"):
                    if pat in text:
                        hints += 2
                return -hints  # higher hints → smaller key → earlier
            results.sort(key=lambda pair: (_pref_train_cards(pair[0]), pair[1]))
    except Exception:
        pass

    print(f"  Found {len(results)} results")
    if results:
        print(f"  Best match score: {results[0][1]:.4f}")
        # Show which games the results come from
        sources = [doc.metadata.get("source", "unknown") for doc, _ in results[:3]]
        unique_sources = list(set([Path(s).name for s in sources if s != "unknown"]))
        print(f"  Top sources: {', '.join(unique_sources[:3])}")
    else:
        print("  ⚠️ No results found - check if database contains relevant documents")

    # Supplement with live web search results (optional)
    effective_web_search = ENABLE_WEB_SEARCH if enable_web is None else enable_web

    web_docs: List[Document] = []
    if effective_web_search:
        # Incorporate game name(s) into web search for better relevance
        quoted_game = ""
        if game_names:
            # Use first game name to keep query concise; wrap in quotes
            quoted_game = f'"{game_names[0]}" '
        pre_query = f"{quoted_game}{query_text}"
        web_query = rewrite_search_query(pre_query)

        print(
            f"🌐 Web search enabled – fetching top {WEB_SEARCH_RESULTS} snippets with query: {web_query!r}…"
        )
        web_docs = perform_web_search(web_query, k=WEB_SEARCH_RESULTS)
        print(f"🌐 Retrieved {len(web_docs)} web snippets")
        # Treat them as zero-score items but include for context.
        if web_docs:
            results.extend([(doc, 0.0) for doc in web_docs])

    # Build the prompt
    print("🔮 Building the prompt …")
    # Build context with a hard character cap
    # Sanitizer to remove TOC-like lines and bare page numbers that can mislead citation generation
    def _sanitize_for_context(text: str) -> str:
        try:
            lines = text.splitlines()
            cleaned = []
            toc_re = re.compile(r"^\s*\d+(?:\.\d+)+\s+.+?\.{2,}\s*\d+\s*$")
            page_num_re = re.compile(r"^\s*\d+\s*$")
            for ln in lines:
                if toc_re.match(ln):
                    continue
                if page_num_re.match(ln):
                    continue
                cleaned.append(ln)
            return "\n".join(cleaned)
        except Exception:
            return text

    parts = []
    included_chunks_debug = []  # Collect details for logs
    included_chunks_payload = []  # Collect structured chunks for client
    used = 0
    for doc, _score in results:
        t = _sanitize_for_context(doc.page_content or "")
        if not t:
            continue
        if used + len(t) + 8 > RAG_CONTEXT_CHAR_LIMIT:
            t = t[: max(0, RAG_CONTEXT_CHAR_LIMIT - used)]
        parts.append(t)
        try:
            meta = getattr(doc, 'metadata', {}) or {}
            # Debug entry for logs
            included_chunks_debug.append({
                "source": (meta.get("source") or "unknown"),
                "page": meta.get("page"),
                "section": (meta.get("section") or "").strip(),
                "section_number": (meta.get("section_number") or "").strip(),
                "length": len(t),
                "preview": t.replace("\n", " ")[:160],
            })
            # Client payload (deliver essential info so the UI can render without refetch)
            from pathlib import Path as _Path
            src_name = _Path((meta.get("source") or "")).name or (meta.get("source") or "unknown")
            payload_item = {
                "text": t,
                "source": src_name,
                "page": meta.get("page"),
                "section": (meta.get("section") or "").strip(),
                "section_number": (meta.get("section_number") or "").strip(),
            }
            # Pass through any normalized rects if present
            rects_norm = meta.get("rects_norm")
            if rects_norm is not None:
                payload_item["rects_norm"] = rects_norm
            included_chunks_payload.append(payload_item)
        except Exception:
            pass
        used += len(t) + 8
        if used >= RAG_CONTEXT_CHAR_LIMIT:
            break
    context_text = "\n\n---\n\n".join(parts)

    # Provide an explicit allowlist of section identifiers derived from retrieved sources
    try:
        allowed_sections = []
        seen_allow = set()
        for doc, _ in results:
            meta = getattr(doc, 'metadata', {}) or {}
            sec = (meta.get('section_number') or '').strip()
            if not sec:
                # Support numeric dotted (3.5, 3.5.1), alphanumeric letter-first (F4, F4.a, A10.2b), and digit-first (1B6, 1B6b)
                label = (meta.get('section') or '').strip()
                m = re.match(r"^\s*(([A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?))\b", label)  # letter-first
                if not m:
                    m = re.match(r"^\s*((\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?))\b", label)  # digit-first
                if not m:
                    m = re.match(r"^\s*((\d+(?:\.[A-Za-z0-9]+)+))\b", label)  # numeric dotted
                if m:
                    sec = m.group(1)
            if sec and sec not in seen_allow:
                allowed_sections.append(sec)
                seen_allow.add(sec)
        if allowed_sections:
            allowline = ''.join(f'[{s}]' for s in allowed_sections[:24])
            context_text = f"ALLOWED_SECTIONS: {allowline}\n\n" + context_text
        else:
            context_text = "ALLOWED_SECTIONS: \n\n" + context_text
    except Exception:
        pass

    # Combine chat history with the latest question so the LLM has conversational context
    if chat_history:
        composite_question = (
            f"Previous conversation (for context):\n{chat_history}\n\nUser's latest question: {query_text}"
        )
    else:
        composite_question = query_text

    # Try improved template first, fallback to original
    try:
        prompt = load_jinja2_prompt(
            context=context_text,
            question=composite_question,
            template_name="rag_query_improved.txt",
        )
    except Exception:
        prompt = load_jinja2_prompt(context=context_text, question=composite_question)
    # Debug: spew full prompt and list included chunks
    try:
        print("\n===== FULL PROMPT (BEGIN) =====")
        print(prompt)
        print("===== FULL PROMPT (END) =====\n")
        print("Included chunks (in order):")
        for idx, info in enumerate(included_chunks_debug, 1):
            try:
                from pathlib import Path as _Path
                src_name = _Path(info.get("source") or "").name
            except Exception:
                src_name = info.get("source") or "unknown"
            print(f"  {idx:02d}. src={src_name} page={info.get('page')} section='{info.get('section')}' num='{info.get('section_number')}' len={info.get('length')}")
            print(f"      {info.get('preview')}")
    except Exception:
        pass


    print("🍳 Generating the response (streaming)…")

    # Prepare streaming callback to flush tokens to stdout immediately.
    try:
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

        callbacks = [StreamingStdOutCallbackHandler()]
    except Exception:
        # Fallback: no streaming callback available
        callbacks = None

    # Temperature is fixed to 0 for deterministic answers that are easier to test.
    model_kwargs = {
        "callbacks": callbacks,
        "streaming": True,
    }

    if cfg.LLM_PROVIDER.lower() == "ollama":
        # Import here to avoid requiring Ollama for OpenAI users.
        from langchain_community.llms.ollama import Ollama  # pylint: disable=import-error

        model = Ollama(model=cfg.GENERATOR_MODEL, base_url=cfg.OLLAMA_URL, num_predict=LLM_MAX_TOKENS, **model_kwargs)
    elif cfg.LLM_PROVIDER.lower() == "anthropic":
        model = ChatAnthropic(model=cfg.GENERATOR_MODEL, temperature=0, max_tokens=LLM_MAX_TOKENS, **model_kwargs)
    elif cfg.LLM_PROVIDER.lower() == "openai":
        if _openai_requires_default_temperature(cfg.GENERATOR_MODEL):
            model = ChatOpenAI(model=cfg.GENERATOR_MODEL, temperature=1, max_tokens=LLM_MAX_TOKENS, timeout=REQUEST_TIMEOUT_S, **model_kwargs)
        else:
            model = ChatOpenAI(model=cfg.GENERATOR_MODEL, temperature=0, max_tokens=LLM_MAX_TOKENS, timeout=REQUEST_TIMEOUT_S, **model_kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {cfg.LLM_PROVIDER}. Must be 'openai', 'anthropic', or 'ollama'"
        )

    # If the user requested a specific numeric section but none were found in results,
    # avoid generating with misleading content.
    if requested_section:
        has_requested = False
        for doc, _ in results:
            sec_num = ((doc.metadata or {}).get("section_number") or "").strip()
            sec_label = ((doc.metadata or {}).get("section") or "").strip()
            if sec_num.startswith(requested_section) or sec_label.startswith(requested_section):
                has_requested = True
                break
        if not has_requested:
            return {
                "response_text": f"Section {requested_section} not found in the selected game.",
                "sources": [],
                "original_query": query_text,
                "context": context_text,
                "prompt": prompt,
            }

    response_raw = model.invoke(prompt)
    # Convert LangChain message objects to string content when necessary.
    if hasattr(response_raw, "content"):
        response_text = response_raw.content
    else:
        response_text = response_raw

    # If user asked for number-only, normalize first numeric token
    try:
        if re.search(r"number\s+only", query_text, re.IGNORECASE):
            m_num = re.search(r"\$?\s*(\d[\d,]*)", str(response_text))
            if m_num:
                digits_only = re.sub(r"\D", "", m_num.group(1))
                if digits_only:
                    response_text = digits_only
    except Exception:
        pass

    # Remove trailing auto-appended numeric citations; do not modify model output here

    # Build structured source metadata for citations without thresholding
    sources = []
    for doc, score in results:
        meta_doc = doc.metadata
        src_path = meta_doc.get("source", "")
        # Always include web results as URLs
        if isinstance(src_path, str) and src_path.startswith("http"):
            sources.append(src_path)
            continue
        # Include all PDF sources with page/section
        sources.append({
            "filepath": src_path,
            "page": meta_doc.get("page"),
            "section": meta_doc.get("section"),
        })
    response = {
        "response_text": response_text,
        "sources": sources,
        "original_query": query_text,
        "context": context_text,
        "chunks": included_chunks_payload,
        "prompt": prompt,
    }

    return response


# ---------------------------------------------------------------------------
# Streaming variant – yields tokens incrementally for UI streaming
# ---------------------------------------------------------------------------


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

    # ---- DB-less branch: use Anthropic Files API on original PDFs ----
    # When DB_LESS is enabled, do NOT fall back to DB. Stream a clear error instead.
    db_less_enabled = True
    try:
        from . import config as _cfg  # type: ignore
        db_less_enabled = bool(getattr(_cfg, "DB_LESS", True))
    except Exception:
        import os as _os
        db_less_enabled = _os.getenv("DB_LESS", "1").lower() in {"1", "true", "yes"}
    if db_less_enabled:
        import os as _os
        api_key = _os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            def _err_gen():
                yield "ANTHROPIC_API_KEY missing for DB-less mode."
            meta = {"sources": [], "context": "", "prompt": "", "original_query": query_text, "spans": []}
            return _err_gen(), meta

        def _normalize_game_input(game_input):
            if isinstance(game_input, list):
                return (game_input[0] if game_input else "").strip()
            if isinstance(game_input, str):
                return game_input.strip()
            return str(game_input or "").strip()

        norm_game = _normalize_game_input(selected_game)

        from pathlib import Path as _P
        try:
            from .catalog import resolve_file_ids_for_game  # type: ignore
            pairs = resolve_file_ids_for_game(norm_game or None)
        except Exception:
            pairs = []

        if not pairs:
            def _no_pdf_gen():
                yield f"No PDFs available for game '{norm_game or 'All'}' in catalog."
            meta = {"sources": [], "context": "", "prompt": "", "original_query": query_text, "spans": []}
            return _no_pdf_gen(), meta

        file_ids = pairs[:2]

        # Build allowed filenames list to force exact matching in citations
        try:
            allowed_filenames = ", ".join([_P(fp).name for fp, _ in file_ids])
        except Exception:
            allowed_filenames = ""
        instruction = (
            "Answer the user's question based ONLY on the attached PDF(s).\n"
            "Do NOT include any preamble or meta commentary. Start directly with the answer.\n"
            "Give the answer a brief title that is a succinct form of the question/ A few words, no more than 5.\n"
            "For EVERY factual claim, make a separate short paragraph and include one inline citation at the end of the paragraph exactly in this form: [<section>: {\"file\":\"<filename.pdf>\", \"page\": <1-based>, \"section\": \"<name of the section>\"}].\n"
            "- <section> is the rule code or range (e.g., 6.2, 6.4.1, 6.41-6.43).\n"
            "- The JSON must include: file (pdf filename), page (1-based), and section (the section number or designation).\n"
            + (f"- Allowed PDF filenames for the file field (use EXACTLY one of these, verbatim): {allowed_filenames}\n" if allowed_filenames else "") +
            "Do NOT include any code blocks; provide prose only with inline citations.\n\n"
            f"Question: {query_text}"
        )
        system_prompt = (
            "You are an expert assistant for boardgame rulebooks. "
            "Provide concise answers with inline citations; do not return JSON blocks."
        )
        import os as _os
        model_name = _os.getenv("OUTLINE_LLM_MODEL", "claude-sonnet-4-20250514")

        aggregated_answer: Optional[str] = None
        aggregated_spans: List[Dict[str, Union[int, str]]] = []  # type: ignore[name-defined]
        sources: List[Dict[str, str]] = []
        last_raw_text: str = ""
        # Spew what we are about to send (system + instruction + files)
        try:
            print("\n===== DB-LESS REQUEST (BEGIN) =====")
            print("-- System Prompt --\n" + system_prompt)
            print("\n-- User Instruction --\n" + instruction)
            try:
                from pathlib import Path as __P
                attached_names = ", ".join([__P(fp).name for fp, _ in file_ids])
            except Exception:
                attached_names = "(unknown files)"
            print(f"\n-- Attached PDFs --\n{attached_names}")
            print("===== DB-LESS REQUEST (END) =====\n")
        except Exception:
            pass

        # Stream each file's response; emit raw tokens only (no server-side parsing of citations)
        from .llm_outline_helpers import anthropic_pdf_messages_with_file_stream  # type: ignore

        def _token_gen():
            for fp, fid in file_ids:
                sources.append({"filepath": fp})
                buffer = ""
                had_text = False
                try:
                    for delta in anthropic_pdf_messages_with_file_stream(api_key, model_name, system_prompt, instruction, fid):
                        txt = str(delta or "")
                        if not txt:
                            continue
                        # Accumulate raw for debug spew; do not parse citations on server
                        buffer += txt
                        # Spew the literal model output chunk as-is
                        had_text = True
                        yield txt
                except Exception as e:
                    # If we hit rate limits or usage caps, surface a clear chat message
                    try:
                        status = getattr(getattr(e, 'response', None), 'status_code', None)
                        body = ''
                        try:
                            body = (getattr(e, 'response', None).text or '') if getattr(e, 'response', None) is not None else ''
                        except Exception:
                            body = ''
                        lower = (str(body) or '').lower()
                        if status == 429 or 'rate limit' in lower or 'usage' in lower or 'quota' in lower:
                            yield "API USAGE LIMIT HIT"
                            # stop processing this file
                            continue
                        # Anthropic overloaded / transient error surfaced by stream helper
                        if 'overloaded' in (str(e).lower() + lower):
                            yield "API OVERLOADED — please retry"
                            continue
                    except Exception:
                        pass
                # After streaming finishes for this file, spew the raw accumulated answer for debug
                try:
                    print("\n===== RAW MODEL RESPONSE (BEGIN) =====")
                    try:
                        from pathlib import Path as __P
                        print(f"-- File --\n{__P(fp).name}")
                    except Exception:
                        pass
                    print("-- Raw Text --\n" + buffer)
                    print("===== RAW MODEL RESPONSE (END) =====\n")
                except Exception:
                    pass

        meta = {"sources": sources, "context": "", "prompt": instruction, "original_query": query_text, "spans": []}
        print("="*80)
        print("🔍 STREAM_QUERY_RAG DEBUG END")
        print("="*80 + "\n")
        return _token_gen(), meta

    # Legacy DB-backed mode removed
    def _err_gen():
        yield "DB-backed mode has been removed. Please enable DB-less mode."
    meta = {"sources": [], "context": "", "prompt": "", "original_query": query_text, "chunks": [], "spans": []}
    return _err_gen(), meta


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
