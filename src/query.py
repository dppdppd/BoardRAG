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
from typing import Dict, List, Optional, Union

import chromadb
import os
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document
# config imports extended
from .config import (
    CHROMA_PATH,
    CITATION_SCORE_THRESHOLD,
    CITATION_MIN_SOURCES,
    disable_chromadb_telemetry,
    get_chromadb_settings,
    suppress_chromadb_telemetry,
    validate_config,
    ENABLE_WEB_SEARCH,
    WEB_SEARCH_RESULTS,
    SEARCH_PROVIDER,
    SERPAPI_API_KEY,
    BRAVE_API_KEY,
    ENABLE_SEARCH_REWRITE,
)
from .embedding_function import get_embedding_function
from templates.load_jinja_template import load_jinja2_prompt
from . import config as cfg

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
    """Return top *k* DuckDuckGo search snippets as Document objects.

    Each Document gets its snippet as *page_content* and carries the URL both
    as *id* and *source* metadata so downstream citation code can display it.
    """
    if not ENABLE_WEB_SEARCH:
        return []

    results = []

    if SEARCH_PROVIDER == "serpapi":
        if not SERPAPI_API_KEY:
            print("⚠️ SERPAPI_API_KEY not set; falling back to DuckDuckGo")
        else:
            try:
                from serpapi import GoogleSearch  # type: ignore

                params = {
                    "q": query,
                    "api_key": SERPAPI_API_KEY,
                    "num": k,
                    "engine": "google",
                }
                search = GoogleSearch(params)
                serp_results = search.get_dict()
                organic = serp_results.get("organic_results", [])
                for item in organic[:k]:
                    results.append({
                        "snippet": item.get("snippet", ""),
                        "url": item.get("link", ""),
                    })
            except Exception as e:
                print(f"⚠️ SerpAPI search failed: {e}")

    elif SEARCH_PROVIDER == "brave":
        if not BRAVE_API_KEY:
            print("⚠️ BRAVE_API_KEY not set; falling back to DuckDuckGo")
        else:
            try:
                import requests  # pylint: disable=import-error

                endpoint = "https://api.search.brave.com/res/v1/web/search"
                headers = {"X-Subscription-Token": BRAVE_API_KEY}
                params = {"q": query, "count": k}
                resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("web", {}).get("results", [])[:k]:
                        results.append({
                            "snippet": item.get("description", ""),
                            "url": item.get("url", ""),
                        })
                else:
                    print(f"⚠️ Brave API HTTP {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                print(f"⚠️ Brave search failed: {e}")

    # If still empty or provider is duckduckgo / brute fallback, use duckduckgo methods
    if not results and SEARCH_PROVIDER in {"duckduckgo", "serpapi", "brave"}:
        if ddg is not None:
            try:
                results = ddg(query, max_results=k) or []
            except Exception as e:
                print(f"⚠️ ddg() failed: {e}")

        # Fallback DDGS
        if not results:
            try:
                from duckduckgo_search import DDGS  # type: ignore

                with DDGS() as search:
                    results = search.text(query, max_results=k) or []
            except Exception as e:
                print(f"⚠️ DDGS fallback failed: {e}")

    if not results:
        print("⚠️ Web search returned 0 results")
        return []

    docs: List[Document] = []
    for res in results:
        snippet = (
            res.get("snippet")
            or res.get("body")
            or res.get("text")
            or res.get("title")
            or ""
        )
        url = res.get("url") or res.get("href") or res.get("link") or ""
        if not snippet or not url:
            continue
        meta = {
            "id": url,  # So it appears in sources list
            "source": url,
            "url": url,
            "web": True,
        }
        docs.append(Document(page_content=snippet, metadata=meta))
    print(f"🌐 Added {len(docs)} web snippets to context")
    return docs
def _openai_requires_default_temperature(model_name: str) -> bool:
    """Return True if this OpenAI model doesn't accept custom temperature (must use default).

    Covers o3/o4 reasoning and future reasoning-style identifiers.
    """
    m = (model_name or "").lower()
    return m.startswith("o3") or m.startswith("o4") or "-reasoning" in m or m.startswith("gpt-5")



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
    """Generate a small set of retrieval queries to improve recall (generic only).

    Strategy:
    - Start with the raw user query.
    - Append plural/singular variants for key tokens (no domain synonyms).
    Returns a list of 1–5 concise query strings.
    """
    raw = (query_text or "").strip()
    if not raw:
        return []

    # Heuristic token selection: words >= 4 chars (avoids stopwords); keep originals
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", raw)]
    variant_terms: List[str] = []
    for w in words[:6]:  # limit to first few meaningful tokens
        variant_terms.extend(_basic_plural_variants(w))

    # Build variants (cap to 4–5 total)
    variants: List[str] = [raw]
    if variant_terms:
        # Append a lightweight tail so embeddings see both forms without changing meaning
        tail = " ".join(dict.fromkeys(variant_terms))[:200]
        variants.append(f"{raw} {tail}")

    # Deduplicate while preserving order and trim excessive variants
    seen: set[str] = set()
    uniq: List[str] = []
    for v in variants:
        v2 = v.strip()
        if v2 and v2 not in seen:
            seen.add(v2)
            uniq.append(v2)
    return uniq[:5]


def normalize_game_title(title: str) -> str:
    """
    Normalize game title by moving leading articles to the end.
    
    Args:
        title (str): Game title like "The Campaign for North Africa"
        
    Returns:
        str: Normalized title like "Campaign for North Africa, The"
    """
    # Only strip leading/trailing whitespace for the final result
    stripped_title = title.strip()
    
    # Check for leading "The " (case insensitive)
    if stripped_title.lower().startswith("the "):
        # Remove "The " from beginning and add ", The" to end
        return stripped_title[4:] + ", The"
    
    # Check for leading "A " (case insensitive) 
    if stripped_title.lower().startswith("a "):
        # Remove "A " from beginning and add ", A" to end
        return stripped_title[2:] + ", A"
    
    return stripped_title


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
    """
    Simple fallback: just return the literal filename cleaned up.

    Args:
        filename (str): The PDF filename

    Returns:
        str: Cleaned filename
    """
    # Just remove .pdf and replace separators with spaces
    return filename.replace(".pdf", "").replace("-", " ").replace("_", " ").title()


def get_available_games() -> List[str]:
    """
    Get a list of available games from the database with proper names.
    Uses stored game names to avoid LLM calls on every app startup.

    Returns:
        List[str]: List of game names (e.g., ['Monopoly', 'Up Front', 'Ticket to Ride'])
    """
    try:
        # Connect to the database
        embedding_function = get_embedding_function()
        with suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(
                path=CHROMA_PATH, settings=get_chromadb_settings()
            )
            db = Chroma(client=persistent_client, embedding_function=embedding_function)
        
        # Get all documents to extract filenames
        all_docs = db.get()
        
        # Only show debug on first call or when empty
        if not hasattr(get_available_games, '_last_count') or len(all_docs['ids']) == 0:
            print(f"[DEBUG] get_available_games: Main collection has {len(all_docs['ids'])} documents")
            if len(all_docs['ids']) == 0:
                print(f"[DEBUG] get_available_games: Main collection is EMPTY - this is the problem!")
            get_available_games._last_count = len(all_docs['ids'])

        # Extract unique filenames from source paths (primary path)
        filenames = set()
        game_to_files = {}

        for doc_id in all_docs["ids"]:
            if ":" in doc_id:
                # Format: "data/monopoly.pdf:6:2" or "data\\monopoly.pdf:6:2" on Windows
                source_path = doc_id.split(":")[0]
                # Handle both Windows (\) and Unix (/) path separators
                if "/" in source_path or "\\" in source_path:
                    # Use os.path.basename to handle both path separators correctly
                    filename = os.path.basename(source_path)
                    if filename.endswith(".pdf"):
                        filenames.add(filename)

        # Fallback: if the vector DB is empty (e.g., invalid/truncated PDFs),
        # derive available entries from disk or stored name mappings so that
        # admin actions (delete/rename) can still operate.
        if not filenames:
            try:
                data_root = Path(CHROMA_PATH).parent  # usually .../data
                disk_files = []
                try:
                    from . import config as _cfg
                    data_root = Path(_cfg.DATA_PATH)
                except Exception:
                    pass
                if data_root.exists():
                    disk_files = [p.name for p in data_root.rglob("*.pdf")]
                stored_names = get_stored_game_names()
                for fname in disk_files or stored_names.keys():
                    base = os.path.basename(fname)
                    if base.endswith(".pdf"):
                        filenames.add(base)
            except Exception:
                pass

        # Get stored game names (fast lookup, no LLM calls)
        stored_names = get_stored_game_names()

        # Build games list using stored names
        games = []
        game_to_files = {}

        for filename in sorted(filenames):
            if filename in stored_names:
                # Use stored game name
                proper_name = stored_names[filename]
                if VERBOSE_LOGGING:
                    print(f"📦 Using stored game name: '{proper_name}' for '{filename}'")
            else:
                # Fallbacks: try to extract name only if we have valid PDFs,
                # otherwise derive from filename so Admin can still manage items.
                try:
                    proper_name = extract_and_store_game_name(filename)
                except Exception:
                    from .query import improve_fallback_name as _fallback  # type: ignore
                    proper_name = _fallback(filename)

            simple_name = filename.replace(".pdf", "").lower().replace(" ", "_")

            # Deduplicate visible game names
            if proper_name not in games:
                games.append(proper_name)

            # Build list of simple filenames per game
            game_to_files.setdefault(proper_name, []).append(simple_name)

        # Store the mapping for use in filtering (game -> list of simple filenames)
        get_available_games._filename_mapping = game_to_files

        return sorted(games)
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Error getting available games: {e}")
        return []


def store_game_name(filename: str, game_name: str):
    """
    Store the extracted game name in the database.

    Args:
        filename (str): PDF filename (e.g., "catan.pdf")
        game_name (str): Extracted game name (e.g., "Catan")
    """
    try:
        with suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(
                path=CHROMA_PATH, settings=get_chromadb_settings()
            )

            # Get or create a collection specifically for game names
            game_names_collection = persistent_client.get_or_create_collection(
                name="game_names",
                metadata={
                    "description": "Stores extracted game names from PDF filenames"
                },
            )

            # Store the game name with filename as ID
            game_names_collection.upsert(
                ids=[filename],
                documents=[game_name],
                metadatas=[{"filename": filename, "game_name": game_name}],
            )

            print(f"✅ Stored game name: '{game_name}' for '{filename}'")

    except Exception as e:
        print(f"❌ Error storing game name for {filename}: {e}")


def get_stored_game_names() -> Dict[str, str]:
    """
    Retrieve all stored game names from the database.

    Returns:
        Dict[str, str]: Mapping of filename to game name
    """
    try:
        with suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(
                path=CHROMA_PATH, settings=get_chromadb_settings()
            )

            # Try to get the game names collection
            try:
                game_names_collection = persistent_client.get_collection("game_names")
                results = game_names_collection.get()

                # Build mapping from filename to game name
                filename_to_game = {}
                for filename, game_name in zip(results["ids"], results["documents"]):
                    filename_to_game[filename] = game_name

                return filename_to_game

            except Exception:
                # Collection doesn't exist yet
                return {}

    except Exception as e:
        print(f"❌ Error retrieving stored game names: {e}")
        return {}


def extract_and_store_game_name(filename: str) -> str:
    """
    Extract game name from filename and store it in the database.

    Args:
        filename (str): PDF filename

    Returns:
        str: Extracted game name
    """
    # First check if we already have it stored
    stored_names = get_stored_game_names()
    if filename in stored_names:
        if VERBOSE_LOGGING:
            print(f"📦 Using stored game name: '{stored_names[filename]}' for '{filename}'")
        return stored_names[filename]

    # Extract the game name using LLM
    game_name = extract_game_name_from_filename(filename)

    # Store it for future use
    store_game_name(filename, game_name)

    return game_name


def rewrite_search_query(raw_query: str) -> str:
    """Optional LLM-powered rewrite of search query for better retrieval."""
    if not ENABLE_SEARCH_REWRITE:
        return raw_query

    try:
        print("✏️  Rewriting web search query via LLM …")

        # Choose provider following same logic as answer generation but simpler
        if cfg.LLM_PROVIDER.lower() == "anthropic":
            model = ChatAnthropic(model=cfg.SEARCH_REWRITE_MODEL, temperature=0, max_tokens=LLM_MAX_TOKENS)
        elif cfg.LLM_PROVIDER.lower() == "ollama":
            from langchain_community.llms.ollama import Ollama  # pylint: disable=import-error

            model = Ollama(model=cfg.SEARCH_REWRITE_MODEL, base_url=cfg.OLLAMA_URL)
        else:  # openai
            # o3 only supports temperature=1
            temp = 1 if cfg.SEARCH_REWRITE_MODEL == "o3" else 0
            model = ChatOpenAI(model=cfg.SEARCH_REWRITE_MODEL, temperature=temp, timeout=REQUEST_TIMEOUT_S)

        prompt = (
            "You are a search expert. Rewrite the following user question into a concise, "
            "effective web search query. Use quotation marks around exact phrases only if "
            "they are essential. Remove polite fluff. Return one line only, no extra text.\n\n"
            f"User question: {raw_query}\n\nSearch query:"
        )

        rewritten = model.invoke(prompt)
        rewritten_text = getattr(rewritten, "content", str(rewritten)).strip()
        # Guard: fall back if result too short or too long
        if 3 <= len(rewritten_text) <= 200:
            print(f"✏️  Rewritten query: {rewritten_text!r}")
            return rewritten_text
        print("⚠️ Rewriter produced unusable output; falling back to raw query")
    except Exception as e:
        print(f"⚠️ Query rewrite failed: {e}")

    return raw_query


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
    
    # Query RAG system

    # Connect to the database
    print("🔗 Connecting to the database...")
    embedding_function = get_embedding_function()
    with suppress_chromadb_telemetry():
        persistent_client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=get_chromadb_settings()
        )
        db = Chroma(client=persistent_client, embedding_function=embedding_function)

    # Database connection successful

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
    variant_queries = generate_query_variants(query_text, game_names)
    if not variant_queries:
        variant_queries = [query_text]
    # Build variant search strings including chat history when present
    search_variants = [
        (f"{chat_history}\n\n{v}" if chat_history else v) for v in variant_queries
    ]
    # Allocate per-variant k to avoid over-fetching
    per_variant_k = max(8, k_results // max(1, len(search_variants)))
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
    included_chunks_debug = []  # Collect details of chunks actually included in context
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
            included_chunks_debug.append({
                "source": (meta.get("source") or "unknown"),
                "page": meta.get("page"),
                "section": (meta.get("section") or "").strip(),
                "section_number": (meta.get("section_number") or "").strip(),
                "length": len(t),
                "preview": t.replace("\n", " ")[:160],
            })
        except Exception:
            pass
        used += len(t) + 8
        if used >= RAG_CONTEXT_CHAR_LIMIT:
            break
    context_text = "\n\n---\n\n".join(parts)

    # Provide an explicit allowlist of section numbers derived from retrieved sources
    try:
        allowed_sections = []
        seen_allow = set()
        for doc, _ in results:
            meta = getattr(doc, 'metadata', {}) or {}
            sec = (meta.get('section_number') or '').strip()
            if not sec:
                m = re.match(r"^(\d+(?:\.\d+)*)\b", (meta.get('section') or '').strip())
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

    # Remove trailing auto-appended numeric citations; do not modify model output here

    # Build structured source metadata for citations (filter by relevance)
    sources = []
    qualifying_sources = []
    all_pdf_sources = []
    
    for doc, score in results:
        meta_doc = doc.metadata
        src_path = meta_doc.get("source", "")
        
        # Handle web results separately (always include regardless of score)
        if isinstance(src_path, str) and src_path.startswith("http"):
            sources.append(src_path)
            continue
        
        # Collect PDF source info
        source_info = {
            "filepath": src_path,
            "page": meta_doc.get("page"),
            "section": meta_doc.get("section"),
            "score": score
        }
        all_pdf_sources.append(source_info)
        
        # Check if it meets the quality threshold (lower scores = better similarity)
        if score <= CITATION_SCORE_THRESHOLD:
            print(f"  📊 Including source in citation: {Path(src_path).name if src_path else 'unknown'} (score: {score:.4f})")
            qualifying_sources.append(source_info)
        else:
            print(f"  📊 Source above threshold: {Path(src_path).name if src_path else 'unknown'} (score: {score:.4f} > threshold {CITATION_SCORE_THRESHOLD})")
    
    # Add qualifying sources
    for source_info in qualifying_sources:
        sources.append({
            "filepath": source_info["filepath"],
            "page": source_info["page"],
            "section": source_info["section"],
        })
    
    # If we don't have enough qualifying sources, add the best remaining ones
    if len(qualifying_sources) < CITATION_MIN_SOURCES and len(all_pdf_sources) > len(qualifying_sources):
        print(f"  📊 Only {len(qualifying_sources)} sources met threshold, adding {CITATION_MIN_SOURCES - len(qualifying_sources)} more from best available")
        # Sort all sources by score (ascending - lower is better) and take the best remaining
        remaining_sources = [s for s in all_pdf_sources if s not in qualifying_sources]
        remaining_sources.sort(key=lambda x: x["score"])
        
        needed = CITATION_MIN_SOURCES - len(qualifying_sources)
        for source_info in remaining_sources[:needed]:
            print(f"  📊 Adding minimum source: {Path(source_info['filepath']).name if source_info['filepath'] else 'unknown'} (score: {source_info['score']:.4f})")
            sources.append({
                "filepath": source_info["filepath"],
                "page": source_info["page"],
                "section": source_info["section"],
            })
    response = {
        "response_text": response_text,
        "sources": sources,
        "original_query": query_text,
        "context": context_text,
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

    # ---- Build prompt and retrieve context (reuse logic from query_rag) ----

    print("🔗 Connecting to the database...")
    embedding_function = get_embedding_function()
    with suppress_chromadb_telemetry():
        persistent_client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=get_chromadb_settings()
        )
        db = Chroma(client=persistent_client, embedding_function=embedding_function)

    search_query = f"{chat_history}\n\n{query_text}" if chat_history else query_text
    
    print("🔍 Searching in the database…")
    print(f"  Search query (first 500 chars): '{search_query[:500]}'")
    if len(search_query) > 500:
        print(f"  ... (truncated, full length: {len(search_query)} chars)")
    if selected_game:
        print(f"  Filtering by game: '{selected_game}'")

    # Fetch DB results (same k logic)
    k_results = 200 if selected_game else 40  # balanced pool size for game-specific searches
    print(f"🔎 Fetching {k_results} results from database...")
            # Try server-side metadata filtering first (if supported)
    metadata_filter = None
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
    # Multi-query retrieval with plural/synonym expansion
    variant_queries = generate_query_variants(query_text, game_names)
    if not variant_queries:
        variant_queries = [query_text]
    search_variants = [
        (f"{chat_history}\n\n{v}" if chat_history else v) for v in variant_queries
    ]
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
        all_results.extend(batch)
    print(f"📊 Database returned {len(all_results)} total results")

    # No hardcoded section fast path: rely on standard retrieval and ranking

    if len(all_results) == 0:
        print("❌ ERROR: No results from database! This will cause 'I don't know' response")
        print("  Check if:")
        print("  - Database has been populated")
        print("  - Embedding function is working")
        print("  - Query is not too specific")

    # Merge similarity hits with simple dedupe
    results = []
    seen_keys = set()
    def _key_for(d):
        m = getattr(d, "metadata", {}) or {}
        return (m.get("source"), m.get("page"), (m.get("section") or "").strip())

    for d, s in all_results:
        k = _key_for(d)
        if k not in seen_keys:
            results.append((d, s))
            seen_keys.add(k)

    # Use only the top-N results and cap total context size to control token usage
    results = results[: RAG_MAX_DOCS]

    # Heuristic re-rank for streaming: prefer numeric-section chunks for phase queries
    try:
        if re.search(r"\bphase(s)?\b", query_text, re.IGNORECASE):
            def _is_numeric_section(doc):
                sec = (getattr(doc, 'metadata', {}) or {}).get('section') or ''
                return bool(re.match(r"^\s*\d+(?:\.\d+)*\b", str(sec)))
            results.sort(key=lambda pair: (not _is_numeric_section(pair[0]), pair[1]))
    except Exception:
        pass
    print(f"📊 Using top {len(results)} results (capped by RAG_MAX_DOCS={RAG_MAX_DOCS})")

    # Show final results summary
    print(f"\n📋 FINAL RESULTS SUMMARY:")
    print(f"  Total results for context: {len(results)}")
    if results:
        print("  Top 5 results:")
        for i, (doc, score) in enumerate(results[:5]):
            source = doc.metadata.get("source", "unknown")
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"    {i+1}. Score: {score:.4f}, Source: {source}")
            print(f"       Content: '{content_preview}...'")
    else:
        print("  ❌ NO RESULTS - This will definitely cause 'I don't know' response!")

    # Supplement with web search if enabled
    effective_web_search = ENABLE_WEB_SEARCH if enable_web is None else enable_web
    print(f"\n🌐 Web search check: effective_web_search={effective_web_search}")
    if effective_web_search:
        quoted_game = f'"{game_names[0]}" ' if game_names else ""
        pre_query = f"{quoted_game}{query_text}"
        web_query = rewrite_search_query(pre_query)
        print(f"🌐 Web search enabled – fetching top {WEB_SEARCH_RESULTS} snippets with query: {web_query!r}…")
        web_docs = perform_web_search(web_query, k=WEB_SEARCH_RESULTS)
        print(f"🌐 Retrieved {len(web_docs)} web snippets")
        if web_docs:
            results.extend([(doc, 0.0) for doc in web_docs])
            print(f"📊 Total results after adding web: {len(results)}")
        else:
            print("⚠️ No web results found")

    # Build prompt
    print("🔮 Building the prompt …")
    # Build context with a hard character cap (same logic as non-streaming path)
    parts = []
    used = 0
    for doc, _score in results:
        t = doc.page_content
        if not t:
            continue
        if used + len(t) + 8 > RAG_CONTEXT_CHAR_LIMIT:
            t = t[: max(0, RAG_CONTEXT_CHAR_LIMIT - used)]
        parts.append(t)
        used += len(t) + 8
        if used >= RAG_CONTEXT_CHAR_LIMIT:
            break
    context_text = "\n\n---\n\n".join(parts)
    print(f"📏 Context length (capped): {len(context_text)} characters (limit={RAG_CONTEXT_CHAR_LIMIT})")
    if len(context_text) == 0:
        print("❌ ERROR: Empty context! This will cause 'I don't know' response")
    else:
        print(f"📄 Context preview (first 300 chars): '{context_text[:300]}...'")

    composite_question = (
        f"Previous conversation (for context):\n{chat_history}\n\nUser's latest question: {query_text}"
        if chat_history
        else query_text
    )
    print(f"❓ Composite question length: {len(composite_question)} characters")

    try:
        print("📝 Loading improved prompt template...")
        prompt = load_jinja2_prompt(
            context=context_text,
            question=composite_question,
            template_name="rag_query_improved.txt",
        )
        print("✅ Loaded improved template successfully")
    except Exception as e:
        print(f"⚠️ Failed to load improved template: {e}")
        print("📝 Falling back to default template...")
        prompt = load_jinja2_prompt(context=context_text, question=composite_question)
        print("✅ Loaded default template successfully")

    print(f"📏 Final prompt length: {len(prompt)} characters")
    print(f"📄 Prompt preview (first 500 chars): '{prompt[:500]}...'")

    # ---- Create LLM (streaming enabled) ----

    print("🍳 Generating the response (streaming)…")
    print(f"🤖 Using LLM Provider: {cfg.LLM_PROVIDER}")
    print(f"🤖 Using Model: {cfg.GENERATOR_MODEL}")

    model_kwargs = {"streaming": True}
    if cfg.LLM_PROVIDER.lower() == "ollama":
        from langchain_community.llms.ollama import Ollama  # pylint: disable=import-error
        print(f"🤖 Creating Ollama model with base_url: {cfg.OLLAMA_URL}")
        model = Ollama(model=cfg.GENERATOR_MODEL, base_url=cfg.OLLAMA_URL, **model_kwargs)
    elif cfg.LLM_PROVIDER.lower() == "anthropic":
        print("🤖 Creating Anthropic model...")
        # Anthropic requires max_tokens for reliable streaming output
        model = ChatAnthropic(
            model=cfg.GENERATOR_MODEL,
            temperature=0,
            max_tokens=LLM_MAX_TOKENS,
            **model_kwargs,
        )
    elif cfg.LLM_PROVIDER.lower() == "openai":
        if _openai_requires_default_temperature(cfg.GENERATOR_MODEL):
            print("🤖 Creating OpenAI model with temperature: 1 (required by reasoning models)")
            model = ChatOpenAI(model=cfg.GENERATOR_MODEL, temperature=1, timeout=REQUEST_TIMEOUT_S, **model_kwargs)
        else:
            print("🤖 Creating OpenAI model with temperature: 0")
            model = ChatOpenAI(model=cfg.GENERATOR_MODEL, temperature=0, timeout=REQUEST_TIMEOUT_S, **model_kwargs)
    else:
        raise ValueError("Unsupported LLM_PROVIDER: " + cfg.LLM_PROVIDER)

    # Pre-compute numeric sections from retrieved results for fallback citation injection
    numeric_sections: List[str] = []
    try:
        seen_sec: set[str] = set()
        for d, _s in results:
            sec_label = (d.metadata or {}).get("section") or ""
            m = re.match(r"^\s*(\d+(?:\.\d+)*)\b", str(sec_label))
            if m:
                sec = m.group(1)
                if sec not in seen_sec:
                    seen_sec.add(sec)
                    numeric_sections.append(sec)
        numeric_sections = numeric_sections[:12]
    except Exception:
        numeric_sections = []

    # ---- Token generator ----

    def _token_gen():
        import random, time as _time

        def _is_rate_limited(err: Exception) -> bool:
            s = str(err).lower()
            return ("429" in s) or ("rate_limit" in s) or ("rate limit" in s) or ("overload" in s)

        max_attempts = 3
        saw_citation = False
        for attempt in range(max_attempts):
            # On retries, shrink prompt to reduce input tokens pressure
            attempt_prompt = prompt
            if attempt > 0:
                shrink_factor = 0.8 ** attempt
                new_len = max(1000, int(len(prompt) * shrink_factor))
                attempt_prompt = prompt[:new_len]
                print(f"⏳ Retry {attempt}/{max_attempts-1}: shrinking prompt to {len(attempt_prompt)} chars")
            try:
                for chunk in model.stream(attempt_prompt):
                    # chunk can be ChatGenerationChunk or similar; extract text portion
                    text_part = getattr(chunk, "content", None)
                    if text_part is None and hasattr(chunk, "message"):
                        text_part = getattr(chunk.message, "content", "")
                    if text_part is None:
                        text_part = str(chunk)
                    if text_part:
                        if not saw_citation and re.search(r"\[\d+\.\d+[^\]]*\]", text_part or ""):
                            saw_citation = True
                        yield text_part
                # Do not append fallback bracket list of numeric citations
                return
            except Exception as e:
                print(f"❌ ERROR during token generation: {e}")
                if _is_rate_limited(e) and attempt < max_attempts - 1:
                    # Exponential backoff with jitter, then retry
                    delay = (2 ** attempt) + random.uniform(0, 0.75)
                    print(f"🔁 Rate limited/overloaded; retrying in {delay:.2f}s …")
                    _time.sleep(delay)
                    continue
                # Non-retryable or out of attempts – propagate
                raise

    # Build structured source metadata for citations (filter by relevance)
    sources = []
    qualifying_sources = []
    all_pdf_sources = []
    
    for doc, score in results:
        meta_doc = doc.metadata
        src_path = meta_doc.get("source", "")
        
        # Handle web results separately (always include regardless of score)
        if isinstance(src_path, str) and src_path.startswith("http"):
            sources.append(src_path)
            continue
        
        # Collect PDF source info
        source_info = {
            "filepath": src_path,
            "page": meta_doc.get("page"),
            "section": meta_doc.get("section"),
            "score": score
        }
        all_pdf_sources.append(source_info)
        
        # Check if it meets the quality threshold (lower scores = better similarity)
        if score <= CITATION_SCORE_THRESHOLD:
            print(f"  📊 Including source in citation: {Path(src_path).name if src_path else 'unknown'} (score: {score:.4f})")
            qualifying_sources.append(source_info)
        else:
            print(f"  📊 Source above threshold: {Path(src_path).name if src_path else 'unknown'} (score: {score:.4f} > threshold {CITATION_SCORE_THRESHOLD})")
    
    # Add qualifying sources
    for source_info in qualifying_sources:
        sources.append({
            "filepath": source_info["filepath"],
            "page": source_info["page"],
            "section": source_info["section"],
        })
    
    # If we don't have enough qualifying sources, add the best remaining ones
    if len(qualifying_sources) < CITATION_MIN_SOURCES and len(all_pdf_sources) > len(qualifying_sources):
        print(f"  📊 Only {len(qualifying_sources)} sources met threshold, adding {CITATION_MIN_SOURCES - len(qualifying_sources)} more from best available")
        # Sort all sources by score (ascending - lower is better) and take the best remaining
        remaining_sources = [s for s in all_pdf_sources if s not in qualifying_sources]
        remaining_sources.sort(key=lambda x: x["score"])
        
        needed = CITATION_MIN_SOURCES - len(qualifying_sources)
        for source_info in remaining_sources[:needed]:
            print(f"  📊 Adding minimum source: {Path(source_info['filepath']).name if source_info['filepath'] else 'unknown'} (score: {source_info['score']:.4f})")
            sources.append({
                "filepath": source_info["filepath"],
                "page": source_info["page"],
                "section": source_info["section"],
            })
    
    print(f"📚 Final sources: {sources}")

    # Post-generation fallback: ensure at least some numeric citations are present
    try:
        has_numeric_cite = re.search(r"\[\d+\.\d+[^\]]*\]", prompt) is not None
        if not has_numeric_cite:
            numeric_sections = []
            for src in sources:
                if isinstance(src, dict):
                    sec = (src.get('section') or '').strip()
                    m = re.match(r"^(\d+(?:\.\d+)*)\b", sec)
                    if m:
                        numeric_sections.append(m.group(1))
            seen = set()
            dedup = []
            for s in numeric_sections:
                if s not in seen:
                    dedup.append(s)
                    seen.add(s)
            if dedup:
                # Prepend minimal bracket list to the context so the model sees them on retry attempts
                context_text = f"{''.join(f'[{s}]' for s in dedup[:12])}\n\n{context_text}"
    except Exception:
        pass

    meta = {
        "sources": sources,
        "context": context_text,
        "prompt": prompt,
        "original_query": query_text,
    }

    print("="*80)
    print("🔍 STREAM_QUERY_RAG DEBUG END")
    print("="*80 + "\n")

    return _token_gen(), meta


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

    response = query_rag(query_text, selected_game, game_names=args.game_names)

    # Extract game name from first source
    if response["sources"]:
        first_src = response["sources"][0]
        if isinstance(first_src, dict):
            src_path = first_src.get("filepath", "")
        else:
            src_path = first_src
        game_name = os.path.basename(src_path).split()[0].capitalize() if src_path else "Game"
    else:
        game_name = "Game"

    response_text = f"🤖 {game_name}: {response['response_text']}"

    if include_sources:
        response_text += f"\n\n\n 📜Sources: {response['sources']}"

    if include_context:
        response_text += f"\n\n\n 🌄Context: {response['context']}"

    print(response_text)


if __name__ == "__main__":
    main()
