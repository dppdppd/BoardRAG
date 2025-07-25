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
from config import (
    CHROMA_PATH,
    GENERATOR_MODEL,
    LLM_PROVIDER,
    OLLAMA_URL,
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
    SEARCH_REWRITE_MODEL,
)
from embedding_function import get_embedding_function
from templates.load_jinja_template import load_jinja2_prompt

# Optional import for web search; only loaded when enabled to avoid extra dependency at runtime
if ENABLE_WEB_SEARCH:
    try:
        from duckduckgo_search import ddg
    except ImportError:  # Fallback if dependency missing
        ddg = None

# Global config ----------------------------------------------------------------
# Importing here to avoid circular deps and keep single source of truth.
from config import (
    CHROMA_PATH,
    GENERATOR_MODEL,
    LLM_PROVIDER,
    OLLAMA_URL,
    disable_chromadb_telemetry,
    get_chromadb_settings,
    suppress_chromadb_telemetry,
    validate_config,
    ENABLE_WEB_SEARCH,
    WEB_SEARCH_RESULTS,
)

# ---------------------------------------------------------------------------
# Verbose logging toggle (set VERBOSE=true to enable extra prints)
# ---------------------------------------------------------------------------
VERBOSE_LOGGING = os.getenv("VERBOSE", "False").lower() in {"1", "true", "yes"}


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
        from config import DATA_PATH

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
            if LLM_PROVIDER.lower() == "anthropic":
                model = ChatAnthropic(model=GENERATOR_MODEL, temperature=0)
            elif LLM_PROVIDER.lower() == "ollama":
                from langchain_community.llms.ollama import Ollama

                model = Ollama(model=GENERATOR_MODEL, base_url=OLLAMA_URL)
            else:  # openai
                # o3 model only supports temperature=1, other OpenAI models use 0 for determinism
                if GENERATOR_MODEL == "o3":
                    model = ChatOpenAI(model=GENERATOR_MODEL, temperature=1)
                else:
                    model = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)

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
                print(
                    f"Successfully extracted game name: '{game_name}' from '{filename}'"
                )
                return game_name
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
    if debug and last_raw_response is not None:
        print("\n❌ Using fallback – last raw LLM response was:\n")
        print(last_raw_response)
        print()

    print(f"Using fallback name: '{fallback_name}' for '{filename}'")
    return fallback_name


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

        # Extract unique filenames from source paths
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
                # Fallback to extracting (this should be rare after initial setup)
                print(f"⚠️ Game name not found in storage, extracting for: '{filename}'")
                proper_name = extract_and_store_game_name(filename)

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
        if LLM_PROVIDER.lower() == "anthropic":
            model = ChatAnthropic(model=SEARCH_REWRITE_MODEL, temperature=0)
        elif LLM_PROVIDER.lower() == "ollama":
            from langchain_community.llms.ollama import Ollama  # pylint: disable=import-error

            model = Ollama(model=SEARCH_REWRITE_MODEL, base_url=OLLAMA_URL)
        else:  # openai
            # o3 only supports temperature=1
            temp = 1 if SEARCH_REWRITE_MODEL == "o3" else 0
            model = ChatOpenAI(model=SEARCH_REWRITE_MODEL, temperature=temp)

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
    k_results = (
        75 if selected_game else 40
    )  # Increased to maintain context with smaller chunks
    all_results = db.similarity_search_with_score(search_query, k=k_results)

    # Apply game filter if specified
    if selected_game:
        # selected_game can be a list of filter strings or a single string
        filters = selected_game if isinstance(selected_game, list) else [selected_game]

        filtered_results = []
        for doc, score in all_results:
            source = doc.metadata.get("source", "").lower()
            # Normalize source filename the same way we create filter names
            import os
            source_filename = os.path.basename(source)
            normalized_source = source_filename.replace(".pdf", "").replace(" ", "_")
            
            if any(f in normalized_source for f in filters):
                filtered_results.append((doc, score))
                if len(filtered_results) >= 40:  # Increased to maintain context
                    break

        results = filtered_results
        print(
            f"  Filtered from {len(all_results)} to {len(results)} results for game '{selected_game}'"
        )
    else:
        results = all_results
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
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

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

    if LLM_PROVIDER.lower() == "ollama":
        # Import here to avoid requiring Ollama for OpenAI users.
        from langchain_community.llms.ollama import Ollama  # pylint: disable=import-error

        model = Ollama(model=GENERATOR_MODEL, base_url=OLLAMA_URL, **model_kwargs)
    elif LLM_PROVIDER.lower() == "anthropic":
        model = ChatAnthropic(model=GENERATOR_MODEL, temperature=0, **model_kwargs)
    elif LLM_PROVIDER.lower() == "openai":
        # o3 model only supports temperature=1, others use 0
        temp = 1 if GENERATOR_MODEL == "o3" else 0
        model = ChatOpenAI(model=GENERATOR_MODEL, temperature=temp, **model_kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Must be 'openai', 'anthropic', or 'ollama'"
        )

    response_raw = model.invoke(prompt)
    # Convert LangChain message objects to string content when necessary.
    if hasattr(response_raw, "content"):
        response_text = response_raw.content
    else:
        response_text = response_raw

    sources = [doc.metadata.get("id", None) for doc, _score in results]
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
    k_results = 75 if selected_game else 40
    print(f"🔎 Fetching {k_results} results from database...")
    all_results = db.similarity_search_with_score(search_query, k=k_results)
    print(f"📊 Database returned {len(all_results)} total results")

    if len(all_results) == 0:
        print("❌ ERROR: No results from database! This will cause 'I don't know' response")
        print("  Check if:")
        print("  - Database has been populated")
        print("  - Embedding function is working")
        print("  - Query is not too specific")

    # Apply optional game filter
    if selected_game:
        filters = [selected_game] if isinstance(selected_game, str) else selected_game
        print(f"🎯 Applying game filter: {filters}")
        filtered_results = []
        for i, (doc, score) in enumerate(all_results):
            source = doc.metadata.get("source", "").lower()
            # Normalize source filename the same way we create filter names
            # Extract just the filename from the path and normalize it
            import os
            source_filename = os.path.basename(source)
            normalized_source = source_filename.replace(".pdf", "").replace(" ", "_")
            
            print(f"  Result {i+1}: source='{source}', normalized='{normalized_source}', score={score:.4f}")
            if any(f in normalized_source for f in filters):
                filtered_results.append((doc, score))
                print(f"    ✅ PASSED filter ('{normalized_source}' contains: {[f for f in filters if f in normalized_source]})")
                if len(filtered_results) >= 40:
                    print(f"    🛑 Reached limit of 40 filtered results")
                    break
            else:
                print(f"    ❌ FAILED filter ('{normalized_source}' doesn't contain any of: {filters})")
        results = filtered_results
        print(f"📊 After filtering: {len(results)} results remain")
        
        if len(results) == 0:
            print("❌ ERROR: No results after game filtering! This will cause 'I don't know' response")
            print("  Check if:")
            print("  - Game filter matches actual source filenames")
            print("  - Documents were ingested with correct source metadata")
            print("  - Filter is not too restrictive")
    else:
        results = all_results
        print(f"📊 Using all {len(results)} results (no game filter)")

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
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    print(f"📏 Context length: {len(context_text)} characters")
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
    print(f"🤖 Using LLM Provider: {LLM_PROVIDER}")
    print(f"🤖 Using Model: {GENERATOR_MODEL}")

    model_kwargs = {"streaming": True}
    if LLM_PROVIDER.lower() == "ollama":
        from langchain_community.llms.ollama import Ollama  # pylint: disable=import-error
        print(f"🤖 Creating Ollama model with base_url: {OLLAMA_URL}")
        model = Ollama(model=GENERATOR_MODEL, base_url=OLLAMA_URL, **model_kwargs)
    elif LLM_PROVIDER.lower() == "anthropic":
        print("🤖 Creating Anthropic model...")
        model = ChatAnthropic(model=GENERATOR_MODEL, temperature=0, **model_kwargs)
    elif LLM_PROVIDER.lower() == "openai":
        temp = 1 if GENERATOR_MODEL == "o3" else 0
        print(f"🤖 Creating OpenAI model with temperature: {temp}")
        model = ChatOpenAI(model=GENERATOR_MODEL, temperature=temp, **model_kwargs)
    else:
        raise ValueError("Unsupported LLM_PROVIDER: " + LLM_PROVIDER)

    # ---- Token generator ----

    def _token_gen():
        try:
            for chunk in model.stream(prompt):
                # chunk can be ChatGenerationChunk or similar; extract text portion
                text_part = getattr(chunk, "content", None)
                if text_part is None and hasattr(chunk, "message"):
                    text_part = getattr(chunk.message, "content", "")
                if text_part is None:
                    text_part = str(chunk)
                if text_part:
                    yield text_part
        except Exception as e:
            print(f"❌ ERROR during token generation: {e}")
            raise

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    print(f"📚 Final sources: {sources}")

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
    if response["sources"] and response["sources"][0]:
        game_name = os.path.basename(response["sources"][0]).split()[0].capitalize()
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
