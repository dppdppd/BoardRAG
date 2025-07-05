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
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

from config import (
    CHROMA_PATH,
    GENERATOR_MODEL,
    LLM_PROVIDER,
    OLLAMA_URL,
    disable_chromadb_telemetry,
    get_chromadb_settings,
    suppress_chromadb_telemetry,
    validate_config,
)
from embedding_function import get_embedding_function
from templates.load_jinja_template import load_jinja2_prompt

# Disable ChromaDB telemetry and validate configuration after imports
disable_chromadb_telemetry()
validate_config()


def extract_game_name_from_filename(filename: str) -> str:
    """
    Use LLM API to extract the proper game name from a PDF filename.
    Also reads the first few pages of the PDF for better accuracy.

    Args:
        filename (str): PDF filename like "up front rulebook bw.pdf"

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
            
            # Get text from first 3 pages (or fewer if PDF is shorter)
            pages_to_read = min(3, len(documents))
            page_texts = []
            
            for i in range(pages_to_read):
                page_text = documents[i].page_content.strip()
                if page_text:
                    # Take first 500 chars per page to keep context manageable
                    page_texts.append(page_text[:500])
            
            if page_texts:
                pdf_context = "\n\n".join(page_texts)
                print(f"📖 Extracted {len(pdf_context)} chars from first {pages_to_read} pages of {filename}")
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

    for attempt in range(max_retries):
        try:
            # Use configured provider for filename extraction
            if LLM_PROVIDER.lower() == "anthropic":
                model = ChatAnthropic(model=GENERATOR_MODEL, temperature=0)
            elif LLM_PROVIDER.lower() == "ollama":
                from langchain_community.llms.ollama import Ollama

                model = Ollama(model=GENERATOR_MODEL, base_url=OLLAMA_URL)
            else:  # openai
                model = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)

            # Build prompt with optional PDF content
            context_section = ""
            if pdf_context:
                context_section = f"""

CONTENT FROM FIRST PAGES:
{pdf_context}

"""

            prompt = f"""You are a board game expert. Extract the proper board game name from this filename: "{filename}"{context_section}

IMPORTANT: Use your knowledge of published board games to identify what this filename refers to.
If PDF content is provided above, use it to confirm the correct game title.

Guidelines:
- Return ONLY the official published game name
- Remove file-related words: "rules", "manual", "rulebook", "complete", "rework", "bw", "color", "v1", "v2"
- Consider common board game abbreviations (CNA, D&D, MTG, etc.)
- Think about what actual published board games exist
- Use proper capitalization for official game titles
- If you see the actual game title in the PDF content, prefer that over filename guessing

Examples of proper extraction:
- "monopoly.pdf" → "Monopoly"
- "catan.pdf" → "Catan"
- "dnd-players-handbook.pdf" → "Dungeons & Dragons"
- "mtg-comprehensive-rules.pdf" → "Magic: The Gathering"
- "up-front-rulebook-bw.pdf" → "Up Front"
- "ticket-to-ride.pdf" → "Ticket to Ride"

For abbreviations, think about famous board games:
- CNA is a well-known abbreviation for a famous complex wargame
- Consider what major published board games use these initials

Filename: {filename}
Official game name:"""

            response = model.invoke(prompt)
            game_name = response.content.strip().strip("\"'")

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
        filename_to_simple = {}  # Map cleaned name back to simple name for filtering

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
        for filename in sorted(filenames):
            if filename in stored_names:
                # Use stored game name
                proper_name = stored_names[filename]
                print(f"📦 Using stored game name: '{proper_name}' for '{filename}'")
            else:
                # Fallback to extracting (this should be rare after initial setup)
                print(f"⚠️ Game name not found in storage, extracting for: '{filename}'")
                proper_name = extract_and_store_game_name(filename)

            simple_name = filename.replace(".pdf", "").split()[0].lower()
            games.append(proper_name)
            filename_to_simple[proper_name] = simple_name

        # Store the mapping for use in filtering
        get_available_games._filename_mapping = filename_to_simple

        return sorted(games)
    except Exception as e:
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
        print(f"📦 Using stored game name: '{stored_names[filename]}' for '{filename}'")
        return stored_names[filename]

    # Extract the game name using LLM
    game_name = extract_game_name_from_filename(filename)

    # Store it for future use
    store_game_name(filename, game_name)

    return game_name


def query_rag(
    query_text: str, selected_game: Optional[str] = None
) -> Dict[str, Union[str, Dict]]:
    """
    Queries the RAG model with the given query and returns the response.

    Args:
        query_text (str): The query to be passed to the RAG model.
        selected_game (Optional[str]): Game to filter results by (e.g., 'monopoly', 'catan')

    Returns:
        Dict: The response from the RAG model.
    """

    # Connect to the database
    print("🔗 Connecting to the database...")
    embedding_function = get_embedding_function()
    with suppress_chromadb_telemetry():
        persistent_client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=get_chromadb_settings()
        )
        db = Chroma(client=persistent_client, embedding_function=embedding_function)

    # Search in the database
    print("🔍 Searching in the database...")
    print(f"  Query text: '{query_text}'")
    if selected_game:
        print(f"  Filtering by game: '{selected_game}'")

    # Test embedding generation
    try:
        test_embedding = embedding_function.embed_query(query_text)
        print(f"  Generated embedding dimensions: {len(test_embedding)}")
        print(f"  First few embedding values: {test_embedding[:3]}")
    except Exception as e:
        print(f"  ❌ Embedding generation failed: {e}")
        return {
            "response_text": "Error generating embeddings",
            "sources": [],
            "context": "",
            "prompt": "",
            "original_query": query_text,
        }

    # Get results from database (more if filtering is needed)
    k_results = (
        75 if selected_game else 40
    )  # Increased to maintain context with smaller chunks
    all_results = db.similarity_search_with_score(query_text, k=k_results)

    # Apply game filter if specified
    if selected_game:
        # Filter results by source path containing the game name
        filtered_results = []
        for doc, score in all_results:
            source = doc.metadata.get("source", "")
            if selected_game in source.lower():
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
        print(f"  Best match content preview: {results[0][0].page_content[:100]}...")
        # Show which games the results come from
        sources = [doc.metadata.get("source", "unknown") for doc, _ in results[:3]]
        unique_sources = list(set([Path(s).name for s in sources if s != "unknown"]))
        print(f"  Top sources: {', '.join(unique_sources[:3])}")
    else:
        print("  No results found in similarity search")

    # Build the prompt
    print("🔮 Building the prompt ...")
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # Try improved template first, fallback to original
    try:
        prompt = load_jinja2_prompt(
            context=context_text,
            question=query_text,
            template_name="rag_query_improved.txt",
        )
    except Exception:
        prompt = load_jinja2_prompt(context=context_text, question=query_text)

    print("🍳 Generating the response...")
    # Temperature is fixed to 0 for deterministic answers that are easier to test.

    if LLM_PROVIDER.lower() == "ollama":
        # Import here to avoid requiring Ollama for OpenAI users.
        from langchain_community.llms.ollama import Ollama  # pylint: disable=import-error

        model = Ollama(model=GENERATOR_MODEL, base_url=OLLAMA_URL)
    elif LLM_PROVIDER.lower() == "anthropic":
        model = ChatAnthropic(model=GENERATOR_MODEL, temperature=0)
    elif LLM_PROVIDER.lower() == "openai":
        model = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)
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
    args = parser.parse_args()
    query_text = args.query_text
    selected_game = args.game
    include_sources = args.include_sources
    include_context = args.include_context

    # Convert proper game name to simple filename for filtering
    game_filter = None
    if selected_game:
        # Get available games to populate the mapping
        get_available_games()  # Call to populate the _filename_mapping
        mapping = getattr(get_available_games, "_filename_mapping", {})
        # Try to find the game in the mapping, or use as-is
        game_filter = mapping.get(selected_game, selected_game.lower())

    response = query_rag(query_text, game_filter)

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
