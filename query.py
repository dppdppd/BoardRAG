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

    Args:
        filename (str): PDF filename like "up front rulebook bw.pdf"

    Returns:
        str: Cleaned game name like "Up Front"
    """
    try:
        # Use configured provider for filename extraction
        if LLM_PROVIDER.lower() == "anthropic":
            model = ChatAnthropic(model=GENERATOR_MODEL, temperature=0)
        elif LLM_PROVIDER.lower() == "ollama":
            from langchain_community.llms.ollama import Ollama

            model = Ollama(model=GENERATOR_MODEL, base_url=OLLAMA_URL)
        else:  # openai
            model = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)

        prompt = f"""Extract the board game name from this PDF filename: "{filename}"

Rules:
- Return only the game name, nothing else
- Use proper capitalization (e.g., "Up Front", "Ticket to Ride")
- Remove words like "rulebook", "manual", "bw", "color", etc.
- If unclear, make your best guess based on common board game names

Examples:
- "monopoly.pdf" → "Monopoly"
- "up front rulebook bw.pdf" → "Up Front"
- "ticket-to-ride.pdf" → "Ticket to Ride"
- "catan.pdf" → "Catan"

Filename: {filename}
Game name:"""

        response = model.invoke(prompt)
        game_name = response.content.strip().strip("\"'")

        # Fallback to simple extraction if API call fails
        if not game_name or len(game_name) > 50:
            game_name = filename.replace(".pdf", "").split()[0].title()

        return game_name
    except Exception as e:
        print(f"Error extracting game name from {filename}: {e}")
        # Fallback to simple extraction
        return filename.replace(".pdf", "").split()[0].title()


def get_available_games() -> List[str]:
    """
    Get a list of available games from the database with proper names.

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
                # Format: "data/monopoly.pdf:6:2"
                source_path = doc_id.split(":")[0]
                if "/" in source_path:
                    filename = source_path.split("/")[-1]
                    if filename.endswith(".pdf"):
                        filenames.add(filename)

        # Use OpenAI to extract proper game names
        games = []
        for filename in sorted(filenames):
            proper_name = extract_game_name_from_filename(filename)
            simple_name = filename.replace(".pdf", "").split()[0].lower()
            games.append(proper_name)
            filename_to_simple[proper_name] = simple_name

        # Store the mapping for use in filtering
        get_available_games._filename_mapping = filename_to_simple

        return sorted(games)
    except Exception as e:
        print(f"Error getting available games: {e}")
        return []


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
    k_results = 50 if selected_game else 25
    all_results = db.similarity_search_with_score(query_text, k=k_results)

    # Apply game filter if specified
    if selected_game:
        # Filter results by source path containing the game name
        filtered_results = []
        for doc, score in all_results:
            source = doc.metadata.get("source", "")
            if selected_game in source.lower():
                filtered_results.append((doc, score))
                if len(filtered_results) >= 25:  # Limit to top 25 for more context
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
