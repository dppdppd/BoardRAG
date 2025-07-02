"""
Python file that populates the database with the documents from the data folder, along with their embeddings.
Embeddings are calculated using either OpenAI or a local Ollama model, depending
on the `LLM_PROVIDER` environment variable.
The documents are split into chunks and added to the database, once the embedding has been calculated.
"""

import argparse
import os
import shutil
import warnings
from typing import List

import chromadb
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    disable_chromadb_telemetry,
    get_chromadb_settings,
    suppress_chromadb_telemetry,
    validate_config,
)
from embedding_function import get_embedding_function

# Ignore deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Disable ChromaDB telemetry and validate configuration after imports
disable_chromadb_telemetry()
validate_config()


def load_documents():
    """
    Extracts the documents from the data folder, processes them using Haystack and loads them into the database.
    """
    return PyPDFDirectoryLoader(DATA_PATH).load()


def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: List[Document]) -> bool:
    """
    Adds chunks passed as argument to the Chroma vector database.

    Args:
        chunks (List[Document]): The chunks to be added to the database.

    Returns:
        bool: True if the chunks were added successfully, False otherwise.
    """
    # Load the existing database with PersistentClient for proper persistence
    with suppress_chromadb_telemetry():
        persistent_client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=get_chromadb_settings()
        )
        db = Chroma(
            client=persistent_client, embedding_function=get_embedding_function()
        )

    # Calculate Page IDs.
    chunks_with_ids = calc_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        try:
            db.add_documents(new_chunks, ids=new_chunk_ids)
            print("✅ Documents added successfully")

            # With PersistentClient, persistence is automatic
            print("📝 Using PersistentClient - auto-persistence enabled")

            # Verify documents were actually added
            verification = db.get()
            print(
                f"📊 Verification: Database now contains {len(verification['ids'])} documents"
            )
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("✅ No new documents to add")


def calc_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    This function will create IDs like "data/monopoly.pdf:6:2", following this pattern:
    `Page Source : Page Number : Chunk Index`. It will add these IDs to the chunks and return them.

    Args:
        chunks (List[Document]): The chunks to be processed.

    Returns:
        List[Document]: The chunks with the IDs added.
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """
    Clears the Chroma vector database.
    """

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main() -> None:
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


if __name__ == "__main__":
    main()
