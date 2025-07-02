"""
Script for loading the database into Argilla and visualizing the data.
"""

import os
import warnings

import argilla as rg
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma

from config import (
    disable_chromadb_telemetry,
    get_chromadb_settings,
    suppress_chromadb_telemetry,
)
from embedding_function import get_embedding_function

# Disable ChromaDB telemetry after imports
disable_chromadb_telemetry()

# Ignore deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH")


def get_all_instances_database() -> dict:
    """
    Returns all the instances in the database.

    Returns:
        dict: The instances in the database.
    """
    # Load the existing database.
    with suppress_chromadb_telemetry():
        persistent_client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=get_chromadb_settings()
        )
        db = Chroma(
            client=persistent_client, embedding_function=get_embedding_function()
        )

    existing_items = db.get()  # IDs are always included by default

    return existing_items


def connect_argilla():
    """
    Connects to the Argilla API and returns the client object.
    """
    return rg.Argilla(
        api_url=ARGILLA_API_URL,
        api_key=ARGILLA_API_KEY,
    )


def create_argilla_dataset(client: rg.client) -> None:
    """
    Creates the Argilla dataset.

    Args:
        client (rg.Client): The Argilla client.
    """

    dataset = client.datasets(name="chunk-dabatase-1")

    if dataset is not None:
        dataset.delete()

    # Create the Argilla dataset
    settings = rg.Settings(
        guidelines="Visualize the BoardRAG database",
        fields=[
            rg.TextField(
                name="documents",
                title="Chunk Content",
                use_markdown=False,
            ),
        ],
        questions=[
            rg.LabelQuestion(
                name="positive_negative",
                title="Is it correctly chunked?",
                labels=["positive", "negative"],
            )
        ],
        metadata=[
            rg.TermsMetadataProperty(
                name="ids",
                title="IDs",
            ),
            rg.TermsMetadataProperty(
                name="metadatas",
                title="Metadatas",
            ),
        ],
    )

    dataset = rg.Dataset(
        name="chunk-dabatase-1",
        workspace="argilla",  # change this to your workspace
        settings=settings,
        client=client,
    )

    dataset.create()

    return dataset


def populate_argilla_dataset(existing_items: dict, dataset: rg.Dataset) -> None:
    """
    Populates the Argilla dataset.

    Args:
        existing_items (dict): The existing items in the database.
        dataset (rg.Dataset): The Argilla dataset.
    """
    records = []

    for ids, metadata, documents in zip(
        existing_items["ids"],
        existing_items["metadatas"],
        existing_items["documents"],
    ):
        records.append(
            rg.Record(
                fields={"documents": documents},
                metadata={"ids": ids, "metadatas": str(metadata)},
            )
        )
    dataset.records.log(records)


def main() -> None:
    print("🚀 Visualizing the database in Argilla...")

    # Get all instances in the database.
    print("🔍 Getting all instances in the database...")
    existing_items = get_all_instances_database()

    print("🔗 Connecting to the Argilla Client...")
    client = connect_argilla()

    # Create the Argilla dataset.
    print("📊 Creating the Argilla dataset...")
    dataset = create_argilla_dataset(client)

    # Populate the Argilla dataset.
    print("📝 Populating the Argilla dataset...")
    populate_argilla_dataset(existing_items, dataset)

    print("✅ Done!")


if __name__ == "__main__":
    main()
