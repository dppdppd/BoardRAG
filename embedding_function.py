"""
Utility for instantiating the embedding function used across the project.

This was originally implemented with `OllamaEmbeddings`, which relied on a
locally-running Ollama server. The implementation has now been swapped for
`OpenAIEmbeddings` so that embeddings are generated through OpenAI's hosted
API instead. The public interface (`get_embedding_function`) remains exactly
the same.
"""

# Base embeddings implementation (OpenAI is the default)
from langchain_openai import OpenAIEmbeddings

# Project configuration
from config import LLM_PROVIDER, EMBEDDER_MODEL, OLLAMA_URL


def get_embedding_function():
    """
    Returns an embedding function instance configured with the model supplied
    via the EMBEDDER_MODEL configuration.
    """

    # Choose between OpenAI (hosted) and Ollama (local) depending on the
    # `LLM_PROVIDER` configuration.
    if LLM_PROVIDER.lower() == "ollama":
        # Import locally to avoid mandatory dependency when provider is OpenAI.
        from langchain_community.embeddings.ollama import OllamaEmbeddings  # pylint: disable=import-error

        embeddings = OllamaEmbeddings(model=EMBEDDER_MODEL, base_url=OLLAMA_URL)
    else:
        embeddings = OpenAIEmbeddings(model=EMBEDDER_MODEL)

    return embeddings
