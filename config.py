"""
BoardRAG Configuration Settings

This file contains all non-sensitive configuration for the BoardRAG application.
Sensitive information like API keys should remain in the .env file.

To customize your setup:
1. Modify the values below directly
2. Or set environment variables to override these defaults
"""

import contextlib
import os
import sys
from dotenv import load_dotenv


def disable_chromadb_telemetry():
    """Properly disable ChromaDB telemetry using official methods."""
    # Set environment variables that ChromaDB checks for telemetry
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["DO_NOT_TRACK"] = "1"


# Disable telemetry immediately
disable_chromadb_telemetry()


def get_chromadb_settings():
    """Get ChromaDB settings with telemetry properly disabled."""
    try:
        import chromadb

        return chromadb.config.Settings(
            anonymized_telemetry=False,
        )
    except ImportError:
        return None


@contextlib.contextmanager
def suppress_chromadb_telemetry():
    """Context manager to suppress ChromaDB telemetry error messages."""
    original_stderr = sys.stderr

    class FilteredStderr:
        def write(self, text):
            # Only suppress specific telemetry error messages
            if any(
                keyword in text.lower()
                for keyword in [
                    "failed to send telemetry event",
                    "capture() takes 1 positional argument but 3 were given",
                    "clientstartevent",
                    "clientcreatecollectionevent",
                    "collectiongetevent",
                ]
            ):
                return
            return original_stderr.write(text)

        def flush(self):
            return original_stderr.flush()

        def __getattr__(self, name):
            return getattr(original_stderr, name)

    try:
        sys.stderr = FilteredStderr()
        yield
    finally:
        sys.stderr = original_stderr


# Load API keys from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Provider and Model Configuration
# ---------------------------------------------------------------------------

# LLM Provider selection: "openai", "anthropic", or "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

# Model configurations by provider
MODEL_CONFIGS = {
    "openai": {
        "generator": "gpt-4o",
        "embedder": "text-embedding-3-small",
        "evaluator": "gpt-4o",
    },
    "anthropic": {
        "generator": "claude-sonnet-4-20250514",
        "embedder": "text-embedding-3-small",  # Still use OpenAI for embeddings
        "evaluator": "claude-sonnet-4-20250514",
    },
    "ollama": {
        "generator": "mistral",
        "embedder": "nomic-embed-text",
        "evaluator": "mistral",
    },
}

# Get model configuration for selected provider
_config = MODEL_CONFIGS.get(LLM_PROVIDER.lower(), MODEL_CONFIGS["openai"])

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", _config["generator"])
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", _config["embedder"])
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", _config["evaluator"])

# ---------------------------------------------------------------------------
# Ollama Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Vector Database Configuration
# ---------------------------------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")

# Chunking parameters optimized by model context window
CHUNKING_CONFIGS = {
    "claude-sonnet-4-20250514": {
        "chunk_size": 1400,
        "chunk_overlap": 250,
    },  # Enhanced for Claude 4's improved reasoning
    "claude-3-5-sonnet-20241022": {
        "chunk_size": 1200,
        "chunk_overlap": 200,
    },  # Reduced for better citations
    "claude-3-5-haiku-20241022": {"chunk_size": 2400, "chunk_overlap": 300},
    "gpt-4o": {"chunk_size": 2400, "chunk_overlap": 300},
    "gpt-4-turbo": {"chunk_size": 2400, "chunk_overlap": 300},
    "gpt-3.5-turbo": {"chunk_size": 800, "chunk_overlap": 80},
    "mistral": {"chunk_size": 800, "chunk_overlap": 80},
    # Add more models as needed
}

# Get chunking config for current generator model, with fallback
_chunking = CHUNKING_CONFIGS.get(
    GENERATOR_MODEL, {"chunk_size": 800, "chunk_overlap": 80}
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", _chunking["chunk_size"]))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", _chunking["chunk_overlap"]))

# ---------------------------------------------------------------------------
# Data and Template Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", "data")
JINJA_TEMPLATE_PATH = os.getenv("JINJA_TEMPLATE_PATH", "rag_query_improved.txt")
EVAL_TEMPLATE_PATH = os.getenv("EVAL_TEMPLATE_PATH", "eval_prompt_tests.txt")

# ---------------------------------------------------------------------------
# Optional - Argilla Settings
# ---------------------------------------------------------------------------
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")

# ---------------------------------------------------------------------------
# API Keys (loaded from .env)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def validate_config():
    """Validate that required configuration is present."""
    errors = []

    # Check provider is valid
    if LLM_PROVIDER.lower() not in MODEL_CONFIGS:
        errors.append(
            f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be one of: {list(MODEL_CONFIGS.keys())}"
        )

    # Check API keys based on provider
    if LLM_PROVIDER.lower() == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY must be set in .env when using OpenAI provider")
    elif LLM_PROVIDER.lower() == "anthropic":
        if not ANTHROPIC_API_KEY:
            errors.append(
                "ANTHROPIC_API_KEY must be set in .env when using Anthropic provider"
            )
        if not OPENAI_API_KEY:
            errors.append(
                "OPENAI_API_KEY must be set in .env for embeddings when using Anthropic provider"
            )

    # Check required models are set
    if not GENERATOR_MODEL:
        errors.append("GENERATOR_MODEL must be configured")
    if not EMBEDDER_MODEL:
        errors.append("EMBEDDER_MODEL must be configured")

    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"- {error}" for error in errors)
        )


def print_config():
    """Print current configuration (hiding API keys)."""
    print("🔧 BoardRAG Configuration:")
    print(f"  Provider: {LLM_PROVIDER}")
    print(f"  Generator: {GENERATOR_MODEL}")
    print(f"  Embedder: {EMBEDDER_MODEL}")
    print(f"  Chunk Size: {CHUNK_SIZE} chars (~{CHUNK_SIZE//4} tokens)")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP} chars")
    print(f"  Database: {CHROMA_PATH}")
    print(f"  Data: {DATA_PATH}")
    print(f"  Template: {JINJA_TEMPLATE_PATH}")

    # Show API key status without revealing keys
    openai_status = "✅ Set" if OPENAI_API_KEY else "❌ Missing"
    anthropic_status = "✅ Set" if ANTHROPIC_API_KEY else "❌ Missing"
    print(f"  OpenAI API Key: {openai_status}")
    print(f"  Anthropic API Key: {anthropic_status}")


if __name__ == "__main__":
    try:
        validate_config()
        print_config()
    except ValueError as e:
        print(f"❌ {e}")
