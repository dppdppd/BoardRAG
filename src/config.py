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
from pathlib import Path


# Load API keys from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Provider and Model Configuration
# ---------------------------------------------------------------------------

# LLM Provider selection: "openai", "anthropic", "ollama", or "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
print(f"[config] LLM_PROVIDER = {LLM_PROVIDER}")

# Model configurations by provider
MODEL_CONFIGS = {
    "openai": {
        "generator": "o3-mini",
        "embedder": "text-embedding-3-small",
        "evaluator": "o3-mini",
    },
    "anthropic": {
        "generator": "claude-sonnet-4-20250514",
        "embedder": "text-embedding-3-small",  # Still use OpenAI for embeddings
        "evaluator": "claude-sonnet-4-20250514",
    },
    # "anthropic": {
    #     "generator": "claude-3-7-sonnet-20250219",
    #     "embedder": "text-embedding-3-small",  # Still use OpenAI for embeddings
    #     "evaluator": "claude-3-7-sonnet-20250219",
    # },
    "ollama": {
        "generator": "mistral",
        "embedder": "nomic-embed-text",
        "evaluator": "mistral",
    },
    "openrouter": {
        # OpenRouter model id for Llama 3.3 70B Instruct
        "generator": "meta-llama/llama-3.3-70b-instruct",
        "embedder": "text-embedding-3-small",
        "evaluator": "meta-llama/llama-3.3-70b-instruct",
    },
}

# Get model configuration for selected provider
_config = MODEL_CONFIGS.get(LLM_PROVIDER.lower())

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", _config["generator"])
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", _config["embedder"])
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", _config["evaluator"])

# ---------------------------------------------------------------------------
# Optional - Web Search Configuration
# ---------------------------------------------------------------------------
# Enable or disable supplementary web search results in responses.
# Set ENABLE_WEB_SEARCH=true (or 1/yes) in the environment to turn this on.

ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "False").lower() in {
    "1",
    "true",
    "yes",
}

# Debug print so users can see what the setting resolved to at import time
print(f"[config] ENABLE_WEB_SEARCH = {ENABLE_WEB_SEARCH}")

# How many web snippets to fetch per query (only used when ENABLE_WEB_SEARCH)
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "5"))

# Search provider selection: "duckduckgo" (default), "serpapi", or "brave"
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "brave").lower()

# API keys for external providers (only required when selected)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ---------------------------------------------------------------------------
# Optional - Search Query Rewrite Configuration
# ---------------------------------------------------------------------------
# Enable to run an extra LLM call that rewrites the user's question into a
# concise web-search query. Turn on via ENABLE_SEARCH_REWRITE=true.

ENABLE_SEARCH_REWRITE = os.getenv("ENABLE_SEARCH_REWRITE", "True").lower() in {
    "1",
    "true",
    "yes",
}

# Model used for query rewriting (falls back to GENERATOR_MODEL)
SEARCH_REWRITE_MODEL = os.getenv("SEARCH_REWRITE_MODEL", GENERATOR_MODEL)

# ---------------------------------------------------------------------------
# Ollama Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Retrieval Mode Configuration (single source of truth)
# ---------------------------------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")

# Retrieval mode (db-less removed) – force vector mode
RAG_MODE = "vector"
IS_VECTOR_MODE = True
IS_DB_LESS_MODE = False

# ---------------------------------------------------------------------------
# Chunking Mode
# ---------------------------------------------------------------------------
# Section-only mode (pages removed downstream). Always True.
USE_SECTION_CHUNKS = True

## Page-based chunking params removed in section-only mode
CHUNK_SIZE = 0
CHUNK_OVERLAP = 0

# ---------------------------------------------------------------------------
# Context Filtering Configuration
# ---------------------------------------------------------------------------
# Page visual importance filtering not applicable in section-only mode
IGNORE_VISUAL_IMPORTANCE = True

# ---------------------------------------------------------------------------
# Optional - Streaming Validation/Repair (detached, togglable)
# ---------------------------------------------------------------------------
# When enabled, server will perform an inline streaming transform that
# validates per-sentence formatting and attempts lightweight repair without
# stalling the outgoing stream.
ENABLE_STREAM_VALIDATION = os.getenv("ENABLE_STREAM_VALIDATION", "False").lower() in {"1", "true", "yes"}
STREAM_REPAIR_USE_LLM = os.getenv("STREAM_REPAIR_USE_LLM", "False").lower() in {"1", "true", "yes"}
STREAM_REPAIR_TIMEOUT_MS = int(os.getenv("STREAM_REPAIR_TIMEOUT_MS", "100"))
STREAM_REPAIR_MAX_CONSECUTIVE_INVALID = int(os.getenv("STREAM_REPAIR_MAX_CONSECUTIVE_INVALID", "3"))

print(f"[config] ENABLE_STREAM_VALIDATION = {ENABLE_STREAM_VALIDATION}")

# ---------------------------------------------------------------------------
# Data and Template Paths
# ---------------------------------------------------------------------------
# Detect if running inside a Hugging Face Space (SPACE_ID env var is always set)
ON_HF_SPACE = bool(os.getenv("SPACE_ID"))

_default_data_path = "/data" if ON_HF_SPACE else "data"

DATA_PATH = os.getenv("DATA_PATH", _default_data_path)

# ---------------------------------------------------------------------------
# Ensure chosen DATA_PATH is writable; if not, fall back to repo-local 'data'
# ---------------------------------------------------------------------------


def _is_writable(dir_path: str) -> bool:
    """Check if we can create and write a temporary file in dir_path."""
    try:
        os.makedirs(dir_path, exist_ok=True)
        test_file = Path(dir_path) / ".write_test"
        with open(test_file, "w") as f:
            f.write("ok")
        test_file.unlink()
        return True
    except Exception:
        return False


# On HF Spaces without persistent storage, '/data' is not writable → fallback
if not _is_writable(DATA_PATH):
    fallback_data = "data"
    DATA_PATH = fallback_data
    os.makedirs(DATA_PATH, exist_ok=True)

# ---------------------------------------------------------------------------
# Optional - PDF Optimization
# ---------------------------------------------------------------------------
# Enable optimizing large PDFs during ingestion to reduce size and speed up
# loading. Tunable via environment variables.
ENABLE_PDF_OPTIMIZATION = os.getenv("ENABLE_PDF_OPTIMIZATION", "True").lower() in {
    "1",
    "true",
    "yes",
}
PDF_OPTIMIZE_MIN_SIZE_MB = float(os.getenv("PDF_OPTIMIZE_MIN_SIZE_MB", "25"))
PDF_LINEARIZE = os.getenv("PDF_LINEARIZE", "True").lower() in {"1", "true", "yes"}
PDF_GARBAGE_LEVEL = int(os.getenv("PDF_GARBAGE_LEVEL", "3"))
PDF_ENABLE_RASTER_FALLBACK = os.getenv("PDF_ENABLE_RASTER_FALLBACK", "False").lower() in {"1", "true", "yes"}
PDF_RASTER_DPI = int(os.getenv("PDF_RASTER_DPI", "150"))
PDF_JPEG_QUALITY = int(os.getenv("PDF_JPEG_QUALITY", "70"))

# ---------------------------------------------------------------------------
# Optional - Pipeline Cleanup
# ---------------------------------------------------------------------------
# When enabled, after successful populate the pipeline deletes intermediate
# artifacts to save disk space: page PDFs, raw.txt, and JSON files.
PIPELINE_DELETE_INTERMEDIATE = os.getenv("PIPELINE_DELETE_INTERMEDIATE", "False").lower() in {"1", "true", "yes"}

# Templates (these are small text files so they can stay in repo)
JINJA_TEMPLATE_PATH = os.getenv("JINJA_TEMPLATE_PATH", "rag_query_improved.txt")
EVAL_TEMPLATE_PATH = os.getenv("EVAL_TEMPLATE_PATH", "eval_prompt_tests.txt")

# ---------------------------------------------------------------------------
# Optional - Argilla Settings
# ---------------------------------------------------------------------------
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")

# ---------------------------------------------------------------------------
# API Keys and Security (loaded from .env)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Access control passwords (set in environment)
USER_PW = os.getenv("USER_PW")
ADMIN_PW = os.getenv("ADMIN_PW")

# Server-issued auth token secret. If not provided, generate a per-process
# ephemeral secret which will invalidate tokens on restart.
AUTH_SECRET = os.getenv("AUTH_SECRET")
if not AUTH_SECRET:
    try:
        import secrets as _secrets
        AUTH_SECRET = _secrets.token_urlsafe(32)
        print("[config] AUTH_SECRET not set; using ephemeral secret (tokens reset on restart)")
    except Exception:
        AUTH_SECRET = "change-me"

# Token time-to-live in seconds
AUTH_TOKEN_TTL_SECS = int(os.getenv("AUTH_TOKEN_TTL_SECS", "43200"))  # 12 hours

# Back-compat shim: expose DB_LESS (always False; db-less removed)
DB_LESS = False

print(f"[config] RAG_MODE = {RAG_MODE}")


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
    print("BoardRAG Configuration:")
    print(f"  Provider: {LLM_PROVIDER}")
    print(f"  Generator: {GENERATOR_MODEL}")
    print(f"  Embedder: {EMBEDDER_MODEL}")
    print(f"  Chunk Size: {CHUNK_SIZE} chars (~{CHUNK_SIZE//4} tokens)")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP} chars")
    # No local DB in DB-less mode
    print(f"  Data: {DATA_PATH}")
    print(f"  Template: {JINJA_TEMPLATE_PATH}")

    # Web search configuration (non-sensitive)
    web_status = "Enabled" if ENABLE_WEB_SEARCH else "Disabled"
    print(f"  Web Search: {web_status} – {WEB_SEARCH_RESULTS} snippets per query")

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
