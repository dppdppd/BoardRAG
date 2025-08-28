---
title: BoardRAG
emoji: 📈
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
license: unknown
---

# 🎲 BoardRAG

A RAG application feeding on board games running locally. Create a database with your favorite board game rulesets and query the RAG model to get answers to your questions!

Based on the [RAG tutorial by pixegami](https://github.com/pixegami/rag-tutorial-v2). 

## Dependencies

This project is based on the following technologies:

-   OpenAI API: hosted LLMs (e.g. GPT-3.5, GPT-4) and embeddings are accessed though the OpenAI API.
-   ChromaDB: the database used to store the chunks of the board game rulesets, alongside their corresponding embeddings.
-   Langchain: used for aligning the LLMs calls, the database and the prompt templates.
-   Pytest: for testing the codebase.
-   Pypdf2: for extracting the text from the PDFs.
-   Argilla: for data visualization.

To install the dependencies, please run the following command to create a virtual environment and install the dependencies (instructions for Unix-based systems):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Choose which backend to use ("openai" or "ollama"). Defaults to "openai".
LLM_PROVIDER="openai"

# Only required for the OpenAI provider
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

If you use the OpenAI provider you will need an API key (`OPENAI_API_KEY`). If
you use the Ollama provider make sure the Ollama server is running locally and
that the selected models are already downloaded.

### Dev Dependencies

If you want to contribute to the project, please install the ruff pre-commit hooks by running the following command:

```bash
pre-commit install
```

This will install the pre-commit hooks that will run the tests and the linters before each commit.

## Running the application

This RAG application is composed of two stages: database population and the RAG model itself. The project also comes with unit tests. Before running any of these processes make sure the environment variable `OPENAI_API_KEY` is available.

### DB-less Flow

The app no longer uses a local vector database. Place rulebook PDFs in the `data/` folder. On API startup, the catalog is scanned and missing PDFs are uploaded to the provider Files API. Queries run directly against the original PDFs with citations.

### Retriever & Generator

The RAG model is composed by a retriever and a generator. The retriever is responsible for finding the most relevant chunks in the database, while the generator is responsible for generating the answer to the user's question.

Run the RAG model:

```bash
python query.py --query_text "How can I build a hotel in Monopoly?"
```

You can also include the flags `--include_sources` and `include_contexts` to include the sources and chunks used to build the answer, respectively. The LLM used for generation is configured via the environment variable `GENERATOR_MODEL`.

### Frontend

The app uses a Next.js frontend in `web/`. Start it with your usual Next workflow after setting `NEXT_PUBLIC_API_BASE` to point at the FastAPI server.

### Tests

To run the tests, please run the following command:

```bash
pytest .
```

In the `test` folder, there is a file for each ruleset in the data folder. Tests rely on the OpenAI evaluator model declared by the `EVALUATOR_MODEL` environment variable (defaults to `gpt-3.5-turbo`).

## Example of .env file

```bash
ARGILLA_API_KEY = "YOUR_API_KEY"
ARGILLA_API_URL = "YOUR_API_URL"
CHUNK_OVERLAP = 80
CHUNK_SIZE = 800
CHROMA_PATH = "chroma"
DATA_PATH = "data"
GENERATOR_MODEL = "gpt-3.5-turbo"
EMBEDDER_MODEL = "text-embedding-3-small"
EVAL_TEMPLATE_PATH = "eval_prompt_tests.txt"
JINJA_TEMPLATE_PATH = "rag_query_pixegami.txt"
```

## Hugging Face Spaces - PDF Renaming

If you're running BoardRAG on Hugging Face Spaces with persistent storage and you rename PDF files:

1. **The issue**: The ChromaDB database contains references to the old filenames, so renamed PDFs won't appear in the game list even though the files exist.

2. **The solution**: After renaming PDFs in the persistent storage, you must rebuild the library:
   - Go to the "Technical Info" section (requires admin access)
   - Click "🔄 Rebuild Library" to reprocess all PDFs with their new names
   - This will update the ChromaDB database and game name mappings

3. **Why this happens**: 
   - ChromaDB stores document references using the original filenames
   - The game name mappings are stored separately and reference the old filenames
   - On HF Spaces, persistent storage preserves the old database even after file renames

4. **Prevention**: Use the built-in rename functionality in the UI instead of manually renaming files, or always rebuild after manual renames.
