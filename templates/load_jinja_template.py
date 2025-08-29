"""
Python script with functions for loading the content of the .jinja2 file selected by environment variable and creating a LangChain PromptTemplate.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

# ---------------------------------------------------------------------------
# Resolve template path
# ---------------------------------------------------------------------------

_env_value = os.getenv("JINJA_TEMPLATE_PATH", "rag_query_pixegami.txt")

# If the supplied path exists as-is, use it. Otherwise, assume it's relative to
# the templates directory bundled with the source tree.
_candidate = Path(_env_value)
if not _candidate.is_file():
    _candidate = Path(__file__).parent / _env_value

JINJA_TEMPLATE_PATH = str(_candidate)


def load_jinja2_prompt(
    context: str, question: str, template_name: str = None
) -> str:
    """
    Loads the content of the .jinja2 file selected by environment variable and creates a LangChain PromptTemplate.

    Args:
        context (str): The context to be passed to the Jinja2 template.
        question (str): The question to be passed to the Jinja2 template
        template_name (str): Optional specific template file name to use

    Returns:
        PromptTemplate: A LangChain PromptTemplate with the loaded content.
    """

    if template_name:
        template_path = Path(__file__).parent / template_name
        if not template_path.is_file():
            raise FileNotFoundError(f"Template {template_name} not found")
    else:
        template_path = JINJA_TEMPLATE_PATH

    with open(template_path, "r", encoding="utf-8") as file:
        template_str = file.read()

    # Perform literal, safe replacement for known placeholders only
    out = template_str.replace("{context}", str(context)).replace("{question}", str(question))
    return out


def render_template(template_name: str, **kwargs: str) -> str:
    """
    Render an arbitrary prompt template from the templates directory with named parameters.

    Args:
        template_name: File name within the templates directory (e.g., 'rag_query_pixegami.txt')
        **kwargs: Named variables available in the template.

    Returns:
        The rendered string.
    """
    # Resolve path: allow absolute/relative elsewhere, else use bundled templates dir
    candidate = Path(template_name)
    if not candidate.is_file():
        candidate = Path(__file__).parent / template_name
    if not candidate.is_file():
        raise FileNotFoundError(f"Template {template_name} not found")

    with open(candidate, "r", encoding="utf-8") as f:
        template_str = f.read()

    # Literal, safe replacement for provided placeholders, without interpreting other braces
    out = template_str
    for k, v in (kwargs or {}).items():
        out = out.replace("{" + str(k) + "}", str(v))
    return out


def read_text_template(template_name: str) -> str:
    """Return raw text contents of a template file in templates/ or absolute path."""
    candidate = Path(template_name)
    if not candidate.is_file():
        candidate = Path(__file__).parent / template_name
    if not candidate.is_file():
        raise FileNotFoundError(f"Template {template_name} not found")
    return candidate.read_text(encoding="utf-8")
