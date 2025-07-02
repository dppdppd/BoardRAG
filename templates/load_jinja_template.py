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
) -> PromptTemplate:
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

    with open(template_path, "r") as file:
        template_str = file.read()

    # Load the template into LangChain's PromptTemplate
    prompt = PromptTemplate.from_template(template_str)
    formatted_prompt = prompt.format(context=context, question=question)

    return formatted_prompt
