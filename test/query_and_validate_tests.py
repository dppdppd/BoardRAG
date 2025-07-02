"""
Test module for the query_and_validate function.
"""

import os
from pathlib import Path

from query import query_rag

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

_eval_env = os.getenv("EVAL_TEMPLATE_PATH", "eval_prompt_tests.txt")
_eval_path = Path(_eval_env)
if not _eval_path.is_file():
    _eval_path = Path(__file__).parent.parent / "templates" / _eval_env

EVAL_TEMPLATE_PATH = str(_eval_path)


def load_jinja2_prompt(expected_response: str, actual_response: str) -> PromptTemplate:
    """
    Loads the content of the .jinja2 file selected by environment variable and creates a LangChain PromptTemplate.

    Args:
        expected_response (str): The expected response to be passed to the Jinja2 template.
        actual_response (str): The actual response to be passed to the Jinja2 template

    Returns:
        PromptTemplate: A LangChain PromptTemplate with the loaded content.
    """

    with open(EVAL_TEMPLATE_PATH, "r") as file:
        template_str = file.read()

    # Load the template into LangChain's PromptTemplate
    prompt = PromptTemplate.from_template(template_str)
    formatted_prompt = prompt.format(
        expected_response=expected_response, actual_response=actual_response
    )

    return formatted_prompt


def query_and_validate(question: str, expected_response: str) -> bool:
    """
    Queries the RAG model with the given question and validates the response.

    Args:
        question (str): The question to be passed to the RAG model.
        expected_response (str): The expected response to be validated.

    Returns:
        bool: True if the response is correct, False otherwise.

    Raises:
        ValueError: If the evaluation result is neither 'true' nor 'false'.
    """
    response_text = query_rag(question)["response_text"]
    prompt = load_jinja2_prompt(expected_response, response_text)

    # Use an OpenAI chat model for evaluation. Falls back to GPT-3.5-Turbo if
    # no specific evaluator model is provided via the environment.
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "ollama":
        from langchain_community.llms.ollama import Ollama  # pylint: disable=import-error

        evaluator_model_name = os.getenv("EVALUATOR_MODEL", "mistral")
        model = Ollama(
            model=evaluator_model_name,
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        )
    else:
        evaluator_model_name = os.getenv("EVALUATOR_MODEL", "gpt-3.5-turbo")
        model = ChatOpenAI(model=evaluator_model_name, temperature=0)
    evaluation_raw = model.invoke(prompt)
    evaluation_results_str = (
        evaluation_raw.content if hasattr(evaluation_raw, "content") else evaluation_raw
    )
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            "Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
