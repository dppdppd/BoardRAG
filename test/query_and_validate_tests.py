"""
Query and validation utilities for tests.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query import query_rag


def query_and_validate(question: str, expected_response: str, selected_game: str = None) -> bool:
    """
    Query the RAG system and validate the response against expected answer.
    
    Args:
        question: The question to ask
        expected_response: The expected answer
        selected_game: Optional game to filter by
        
    Returns:
        bool: True if the response matches expectations, False otherwise
    """
    try:
        # Query the RAG system
        result = query_rag(
            query_text=question,
            selected_game=selected_game,
            chat_history=None,
            game_names=None,
            enable_web=False
        )
        
        response_text = result.get("response_text", "").strip()
        expected = expected_response.strip()
        
        print(f"Question: {question}")
        print(f"Expected: {expected}")
        print(f"Got: {response_text}")
        
        # Simple string matching - could be enhanced with fuzzy matching
        if expected.lower() in response_text.lower():
            print("✅ PASS")
            return True
        else:
            print("❌ FAIL")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False