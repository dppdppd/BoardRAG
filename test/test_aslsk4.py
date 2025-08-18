"""
Tests for ASLSK4: ensure section 3.0 is retrievable via quote query.
"""

import importlib


def test_aslsk4_quote_section_3_0():
    # Reload config and query to pick up CHROMA_PATH
    import src.config as cfg
    importlib.reload(cfg)
    import src.query as qr
    importlib.reload(qr)

    # Ask for the section and expect a distinctive sentence from 3.0
    token_gen, meta = qr.stream_query_rag(
        query_text="quote 3.0",
        selected_game="Advanced Squad Leader Starter Kit #4",
        chat_history=None,
        game_names=["Advanced Squad Leader Starter Kit #4"],
        enable_web=False,
    )
    response_text = "".join(list(token_gen)).lower()
    assert "there are eight distinct phases in each player" in response_text


def test_aslsk4_quote_section_3_3():
    import src.config as cfg  # noqa: F401
    import src.query as qr
    importlib.reload(qr)

    token_gen, _meta = qr.stream_query_rag(
        query_text="quote 3.3",
        selected_game="Advanced Squad Leader Starter Kit #4",
        chat_history=None,
        game_names=["Advanced Squad Leader Starter Kit #4"],
        enable_web=False,
    )
    response_text = "".join(list(token_gen)).lower()
    assert "movement phase (mph)" in response_text or "during the mph" in response_text


def test_aslsk4_quote_section_3_5():
    import src.config as cfg  # noqa: F401
    import src.query as qr
    importlib.reload(qr)

    token_gen, _meta = qr.stream_query_rag(
        query_text="quote 3.5",
        selected_game="Advanced Squad Leader Starter Kit #4",
        chat_history=None,
        game_names=["Advanced Squad Leader Starter Kit #4"],
        enable_web=False,
    )
    response_text = "".join(list(token_gen)).lower()
    assert "advancing fire phase" in response_text


def test_aslsk4_quote_section_3_6():
    import src.config as cfg  # noqa: F401
    import src.query as qr
    importlib.reload(qr)

    token_gen, _meta = qr.stream_query_rag(
        query_text="quote 3.6",
        selected_game="Advanced Squad Leader Starter Kit #4",
        chat_history=None,
        game_names=["Advanced Squad Leader Starter Kit #4"],
        enable_web=False,
    )
    response_text = "".join(list(token_gen)).lower()
    assert "rout phase (rtph)" in response_text or "rout phase" in response_text


