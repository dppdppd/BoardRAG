#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.query import stream_query_rag  # type: ignore
from src import config as cfg  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a vector-RAG query and stream the LLM answer")
    ap.add_argument("--query", required=True, help="user question")
    ap.add_argument("--game", help="optional game/pdf filter (e.g., catan)")
    args = ap.parse_args()

    # Ensure vector mode
    if not getattr(cfg, "IS_VECTOR_MODE", False):
        print("ERROR: RAG_MODE must be 'vector' to use vector retrieval.")
        return 2

    token_gen, meta = stream_query_rag(
        query_text=args.query,
        selected_game=args.game,
        chat_history=None,
        game_names=[args.game] if args.game else None,
        enable_web=False,
    )

    try:
        for chunk in token_gen:
            s = str(chunk)
            if not s:
                continue
            sys.stdout.write(s)
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


