#!/usr/bin/env python
"""Quick helper to test the game-name extraction prompt/function.

Usage:
    python temp_tests/extract_name.py cna.pdf another-file.pdf

The script passes each given filename to
`query.extract_game_name_from_filename` and prints the result.
"""

import argparse
import os
import sys
from typing import List

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from query import extract_game_name_from_filename


def main(filenames: List[str]):
    for fn in filenames:
        try:
            name = extract_game_name_from_filename(fn, debug=True)
            print(f"{fn}  →  {name}")
        except Exception as exc:
            print(f"{fn}  →  ❌  {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test game-name extraction")
    parser.add_argument("filenames", nargs="+", help="PDF filenames to test")
    args = parser.parse_args()
    main(args.filenames) 