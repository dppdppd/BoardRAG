#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

# Ensure repository root on path for `import src` in child scripts as well
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src import config as cfg  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser(description="Process all PDFs in data folder")
    _ = ap.parse_args()

    data = Path(cfg.DATA_PATH)
    pdfs = sorted([p for p in data.glob("*.pdf") if p.is_file()])
    if not pdfs:
        print("No PDFs found.")
        return 0
    for p in pdfs:
        print(f"==> {p.name}")
        # 1) Split pages (missing)
        code = subprocess.call([sys.executable, str(repo_root / "scripts" / "split_pages.py"), str(p)])
        if code != 0:
            print(f"❌ Split failed {p.name} (exit {code})")
            return code
        # 2) LLM eval (missing)
        code = subprocess.call([sys.executable, str(repo_root / "scripts" / "eval_pages.py"), str(p)])
        if code != 0:
            print(f"❌ Eval failed {p.name} (exit {code})")
            return code
        # 3) Populate DB (missing)
        code = subprocess.call([sys.executable, str(repo_root / "scripts" / "populate_from_processed.py"), str(p)])
        if code != 0:
            print(f"❌ Populate failed {p.name} (exit {code})")
            return code
    print("🎉 All PDFs processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


