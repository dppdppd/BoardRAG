#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repo root on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vector_store import _get_native_collection  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser(description="List Chroma DB chunk ids for a given PDF (by source metadata)")
    ap.add_argument("pdf", type=str, help="PDF filename (e.g., catan.pdf)")
    args = ap.parse_args()

    fname = Path(args.pdf).name
    coll = _get_native_collection()
    got = coll.get(where={"source": fname}, include=["metadatas"])  # type: ignore[arg-type]
    ids = (got or {}).get("ids") or []
    metas = (got or {}).get("metadatas") or []
    print(f"source={fname}")
    print(f"count={len(ids)}")
    for i, cid in enumerate(ids):
        ver = ""
        try:
            md = metas[i] if i < len(metas) else {}
            ver = str((md or {}).get("version") or "")
        except Exception:
            ver = ""
        print(f"{i+1:>3}. id={cid} version={ver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


