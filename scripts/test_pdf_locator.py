import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root on sys.path for `import src` modules
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _print(title: str, payload: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    try:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        print(str(payload))


def _norm(s: str) -> str:
    try:
        return " ".join((s or "").replace("\u00a0", " ").split())
    except Exception:
        return s


def _norm_token(s: str) -> str:
    import re as _re
    try:
        s2 = _norm(s).lower()
        return _re.sub(r"^[\s\-–—:;,.]+|[\s\-–—:;,.]+$", "", s2)
    except Exception:
        return s


def _tokens(s: str) -> List[str]:
    try:
        return [_norm_token(t) for t in _norm(s).split() if t]
    except Exception:
        return [s]


def _normalize_bbox(
    page_rect: Tuple[float, float, float, float],
    rect: Tuple[float, float, float, float],
    cal_margin_pct_h: float,
    cal_margin_pct_v: float,
) -> Tuple[float, float, float, float]:
    x0p, y0p, x1p, y1p = page_rect
    W = float(x1p - x0p)
    H = float(y1p - y0p)
    inset_x = max(0.0, cal_margin_pct_h) * 0.01 * W
    inset_y = max(0.0, cal_margin_pct_v) * 0.01 * H
    frame_left = x0p + inset_x
    frame_top = y0p + inset_y
    frame_w = max(1.0, W - 2 * inset_x)
    frame_h = max(1.0, H - 2 * inset_y)

    x0, y0, x1, y1 = rect
    nx = max(0.0, (y0 - frame_top) * 100.0 / frame_h)
    ny = max(0.0, (x0 - frame_left) * 100.0 / frame_w)
    nw = max(0.1, (x1 - x0) * 100.0 / frame_w)
    nh = max(0.1, (y1 - y0) * 100.0 / frame_h)
    return (nx, ny, nw, nh)


# NEW: reusable search function
def search_pdf_for_string(
    pdf_path: Path,
    target_text: str,
    page_number_1_based: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return merged rect matches for target_text on the given PDF (optionally one page).

    Each match item: { page: int, rect_pdf: [x0,y0,x1,y1], rect_pct: [x,y,w,h] }
    """
    import fitz  # type: ignore

    results: List[Dict[str, Any]] = []

    def _variants(s: str) -> List[str]:
        out: List[str] = []
        base = _norm(s)
        # 1) as-is
        out.append(base)
        # 2) smart apostrophes
        out.append(base.replace("'", "\u2019"))
        out.append(base.replace("'", "\u02BC"))
        # 3) collapse multiple spaces (already normalized) + prefix of first ~7 tokens
        try:
            toks = [t for t in base.split() if t]
            if len(toks) >= 4:
                prefix = " ".join(toks[:min(7, len(toks))])
                out.append(prefix)
        except Exception:
            pass
        # Deduplicate while preserving order
        seen: set[str] = set()
        uniq: List[str] = []
        for v in out:
            if v not in seen and v:
                uniq.append(v)
                seen.add(v)
        return uniq
    with fitz.open(pdf_path.as_posix()) as doc:
        pages = range(len(doc)) if page_number_1_based is None else [max(1, int(page_number_1_based)) - 1]
        for idx in pages:
            if idx < 0 or idx >= len(doc):
                continue
            page = doc.load_page(idx)
            prect = page.rect
            page_rect = (float(prect.x0), float(prect.y0), float(prect.x1), float(prect.y1))
            rects = []
            for needle in _variants(target_text):
                try:
                    rects = page.search_for(needle)
                except Exception:
                    rects = []
                if rects:
                    break
            # Merge adjacent rects on the same line into a single logical hit
            merged: List[Tuple[float, float, float, float]] = []
            rects_sorted = sorted(rects or [], key=lambda rr: (float(rr.y0), float(rr.x0)))
            def _close(a: float, b: float, tol: float) -> bool:
                return abs(a - b) <= tol
            for r in rects_sorted:
                x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                if not merged:
                    merged.append((x0, y0, x1, y1))
                    continue
                px0, py0, px1, py1 = merged[-1]
                # Consider same line if vertical alignment is very close
                same_line = _close(y0, py0, tol=2.0) or _close(y1, py1, tol=2.0)
                # Small horizontal gap tolerance (<= 6pt or 10% of height)
                gap_tol = max(6.0, 0.1 * max(1.0, py1 - py0))
                contiguous = (x0 - px1) <= gap_tol
                if same_line and contiguous:
                    # Union horizontally
                    nx0 = min(px0, x0)
                    ny0 = min(py0, y0)
                    nx1 = max(px1, x1)
                    ny1 = max(py1, y1)
                    merged[-1] = (nx0, ny0, nx1, ny1)
                else:
                    merged.append((x0, y0, x1, y1))

            for x0, y0, x1, y1 in merged:
                rect_pdf = (x0, y0, x1, y1)
                results.append({
                    "page": idx + 1,
                    "rect_pdf": rect_pdf,
                    "rect_pct": _normalize_bbox(page_rect, rect_pdf, 2.0, 2.0),
                })
    return results


def run_tests(
    pdf_path: Path,
    page_number_1_based: Optional[int],
    target_text: str,
) -> None:
    results = search_pdf_for_string(pdf_path, target_text, page_number_1_based)

    _print("search_results", {
        "file": pdf_path.name,
        "needle": target_text,
        "matches": results,
        "count": len(results),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Search for exact occurrences of a section_start string in a PDF. Outputs page and bbox (PDF and normalized pct).")
    ap.add_argument("pdf", type=str, help="Path to PDF file (relative defaults to data/)")
    ap.add_argument("text", type=str, help="Section_start string to search for")
    ap.add_argument("--page", type=int, default=None, help="Optional 1-based page number filter (search only this page)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    # Default to data/ for relative paths (align with other project scripts)
    if not pdf_path.is_absolute():
        first = (pdf_path.parts[0].lower() if pdf_path.parts else "")
        if first != "data":
            pdf_path = Path("data") / pdf_path
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(2)

    run_tests(
        pdf_path,
        int(args.page) if args.page is not None else None,
        str(args.text),
    )


if __name__ == "__main__":
    main()


