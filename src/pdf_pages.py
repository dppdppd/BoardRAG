from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple, List


def _slugify(text: str) -> str:
    s = (text or "").lower()
    out = []
    prev_dash = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "file"


def page_slug_from_pdf(pdf_path: Path) -> str:
    return _slugify(pdf_path.stem)


def make_page_filename(slug: str, page_1based: int) -> str:
    return f"{slug}_p{max(1, int(page_1based)):04}.pdf"


def parse_page_1based_from_name(name: str) -> int:
    try:
        base = Path(name).stem
        # Expect pattern: <slug>_pNNNN
        idx = base.rfind("_p")
        if idx >= 0 and idx + 6 <= len(base):
            num = int(base[idx+2:idx+6])
            return max(1, num)
    except Exception:
        pass
    try:
        # Fallback: legacy pNNNN
        base = Path(name).stem
        if base.startswith("p"):
            return max(1, int(base[1:]))
    except Exception:
        pass
    return 1


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_pages_dir(pdf_path: Path, data_path: Path) -> Path:
    base = pdf_path.stem
    # New hierarchy: per-PDF folder with ordered step directories
    out_dir = data_path / base / "1_pdf_pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def export_single_page_pdfs(pdf_path: Path, pages_dir: Path) -> Tuple[int, List[Path]]:
    try:
        import fitz  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PyMuPDF (fitz) is required for page export: {e}")

    def _validate_page_pdf(p: Path) -> None:
        """Open the page PDF and load page 0 to ensure it is readable."""
        try:
            if p.stat().st_size <= 0:
                raise RuntimeError("file size is 0")
        except Exception:
            raise RuntimeError("file not found or unreadable size")
        try:
            with fitz.open(str(p)) as d:
                if getattr(d, "page_count", len(d)) < 1:
                    raise RuntimeError("no pages present")
                # Attempt to load first page to ensure structure is valid
                _ = d.load_page(0)
        except Exception as _e:
            raise RuntimeError(f"invalid page pdf: {_e}")

    doc = fitz.open(str(pdf_path))
    count = doc.page_count
    out_paths: List[Path] = []
    slug = page_slug_from_pdf(pdf_path)

    for i in range(count):
        name = make_page_filename(slug, i + 1)
        out = pages_dir / name
        if not out.exists():
            single = fitz.open()
            single.insert_pdf(doc, from_page=i, to_page=i)
            single.save(str(out))
            single.close()
            _validate_page_pdf(out)
            try:
                size = out.stat().st_size
                print(f"[pages] Created page {i+1}: {out} size={size} bytes")
            except Exception:
                print(f"[pages] Created page {i+1}: {out} (size unavailable)")
        else:
            # Validate existing file as well to catch prior corrupt outputs
            _validate_page_pdf(out)
            try:
                size = out.stat().st_size
                print(f"[pages] Reused page {i+1}: {out} size={size} bytes")
            except Exception:
                print(f"[pages] Reused page {i+1}: {out} (size unavailable)")
        out_paths.append(out)
    doc.close()
    return count, out_paths


def get_page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        from PyPDF2 import PdfReader  # type: ignore
    reader = PdfReader(str(pdf_path))
    return len(getattr(reader, "pages", []))


