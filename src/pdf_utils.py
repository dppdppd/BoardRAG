"""
Utilities for optimizing PDF files to reduce size and improve load performance.

Uses PyMuPDF (fitz) to rewrite PDFs with linearization, stream deflation,
and garbage collection. Performs safe, atomic in-place replacement via a
temporary file and only replaces the original when the optimized file is
smaller.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import fitz  # PyMuPDF


def _log_default(_: str) -> None:
    pass


def optimize_pdf(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    linearize: bool = True,
    garbage_level: int = 3,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, int, int, str]:
    """
    Optimize a PDF and write to output_path.

    Returns (success, original_size, optimized_size, message).
    """
    logger = log or _log_default
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists() or not in_path.is_file():
        return (False, 0, 0, f"Input not found: {in_path}")

    try:
        original_size = in_path.stat().st_size
    except Exception:
        original_size = 0

    try:
        with fitz.open(in_path) as doc:
            if doc.is_encrypted:
                # Skip encrypted PDFs as we don't have credentials
                return (False, original_size, original_size, "Encrypted PDF – skipped optimization")

            # Rewrite with maximum safe cleanup and linearization
            save_kwargs = {
                "garbage": max(0, min(int(garbage_level), 4)),
                "deflate": True,
                "clean": True,
                "linear": bool(linearize),
                "incremental": False,
            }

            # Ensure output directory exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(out_path.as_posix(), **save_kwargs)

        optimized_size = out_path.stat().st_size if out_path.exists() else original_size
        if optimized_size <= 0:
            return (False, original_size, optimized_size, "Optimized output missing or empty")

        if optimized_size < original_size:
            ratio = optimized_size / max(1, original_size)
            return (True, original_size, optimized_size, f"Reduced to {ratio:.1%} of original")
        else:
            return (True, original_size, optimized_size, "No size reduction achieved")
    except Exception as e:
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return (False, original_size, 0, f"Optimization error: {e}")


def optimize_pdf_inplace(
    input_path: str | os.PathLike,
    *,
    linearize: bool = True,
    garbage_level: int = 3,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, int, int, str]:
    """
    Optimize a PDF in place using a temporary file. Only replaces the original
    if the optimized file is smaller. Returns (replaced, original_size,
    optimized_size, message).
    """
    in_path = Path(input_path)
    if not in_path.exists():
        return (False, 0, 0, "Input not found")

    # Create temp file in the SAME DIRECTORY to ensure same filesystem/drive
    try:
        with tempfile.NamedTemporaryFile(
            prefix=in_path.stem + ".", suffix=".opt.pdf", dir=in_path.parent, delete=False
        ) as tf:
            temp_out = Path(tf.name)
    except Exception as e:
        return (False, 0, 0, f"Failed to create temp file: {e}")

    success, orig, opt, msg = optimize_pdf(
        in_path, temp_out, linearize=linearize, garbage_level=garbage_level, log=log
    )
    if not success:
        try:
            if temp_out.exists():
                temp_out.unlink()
        except Exception:
            pass
        return (False, orig, opt, msg)

    try:
        # Only replace if smaller; otherwise keep original
        if opt < orig:
            perms = None
            try:
                perms = in_path.stat().st_mode
            except Exception:
                pass
            os.replace(temp_out, in_path)
            if perms is not None:
                try:
                    os.chmod(in_path, perms)
                except Exception:
                    pass
            return (True, orig, opt, msg)
        else:
            # No gain; remove temp
            try:
                temp_out.unlink()
            except Exception:
                pass
            return (False, orig, opt, msg)
    except Exception as e:
        try:
            if temp_out.exists():
                temp_out.unlink()
        except Exception:
            pass
        return (False, orig, opt, f"Failed to replace original: {e}")


def optimize_if_large(
    input_path: str | os.PathLike,
    *,
    min_size_mb: float = 20.0,
    linearize: bool = True,
    garbage_level: int = 3,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, int, int, str]:
    """
    Optimize the PDF in place only if its size is >= min_size_mb.
    Returns (replaced, original_size, optimized_size, message).
    """
    path = Path(input_path)
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    threshold = int(min_size_mb * 1024 * 1024)
    if size >= threshold:
        return optimize_pdf_inplace(
            path, linearize=linearize, garbage_level=garbage_level, log=log
        )
    return (False, size, size, "Below size threshold – skipped")


def rasterize_pdf(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    dpi: int = 150,
    jpeg_quality: int = 70,
    linearize: bool = True,
    garbage_level: int = 3,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, int, int, str]:
    """
    Aggressively reduce size by rasterizing pages to images and rebuilding PDF.
    This will remove text/selectability. Returns (success, original_size,
    rasterized_size, message).
    """
    logger = log or _log_default
    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.exists():
        return (False, 0, 0, "Input not found")
    try:
        original_size = in_path.stat().st_size
    except Exception:
        original_size = 0

    try:
        scale = max(36, int(dpi)) / 72.0
        mat = fitz.Matrix(scale, scale)
        with fitz.open(in_path) as src:
            new_pdf = fitz.open()
            for page in src:
                rect = page.rect
                pm = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pm.tobytes("jpeg", quality=max(30, min(int(jpeg_quality), 95)))
                new_page = new_pdf.new_page(width=rect.width, height=rect.height)
                new_page.insert_image(rect, stream=img_bytes)
            save_kwargs = {
                "garbage": max(0, min(int(garbage_level), 4)),
                "deflate": True,
                "clean": True,
                "linear": bool(linearize),
                "incremental": False,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            new_pdf.save(out_path.as_posix(), **save_kwargs)
            new_pdf.close()
        raster_size = out_path.stat().st_size if out_path.exists() else 0
        if raster_size <= 0:
            return (False, original_size, 0, "Raster output missing or empty")
        ratio = raster_size / max(1, original_size)
        return (True, original_size, raster_size, f"Rasterized to {ratio:.1%} of original")
    except Exception as e:
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return (False, original_size, 0, f"Rasterization error: {e}")


def rasterize_pdf_inplace(
    input_path: str | os.PathLike,
    *,
    dpi: int = 150,
    jpeg_quality: int = 70,
    linearize: bool = True,
    garbage_level: int = 3,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, int, int, str]:
    """
    Rasterize PDF in place using a temporary file; replace original only if
    rasterized file is smaller.
    """
    in_path = Path(input_path)
    try:
        with tempfile.NamedTemporaryFile(
            prefix=in_path.stem + ".", suffix=".ras.pdf", dir=in_path.parent, delete=False
        ) as tf:
            temp_out = Path(tf.name)
    except Exception as e:
        return (False, 0, 0, f"Failed to create temp file: {e}")

    success, orig, ras, msg = rasterize_pdf(
        in_path,
        temp_out,
        dpi=dpi,
        jpeg_quality=jpeg_quality,
        linearize=linearize,
        garbage_level=garbage_level,
        log=log,
    )
    if not success:
        try:
            if temp_out.exists():
                temp_out.unlink()
        except Exception:
            pass
        return (False, orig, ras, msg)

    try:
        if ras < orig:
            perms = None
            try:
                perms = in_path.stat().st_mode
            except Exception:
                pass
            os.replace(temp_out, in_path)
            if perms is not None:
                try:
                    os.chmod(in_path, perms)
                except Exception:
                    pass
            return (True, orig, ras, msg)
        try:
            temp_out.unlink()
        except Exception:
            pass
        return (False, orig, ras, msg)
    except Exception as e:
        try:
            if temp_out.exists():
                temp_out.unlink()
        except Exception:
            pass
        return (False, orig, ras, f"Failed to replace original: {e}")


def optimize_with_raster_fallback_if_large(
    input_path: str | os.PathLike,
    *,
    min_size_mb: float = 20.0,
    linearize: bool = True,
    garbage_level: int = 3,
    enable_raster_fallback: bool = False,
    raster_dpi: int = 150,
    jpeg_quality: int = 70,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, int, int, str]:
    """
    First try structural optimization; if no reduction and fallback enabled,
    rasterize at given dpi/quality. Returns final (replaced, orig, new, msg).
    """
    # Try standard optimization first
    replaced, orig, opt, msg = optimize_if_large(
        input_path,
        min_size_mb=min_size_mb,
        linearize=linearize,
        garbage_level=garbage_level,
        log=log,
    )
    # If below threshold, just return
    if orig == opt and msg.startswith("Below size threshold"):
        return (replaced, orig, opt, msg)

    # If not reduced and fallback requested, rasterize
    if enable_raster_fallback and (not replaced) and opt >= orig:
        r_replaced, r_orig, r_new, r_msg = rasterize_pdf_inplace(
            input_path,
            dpi=raster_dpi,
            jpeg_quality=jpeg_quality,
            linearize=linearize,
            garbage_level=garbage_level,
            log=log,
        )
        return (r_replaced, r_orig, r_new, r_msg)

    return (replaced, orig, opt, msg)


# Spotlight utilities
def _normalize_search_text(s: str) -> str:
    try:
        s2 = (s or "").replace("\u00a0", " ").strip()
        # Prefer substring before first colon and trim trailing punctuation
        if ":" in s2:
            s2 = s2.split(":", 1)[0].strip()
        return s2.rstrip(" :.")
    except Exception:
        return s


def _extract_section_code(header_text: str) -> str:
    """Return a best-effort section code like 3.5, 3.5.1, F4, F4.a, 1B6b from a header string.

    Lowercased, no trailing punctuation.
    """
    import re as _re
    try:
        s = (header_text or "").strip()
        # Prefer letter-first like F4 or F4.a or A10.2b
        m = _re.search(r"[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?", s)
        if m:
            return m.group(0).rstrip(" :.").lower()
        # Digit-letter like 1B6 or 1B6b
        m = _re.search(r"\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?", s)
        if m:
            return m.group(0).rstrip(" :.").lower()
        # Numeric dotted like 3.5, 3.5.1
        m = _re.search(r"\d+(?:\.[A-Za-z0-9]+)+", s)
        if m:
            return m.group(0).rstrip(" :.").lower()
    except Exception:
        pass
    return ""


def _tokenize_words_for_match(s: str) -> List[str]:
    """Lightweight tokenizer for matching header strings against PDF words.

    - Lowercases
    - Collapses whitespace
    - Strips leading/trailing punctuation
    - Keeps dots inside tokens (so 3.5.1 stays intact)
    """
    import re as _re
    try:
        s2 = (s or "").replace("\u00a0", " ")
        s2 = _re.sub(r"\s+", " ", s2).strip().lower()
        # Split on spaces, then strip common punctuation around tokens
        toks = []
        for t in s2.split():
            t2 = _re.sub(r"^[\s\-–—:;,.]+|[\s\-–—:;,.]+$", "", t)
            if t2:
                toks.append(t2)
        return toks
    except Exception:
        return [s]


def find_header_anchor(pdf_path: str | os.PathLike, page_number_1_based: int, header_text: str) -> Tuple[float, float] | None:
    """
    Return (x, y) top-left coordinates of the first occurrence of header_text on the page.
    Falls back to searching only the leading code token if full header match fails.
    Coordinates are in PDF points (72 dpi) in page space.
    """
    try:
        pg_idx = max(1, int(page_number_1_based)) - 1
    except Exception:
        return None
    needle_full = _normalize_search_text(str(header_text or ""))
    code_only = _extract_section_code(needle_full) or (needle_full.split()[0].lower() if needle_full else "")
    try:
        with fitz.open(str(pdf_path)) as doc:
            if pg_idx < 0 or pg_idx >= len(doc):
                return None
            page = doc.load_page(pg_idx)
            # Robust word-based matching (case-insensitive, punctuation-insensitive)
            try:
                words_raw = page.get_text("words")  # [x0, y0, x1, y1, word, block, line, word_no]
            except Exception:
                words_raw = []
            words: List[Tuple[float, float, float, float, str, int, int, int]] = []  # type: ignore
            for w in words_raw or []:
                try:
                    x0, y0, x1, y1, t, b, l, wn = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4]), int(w[5]), int(w[6]), int(w[7])
                except Exception:
                    # Be defensive with tuple layout
                    try:
                        x0 = float(w[0]); y0 = float(w[1]); x1 = float(w[2]); y1 = float(w[3]); t = str(w[4])
                        b = int(w[5]) if len(w) > 5 else 0
                        l = int(w[6]) if len(w) > 6 else 0
                        wn = int(w[7]) if len(w) > 7 else 0
                    except Exception:
                        continue
                if not t or not t.strip():
                    continue
                words.append((x0, y0, x1, y1, t, b, l, wn))
            # Sort words by line and then by word number to approximate reading order
            words.sort(key=lambda w: (w[5], w[6], w[7], w[1], w[0]))

            # Phase 1: try to locate by section code token
            if code_only:
                code_low = code_only.lower()
                for x0, y0, x1, y1, t, _b, _l, _wn in words:
                    tt = (t or "").replace("\u00a0", " ").strip().rstrip(" :.").lower()
                    if tt.startswith(code_low) or tt == code_low:
                        return (float(x0), float(y0))

            # Phase 2: try to match first few header tokens in sequence
            header_tokens = _tokenize_words_for_match(needle_full)[:6]
            if header_tokens:
                page_tokens = _tokenize_words_for_match(" ".join([w[4] for w in words]))
                # Build index from token position to word index (approximate: map by linear scan)
                # We will try a sliding window over the words list comparing normalized tokens
                def _norm_word(s: str) -> str:
                    import re as _re
                    s2 = (s or "").lower().replace("\u00a0", " ")
                    return _re.sub(r"^[\s\-–—:;,.]+|[\s\-–—:;,.]+$", "", s2)
                norm_words: List[str] = [_norm_word(w[4]) for w in words]
                mlen = len(header_tokens)
                if mlen > 0 and len(norm_words) >= mlen:
                    for i in range(0, len(norm_words) - mlen + 1):
                        window = norm_words[i:i+mlen]
                        # Require that the window starts with the first header token and that
                        # at least half of the tokens match to reduce false positives
                        if window and window[0] == header_tokens[0].lower():
                            eq = sum(1 for a, b in zip(window, [t.lower() for t in header_tokens]) if a == b)
                            if eq >= max(2, mlen // 2):
                                x0, y0 = float(words[i][0]), float(words[i][1])
                                return (x0, y0)

            # Phase 3: fall back to built-in search_for for the full header
            if needle_full:
                try:
                    rects = page.search_for(needle_full)
                    if rects:
                        r = rects[0]
                        return (float(r.x0), float(r.y0))
                except Exception:
                    pass
            # Phase 4: final fallback – search by code token literally
            if code_only:
                try:
                    rects2 = page.search_for(code_only)
                    if rects2:
                        r = rects2[0]
                        return (float(r.x0), float(r.y0))
                except Exception:
                    pass
    except Exception:
        return None
    return None

