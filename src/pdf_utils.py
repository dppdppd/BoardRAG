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
from typing import Callable, Optional, Tuple, List, Dict

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


def compute_normalized_header_bbox(
    pdf_path: str | os.PathLike,
    page_number_1_based: int,
    header_text: str,
    *,
    cal_margin_pct_h: float = 2.0,
    cal_margin_pct_v: float = 2.0,
) -> Tuple[float, float, float, float] | None:
    """
    Compute a normalized [x,y,w,h] bbox in [0-100] percentages for the header
    within a 2% inset frame to avoid page edges.

    - x is vertical from top inset origin; y is horizontal from left inset origin
    - Uses PyMuPDF word positions; approximates width from the span of header tokens.
    - Falls back to an estimated width if token matching is poor.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    try:
        pg_idx = max(1, int(page_number_1_based)) - 1
    except Exception:
        return None
    text_norm = _normalize_search_text(str(header_text or ""))
    if not text_norm:
        return None
    tokens = _tokenize_words_for_match(text_norm)
    if not tokens:
        return None
    try:
        with fitz.open(str(pdf_path)) as doc:
            if pg_idx < 0 or pg_idx >= len(doc):
                return None
            page = doc.load_page(pg_idx)
            rect = page.rect
            w = float(rect.width)
            h = float(rect.height)
            inset_x = (max(0.0, cal_margin_pct_h) / 100.0) * w
            inset_y = (max(0.0, cal_margin_pct_v) / 100.0) * h
            frame_left = rect.x0 + inset_x
            frame_top = rect.y0 + inset_y
            frame_w = max(1.0, w - 2 * inset_x)
            frame_h = max(1.0, h - 2 * inset_y)

            try:
                words_raw = page.get_text("words")
            except Exception:
                words_raw = []
            words: List[Tuple[float, float, float, float, str, int, int, int]] = []  # type: ignore
            for wrec in words_raw or []:
                try:
                    x0, y0, x1, y1, t, b, l, wn = float(wrec[0]), float(wrec[1]), float(wrec[2]), float(wrec[3]), str(wrec[4]), int(wrec[5]), int(wrec[6]), int(wrec[7])
                except Exception:
                    try:
                        x0 = float(wrec[0]); y0 = float(wrec[1]); x1 = float(wrec[2]); y1 = float(wrec[3]); t = str(wrec[4])
                        b = int(wrec[5]) if len(wrec) > 5 else 0
                        l = int(wrec[6]) if len(wrec) > 6 else 0
                        wn = int(wrec[7]) if len(wrec) > 7 else 0
                    except Exception:
                        continue
                if not t or not t.strip():
                    continue
                words.append((x0, y0, x1, y1, t, b, l, wn))
            words.sort(key=lambda ww: (ww[6], ww[7], ww[5], ww[1], ww[0]))

            # If a section code token exists (e.g., 6/3, 3.1.2), highlight ONLY that token
            try:
                code_only = _extract_section_code(text_norm)
            except Exception:
                code_only = ""
            if code_only:
                def _norm_word(s: str) -> str:
                    import re as _re
                    s2 = (s or "").lower().replace("\u00a0", " ")
                    return _re.sub(r"^[\s\-–—:;,.]+|[\s\-–—:;,.]+$", "", s2)
                for wrec in words:
                    try:
                        x0, y0, x1, y1, t = float(wrec[0]), float(wrec[1]), float(wrec[2]), float(wrec[3]), str(wrec[4])
                    except Exception:
                        continue
                    tw = _norm_word(t)
                    if not tw:
                        continue
                    if tw == code_only or tw.startswith(code_only):
                        # Convert this single word rect to normalized percentages and return immediately
                        y_pct = max(0.0, min(100.0, ((float(x0) - frame_left) / frame_w) * 100.0))
                        x_pct = max(0.0, min(100.0, ((float(y0) - frame_top) / frame_h) * 100.0))
                        w_pct = max(0.2, min(100.0, ((float(x1) - float(x0)) / frame_w) * 100.0))
                        h_pct = max(0.2, min(100.0, ((float(y1) - float(y0)) / frame_h) * 100.0))
                        return (x_pct, y_pct, w_pct, h_pct)

            # Find a contiguous sequence matching the exact header tokens (case/space/punct-insensitive)
            header_tokens = tokens
            def _norm_word(s: str) -> str:
                import re as _re
                s2 = (s or "").lower().replace("\u00a0", " ")
                return _re.sub(r"^[\s\-–—:;,.]+|[\s\-–—:;,.]+$", "", s2)
            norm_words: List[str] = [_norm_word(wr[4]) for wr in words]
            mlen = len(header_tokens)
            i0: Optional[int] = None
            i1: Optional[int] = None
            matched_indices: Optional[List[int]] = None
            if mlen > 0 and len(norm_words) >= mlen:
                # Pass 1: exact contiguous match
                for i in range(0, len(norm_words) - mlen + 1):
                    window = norm_words[i:i+mlen]
                    if all(a == b.lower() for a, b in zip(window, header_tokens)):
                        i0, i1 = i, i + mlen - 1
                        matched_indices = list(range(i0, i1 + 1))
                        break
                # Pass 2: allow small gaps across line/block breaks within vertical threshold
                if i0 is None or i1 is None:
                    # Estimate allowed vertical jump as ~1.6x median word height
                    try:
                        import statistics as _stats
                        heights = [max(0.5, float(wr[3]) - float(wr[1])) for wr in words]
                        med_h = _stats.median(heights) if heights else 8.0
                        allowed_gap = max(6.0, 1.6 * float(med_h))
                    except Exception:
                        allowed_gap = 12.0
                    for i in range(0, len(norm_words)):
                        cur: List[int] = []
                        k = 0
                        last_y: Optional[float] = None
                        last_line: Optional[int] = None
                        # Advance j collecting tokens in order; skip non-matching words, but enforce vertical proximity when line changes
                        for j in range(i, len(norm_words)):
                            if k >= mlen:
                                break
                            if norm_words[j] == header_tokens[k].lower():
                                y0j = float(words[j][1])
                                line_j = int(words[j][6]) if len(words[j]) > 6 else None  # type: ignore
                                if last_y is not None and last_line is not None and line_j is not None:
                                    if line_j != last_line:
                                        if abs(y0j - last_y) > allowed_gap:
                                            # too far; abandon this start
                                            cur = []
                                            break
                                cur.append(j)
                                last_y = y0j
                                last_line = line_j
                                k += 1
                                if k == mlen:
                                    break
                        if k == mlen and cur:
                            # Ensure tokens are on at most two lines to avoid spanning large blocks
                            try:
                                lines_used = len(set(int(words[idx][6]) for idx in cur))
                            except Exception:
                                lines_used = 2
                            if lines_used <= 2:
                                matched_indices = cur
                                i0, i1 = cur[0], cur[-1]
                                break
            # If no exact contiguous match, try PyMuPDF search_for on the full normalized header
            if i0 is None or i1 is None or not matched_indices:
                try:
                    rects = page.search_for(text_norm)
                    if rects:
                        r = rects[0]
                        y_pct = max(0.0, min(100.0, ((float(r.x0) - frame_left) / frame_w) * 100.0))
                        x_pct = max(0.0, min(100.0, ((float(r.y0) - frame_top) / frame_h) * 100.0))
                        w_pct = max(0.2, min(100.0, ((float(r.x1) - float(r.x0)) / frame_w) * 100.0))
                        h_pct = max(0.2, min(100.0, ((float(r.y1) - float(r.y0)) / frame_h) * 100.0))
                        return (x_pct, y_pct, w_pct, h_pct)
                except Exception:
                    pass
                return None

            # Union bbox across matched word rectangles (exact words only)
            xs0 = [float(words[idx][0]) for idx in matched_indices]
            ys0 = [float(words[idx][1]) for idx in matched_indices]
            xs1 = [float(words[idx][2]) for idx in matched_indices]
            ys1 = [float(words[idx][3]) for idx in matched_indices]
            x0 = min(min(xs0), min(xs1))
            y0 = min(ys0)
            x1 = max(max(xs0), max(xs1))
            y1 = max(ys1)
            # Convert to normalized percentages within inset frame
            y_pct = max(0.0, min(100.0, ((x0 - frame_left) / frame_w) * 100.0))
            x_pct = max(0.0, min(100.0, ((y0 - frame_top) / frame_h) * 100.0))
            w_pct = max(0.2, min(100.0, ((x1 - x0) / frame_w) * 100.0))
            h_pct = max(0.2, min(100.0, ((y1 - y0) / frame_h) * 100.0))
            return (x_pct, y_pct, w_pct, h_pct)
    except Exception:
        return None
    return None


