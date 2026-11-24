"""
Standalone PDF redactor for Vercel serverless functions.
Adds watermark and footer to every page. No content removal.
Extracted from ai_enhanced_redactor.py — no Streamlit dependency.
"""

import io
import os
import sys
import gc
import logging
import time
import requests
from pathlib import Path
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger("redactor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Branding configuration
# ---------------------------------------------------------------------------
DEFAULT_WATERMARK_TEXT = "Precon Factory\n(647) 956-4063"
DEFAULT_FOOTER_IMAGE_URL = "https://cfzuypbljirmibmxpabi.supabase.co/storage/v1/object/public/email-images/footer/footer%20precon%20factory.png"

FAHAD_WATERMARK_TEXT = "Fahad Javed\n(289) 536-9724"
FAHAD_FOOTER_IMAGE_URL = "https://images.preconfactory.com/storage/v1/object/public/rental-documents/1790030570491_0_new_footer_fj.png"
FAHAD_FOOTER_LOCAL_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "fahad_javed_footer.png"

GTA_LOWRISE_WATERMARK_TEXT = "GTA Lowrise\n(289) 536-9724"
GTA_LOWRISE_FOOTER_IMAGE_URL = "https://images.preconfactory.com/storage/v1/object/public/rental-documents/1794798498507_0_gta_lowrise_footer.png"
GTA_LOWRISE_FOOTER_LOCAL_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "gta_lowrise_footer.png"

WATERMARK_OPACITY = 0.30
WATERMARK_ANGLE_DEGREES = 35
WATERMARK_FONT_FRACTION_OF_MIN_SIDE = 0.14


# ---------------------------------------------------------------------------
# Branding routing
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in (text or "") if ch.isalnum())


def _is_fahad(path: str) -> bool:
    name = (path or "").split("/")[0]
    lower = name.lower()
    compact = _normalize(name)
    return "fahad" in lower or compact.startswith("fj")


def _get_branding_config(project_folder_path: str):
    folder_path_lower = project_folder_path.lower()
    if "gta" in folder_path_lower and "lowrise" in folder_path_lower:
        return GTA_LOWRISE_WATERMARK_TEXT, GTA_LOWRISE_FOOTER_IMAGE_URL
    if _is_fahad(project_folder_path):
        return FAHAD_WATERMARK_TEXT, FAHAD_FOOTER_IMAGE_URL
    return DEFAULT_WATERMARK_TEXT, DEFAULT_FOOTER_IMAGE_URL


# ---------------------------------------------------------------------------
# Footer / watermark helpers
# ---------------------------------------------------------------------------
def _download_image_bytes(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(
            url, timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PDFProcessor/1.0)",
                "Accept": "image/png,image/*,*/*;q=0.8",
                "Cache-Control": "no-cache",
            },
        )
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.error("Failed to download image from %s: %s", url, e)
        return None


def _load_footer_bytes(watermark_text: str, footer_image_url: str) -> Optional[bytes]:
    if watermark_text == FAHAD_WATERMARK_TEXT and FAHAD_FOOTER_LOCAL_PATH.exists():
        return FAHAD_FOOTER_LOCAL_PATH.read_bytes()
    if watermark_text == GTA_LOWRISE_WATERMARK_TEXT and GTA_LOWRISE_FOOTER_LOCAL_PATH.exists():
        return GTA_LOWRISE_FOOTER_LOCAL_PATH.read_bytes()
    return _download_image_bytes(footer_image_url)


def _font_candidates() -> List[str]:
    base = Path(__file__).resolve().parent
    candidates = [
        str(base / "fonts" / "DejaVuSans-Bold.ttf"),
        str(base / "fonts" / "DejaVuSans.ttf"),
    ]
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ])
    return candidates


def _load_font() -> fitz.Font:
    for path in _font_candidates():
        if not path or not os.path.isfile(path):
            continue
        try:
            return fitz.Font(fontfile=path)
        except Exception:
            try:
                with open(path, "rb") as fp:
                    return fitz.Font(fontbuffer=fp.read())
            except Exception:
                continue
    return fitz.Font("hebo")


def _watermark_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]


def _apply_watermark(page: fitz.Page, text: str, opacity: float, angle_deg: float):
    lines = _watermark_lines(text)
    if not lines:
        return
    font = _load_font()
    rect = page.rect
    w, h = rect.width, rect.height
    cx, cy = w / 2.0, h / 2.0
    fontsize = max(24.0, min(w, h) * float(WATERMARK_FONT_FRACTION_OF_MIN_SIDE))
    asc, desc = font.ascender, font.descender
    line_height = fontsize * (asc - desc) if (asc > desc and (asc - desc) > 0.01) else fontsize * 1.2
    span = (len(lines) - 1) * line_height
    y_first = cy - span / 2.0
    tw = fitz.TextWriter(rect)
    for i, line in enumerate(lines):
        lw = font.text_length(line, fontsize)
        tw.append(fitz.Point(cx - lw / 2.0, y_first + i * line_height), line, font=font, fontsize=fontsize)
    morph = (fitz.Point(cx, cy), fitz.Matrix(angle_deg))
    tw.write_text(page, color=(0, 0, 0), opacity=max(0.0, min(1.0, float(opacity))), overlay=True, morph=morph)


def _insert_footer(page: fitz.Page, footer_png: Optional[bytes]):
    if not footer_png:
        return
    try:
        pw, ph = page.rect.width, page.rect.height
        with Image.open(io.BytesIO(footer_png)) as im:
            iw, ih = im.size
        scale = pw / float(iw)
        th = ih * scale
        rect = fitz.Rect(0, ph - th, pw, ph)
        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), overlay=True)
        page.insert_image(rect, stream=footer_png, keep_proportion=True, overlay=True)
    except Exception as e:
        logger.warning("Failed to insert footer: %s", e)


# ---------------------------------------------------------------------------
# Broken-PDF repair
# ---------------------------------------------------------------------------
def _needs_repair(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "xref",
            "cannot find object",
            "code=7",
            "format error",
            "cannot open",
            "source object number out of range",
        )
    )


def _probe_pdf(path: str) -> bool:
    """True when MuPDF can open every page."""
    try:
        doc = fitz.open(path)
    except Exception:
        return False
    try:
        if doc.page_count < 1:
            return False
        for i in range(doc.page_count):
            page = doc[i]
            if page.rect.width <= 0 or page.rect.height <= 0:
                return False
            page.get_contents()
        return True
    except Exception:
        return False
    finally:
        doc.close()


def _repair_with_pikepdf(src: str, dst: str) -> bool:
    import pikepdf

    with pikepdf.open(src, attempt_recovery=True) as pdf:
        pdf.save(dst)
    return os.path.isfile(dst) and os.path.getsize(dst) > 100 and _probe_pdf(dst)


def _repair_with_pypdf(src: str, dst: str) -> bool:
    from pypdf import PdfReader, PdfWriter

    try:
        reader = PdfReader(src, strict=False)
    except TypeError:
        reader = PdfReader(src)
    writer = PdfWriter()
    kept = 0
    for i, page in enumerate(reader.pages):
        try:
            writer.add_page(page)
            kept += 1
        except Exception as exc:
            logger.warning("Dropping unreadable page %s: %s", i + 1, exc)
    if kept < 1:
        return False
    with open(dst, "wb") as handle:
        writer.write(handle)
    return os.path.isfile(dst) and os.path.getsize(dst) > 100 and _probe_pdf(dst)


def _repair_by_raster(src: str, dst: str) -> bool:
    """Last resort: redraw each readable page as an image."""
    try:
        doc = fitz.open(src)
    except Exception as exc:
        logger.warning("Raster open failed: %s", exc)
        return False
    out = fitz.open()
    kept = 0
    try:
        for i in range(doc.page_count):
            try:
                page = doc[i]
                longest = max(page.rect.width, page.rect.height) or 1
                zoom = min(2.0, 2400.0 / longest)
                zoom = max(0.6, zoom)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                new_page = out.new_page(width=page.rect.width, height=page.rect.height)
                new_page.insert_image(new_page.rect, stream=pix.tobytes("jpeg", jpg_quality=82))
                kept += 1
            except Exception as exc:
                logger.warning("Raster page %s failed: %s", i + 1, exc)
        if kept < 1:
            return False
        out.save(dst, garbage=4, deflate=True)
        return _probe_pdf(dst)
    finally:
        out.close()
        doc.close()


def _repair_pdf(src: str) -> str:
    dst = src + ".repaired.pdf"
    errors = []
    for name, fn in (
        ("pikepdf", _repair_with_pikepdf),
        ("pypdf", _repair_with_pypdf),
        ("raster", _repair_by_raster),
    ):
        try:
            if fn(src, dst):
                logger.info("Repaired PDF with %s", name)
                return dst
            errors.append(f"{name}: output still unreadable")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("%s repair failed: %s", name, exc)
    raise RuntimeError("Could not repair PDF (" + "; ".join(errors) + ")")


# ---------------------------------------------------------------------------
# Manual white-out
# ---------------------------------------------------------------------------
MAX_WHITEOUTS = 1000


def parse_whiteouts(raw) -> List[dict]:
    """Validate boxes sent by the browser.

    Each box is {page, x0, y0, x1, y1}: a 0-based page index and corners as
    fractions (0..1) of the page as displayed, origin top-left.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        import json
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("whiteouts must be a list")
    boxes = []
    for item in raw[:MAX_WHITEOUTS]:
        try:
            page = int(item["page"])
            xs = sorted(min(1.0, max(0.0, float(item[k]))) for k in ("x0", "x1"))
            ys = sorted(min(1.0, max(0.0, float(item[k]))) for k in ("y0", "y1"))
        except (KeyError, TypeError, ValueError):
            continue
        if page < 0 or xs[1] - xs[0] < 0.002 or ys[1] - ys[0] < 0.002:
            continue
        boxes.append({"page": page, "x0": xs[0], "y0": ys[0], "x1": xs[1], "y1": ys[1]})
    return boxes


def _apply_whiteouts(doc: fitz.Document, whiteouts: List[dict]) -> int:
    by_page = {}
    for box in whiteouts:
        by_page.setdefault(box["page"], []).append(box)

    applied = 0
    for pno, boxes in by_page.items():
        if pno >= doc.page_count:
            continue
        page = doc[pno]
        visible = page.rect
        rects = []
        for b in boxes:
            shown = fitz.Rect(
                visible.x0 + b["x0"] * visible.width,
                visible.y0 + b["y0"] * visible.height,
                visible.x0 + b["x1"] * visible.width,
                visible.y0 + b["y1"] * visible.height,
            )
            rects.append(shown * page.derotation_matrix)
        try:
            for rect in rects:
                page.add_redact_annot(rect, fill=(1, 1, 1))
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_PIXELS,
                graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
            )
        except Exception as exc:
            logger.warning("Redaction failed on page %s, covering instead: %s", pno + 1, exc)
            for annot in list(page.annots(types=[fitz.PDF_ANNOT_REDACT]) or []):
                page.delete_annot(annot)
        # A white box on top guarantees the area is blank even if the
        # underlying content could not be removed.
        for rect in rects:
            page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)
        applied += len(rects)
    return applied


def _stamp_pdf(
    source: str,
    output_path: str,
    watermark_text: str,
    footer_png: Optional[bytes],
    include_footer: bool,
    whiteouts: Optional[List[dict]] = None,
):
    doc = fitz.open(source)
    try:
        if whiteouts:
            count = _apply_whiteouts(doc, whiteouts)
            logger.info("Applied %s white-out area(s)", count)
        for page_num, page in enumerate(doc):
            _apply_watermark(page, watermark_text, WATERMARK_OPACITY, WATERMARK_ANGLE_DEGREES)
            if include_footer:
                _insert_footer(page, footer_png)
            if (page_num + 1) % 10 == 0:
                gc.collect()
        doc.save(output_path, garbage=4, deflate=True)
    finally:
        doc.close()
        del doc
        gc.collect()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def process_pdf_enhanced(
    input_path: str,
    output_path: str,
    project_folder_path: str = "",
    include_footer: bool = True,
    whiteouts: Optional[List[dict]] = None,
):
    """White out requested areas, then add watermark (and optional footer) to every page."""
    start = time.time()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    watermark_text, footer_url = _get_branding_config(project_folder_path)
    footer_png = _load_footer_bytes(watermark_text, footer_url) if include_footer else None
    if include_footer and not footer_png:
        logger.error("Footer image could not be loaded")

    source = input_path
    if not _probe_pdf(source):
        logger.warning("PDF failed to open cleanly, repairing first: %s", input_path)
        source = _repair_pdf(input_path)

    try:
        _stamp_pdf(source, output_path, watermark_text, footer_png, include_footer, whiteouts)
    except Exception as exc:
        if source != input_path or not _needs_repair(exc):
            raise
        logger.warning("Stamp failed on original PDF (%s), repairing and retrying", exc)
        source = _repair_pdf(input_path)
        _stamp_pdf(source, output_path, watermark_text, footer_png, include_footer, whiteouts)
    finally:
        if footer_png:
            del footer_png
        gc.collect()

    logger.info("Processed %s in %.2fs", input_path, time.time() - start)
