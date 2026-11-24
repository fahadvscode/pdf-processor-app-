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
from PIL import Image, ImageOps

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


def _probe_pdf(data: bytes) -> bool:
    """True when MuPDF can open every page."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
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


def _repair_with_pikepdf(data: bytes) -> Optional[bytes]:
    import pikepdf

    out = io.BytesIO()
    with pikepdf.open(io.BytesIO(data), attempt_recovery=True) as pdf:
        pdf.save(out)
    fixed = out.getvalue()
    return fixed if len(fixed) > 100 and _probe_pdf(fixed) else None


def _repair_with_pypdf(data: bytes) -> Optional[bytes]:
    from pypdf import PdfReader, PdfWriter

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except TypeError:
        reader = PdfReader(io.BytesIO(data))
    writer = PdfWriter()
    kept = 0
    for i, page in enumerate(reader.pages):
        try:
            writer.add_page(page)
            kept += 1
        except Exception as exc:
            logger.warning("Dropping unreadable page %s: %s", i + 1, exc)
    if kept < 1:
        return None
    out = io.BytesIO()
    writer.write(out)
    fixed = out.getvalue()
    return fixed if len(fixed) > 100 and _probe_pdf(fixed) else None


def _repair_by_raster(data: bytes) -> Optional[bytes]:
    """Last resort: redraw each readable page as an image."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        logger.warning("Raster open failed: %s", exc)
        return None
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
            return None
        fixed = out.tobytes(garbage=1, deflate=True)
        return fixed if _probe_pdf(fixed) else None
    finally:
        out.close()
        doc.close()


def _repair_pdf(data: bytes) -> bytes:
    errors = []
    for name, fn in (
        ("pikepdf", _repair_with_pikepdf),
        ("pypdf", _repair_with_pypdf),
        ("raster", _repair_by_raster),
    ):
        try:
            fixed = fn(data)
            if fixed:
                logger.info("Repaired PDF with %s", name)
                return fixed
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
    data: bytes,
    watermark_text: str,
    footer_png: Optional[bytes],
    include_footer: bool,
    whiteouts: Optional[List[dict]] = None,
) -> bytes:
    """Everything happens in memory; the server's temporary disk is too small for site plans."""
    doc = fitz.open(stream=data, filetype="pdf")
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
        # garbage=1 drops the content a white-out removed. Without white-out,
        # existing streams are copied as they are, so the file stays its size.
        return doc.tobytes(garbage=1 if whiteouts else 0, deflate=True)
    finally:
        doc.close()
        del doc
        gc.collect()


# ---------------------------------------------------------------------------
# Images → one PDF, so white-out / watermark / footer stay on one path
# ---------------------------------------------------------------------------
MAX_IMAGE_EDGE = 4096
MAX_IMAGE_PAGES = 100

_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".apng", ".gif", ".webp",
    ".bmp", ".dib", ".tif", ".tiff", ".heic", ".heif", ".avif", ".svg",
    ".ico", ".cur", ".jxl",
}
_HEIF_BRANDS = {b"heic", b"heix", b"heif", b"hevc", b"mif1", b"msf1", b"avif", b"avis"}
_heif_ready = False


def _enable_heif() -> None:
    """Turn on HEIC/AVIF in Pillow when the optional decoder is installed."""
    global _heif_ready
    if _heif_ready:
        return
    _heif_ready = True
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        logger.info("HEIC/AVIF decoder is not installed; those files are converted in the browser")


def _looks_like_svg(data: bytes) -> bool:
    sample = data[:4000].lstrip().lower()
    if sample.startswith(b"<svg"):
        return True
    if sample.startswith((b"<?xml", b"<!doctype")) and b"<svg" in sample:
        return True
    return False


def file_kind(data: bytes, name: str = "") -> str:
    """'pdf', 'image', or 'unknown' from the file header, then the extension."""
    if not data:
        return "unknown"
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:3] == b"\xff\xd8\xff":
        return "image"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image"
    if data[:2] == b"BM":
        return "image"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image"
    if data[:4] == b"\x00\x00\x01\x00":
        return "image"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image"
    if len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12].lower() in _HEIF_BRANDS:
        return "image"
    if _looks_like_svg(data):
        return "image"
    ext = os.path.splitext(name or "")[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    return "unknown"


def _flatten_image(im: Image.Image) -> Image.Image:
    """Apply the camera orientation and paint any transparency onto white."""
    im = ImageOps.exif_transpose(im)
    if im.mode == "RGB":
        return im
    rgba = im.convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.getchannel("A"))
    return bg


def image_bytes_to_pdf(data: bytes) -> bytes:
    """Wrap an image in a PDF page per frame. The picture fills each page.

    White-out boxes are fractions of the page, so they land on the same place
    the browser showed. TIFF keeps every page; everything else uses the first
    frame, which is the frame the browser previews.
    """
    if not data:
        raise ValueError("Image was empty")
    _enable_heif()

    if _looks_like_svg(data):
        try:
            import cairosvg
            data = cairosvg.svg2png(bytestring=data, output_width=MAX_IMAGE_EDGE)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read this SVG ({exc})") from exc

    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        raise ValueError(
            "Could not read this image. JPG, PNG, WEBP, GIF, BMP, TIFF, HEIC, AVIF and SVG work."
        ) from exc

    is_tiff = (im.format or "").upper() == "TIFF"
    try:
        count = im.n_frames if is_tiff else 1
    except Exception:
        count = 1
    if count > MAX_IMAGE_PAGES:
        im.close()
        raise ValueError(f"This image has too many pages (limit is {MAX_IMAGE_PAGES})")

    doc = fitz.open()
    try:
        for index in range(count):
            im.seek(index)
            flat = _flatten_image(im.copy())
            if max(flat.size) > MAX_IMAGE_EDGE:
                flat.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
            if flat.width < 1 or flat.height < 1:
                raise ValueError(f"Image page {index + 1} has no size")
            buf = io.BytesIO()
            flat.save(buf, format="JPEG", quality=92, optimize=True)
            page = doc.new_page(width=flat.width, height=flat.height)
            page.insert_image(page.rect, stream=buf.getvalue())
        if doc.page_count < 1:
            raise ValueError("Image had no pages")
        logger.info("Converted image to a %s-page PDF", doc.page_count)
        return doc.tobytes(deflate=True, garbage=4)
    finally:
        doc.close()
        im.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def process_pdf_bytes(
    data: bytes,
    project_folder_path: str = "",
    include_footer: bool = True,
    whiteouts: Optional[List[dict]] = None,
) -> bytes:
    """White out requested areas, then add watermark (and optional footer) to every page."""
    start = time.time()
    if not data:
        raise ValueError("Uploaded file was empty")

    watermark_text, footer_url = _get_branding_config(project_folder_path)
    footer_png = _load_footer_bytes(watermark_text, footer_url) if include_footer else None
    if include_footer and not footer_png:
        logger.error("Footer image could not be loaded")

    source = data
    if not _probe_pdf(source):
        logger.warning("PDF failed to open cleanly, repairing first")
        source = _repair_pdf(data)

    try:
        result = _stamp_pdf(source, watermark_text, footer_png, include_footer, whiteouts)
    except Exception as exc:
        if source is not data or not _needs_repair(exc):
            raise
        logger.warning("Stamp failed on original PDF (%s), repairing and retrying", exc)
        source = _repair_pdf(data)
        result = _stamp_pdf(source, watermark_text, footer_png, include_footer, whiteouts)
    finally:
        footer_png = None
        gc.collect()

    logger.info("Processed %.1f MB in %.2fs", len(data) / 1e6, time.time() - start)
    return result
