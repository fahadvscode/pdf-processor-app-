"""
POST /api/process — process one PDF or image for one branding.

Accepts multipart form with:
  file: the PDF or image binary (JPG, PNG, WEBP, GIF, BMP, TIFF, HEIC, AVIF, SVG, …)
  project_name, branding_name, file_type, include_footer (0|1), replace_existing (0|1)

Images are wrapped in a PDF first, then white-out, watermark and footer run as usual.
Returns JSON with drive_url, file_name, processing_time, file_size, output_path.
"""
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
import cgi
import json
import os
import shutil
import sys
import tempfile
import time
import gc
import requests

# MuPDF scratch files must land on the one writable directory.
os.environ["TMPDIR"] = "/tmp"
os.environ["TEMP"] = "/tmp"
os.environ["TMP"] = "/tmp"
tempfile.tempdir = "/tmp"

# Make sure the api package root is importable so the redactor can find assets
API_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(API_DIR), "assets")

# Put project root on path so ai_enhanced_redactor resolves local footer PNGs
sys.path.insert(0, API_DIR)

from lib.drive import resolve_destination, trash_files, upload_file, _build_service


def _process_pdf(input_path: str, output_path: str, project_folder_path: str, include_footer: bool, whiteouts_raw: str):
    """Run the redactor. Imported lazily to keep cold starts fast for other endpoints."""
    from lib.redactor import parse_whiteouts, process_pdf_enhanced
    whiteouts = parse_whiteouts(whiteouts_raw)
    process_pdf_enhanced(input_path, output_path, project_folder_path, include_footer, whiteouts)
    return len(whiteouts)


def _output_pdf_name(name: str) -> str:
    """Drive name. Images are stored as a PDF with the same base name."""
    base = os.path.basename((name or "upload").replace("\x00", "")).strip() or "upload"
    stem = os.path.splitext(base)[0].strip() or "upload"
    return f"{stem[:180]}.pdf"


def _as_pdf(data: bytes, name: str) -> bytes:
    from lib.redactor import file_kind, image_bytes_to_pdf
    kind = file_kind(data, name)
    if kind == "pdf":
        return data
    if kind == "image":
        return image_bytes_to_pdf(data)
    raise ValueError(
        "Upload a PDF or an image (JPG, PNG, WEBP, GIF, BMP, TIFF, HEIC, AVIF, SVG)"
    )


def _is_blob_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return url.startswith("https://") and host.endswith(".blob.vercel-storage.com")


def _download_blob(url: str) -> bytes:
    if not _is_blob_url(url):
        raise RuntimeError("Invalid upload URL")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    if not resp.content:
        raise ValueError("Uploaded file was empty")
    return resp.content


def _reclaim_tmp() -> None:
    """Drop leftovers from an earlier run on this reused server instance."""
    if not os.environ.get("VERCEL"):
        return
    root = "/tmp"
    if not os.path.isdir(root):
        return
    try:
        free = shutil.disk_usage(root).free
    except Exception:
        free = 0
    aggressive = free < 200 * 1024 * 1024
    for name in os.listdir(root):
        leftover = name.startswith("tmp") or name.endswith(".pdf") or name.startswith("fitz")
        if not aggressive and not leftover:
            continue
        path = os.path.join(root, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except Exception:
            pass


def _public_error(exc: BaseException) -> str:
    msg = str(exc)
    low = msg.lower()
    if "no space left" in low or "cannot fwrite" in low or "temporary disk" in low:
        return (
            "This file is too large for the server's temporary disk. "
            "Export a smaller PDF (or a lower-resolution image) and try again."
        )
    return msg


def _delete_blob(url: str) -> None:
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
    if not token or not _is_blob_url(url):
        return
    try:
        requests.post(
            "https://vercel.com/api/blob/delete",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"urls": [url]},
            timeout=30,
        )
    except Exception:
        pass


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        start = time.time()
        try:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._json(400, {"error": "multipart/form-data required"})
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )

            project_name = form.getfirst("project_name", "")
            branding_name = form.getfirst("branding_name", "")
            file_type = form.getfirst("file_type", "")
            include_footer = form.getfirst("include_footer", "1") == "1"
            replace_existing = form.getfirst("replace_existing", "0") == "1"
            whiteouts_raw = form.getfirst("whiteouts", "") or ""
            blob_url = (form.getfirst("blob_url", "") or "").strip()
            delete_blob = form.getfirst("delete_blob", "0") == "1"

            if blob_url:
                raw_bytes = _download_blob(blob_url)
                raw_name = form.getfirst("file_name", "") or "upload"
            elif "file" in form:
                pdf_field = form["file"]
                raw_bytes = pdf_field.file.read()
                raw_name = pdf_field.filename or "upload"
            else:
                self._json(400, {"error": "A PDF or image is required"})
                return

            if not raw_bytes:
                raise ValueError("Uploaded file was empty")
            pdf_bytes = _as_pdf(raw_bytes, raw_name)
            file_name = _output_pdf_name(raw_name)
            del raw_bytes

            if not all([project_name, branding_name, file_type]):
                self._json(400, {"error": "project_name, branding_name, file_type required"})
                return

            _reclaim_tmp()
            with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
                in_path = os.path.join(tmp, "input.pdf")
                out_path = os.path.join(tmp, "output.pdf")

                with open(in_path, "wb") as f:
                    f.write(pdf_bytes)
                del pdf_bytes
                gc.collect()

                whiteout_count = _process_pdf(
                    in_path,
                    out_path,
                    f"{branding_name}/{project_name}",
                    include_footer,
                    whiteouts_raw,
                )

                if not os.path.exists(out_path):
                    self._json(500, {"error": "Processed file was not created"})
                    return

                file_size = os.path.getsize(out_path)

                dest = resolve_destination(project_name, branding_name, file_type)
                svc = dest["svc"]

                if replace_existing:
                    trash_files(svc, dest["file_type_folder_id"])

                uploaded = upload_file(svc, out_path, file_name, dest["file_type_folder_id"])

            processing_time = time.time() - start
            gc.collect()

            self._json(200, {
                "success": True,
                "branding_name": branding_name,
                "file_name": uploaded["name"],
                "file_id": uploaded["id"],
                "drive_url": uploaded["webViewLink"],
                "output_path": f"{dest['output_path']}/{file_name}",
                "processing_time": round(processing_time, 2),
                "file_size": file_size,
                "whiteouts": whiteout_count,
            })

        except ValueError as e:
            gc.collect()
            self._json(400, {"error": _public_error(e), "processing_time": round(time.time() - start, 2)})
        except Exception as e:
            gc.collect()
            self._json(500, {"error": _public_error(e), "processing_time": round(time.time() - start, 2)})
        finally:
            if "delete_blob" in locals() and delete_blob and blob_url:
                _delete_blob(blob_url)

    def _json(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
