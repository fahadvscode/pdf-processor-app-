"""
POST /api/process — process one PDF for one branding.

Accepts multipart form with:
  file: the PDF binary
  project_name, branding_name, file_type, include_footer (0|1), replace_existing (0|1)

Returns JSON with drive_url, file_name, processing_time, file_size, output_path.
"""
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
import cgi
import json
import os
import sys
import tempfile
import time
import gc
import requests

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


def _safe_name(name: str) -> str:
    name = os.path.basename((name or "upload.pdf").replace("\x00", "")).strip() or "upload.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def _is_blob_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return url.startswith("https://") and host.endswith(".blob.vercel-storage.com")


def _download_blob(url: str) -> bytes:
    if not _is_blob_url(url):
        raise RuntimeError("Invalid upload URL")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    if not resp.content:
        raise RuntimeError("Uploaded PDF was empty")
    return resp.content


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
                pdf_bytes = _download_blob(blob_url)
                file_name = _safe_name(form.getfirst("file_name", "") or "upload.pdf")
            elif "file" in form:
                pdf_field = form["file"]
                pdf_bytes = pdf_field.file.read()
                file_name = _safe_name(pdf_field.filename or "upload.pdf")
            else:
                self._json(400, {"error": "PDF file is required"})
                return

            if not all([project_name, branding_name, file_type]):
                self._json(400, {"error": "project_name, branding_name, file_type required"})
                return

            with tempfile.TemporaryDirectory() as tmp:
                in_path = os.path.join(tmp, file_name)
                out_path = os.path.join(tmp, f"processed_{file_name}")

                with open(in_path, "wb") as f:
                    f.write(pdf_bytes)

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

        except Exception as e:
            gc.collect()
            self._json(500, {"error": str(e), "processing_time": round(time.time() - start, 2)})
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
