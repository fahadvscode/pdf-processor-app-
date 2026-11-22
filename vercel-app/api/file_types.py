"""GET /api/file_types?branding_id=X — list Raw/ subfolders."""
from http.server import BaseHTTPRequestHandler
import json
from urllib.parse import urlparse, parse_qs
from .lib.drive import get_file_types


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        branding_id = (qs.get("branding_id") or [None])[0]
        if not branding_id:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "branding_id required"}).encode())
            return
        try:
            types = get_file_types(branding_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"file_types": types}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
