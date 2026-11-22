"""GET /api/existing_files?project_name=X&branding_name=Y&file_type=Z"""
from http.server import BaseHTTPRequestHandler
import json
from urllib.parse import urlparse, parse_qs
from .lib.drive import inspect_output


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        project_name = (qs.get("project_name") or [None])[0]
        branding_name = (qs.get("branding_name") or [None])[0]
        file_type = (qs.get("file_type") or [None])[0]
        if not all([project_name, branding_name, file_type]):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "project_name, branding_name, file_type required"}).encode())
            return
        try:
            result = inspect_output(project_name, branding_name, file_type)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
