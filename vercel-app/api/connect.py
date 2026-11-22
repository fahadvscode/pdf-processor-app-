"""GET /api/connect — load all projects from Drive."""
from http.server import BaseHTTPRequestHandler
import json
from .lib.drive import get_all_projects


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            projects = get_all_projects()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"projects": projects}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
