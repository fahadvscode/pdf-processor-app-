"""POST /api/auth — verify password."""
from http.server import BaseHTTPRequestHandler
import json, os


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        expected = os.environ.get("APP_PASSWORD", "")
        ok = bool(expected and body.get("password") == expected)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": ok}).encode())
