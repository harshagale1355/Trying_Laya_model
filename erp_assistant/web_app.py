#!/usr/bin/env python3
"""
Web frontend for the hybrid ERP chatbot (stdlib only). Serves web/index.html and a JSON API
that wraps chatbot.ask().

  python -m erp_assistant.web_app      # then open http://127.0.0.1:8000

POST /api/ask {"question": "..."} -> {"route", "confidence", "answer", "sql"?, "columns"?, "rows"?, "sources"?}
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import chatbot

INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"
MAX_BODY = 4096


def make_handler(ask):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                return self.send_json(404, {"error": "not found"})
            body = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/ask":
                return self.send_json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self.send_json(400, {"error": "request too large"})
            try:
                question = str(json.loads(self.rfile.read(length) or b"{}").get("question", "")).strip()
            except (ValueError, AttributeError):
                return self.send_json(400, {"error": "body must be JSON like {\"question\": \"...\"}"})
            if not question:
                return self.send_json(400, {"error": "question is empty"})
            try:
                self.send_json(200, ask(question))
            except Exception as err:
                self.send_json(500, {"error": str(err)})

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    client = chatbot.make_client()
    route = chatbot.load_router()
    schema = chatbot.SCHEMA_PATH.read_text(encoding="utf-8")
    index = chatbot.load_index(client)

    lock = threading.Lock()  # Laya agent and Gemini client are shared; answer one question at a time

    def ask(question):
        with lock:
            return chatbot.ask(client, route, schema, index, question)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(ask))
    print(f"ERP assistant running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
