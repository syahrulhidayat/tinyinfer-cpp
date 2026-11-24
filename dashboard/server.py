#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = Path(__file__).with_name("index.html")
DEFAULT_RESULTS = REPO_ROOT / "benchmarks" / "results" / "latest.json"
BENCH_SCRIPT = REPO_ROOT / "benchmarks" / "bench_e2e_compare.py"


def _make_handler(results_path: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", body)

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                html = HTML_PATH.read_bytes()
                self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", html)
                return

            if self.path == "/api/results":
                if not results_path.exists():
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "No benchmark results found. Run 'make bench-e2e' first."},
                    )
                    return
                body = results_path.read_bytes()
                self._send_bytes(HTTPStatus.OK, "application/json; charset=utf-8", body)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/api/rerun":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(BENCH_SCRIPT),
                        "--out-json",
                        str(results_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Benchmark execution failed: {exc}"},
                )
                return

            body = results_path.read_bytes()
            self._send_bytes(HTTPStatus.OK, "application/json; charset=utf-8", body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    return DashboardHandler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    results_path: Path = DEFAULT_RESULTS,
) -> ThreadingHTTPServer:
    handler_cls = _make_handler(results_path)
    return ThreadingHTTPServer((host, port), handler_cls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve tinyinfer-cpp benchmark dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    server = create_server(args.host, args.port, args.results)
    print(f"tinyinfer-cpp benchmark dashboard listening at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
