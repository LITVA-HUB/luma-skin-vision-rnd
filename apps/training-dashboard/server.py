"""Loopback-only static dashboard and read-only JSON telemetry API."""

import argparse
import csv
import io
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from face_transfer_monitor import snapshot as face_transfer_snapshot
from facial_monitor import snapshot as facial_snapshot
from head_range_monitor import snapshot as head_range_snapshot
from monitor import Monitor
from palette_transfer_monitor import snapshot as palette_transfer_snapshot


def export_csv(packages, filters):
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=";")
    writer.writerow(
        [
            "Архитектура",
            "Выборка",
            "Этап",
            "Шаги",
            "Всего шагов",
            "Время, сек",
            "Статус",
            "Прогресс оценочный",
        ]
    )
    labels = {"mixed": "Смешанная", "slr_to_ipod": "SLR → iPod", "ipod_to_slr": "iPod → SLR"}
    statuses = {"completed": "Завершён", "running": "Обучается", "queued": "В очереди"}
    for p in packages:
        if filters.get("status", "all") not in ("all", p["status"]):
            continue
        if filters.get("role", "all") not in ("all", p["role"]):
            continue
        if filters.get("query", "").casefold() not in f"{p['variant']} {p['label']}".casefold():
            continue
        writer.writerow(
            [
                p["variant"],
                labels[p["role"]],
                f"Внутренний · {p['fold'] + 1}/3" if p["stage"] == "inner" else "Финальный",
                p["step"] if p["status"] != "queued" else "",
                p["target_steps"],
                p["seconds"],
                statuses[p["status"]],
                "Да" if p["estimated"] else "Нет",
            ]
        )
    return output.getvalue().encode("utf-8-sig")


def make_handler(monitor, ui):
    mutex = threading.Lock()
    ui = Path(ui).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hostname = self.headers.get("Host", "").split(":")[0].lower()
            if hostname not in ("localhost", "127.0.0.1"):
                self.send_error(403)
                return
            path = unquote(urlsplit(self.path).path)
            if path == "/api/export.csv":
                filters = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                with mutex:
                    content = export_csv(monitor.snapshot()["packages"], filters)
                self.respond(
                    content,
                    "text/csv; charset=utf-8",
                    {"Content-Disposition": 'attachment; filename="luma-training-packages.csv"'},
                )
                return
            if path in ("/api/status", "/api/health"):
                with mutex:
                    value = (
                        monitor.snapshot()
                        if path.endswith("status")
                        else {"app": "luma-training-monitor", "ok": True}
                    )
                    if path.endswith('status'):
                        value['facial_skin'] = facial_snapshot()
                        value['head_range'] = head_range_snapshot()
                        value['palette_transfer'] = palette_transfer_snapshot()
                        value['face_transfer'] = face_transfer_snapshot()
                self.respond(
                    json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return
            if path.startswith("/api/"):
                self.send_error(404)
                return
            target = (ui / path.lstrip("/")).resolve()
            if not target.is_relative_to(ui):
                self.send_error(403)
                return
            if path == "/":
                target = ui / "index.html"
            if not target.is_file():
                self.send_error(404)
                return
            self.respond(
                target.read_bytes(), mimetypes.guess_type(target)[0] or "application/octet-stream"
            )

        def respond(self, content, content_type, extra_headers=None):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, format, *args):
            if args and str(args[1]) not in ("200", "304"):
                super().log_message(format, *args)

    return Handler


def main():
    app = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--root", type=Path, default=app.parents[1])
    parser.add_argument("--ui", type=Path, default=app / "dist")
    args = parser.parse_args()
    if not (args.ui / "index.html").exists():
        raise SystemExit(
            "Dashboard build is missing. Run npm install and npm run build in the dashboard folder."
        )
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(Monitor(args.root), args.ui)
    )
    server.daemon_threads = True
    print(f"Luma training monitor: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
