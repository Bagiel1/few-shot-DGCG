"""Servidor local do simulador. Serve apenas a interface e a API de simulação."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch

from .engine import SimulationConfig, load_features, parse_config, simulate
from .source_code import code_catalog


STATIC = Path(__file__).parent / "static"
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def make_handler(cached_features=None):
    simulation_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Evita imprimir uma linha por recurso estático.
            if args and str(args[1]) not in ("200", "304"):
                super().log_message(format, *args)

        def _local_request(self):
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in expected:
                self._json({"error": "Use o endereço local mostrado no terminal."}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in expected}:
                self._json({"error": "Origem não autorizada."}, 403)
                return False
            return True

        def _send(self, data: bytes, content_type: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, value, status=200):
            self._send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            if not self._local_request():
                return
            path = urlsplit(self.path).path
            if path == "/api/config":
                self._json({"defaults": asdict(SimulationConfig()), "source": "cache" if cached_features is not None else "synthetic"})
            elif path == "/api/code":
                self._json(code_catalog())
            elif path in STATIC_ROUTES:
                filename, content_type = STATIC_ROUTES[path]
                self._send((STATIC / filename).read_bytes(), content_type)
            else:
                self._json({"error": "Página não encontrada."}, 404)

        def do_POST(self):
            if not self._local_request():
                return
            if self.path != "/api/simulate":
                self._json({"error": "Rota não encontrada."}, 404)
                return
            if not simulation_lock.acquire(blocking=False):
                self._json({"error": "Há uma simulação em andamento. Aguarde e tente novamente."}, 409)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Tamanho de configuração inválido.")
                cfg = parse_config(json.loads(self.rfile.read(length)))
                self._json(simulate(cfg, cached_features))
            except (ValueError, TypeError, KeyError) as error:
                self._json({"error": str(error)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A aba foi fechada enquanto o modelo calculava.
            except Exception:
                import traceback
                traceback.print_exc()
                self._json({"error": "Não foi possível executar o modelo. Consulte o terminal."}, 500)
            finally:
                simulation_lock.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Simulador local e interativo do ProtoSGC.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--features-cache", type=Path, help="Cache features_resnet18_*.pt do experimento, opcional.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port deve estar entre 1 e 65535.")
    torch.set_num_threads(1)
    cached = load_features(args.features_cache) if args.features_cache else None
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(cached))
    except OSError as error:
        parser.error(f"Não foi possível abrir a porta {args.port}: {error}. Tente --port 8766.")
    print(f"Simulador: http://127.0.0.1:{args.port}", flush=True)
    print("Execução local em CPU. Ctrl+C encerra o servidor.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
