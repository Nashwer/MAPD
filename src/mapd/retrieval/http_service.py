from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from mapd.retrieval.sqlite_fts import SQLiteFTSRetriever


class RetrievalHTTPServer(ThreadingHTTPServer):
    retriever: SQLiteFTSRetriever
    default_top_k: int


class RetrievalHandler(BaseHTTPRequestHandler):
    server: RetrievalHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(
            200,
            {
                "status": "ok",
                "backend": "sqlite-fts5",
                "documents": self.server.retriever.document_count,
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != "/retrieve":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            queries = request["queries"]
            if not isinstance(queries, list) or not all(isinstance(item, str) for item in queries):
                raise ValueError("queries must be a list of strings")
            top_k = int(request.get("topk") or self.server.default_top_k)
            if not 1 <= top_k <= 100:
                raise ValueError("topk must be between 1 and 100")
            return_scores = bool(request.get("return_scores", False))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return

        result: list[list[Any]] = []
        for query in queries:
            passages = self.server.retriever.search(query, top_k)
            documents = []
            for passage in passages:
                document = {"id": passage.id, "contents": passage.contents}
                documents.append(
                    {"document": document, "score": passage.score}
                    if return_scores
                    else document
                )
            result.append(documents)
        self._send(200, {"result": result})

    def log_message(self, format: str, *args: object) -> None:
        print(f"retriever {self.client_address[0]} {format % args}", flush=True)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve_retriever(
    index_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    default_top_k: int = 3,
) -> None:
    retriever = SQLiteFTSRetriever(index_path)
    server = RetrievalHTTPServer((host, port), RetrievalHandler)
    server.retriever = retriever
    server.default_top_k = default_top_k
    print(
        f"retriever ready: http://{host}:{port}/retrieve "
        f"({server.retriever.document_count:,} documents)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
