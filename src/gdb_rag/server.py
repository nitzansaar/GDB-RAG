from __future__ import annotations

import json
from pathlib import Path

import requests as http_requests
from flask import Flask, Response, request, send_from_directory, stream_with_context

from gdb_rag.config import DEFAULT_SETTINGS
from gdb_rag.index import query_index
from gdb_rag.llm import generate_answer_stream

_STATIC = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=None)


@app.route("/")
def index() -> Response:
    return send_from_directory(_STATIC, "index.html")


@app.route("/api/ask", methods=["POST"])
def ask() -> Response:
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()

    if not question:
        return Response(json.dumps({"error": "question is required"}), status=400, mimetype="application/json")

    model = body.get("model") or DEFAULT_SETTINGS.ollama_model
    top_k = int(body.get("top_k") or 5)
    history = body.get("history") or []

    def event_stream():
        try:
            results = query_index(question, settings=DEFAULT_SETTINGS, top_k=top_k)
        except Exception as exc:
            yield _sse("error", {"message": f"Index error: {exc}"})
            return

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            yield _sse("error", {"message": "No relevant sections found."})
            return

        sources = [
            {
                "page_title": m.get("page_title") or "GDB Manual",
                "heading_path": m.get("heading_path") or "",
                "url": m.get("anchor") or m.get("source_url") or "",
                "distance": round(float(d), 4),
            }
            for m, d in zip(metadatas, distances)
        ]
        yield _sse("sources", sources)

        try:
            full_answer = []
            for token in generate_answer_stream(question, documents, model=model, history=history):
                full_answer.append(token)
                yield _sse("token", {"text": token})
        except http_requests.ConnectionError:
            yield _sse("error", {"message": "Cannot reach Ollama at localhost:11434. Is it running?"})
            return
        except http_requests.HTTPError as exc:
            yield _sse("error", {"message": f"Ollama error: {exc.response.status_code}"})
            return
        except Exception as exc:
            yield _sse("error", {"message": f"LLM error: {exc}"})
            return

        yield _sse("done", {"answer": "".join(full_answer)})

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
