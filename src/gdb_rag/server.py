from __future__ import annotations

import json
from pathlib import Path

import groq as groq_lib
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

    model = body.get("model") or DEFAULT_SETTINGS.llm_model
    try:
        top_k = max(1, int(body.get("top_k") or DEFAULT_SETTINGS.top_k))
    except (ValueError, TypeError):
        return Response(json.dumps({"error": "top_k must be an integer"}), status=400, mimetype="application/json")
    history = _validate_history(body.get("history"))

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
        except groq_lib.AuthenticationError:
            yield _sse("error", {"message": "Invalid GROQ_API_KEY. Check your environment variable."})
            return
        except groq_lib.RateLimitError:
            yield _sse("error", {"message": "Groq rate limit reached. Try again in a moment."})
            return
        except groq_lib.APIConnectionError:
            yield _sse("error", {"message": "Could not connect to Groq API."})
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


def _validate_history(raw) -> list[dict]:
    if not isinstance(raw, list):
        return []
    result = []
    for entry in raw:
        if (
            isinstance(entry, dict)
            and entry.get("role") in ("user", "assistant")
            and isinstance(entry.get("content"), str)
            and len(entry["content"]) <= 4000
        ):
            result.append({"role": entry["role"], "content": entry["content"]})
    return result
