from __future__ import annotations

import json
from typing import Iterator

import requests

_SYSTEM = (
    "You are a GDB debugger expert. Answer the user's question using only the "
    "provided GDB manual excerpts. Be concise and precise. "
    "If the answer is not covered by the excerpts, say so explicitly."
)


def generate_answer(question: str, chunks: list[str], model: str) -> str:
    context = "\n\n---\n\n".join(chunks)
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"GDB Manual excerpts:\n\n{context}\n\nQuestion: {question}"},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def generate_answer_stream(question: str, chunks: list[str], model: str) -> Iterator[str]:
    context = "\n\n---\n\n".join(chunks)
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "stream": True,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"GDB Manual excerpts:\n\n{context}\n\nQuestion: {question}"},
            ],
        },
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        token = data.get("message", {}).get("content", "")
        if token:
            yield token
        if data.get("done", False):
            break
