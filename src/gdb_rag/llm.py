from __future__ import annotations

from typing import Iterator

from groq import Groq

_SYSTEM = (
    "You are a GDB manual assistant. Answer questions using the excerpts provided as your source.\n\n"
    "- Lead with a clear, direct explanation. Do not quote raw fragments or echo the question back.\n"
    "- Synthesize the excerpts into a coherent answer in your own words.\n"
    "- If the excerpts don't contain enough information to answer, say so in one sentence.\n"
    "- If the question is not about GDB, decline in one sentence."
)


def _build_messages(question: str, chunks: list[str], history: list[dict] | None) -> list[dict]:
    context = "\n\n---\n\n".join(chunks)
    messages = [{"role": "system", "content": _SYSTEM}]
    if history:
        messages.extend(history[-10:])
    messages.append({
        "role": "user",
        "content": f"GDB Manual excerpts:\n\n{context}\n\nQuestion: {question}",
    })
    return messages


def generate_answer(question: str, chunks: list[str], model: str) -> str:
    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=_build_messages(question, chunks, None),
        temperature=0.2,
    )
    return response.choices[0].message.content


def generate_answer_stream(
    question: str,
    chunks: list[str],
    model: str,
    history: list[dict] | None = None,
) -> Iterator[str]:
    client = Groq()
    stream = client.chat.completions.create(
        model=model,
        messages=_build_messages(question, chunks, history),
        stream=True,
        temperature=0.2,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token
