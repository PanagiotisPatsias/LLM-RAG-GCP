# rag/generator.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE
from rag.retriever import Chunk, retrieve, format_context


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    chunks: List[Chunk]  # retrieved chunks used


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    return OpenAI(api_key=api_key)


def answer_question(
    question: str,
    *,
    top_k: int = 4,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
) -> RAGAnswer:
    """
    End-to-end: retrieve context -> ask LLM -> return answer with citations.
    """
    chunks = retrieve(question, top_k=top_k)
    context = format_context(chunks)

    user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)

    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    text = resp.choices[0].message.content or ""
    return RAGAnswer(question=question, answer=text, chunks=chunks)
