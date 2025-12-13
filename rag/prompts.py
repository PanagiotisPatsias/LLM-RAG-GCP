# rag/prompts.py

RAG_SYSTEM_PROMPT = """You are a helpful assistant.
You must answer ONLY using the provided context.
If the answer is not clearly supported by the context, say you do not have enough information.
Write in English.
"""

RAG_USER_PROMPT_TEMPLATE = """Context:
{context}

User question:
{question}

Instructions:
- Use ONLY the context above.
- If you use information from a specific excerpt, add its citation at the end of the sentence, e.g. [1].
- If the context is insufficient, explicitly say so.
- Keep the answer concise and factual.
- If the context is insufficient, reply with exactly:
  "The provided context does not contain enough information to answer this question."
- Only answer if you can directly support the answer with citations like [1]. If you cannot provide citations, reply with exactly:
  "The provided context does not contain enough information to answer this question."

"""
