# app/streamlit_app.py
import streamlit as st
from dotenv import load_dotenv

from rag.ingest import ingest_pdf_bytes
from rag.generator import answer_question
from rag.store import collection_count

load_dotenv()

st.set_page_config(page_title="RAG Evaluation Demo", page_icon="📚")

st.title("📚 RAG Demo (GDPR) — Retrieval + Citations")
st.caption("UI-only app: all RAG logic lives under the `rag/` package.")

# -----------------------------
# Upload & Ingest
# -----------------------------
st.subheader("1) Upload a PDF to index")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    reset_index = st.checkbox("Reset index before ingest", value=False)
with col2:
    st.write(f"Indexed chunks: **{collection_count()}**")

if uploaded_file is not None:
    with st.spinner("Ingesting PDF into Chroma..."):
        n_chunks = ingest_pdf_bytes(
            uploaded_file.getvalue(),
            filename=uploaded_file.name,
            reset=reset_index,
        )
    if n_chunks == 0:
        st.error("No text found in the PDF.")
    else:
        st.success(f"Indexed **{n_chunks}** chunks from `{uploaded_file.name}`.")
        st.write(f"New total chunks: **{collection_count()}**")

st.divider()

# -----------------------------
# Chat
# -----------------------------
st.subheader("2) Ask questions over the indexed documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if collection_count() == 0:
        msg = "No documents indexed yet. Upload a PDF first."
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                result = answer_question(user_input, top_k=4)

            st.markdown(result.answer)

            with st.expander("Retrieved context (debug)"):
                for i, ch in enumerate(result.chunks, start=1):
                    st.markdown(
                        f"**[{i}]** source=`{ch.source}` chunk_index=`{ch.chunk_index}` "
                        f"distance=`{ch.distance:.4f}` id=`{ch.id}`"
                    )
                    st.write(ch.text)

        st.session_state.messages.append({"role": "assistant", "content": result.answer})
