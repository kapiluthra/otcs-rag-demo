"""
app.py — Streamlit chatbot UI for the OTCS RAG pipeline

Run: streamlit run src/app.py
"""

import os
import requests
import streamlit as st
from retriever import HybridRetriever

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3")

st.set_page_config(
    page_title="OTCS Knowledge Base",
    page_icon="📄",
    layout="wide",
)

st.title("OpenText Content Server — Knowledge Base")
st.caption("Ask questions about your enterprise documents. Answers are grounded in retrieved content.")

# ── Initialise retriever (cached across reruns) ───────────────────────────────
@st.cache_resource
def get_retriever():
    return HybridRetriever()

retriever = get_retriever()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            results = retriever.search(query, top_k=4)

        if not results:
            answer = "No relevant documents found for your query."
            st.markdown(answer)
        else:
            # Build context from parent chunks (more context than child chunks)
            context_parts = []
            for i, r in enumerate(results, 1):
                context_parts.append(
                    f"[{i}] {r['doc_name']}\n{r['parent_text']}"
                )
            context = "\n\n---\n\n".join(context_parts)

            # Build the prompt
            system_prompt = (
                "You are a helpful assistant answering questions about enterprise documents. "
                "Use ONLY the provided context to answer. If the answer is not in the context, "
                "say so clearly. Cite the document name when referencing specific information."
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

            # Stream the response from Ollama
            response_text = ""
            response_placeholder = st.empty()

            try:
                resp = requests.post(
                    f"{OLLAMA_BASE}/api/chat",
                    json={
                        "model": CHAT_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": True,
                    },
                    stream=True,
                    timeout=120,
                )
                resp.raise_for_status()

                import json
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if token := chunk.get("message", {}).get("content", ""):
                            response_text += token
                            response_placeholder.markdown(response_text + "▌")

                response_placeholder.markdown(response_text)

            except Exception as e:
                response_text = f"Error calling language model: {e}"
                response_placeholder.error(response_text)

            # Show source documents
            with st.expander(f"Sources ({len(results)} documents)"):
                for i, r in enumerate(results, 1):
                    st.markdown(f"**[{i}] {r['doc_name']}**")
                    st.caption(f"RRF score: {r['rrf_score']:.4f} · chunk: `{r['chunk_id']}`")
                    st.text(r["child_text"][:300] + "...")
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response_text})
