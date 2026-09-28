# ---------------------------------------------------------------------------
# RAG CHATBOT - STREAMLIT FRONT-END
# ---------------------------------------------------------------------------
# Week 9 deliverable: Working RAG chatbot with custom knowledge base.
#
# Start it from this folder with:
#     python -m streamlit run app.py
#
# ("python -m" makes Streamlit run with the same Python that has the packages
# from requirements.txt installed; a bare "streamlit" command can belong to a
# different Python install on the PATH.)
#
# Features
#   - chat interface with conversation history (st.chat_message / session_state)
#   - sidebar controls for retrieval tuning: search type, top_k, fetch_k,
#     reranker on/off, relevance threshold, LLM model and temperature
#   - upload your own .md / .txt / .pdf files to extend the knowledge base
#   - every answer shows its sources, similarity / rerank scores and a
#     groundedness score that flags possibly hallucinated sentences
# ---------------------------------------------------------------------------

import os
from dataclasses import replace
from pathlib import Path

import streamlit as st

from rag_engine import (GROQ_MODELS, KB_DIR, SUPPORTED_EXTENSIONS, RAGChatbot,
                        RAGConfig, load_env_file)

st.set_page_config(page_title="Data Science RAG Chatbot", page_icon="💬", layout="wide")
load_env_file()


def kb_fingerprint():
    """Changes whenever a knowledge-base file is added, removed or edited."""
    files = sorted(p for p in KB_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return tuple((p.name, p.stat().st_mtime) for p in files)


@st.cache_resource(show_spinner="Building the vector index...")
def get_bot(fingerprint, chunk_size, chunk_overlap):
    # cached: the embedding model and FAISS index are only rebuilt when the
    # knowledge base or the chunking settings change
    return RAGChatbot(KB_DIR, RAGConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))


# ---------------------------------------------------------------------------
# SIDEBAR - settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input("Groq API key", type="password",
                            value=os.environ.get("GROQ_API_KEY", ""),
                            help="Free key at console.groq.com. Without it the bot "
                                 "answers with the most relevant sentences from the documents.")
    model = st.selectbox("LLM model", GROQ_MODELS)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05,
                            help="Low values keep answers close to the sources.")

    st.subheader("Retrieval tuning")
    search_type = st.radio("Search type", ["similarity", "mmr"], horizontal=True,
                           help="MMR picks relevant but diverse chunks.")
    top_k = st.slider("top_k (chunks sent to the LLM)", 1, 8, 3)
    use_reranker = st.toggle("Cross-encoder reranking", value=True)
    fetch_k = st.slider("fetch_k (candidates before reranking / MMR)", top_k, 25, max(10, top_k))
    min_relevance = st.slider("Relevance threshold (cosine)", 0.0, 0.8, 0.30, 0.05,
                              help="If no chunk scores above this, the bot refuses to answer.")

    st.subheader("Chunking")
    chunk_size = st.select_slider("Chunk size (characters)", [250, 500, 750, 1000], 500)
    chunk_overlap = st.select_slider("Chunk overlap", [0, 50, 75, 100, 150], 75)

    st.subheader("📄 Knowledge base")
    uploads = st.file_uploader("Add documents", type=[e.strip(".") for e in SUPPORTED_EXTENSIONS],
                               accept_multiple_files=True)
    if uploads and st.button("Add to knowledge base", use_container_width=True):
        upload_dir = KB_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)
        for f in uploads:
            (upload_dir / Path(f.name).name).write_bytes(f.getbuffer())
        st.success(f"Added {len(uploads)} file(s) - the index will rebuild.")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []

bot = get_bot(kb_fingerprint(), chunk_size, chunk_overlap)
bot.api_key = api_key or None
bot.config = replace(bot.config, search_type=search_type, top_k=top_k, fetch_k=fetch_k,
                     use_reranker=use_reranker, min_relevance=min_relevance,
                     model=model, temperature=temperature)

with st.sidebar:
    st.caption(f"{bot.n_files} files · {len(bot.chunks)} chunks indexed")
    for name in sorted({c.metadata['source'] for c in bot.chunks}):
        st.caption(f"• {name}")


# ---------------------------------------------------------------------------
# MAIN - chat
# ---------------------------------------------------------------------------
st.title("💬 Data Science RAG Chatbot")
st.caption(f"Answers are grounded in the internship knowledge base · mode: **{bot.mode}**")


def render_details(meta):
    """Groundedness badge + expandable sources under an assistant answer."""
    if meta["refused"]:
        st.info("No sufficiently relevant passage found, so the bot declined to answer "
                "instead of guessing.")
        return
    g = meta["groundedness"]
    if g is not None:
        label = f"Groundedness: {g:.0%} of answer sentences supported by the sources"
        (st.success if g >= 0.8 else st.warning if g >= 0.5 else st.error)(label)
        flagged = [s for s, _, ok in meta["support"] if not ok]
        for s in flagged:
            st.caption(f"⚠️ Possibly unsupported: _{s}_")
    with st.expander(f"Sources ({len(meta['sources'])})"):
        for i, c in enumerate(meta["sources"], 1):
            rr = f" · rerank {c.rerank_score:.2f}" if c.rerank_score is not None else ""
            st.markdown(f"**[{i}] {c.source} › {c.section}**  \ncosine {c.similarity:.2f}{rr}")
            st.caption(c.text)


if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    examples = ["What is the difference between precision and recall?",
                "How does reranking improve retrieval?",
                "Why is Adam a good default optimizer?",
                "How can I reduce hallucinations in a RAG chatbot?"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_details(msg["meta"])

question = st.chat_input("Ask about Python, ML, neural networks, LLMs or RAG...")
question = question or st.session_state.pop("pending", None)

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                r = bot.answer(question, history=history)
                meta = {"refused": r.refused, "groundedness": r.groundedness,
                        "support": r.sentence_support, "sources": r.sources}
                content = r.answer
            except Exception as e:  # e.g. invalid API key or network error
                content = f"Error while generating the answer: {e}"
                meta = {"refused": True, "groundedness": None, "support": [], "sources": []}
        st.markdown(content)
        render_details(meta)
    st.session_state.messages.append({"role": "assistant", "content": content, "meta": meta})
