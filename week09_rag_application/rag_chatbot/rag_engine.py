# ---------------------------------------------------------------------------
# RAG CHATBOT ENGINE - LANGCHAIN + FAISS + CROSS-ENCODER RERANKING + GROQ
# ---------------------------------------------------------------------------
# Week 9 deliverable: End-to-End RAG Application.
#
# This module is the "back end" used by both the Streamlit app (app.py) and
# the evaluation script (evaluate_rag.py). Week 9 topics covered here:
#
#   1. LangChain              -> loaders, text splitters, HuggingFace embeddings,
#                                FAISS vector store, prompt | llm | parser chain
#   2. Retrieval tuning       -> top_k, fetch_k, similarity vs MMR search,
#                                chunk size / overlap, relevance threshold
#      & reranking            -> cross-encoder reranks the fetched candidates
#   3. Hallucination &        -> grounded system prompt with citations,
#      grounding                 refusal when nothing relevant is retrieved,
#                                sentence-level groundedness check of the answer
#
# Quick test from a terminal:
#     python rag_engine.py "What is the bias-variance tradeoff?"
# ---------------------------------------------------------------------------

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

# the FAISS wrapper still lives in langchain-community, which warns that it is
# being sunset; the wrapper itself works fine, so that one warning is silenced
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
REFUSAL = "I don't know based on the knowledge base."
SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf")


@dataclass
class RAGConfig:
    # chunking
    chunk_size: int = 500
    chunk_overlap: int = 75
    # retrieval tuning
    search_type: str = "similarity"      # "similarity" or "mmr"
    top_k: int = 3                       # chunks passed to the LLM
    fetch_k: int = 10                    # candidates fetched before MMR / reranking
    mmr_lambda: float = 0.6              # 1 = pure relevance, 0 = pure diversity
    use_reranker: bool = True
    min_rerank_score: float = -5.0       # cross-encoder logit; off-topic chunks score ~ -11
    min_relevance: float = 0.30          # cosine threshold - below it we refuse
    # generation
    model: str = GROQ_MODELS[0]
    temperature: float = 0.1
    # grounding check
    support_threshold: float = 0.55      # sentence counts as supported above this


@dataclass
class RetrievedChunk:
    text: str
    source: str
    section: str
    similarity: float                    # cosine similarity to the question
    rerank_score: float | None = None    # cross-encoder score (higher = better)


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list
    refused: bool
    groundedness: float | None           # share of answer sentences supported by context
    sentence_support: list = field(default_factory=list)   # (sentence, best score, supported)
    mode: str = ""


def load_env_file(path=BASE_DIR / ".env"):
    """Minimal .env reader so GROQ_API_KEY can live in a file instead of the shell."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 3]


# ---------------------------------------------------------------------------
# STEP 1 : LOAD + CHUNK DOCUMENTS (LangChain loaders & splitters)
# ---------------------------------------------------------------------------
def load_and_split(kb_dir, chunk_size, chunk_overlap):
    """Markdown is first split on headings (so each chunk remembers its section),
    then every piece is split recursively to the target chunk size."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")])
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""])

    pieces = []
    for path in sorted(Path(kb_dir).rglob("*")):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        if suffix == ".pdf":
            for n, page in enumerate(PdfReader(path).pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pieces.append(Document(text, metadata={"source": path.name,
                                                           "section": f"page {n}"}))
            continue
        text = path.read_text(encoding="utf-8")
        if suffix == ".md":
            for doc in header_splitter.split_text(text):
                section = doc.metadata.get("section") or doc.metadata.get("title") or path.stem
                pieces.append(Document(doc.page_content, metadata={"source": path.name,
                                                                   "section": section}))
        else:
            pieces.append(Document(text, metadata={"source": path.name, "section": path.stem}))

    chunks = recursive.split_documents(pieces)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


# ---------------------------------------------------------------------------
# THE CHATBOT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful teaching assistant for a data science internship.
Answer the user's question using ONLY the numbered context passages below.

Rules:
- Every factual sentence must be supported by the context and end with a citation like [1] or [2].
- If the context does not contain the answer, reply exactly: "{refusal}"
- Do not use outside knowledge, do not guess, and keep the answer concise (2-5 sentences).

Context:
{context}"""

USER_PROMPT = """Recent conversation (for resolving follow-up questions only):
{history}

Question: {question}"""


class RAGChatbot:
    def __init__(self, kb_dir=KB_DIR, config=None, api_key=None):
        self.kb_dir = Path(kb_dir)
        self.config = config or RAGConfig()
        load_env_file()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
        self._reranker = None
        self.build_index()

    # ----- indexing -----
    def build_index(self):
        self.chunks = load_and_split(self.kb_dir, self.config.chunk_size, self.config.chunk_overlap)
        if not self.chunks:
            raise ValueError(f"No {SUPPORTED_EXTENSIONS} files found in {self.kb_dir}")
        # normalised vectors + inner product = cosine similarity
        self.vectorstore = FAISS.from_documents(
            self.chunks, self.embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        return self

    @property
    def n_files(self):
        return len({c.metadata["source"] for c in self.chunks})

    @property
    def reranker(self):
        if self._reranker is None:  # loaded lazily - only needed when reranking is on
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    # ----- retrieval: fetch candidates -> (MMR) -> (rerank) -> top_k -----
    def retrieve(self, question, config=None):
        cfg = config or self.config
        n_candidates = cfg.fetch_k if (cfg.use_reranker or cfg.search_type == "mmr") else cfg.top_k

        if cfg.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                question, k=cfg.fetch_k if cfg.use_reranker else cfg.top_k,
                fetch_k=max(cfg.fetch_k * 2, 20), lambda_mult=cfg.mmr_lambda)
            q = np.array(self.embeddings.embed_query(question))
            vecs = np.array(self.embeddings.embed_documents([d.page_content for d in docs]))
            scored = list(zip(docs, (vecs @ q).tolist()))
        else:
            scored = self.vectorstore.similarity_search_with_score(question, k=n_candidates)

        candidates = [RetrievedChunk(d.page_content, d.metadata["source"], d.metadata["section"],
                                     float(s)) for d, s in scored]
        # drop chunks that are not similar enough to the question at all
        candidates = [c for c in candidates if c.similarity >= cfg.min_relevance]
        if not candidates:
            return []

        if cfg.use_reranker:
            scores = self.reranker.predict([(question, c.text) for c in candidates])
            for c, s in zip(candidates, scores):
                c.rerank_score = float(s)
            candidates.sort(key=lambda c: c.rerank_score, reverse=True)
            # the reranker is the stricter judge: drop what it scores as irrelevant
            candidates = [c for c in candidates if c.rerank_score >= cfg.min_rerank_score]
        else:
            candidates.sort(key=lambda c: c.similarity, reverse=True)
        return candidates[:cfg.top_k]

    # ----- generation -----
    @property
    def mode(self):
        return f"LLM: {self.config.model} (Groq)" if self.api_key else "Retrieval-only (no Groq API key)"

    def _llm_chain(self):
        from langchain_groq import ChatGroq
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
        llm = ChatGroq(model=self.config.model, temperature=self.config.temperature,
                       api_key=self.api_key)
        return prompt | llm | StrOutputParser()   # LCEL chain

    @staticmethod
    def _format_context(sources):
        return "\n\n".join(f"[{i}] ({c.source} > {c.section})\n{c.text}"
                           for i, c in enumerate(sources, 1))

    @staticmethod
    def _format_history(history, max_turns=3):
        if not history:
            return "(none)"
        recent = history[-max_turns * 2:]
        return "\n".join(f"{m['role']}: {m['content'][:300]}" for m in recent)

    def _extractive_answer(self, question, sources, n_sentences=3):
        """Fallback without an LLM: the context sentences closest to the question."""
        candidates, seen = [], set()
        for i, c in enumerate(sources, 1):
            for s in split_sentences(c.text):
                if len(s) > 20 and s not in seen:
                    seen.add(s)
                    candidates.append((s, i))
        q = np.array(self.embeddings.embed_query(question))
        vecs = np.array(self.embeddings.embed_documents([s for s, _ in candidates]))
        best = sorted(np.argsort(-(vecs @ q))[:n_sentences])
        return " ".join(f"{candidates[j][0]} [{candidates[j][1]}]" for j in best)

    def answer(self, question, history=None, config=None):
        sources = self.retrieve(question, config)
        if not sources:  # grounding rule 1: nothing relevant retrieved -> refuse, don't call the LLM
            return RAGResponse(question, REFUSAL, [], True, None, mode=self.mode)

        if self.api_key:
            text = self._llm_chain().invoke({
                "context": self._format_context(sources), "refusal": REFUSAL,
                "history": self._format_history(history), "question": question}).strip()
        else:
            text = self._extractive_answer(question, sources)

        refused = REFUSAL.lower().rstrip(".") in text.lower()
        support, score = ([], None) if refused else self.check_grounding(text, sources)
        return RAGResponse(question, text, sources, refused, score, support, self.mode)

    # ----- hallucination check -----
    def check_grounding(self, answer, sources):
        """Grounding rule 2: every answer sentence must be close (cosine) to at
        least one sentence of the retrieved context, otherwise it is flagged as a
        possible hallucination. Returns per-sentence support and the overall share."""
        answer_sents = [re.sub(r"\s*\[\d+\]", "", s).strip() for s in split_sentences(answer)]
        answer_sents = [s for s in answer_sents if len(s) > 15]
        context_sents = [s for c in sources for s in split_sentences(c.text)]
        if not answer_sents or not context_sents:
            return [], None
        a = np.array(self.embeddings.embed_documents(answer_sents))
        c = np.array(self.embeddings.embed_documents(context_sents))
        best = (a @ c.T).max(axis=1)
        support = [(s, float(b), bool(b >= self.config.support_threshold))
                   for s, b in zip(answer_sents, best)]
        return support, float(np.mean([ok for _, _, ok in support]))


def print_response(r):
    print(f"\nQ: {r.question}\nA: {r.answer}")
    if r.groundedness is not None:
        print(f"Groundedness: {r.groundedness:.0%}")
        for sent, score, ok in r.sentence_support:
            if not ok:
                print(f"  [possible hallucination] ({score:.2f}) {sent}")
    for i, c in enumerate(r.sources, 1):
        rr = f"  rerank {c.rerank_score:.2f}" if c.rerank_score is not None else ""
        print(f"  [{i}] {c.source} > {c.section}  cosine {c.similarity:.2f}{rr}")
    print(f"({r.mode})")


if __name__ == "__main__":
    bot = RAGChatbot()
    print(f"Indexed {len(bot.chunks)} chunks from {bot.n_files} files.")
    questions = sys.argv[1:] or ["What does a cross-encoder reranker do?"]
    for q in questions:
        print_response(bot.answer(q))
