# ---------------------------------------------------------------------------
# RAG PIPELINE ARCHITECTURE - EMBEDDINGS, VECTOR SEARCH & RETRIEVAL
# ---------------------------------------------------------------------------
# Week 8 deliverable: Embeddings, Vector Search & RAG Pipeline.
#
# The pipeline has two stages:
#
#   INDEXING (offline, once)
#     documents -> load -> chunk -> embed -> store vectors in FAISS / Chroma
#
#   QUERY (online, per question)
#     question -> embed -> similarity search (top-k) -> build grounded prompt
#              -> LLM answer with cited sources
#
# Every Week 8 topic has its own component:
#   1. Text embeddings & similarity -> Embedder (sentence-transformers, cosine)
#   2. Vector databases             -> FaissVectorStore / ChromaVectorStore
#   3. Semantic search              -> SemanticRetriever (+ keyword & hybrid)
#   4. Chunking strategies          -> fixed-size, sentence and recursive chunkers
#   5. Retrieval basics             -> top-k, score threshold, Hit Rate@k, MRR
#
# Usage:
#     python rag_pipeline.py                          # full demo + evaluation
#     python rag_pipeline.py --ask "What is Adam?"    # answer one question
#     python rag_pipeline.py --interactive            # ask questions in a loop
#     python rag_pipeline.py --store chroma           # use Chroma instead of FAISS
#
# Answers are generated with Groq when the GROQ_API_KEY environment variable
# (or a .env file) is set. Without a key the pipeline runs in retrieval-only
# mode and returns the most relevant sentences from the retrieved chunks.
# ---------------------------------------------------------------------------

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, do not open a window

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# paths are resolved from this file, so the script works from any directory
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
INDEX_DIR = BASE_DIR / "vector_index"
PLOTS_DIR = BASE_DIR / "plots"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim embeddings
GROQ_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 400        # characters
CHUNK_OVERLAP = 60      # characters (~15%)
TOP_K = 3
MIN_SCORE = 0.25        # below this cosine similarity a chunk is treated as irrelevant


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# STEP 1 : DOCUMENT LOADING
# ---------------------------------------------------------------------------
@dataclass
class Document:
    text: str
    source: str          # file name
    section: str         # markdown heading the text belongs to


def load_documents(folder=KB_DIR):
    """Load .md / .txt / .pdf files. Markdown is split by '## ' headings so every
    piece keeps its section title as metadata (structure-aware loading)."""
    docs = []
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        if suffix in (".md", ".txt"):
            raw = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            raw = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            continue

        section, buffer = path.stem, []
        for line in raw.splitlines():
            if line.startswith("## "):
                if "".join(buffer).strip():
                    docs.append(Document(" ".join(buffer).strip(), path.name, section))
                section, buffer = line[3:].strip(), []
            elif not line.startswith("# "):
                buffer.append(line.strip())
        if "".join(buffer).strip():
            docs.append(Document(" ".join(buffer).strip(), path.name, section))
    return docs


# ---------------------------------------------------------------------------
# STEP 2 : CHUNKING STRATEGIES
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    section: str
    metadata: dict = field(default_factory=dict)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def fixed_size_chunker(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cut every `size` characters; consecutive chunks share `overlap` characters
    so a sentence cut at a boundary still appears whole in one of them."""
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def sentence_chunker(text, size=CHUNK_SIZE, overlap_sentences=1):
    """Group whole sentences until the size limit; repeat the last sentence
    of a chunk at the start of the next one as overlap."""
    sentences = split_sentences(text)
    chunks, current = [], []
    for sent in sentences:
        if current and len(" ".join(current + [sent])) > size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def recursive_chunker(text, size=CHUNK_SIZE, separators=("\n\n", ". ", " ")):
    """Try the coarsest separator first (paragraphs), fall back to finer ones
    (sentences, then words) only for pieces that are still too long."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    sep, *rest = separators if separators else (" ",)
    parts = text.split(sep)
    chunks, current = [], ""
    for part in parts:
        piece = part + (sep if sep != " " else " ")
        if len(current) + len(piece) <= size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        if len(piece) > size and rest:
            chunks.extend(recursive_chunker(piece, size, tuple(rest)))
            current = ""
        else:
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


CHUNKERS = {"fixed": fixed_size_chunker,
            "sentence": sentence_chunker,
            "recursive": recursive_chunker}


def chunk_documents(docs, strategy="sentence"):
    chunker = CHUNKERS[strategy]
    chunks = []
    for doc in docs:
        for piece in chunker(doc.text):
            chunks.append(Chunk(len(chunks), piece, doc.source, doc.section,
                                {"strategy": strategy, "chars": len(piece)}))
    return chunks


# ---------------------------------------------------------------------------
# STEP 3 : TEXT EMBEDDINGS & SIMILARITY
# ---------------------------------------------------------------------------
class Embedder:
    """Wraps a sentence-transformers model. Vectors are L2-normalised, so the
    dot product of two vectors IS their cosine similarity."""

    def __init__(self, model_name=EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or             self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# STEP 4 : VECTOR DATABASES
# ---------------------------------------------------------------------------
class FaissVectorStore:
    """Exact nearest-neighbour search with FAISS IndexFlatIP (inner product on
    normalised vectors = cosine similarity). Chunks are kept alongside the index."""

    def __init__(self, dim):
        import faiss
        self.faiss = faiss
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []

    def add(self, vectors, chunks):
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector, k):
        scores, ids = self.index.search(query_vector.reshape(1, -1), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, folder=INDEX_DIR):
        folder.mkdir(exist_ok=True)
        self.faiss.write_index(self.index, str(folder / "faiss.index"))
        (folder / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder=INDEX_DIR):
        import faiss
        store = cls.__new__(cls)
        store.faiss = faiss
        store.index = faiss.read_index(str(folder / "faiss.index"))
        store.chunks = [Chunk(**c) for c in
                        json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
        return store

    def __len__(self):
        return self.index.ntotal


class ChromaVectorStore:
    """Chroma keeps documents, embeddings and metadata together and persists
    them to disk. Cosine distance is converted back to similarity (1 - d)."""

    def __init__(self, dim=None, folder=INDEX_DIR / "chroma", collection="knowledge_base"):
        import chromadb
        client = chromadb.PersistentClient(path=str(folder))
        try:
            client.delete_collection(collection)  # rebuild from scratch each run
        except Exception:
            pass
        self.collection = client.create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.chunks = {}

    def add(self, vectors, chunks):
        self.collection.add(
            ids=[str(c.chunk_id) for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "section": c.section} for c in chunks])
        self.chunks.update({str(c.chunk_id): c for c in chunks})

    def search(self, query_vector, k):
        res = self.collection.query(query_embeddings=[query_vector.tolist()], n_results=k)
        return [(self.chunks[i], 1.0 - float(d))
                for i, d in zip(res["ids"][0], res["distances"][0])]

    def save(self, folder=INDEX_DIR):
        pass  # PersistentClient already writes to disk

    def __len__(self):
        return self.collection.count()


VECTOR_STORES = {"faiss": FaissVectorStore, "chroma": ChromaVectorStore}


# ---------------------------------------------------------------------------
# STEP 5 : RETRIEVAL - SEMANTIC, KEYWORD AND HYBRID SEARCH
# ---------------------------------------------------------------------------
class SemanticRetriever:
    """Embed the query and return the top-k most similar chunks above a threshold."""

    def __init__(self, embedder, store, min_score=MIN_SCORE):
        self.embedder = embedder
        self.store = store
        self.min_score = min_score

    def retrieve(self, query, k=TOP_K):
        q = self.embedder.embed([query])[0]
        return [(c, s) for c, s in self.store.search(q, k) if s >= self.min_score]


class KeywordRetriever:
    """TF-IDF keyword search - the classic baseline semantic search improves on."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def scores(self, query):
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()

    def retrieve(self, query, k=TOP_K):
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in top]


class HybridRetriever:
    """Weighted mix of semantic and keyword scores (both min-max normalised)."""

    def __init__(self, embedder, chunks, chunk_vectors, alpha=0.7):
        self.embedder = embedder
        self.chunks = chunks
        self.vectors = chunk_vectors
        self.keyword = KeywordRetriever(chunks)
        self.alpha = alpha  # weight of the semantic score

    @staticmethod
    def _norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    def retrieve(self, query, k=TOP_K):
        semantic = self.vectors @ self.embedder.embed([query])[0]
        combined = self.alpha * self._norm(semantic) + (1 - self.alpha) * self._norm(
            self.keyword.scores(query))
        top = np.argsort(-combined)[:k]
        return [(self.chunks[i], float(combined[i])) for i in top]


# ---------------------------------------------------------------------------
# STEP 6 : PROMPT CONSTRUCTION + GENERATION
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful teaching assistant for a data science internship. "
    "Answer the question using ONLY the numbered context passages. "
    "Cite the passages you used like [1] or [2]. If the context does not contain "
    "the answer, reply exactly: \"I don't know based on the knowledge base.\"")


def build_prompt(question, retrieved):
    context = "\n\n".join(f"[{i}] (source: {c.source} > {c.section})\n{c.text}"
                          for i, (c, _) in enumerate(retrieved, 1))
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def load_env_file(path=BASE_DIR / ".env"):
    """Minimal .env reader so GROQ_API_KEY can live in a file instead of the shell."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Generator:
    """Groq LLM when an API key is available, otherwise an extractive fallback."""

    def __init__(self, embedder):
        load_env_file()
        self.embedder = embedder
        self.client = None
        if os.environ.get("GROQ_API_KEY"):
            try:
                from groq import Groq
                self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
            except ImportError:
                print("  [!] groq package not installed - using retrieval-only mode")

    @property
    def mode(self):
        return f"LLM ({GROQ_MODEL} via Groq)" if self.client else "retrieval-only (no GROQ_API_KEY)"

    def generate(self, question, retrieved):
        if not retrieved:
            return "I don't know based on the knowledge base."
        if self.client:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.1,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": build_prompt(question, retrieved)}])
            return response.choices[0].message.content.strip()
        return self._extractive_answer(question, retrieved)

    def _extractive_answer(self, question, retrieved, n_sentences=2):
        """Pick the sentences from the retrieved chunks closest to the question."""
        candidates, seen = [], set()
        for i, (c, _) in enumerate(retrieved, 1):
            for s in split_sentences(c.text):
                if len(s) > 20 and s not in seen:  # overlapping chunks repeat sentences
                    seen.add(s)
                    candidates.append((s, i))
        vectors = self.embedder.embed([s for s, _ in candidates])
        scores = vectors @ self.embedder.embed([question])[0]
        best = sorted(np.argsort(-scores)[:n_sentences])
        return " ".join(f"{candidates[j][0]} [{candidates[j][1]}]" for j in best)


# ---------------------------------------------------------------------------
# THE PIPELINE - ties all components together
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self, chunk_strategy="sentence", store="faiss", top_k=TOP_K, embedder=None):
        self.chunk_strategy = chunk_strategy
        self.store_name = store
        self.top_k = top_k
        self.embedder = embedder or Embedder()
        self.generator = Generator(self.embedder)

    def build_index(self, verbose=True):
        docs = load_documents()
        self.chunks = chunk_documents(docs, self.chunk_strategy)
        self.chunk_vectors = self.embedder.embed([c.text for c in self.chunks])
        self.store = VECTOR_STORES[self.store_name](self.embedder.dim)
        self.store.add(self.chunk_vectors, self.chunks)
        self.store.save()
        self.retriever = SemanticRetriever(self.embedder, self.store)
        if verbose:
            print(f"  loaded   {len(docs)} sections from {len({d.source for d in docs})} files")
            print(f"  chunked  {len(self.chunks)} chunks  (strategy: {self.chunk_strategy})")
            print(f"  embedded {self.chunk_vectors.shape[0]} x {self.chunk_vectors.shape[1]} "
                  f"vectors with {self.embedder.model_name}")
            print(f"  stored   {len(self.store)} vectors in {self.store_name.upper()}"
                  f" -> {INDEX_DIR.name}/")
        return self

    def query(self, question):
        retrieved = self.retriever.retrieve(question, self.top_k)
        answer = self.generator.generate(question, retrieved)
        return answer, retrieved


def print_answer(question, answer, retrieved):
    print(f"\n  Q: {question}")
    print(f"  A: {answer}")
    print("  Sources:")
    if not retrieved:
        print("    (no chunk passed the similarity threshold)")
    for i, (c, score) in enumerate(retrieved, 1):
        print(f"    [{i}] {c.source} > {c.section}  (cosine {score:.3f})")


# ---------------------------------------------------------------------------
# RETRIEVAL EVALUATION
# ---------------------------------------------------------------------------
# Each question is paraphrased on purpose (few exact words from the text) and
# labelled with the section that contains the answer.
EVAL_SET = [
    ("Which optimizer mixes momentum with a learning rate that adapts per weight?", "Loss functions and optimizers"),
    ("Why can't one neuron solve the exclusive-or problem?", "Perceptron and layers"),
    ("How do I stop my model doing great on training data but badly on new data?", "Bias-variance tradeoff"),
    ("What metric tells me how many of the real positive cases my model caught?", "Classification metrics"),
    ("How does a forest of trees combine predictions?", "Random forest"),
    ("Which boosting method adds trees one after another to fix earlier mistakes?", "XGBoost and gradient boosting"),
    ("Why should the test data only be used once at the very end?", "Train, validation and test split"),
    ("How do I measure how close two sentence vectors are?", "Text embeddings and similarity"),
    ("What tool from Meta finds nearest neighbours among millions of vectors?", "Vector databases"),
    ("Why do we cut long documents into smaller pieces before indexing?", "Chunking strategies"),
    ("How does an LLM look at every other word when processing a token?", "Large language models and transformers"),
    ("Which average is not affected much by extreme values?", "Descriptive statistics"),
    ("How do I combine a table of customers with a table of orders in pandas?", "NumPy and Pandas"),
    ("What is the rule Python uses to look up variable names?", "Functions and scope"),
    ("How does giving the model a few examples in the prompt help?", "Prompt engineering"),
    ("How does grounding a chatbot in my own documents reduce made-up answers?", "RAG pipeline"),
]


def evaluate(retrieve_fn, k=TOP_K):
    """Hit Rate@1 / Hit Rate@k: fraction of questions whose correct section is
    ranked first / appears in the top k. MRR: mean of 1/rank of the first
    correct chunk (0 if not found)."""
    top1, hits, reciprocal_ranks = 0, 0, []
    for question, section in EVAL_SET:
        ranked = [c.section for c, _ in retrieve_fn(question, k)]
        rank = ranked.index(section) + 1 if section in ranked else None
        top1 += rank == 1
        hits += rank is not None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    n = len(EVAL_SET)
    return top1 / n, hits / n, float(np.mean(reciprocal_ranks))


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def save_fig(fig, name):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> plots/{name}")


def plot_architecture():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)

    def box(x, y, text, color):
        ax.add_patch(plt.Rectangle((x, y), 2.0, 1.1, facecolor=color, edgecolor="black",
                                   lw=1.2, zorder=2))
        ax.text(x + 1.0, y + 0.55, text, ha="center", va="center", fontsize=9, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))

    ax.text(0.2, 6.1, "INDEXING STAGE (offline)", fontsize=12, weight="bold", color="#2a6fdb")
    idx = [("Documents\n(.md .txt .pdf)", 0.2), ("Loader\n(split by heading)", 2.9),
           ("Chunker\nfixed / sentence /\nrecursive", 5.6), ("Embedder\nall-MiniLM-L6-v2\n(384-d)", 8.3),
           ("Vector DB\nFAISS IndexFlatIP\nor Chroma", 11.1)]
    for i, (t, x) in enumerate(idx):
        box(x, 4.6, t, "#d6e4fb")
        if i:
            arrow(idx[i - 1][1] + 2.0, 5.15, x, 5.15)

    ax.text(0.2, 3.3, "QUERY STAGE (online)", fontsize=12, weight="bold", color="#e07a1f")
    qry = [("User\nquestion", 0.2), ("Embed query\n(same model)", 2.9),
           ("Top-k search\ncosine similarity\n+ threshold", 5.6), ("Prompt builder\ncontext + sources",
                                                                   8.3),
           ("LLM (Groq)\ngrounded answer\n+ citations", 11.1)]
    for i, (t, x) in enumerate(qry):
        box(x, 1.8, t, "#fde2c8")
        if i:
            arrow(qry[i - 1][1] + 2.0, 2.35, x, 2.35)
    arrow(12.1, 4.6, 6.6, 2.9)  # vector DB feeds the search step
    ax.text(9.3, 4.15, "nearest-neighbour lookup", fontsize=8, color="#555555", rotation=-11)
    ax.text(0.2, 0.9, "Evaluation: Hit Rate@k and MRR on a labelled question set;"
            " keyword (TF-IDF) vs semantic vs hybrid retrieval.", fontsize=9, color="#555555")
    ax.set_title("RAG Pipeline Architecture", fontsize=14, weight="bold")
    save_fig(fig, "01_rag_architecture.png")


def plot_similarity_demo(embedder):
    sentences = ["The cat sat on the mat.",
                 "A kitten is resting on the rug.",
                 "Adam is an optimizer for neural networks.",
                 "Gradient descent updates model weights.",
                 "The stock market fell sharply today."]
    vectors = embedder.embed(sentences)
    sim = vectors @ vectors.T
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
    labels = [s[:28] + ("..." if len(s) > 28 else "") for s in sentences]
    ax.set_xticks(range(len(sentences)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(sentences)), labels)
    for (i, j), v in np.ndenumerate(sim):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("Embedding similarity: same meaning -> high score, even with different words")
    save_fig(fig, "02_embedding_similarity.png")
    return sentences, sim


def plot_bar(results, name, title):
    labels = list(results)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    series = [("Hit Rate@1", "#3fa34d"), (f"Hit Rate@{TOP_K}", "#2a6fdb"), ("MRR", "#e07a1f")]
    for j, (metric, color) in enumerate(series):
        values = [results[l][j] for l in labels]
        offset = (j - 1) * 0.26
        ax.bar(x + offset, values, 0.26, label=metric, color=color)
        for xi, v in zip(x, values):
            ax.text(xi + offset, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, name)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def run_demo(store):
    banner("WEEK 8 : RAG PIPELINE ARCHITECTURE")
    plot_architecture()

    banner("STEP 1 : TEXT EMBEDDINGS & COSINE SIMILARITY")
    embedder = Embedder()
    print(f"  model: {embedder.model_name}   vector size: {embedder.dim}")
    sentences, sim = plot_similarity_demo(embedder)
    print(f"  '{sentences[0]}' vs '{sentences[1]}'  -> {sim[0, 1]:.3f}  (same meaning)")
    print(f"  '{sentences[0]}' vs '{sentences[4]}'  -> {sim[0, 4]:.3f}  (unrelated)")

    banner("STEP 2 : CHUNKING STRATEGIES COMPARED")
    docs = load_documents()
    chunking_results = {}
    print(f"  {'strategy':<11}{'chunks':>8}{'avg chars':>11}{'Hit@1':>8}"
          f"{'Hit@' + str(TOP_K):>8}{'MRR':>8}")
    for strategy in CHUNKERS:
        rag = RAGPipeline(strategy, store="faiss", embedder=embedder).build_index(verbose=False)
        top1, hit, mrr = chunking_results[strategy] = evaluate(rag.retriever.retrieve)
        avg = np.mean([len(c.text) for c in rag.chunks])
        print(f"  {strategy:<11}{len(rag.chunks):>8}{avg:>11.0f}{top1:>8.2f}{hit:>8.2f}{mrr:>8.2f}")
    plot_bar(chunking_results, "03_chunking_strategies.png",
             "Retrieval quality by chunking strategy (semantic search)")

    banner(f"STEP 3 : BUILD THE INDEX ({store.upper()} vector database)")
    rag = RAGPipeline("sentence", store=store, embedder=embedder).build_index()

    banner("STEP 4 : KEYWORD vs SEMANTIC vs HYBRID RETRIEVAL")
    retrievers = {
        "keyword (TF-IDF)": KeywordRetriever(rag.chunks).retrieve,
        "semantic": rag.retriever.retrieve,
        "hybrid": HybridRetriever(embedder, rag.chunks, rag.chunk_vectors).retrieve,
    }
    retrieval_results = {}
    for name, fn in retrievers.items():
        top1, hit, mrr = retrieval_results[name] = evaluate(fn)
        print(f"  {name:<18} Hit@1 {top1:.2f}   Hit@{TOP_K} {hit:.2f}   MRR {mrr:.2f}")
    plot_bar(retrieval_results, "04_retrieval_comparison.png",
             f"Keyword vs semantic vs hybrid search ({len(EVAL_SET)} paraphrased questions)")

    q = EVAL_SET[1][0]
    print(f"\n  Example - \"{q}\"")
    for name in ("keyword (TF-IDF)", "semantic"):
        top = retrievers[name](q, 1)[0][0]
        print(f"    {name:<18} top result -> {top.section}")

    banner(f"STEP 5 : END-TO-END QUESTION ANSWERING  [{rag.generator.mode}]")
    for question in ["What is the difference between precision and recall?",
                     "Why is ReLU used in hidden layers?",
                     "What are the two stages of a RAG pipeline?",
                     "Who won the football world cup in 2022?"]:   # out-of-scope question
        print_answer(question, *rag.query(question))

    banner("DONE - see plots/ for figures and vector_index/ for the saved index")


def main():
    parser = argparse.ArgumentParser(description="Week 8 RAG pipeline")
    parser.add_argument("--ask", help="answer a single question")
    parser.add_argument("--interactive", action="store_true", help="ask questions in a loop")
    parser.add_argument("--store", choices=list(VECTOR_STORES), default="faiss")
    parser.add_argument("--chunking", choices=list(CHUNKERS), default="sentence")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    if not (args.ask or args.interactive):
        run_demo(args.store)
        return

    rag = RAGPipeline(args.chunking, args.store, args.top_k).build_index()
    print(f"  generator: {rag.generator.mode}")
    if args.ask:
        print_answer(args.ask, *rag.query(args.ask))
    while args.interactive:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        print_answer(question, *rag.query(question))


if __name__ == "__main__":
    main()
