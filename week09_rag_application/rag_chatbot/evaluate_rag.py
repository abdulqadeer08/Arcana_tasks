# ---------------------------------------------------------------------------
# EVALUATION OF RAG OUTPUTS
# ---------------------------------------------------------------------------
# Week 9 topic: Evaluation of RAG outputs.
#
# Runs from a terminal:
#     python evaluate_rag.py
#
# Part A - retrieval tuning experiments (which settings find the right chunks?)
#   * similarity vs MMR search, with and without cross-encoder reranking
#   * chunk size sweep
#   metrics: Hit Rate@1, Hit Rate@k, MRR, context precision
#
# Part B - end-to-end answer quality on the best configuration
#   * faithfulness / groundedness  - answer sentences supported by the context
#   * answer relevance            - cosine similarity question <-> answer
#   * answer correctness          - share of expected key facts in the answer
#   * refusal accuracy            - out-of-scope questions must get "I don't know"
#   * LLM-as-judge faithfulness   - only when GROQ_API_KEY is set
#
# Results are saved to eval_results/ as CSV files and PNG plots.
# ---------------------------------------------------------------------------

import re
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")  # save figures to files, do not open a window

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rag_engine import BASE_DIR, KB_DIR, RAGChatbot, RAGConfig

RESULTS_DIR = BASE_DIR / "eval_results"

# (question, section that contains the answer, key facts a correct answer mentions)
# Questions are paraphrased on purpose so exact keyword matching is not enough.
EVAL_SET = [
    ("Which optimizer mixes momentum with a learning rate that adapts per weight?",
     "Loss functions and optimizers", ["adam", "momentum", "adaptive"]),
    ("Why can't one neuron solve the exclusive-or problem?",
     "Perceptron and layers", ["straight line", "xor", "hidden"]),
    ("How do I stop my model doing great on training data but badly on new data?",
     "Bias-variance tradeoff", ["overfit", "regularisation", "more data"]),
    ("What metric tells me how many of the real positive cases my model caught?",
     "Classification metrics", ["recall", "fn"]),
    ("How does a forest of trees combine predictions?",
     "Random forest", ["vote", "average", "bootstrap"]),
    ("Which boosting method adds trees one after another to fix earlier mistakes?",
     "XGBoost and gradient boosting", ["gradient boosting", "residuals", "sequential"]),
    ("Why should the test data only be used once at the very end?",
     "Train, validation and test split", ["leak", "unseen", "over-optimistic"]),
    ("How do I measure how close two sentence vectors are?",
     "Text embeddings and similarity", ["cosine", "dot product"]),
    ("What tool from Meta finds nearest neighbours among millions of vectors?",
     "Vector databases", ["faiss"]),
    ("Why do we cut long documents into smaller pieces before indexing?",
     "Chunking strategies", ["context window", "precise", "limited"]),
    ("How does a second model reorder the retrieved passages?",
     "Reranking", ["cross-encoder", "candidates", "reorder"]),
    ("What can I do so the chatbot stops inventing facts?",
     "Hallucination and grounding", ["context", "cite", "don't know"]),
    ("Which Streamlit function keeps the chat history between reruns?",
     "Streamlit front-end", ["session_state"]),
    ("How do I check whether the answer is actually backed by the retrieved text?",
     "Evaluating RAG outputs", ["faithfulness", "supported"]),
    ("How do you connect a prompt, a model and a parser in LangChain?",
     "LangChain and LlamaIndex", ["lcel", "pipe", "|"]),
    ("Which search picks relevant chunks that are also different from each other?",
     "Retrieval tuning", ["mmr", "maximal marginal relevance", "diverse"]),
]

OUT_OF_SCOPE = [
    "Who won the football world cup in 2022?",
    "What is the capital city of Australia?",
    "Give me a recipe for sourdough bread.",
    "What will the weather be like in Karachi tomorrow?",
]

RETRIEVAL_CONFIGS = {
    "similarity": dict(search_type="similarity", use_reranker=False),
    "MMR": dict(search_type="mmr", use_reranker=False),
    "similarity + rerank": dict(search_type="similarity", use_reranker=True),
    "MMR + rerank": dict(search_type="mmr", use_reranker=True),
}


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def save_fig(fig, name):
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> eval_results/{name}")


# ---------------------------------------------------------------------------
# PART A : RETRIEVAL METRICS
# ---------------------------------------------------------------------------
def retrieval_metrics(bot, config):
    top1 = hits = 0
    rr, precision = [], []
    for question, section, _ in EVAL_SET:
        ranked = [c.section for c in bot.retrieve(question, config)]
        rank = ranked.index(section) + 1 if section in ranked else None
        top1 += rank == 1
        hits += rank is not None
        rr.append(1.0 / rank if rank else 0.0)
        # context precision: share of retrieved chunks that come from the right section
        precision.append(ranked.count(section) / len(ranked) if ranked else 0.0)
    n = len(EVAL_SET)
    return {"hit@1": top1 / n, f"hit@{config.top_k}": hits / n,
            "mrr": float(np.mean(rr)), "context_precision": float(np.mean(precision))}


def plot_metrics(df, name, title):
    ax = df.plot(kind="bar", figsize=(10, 4.8), rot=0, width=0.8,
                 color=["#3fa34d", "#2a6fdb", "#e07a1f", "#8e5bd6"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    save_fig(ax.get_figure(), name)


def part_a(bot):
    banner("PART A1 : RETRIEVAL TUNING - SEARCH TYPE & RERANKING (top_k=3)")
    rows = {name: retrieval_metrics(bot, replace(bot.config, **opts))
            for name, opts in RETRIEVAL_CONFIGS.items()}
    df = pd.DataFrame(rows).T
    print(df.round(3).to_string())
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "retrieval_configs.csv")
    plot_metrics(df, "01_retrieval_configs.png",
                 f"Retrieval quality by configuration ({len(EVAL_SET)} paraphrased questions)")

    banner("PART A2 : RETRIEVAL TUNING - CHUNK SIZE (similarity + rerank)")
    rows = {}
    for size in (250, 500, 1000):
        cfg = replace(bot.config, chunk_size=size, chunk_overlap=int(size * 0.15))
        sized = RAGChatbot(KB_DIR, cfg, api_key=bot.api_key)
        sized._reranker = bot._reranker   # reuse the loaded cross-encoder
        rows[f"{size} chars ({len(sized.chunks)} chunks)"] = retrieval_metrics(sized, cfg)
    df_size = pd.DataFrame(rows).T
    print(df_size.round(3).to_string())
    df_size.to_csv(RESULTS_DIR / "chunk_sizes.csv")
    plot_metrics(df_size, "02_chunk_sizes.png", "Retrieval quality by chunk size")


# ---------------------------------------------------------------------------
# PART B : END-TO-END ANSWER QUALITY
# ---------------------------------------------------------------------------
def answer_correctness(answer, facts):
    text = answer.lower()
    return sum(f.lower() in text for f in facts) / len(facts)


def answer_relevance(bot, question, answer):
    q = np.array(bot.embeddings.embed_query(question))
    a = np.array(bot.embeddings.embed_query(re.sub(r"\[\d+\]", "", answer)))
    return float(q @ a)


def llm_judge_faithfulness(bot, answer, sources):
    """LLM-as-judge: rate 1-5 how fully the answer is supported by the context."""
    from langchain_groq import ChatGroq
    judge = ChatGroq(model=bot.config.model, temperature=0, api_key=bot.api_key)
    context = "\n\n".join(c.text for c in sources)
    prompt = (f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
              "On a scale of 1 to 5, how fully is every claim in the ANSWER supported by the "
              "CONTEXT? 5 = fully supported, 1 = mostly unsupported. Reply with the number only.")
    match = re.search(r"[1-5]", judge.invoke(prompt).content)
    return int(match.group()) if match else None


def part_b(bot):
    banner(f"PART B : END-TO-END ANSWER QUALITY  [{bot.mode}]")
    rows = []
    for question, section, facts in EVAL_SET:
        r = bot.answer(question)
        row = {"question": question,
               "answer": r.answer,
               "refused": r.refused,
               "retrieved_correct_section": any(c.section == section for c in r.sources),
               "groundedness": r.groundedness,
               "answer_relevance": None if r.refused else answer_relevance(bot, question, r.answer),
               "answer_correctness": answer_correctness(r.answer, facts)}
        if bot.api_key and not r.refused:
            row["llm_judge_faithfulness_1to5"] = llm_judge_faithfulness(bot, r.answer, r.sources)
        rows.append(row)
    df = pd.DataFrame(rows)

    oos = [bot.answer(q) for q in OUT_OF_SCOPE]
    oos_df = pd.DataFrame({"question": OUT_OF_SCOPE, "answer": [r.answer for r in oos],
                           "refused": [r.refused for r in oos]})

    summary = {
        "faithfulness (groundedness)": df["groundedness"].mean(),
        "answer relevance": df["answer_relevance"].mean(),
        "answer correctness (key facts)": df["answer_correctness"].mean(),
        "in-scope answered (no false refusal)": 1 - df["refused"].mean(),
        "out-of-scope refused": oos_df["refused"].mean(),
    }
    if "llm_judge_faithfulness_1to5" in df:
        summary["LLM-judge faithfulness (1-5)"] = df["llm_judge_faithfulness_1to5"].mean()

    for name, value in summary.items():
        print(f"  {name:<40}{value:.3f}")
    print("\n  Out-of-scope questions:")
    for _, row in oos_df.iterrows():
        print(f"    {'REFUSED ' if row.refused else 'ANSWERED'}  {row.question}")

    df.to_csv(RESULTS_DIR / "answers.csv", index=False)
    oos_df.to_csv(RESULTS_DIR / "out_of_scope.csv", index=False)
    pd.Series(summary, name="score").to_csv(RESULTS_DIR / "summary.csv")
    print("\n  per-question answers -> eval_results/answers.csv")

    names = [n for n in summary if "1-5" not in n]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(names, [summary[n] for n in names], color="#2a6fdb")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.set_title(f"RAG answer quality ({bot.mode})")
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "03_answer_quality.png")


def main():
    banner("WEEK 9 : EVALUATION OF RAG OUTPUTS")
    bot = RAGChatbot()
    print(f"  knowledge base: {bot.n_files} files, {len(bot.chunks)} chunks")
    print(f"  generator: {bot.mode}")
    part_a(bot)
    part_b(bot)
    banner("DONE - results saved in eval_results/")


if __name__ == "__main__":
    main()
