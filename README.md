# Arcana_tasks

Coursework for the **Arcana Info Data Science & AI Internship**. Each folder is
one week of the [training roadmap](roadmap/Arcana_Info_-_Data_Science_AI_Internship_Roadmap.docx).

## Weekly progress

| Week | Topic | Deliverable |
|------|-------|-------------|
| [01](week01_python_fundamentals/) | Python Fundamentals + Secure Coding add-on | [To-Do List](week01_python_fundamentals/todo_list/), [Code audit](week01_python_fundamentals/secure_coding_addon/) |
| [02](week02_python_for_data_and_statistics/) | Python for Data & Intro Statistics | [EDA notebooks](week02_python_for_data_and_statistics/assignments/notebooks/) |
| [03](week03_sql_essentials/) | SQL Essentials | Notes |
| [04](week04_ml_supervised_learning/) | ML Concepts & Supervised Learning | [Housing price regression model](week04_ml_supervised_learning/housing_model/) |
| [05](week05_tree_models_and_evaluation/) | Tree Models, Ensembles & Evaluation | [Breast cancer classification challenge](week05_tree_models_and_evaluation/classification_model/) |
| [06](week06_neural_networks/) | Neural Networks & Deep Learning Basics | [Neural network from scratch](week06_neural_networks/neural_network/) |
| [07](week07_llm/) | LLMs & Generative AI Fundamentals | Notes |
| [08](week08_embeddings_and_rag_pipeline/) | Embeddings, Vector Search & RAG Pipeline | [RAG pipeline](week08_embeddings_and_rag_pipeline/rag_pipeline/) |
| [09](week09_rag_application/) | End-to-End RAG Application | [RAG chatbot (Streamlit)](week09_rag_application/rag_chatbot/) |

## Folder layout

Every week follows the same pattern, so you always know where to look:

```
weekNN_topic/
├── notes/          concept summaries (images, .txt / .md notes), numbered by topic
├── practice/       small practice scripts for each topic
├── notebooks/      Jupyter notebooks
├── data/           datasets used by the scripts / notebooks
├── plots/          generated charts
└── <project>/      the week's deliverable - self-contained, with its own
                    requirements.txt, data, plots/ and saved model
```

A week only contains the sub-folders it needs.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
```

Each project folder has its own `requirements.txt`. Install it and run the
script from inside that folder:

```bash
cd week04_ml_supervised_learning/housing_model
pip install -r requirements.txt
python housing_regression.py
pytest -v                       # week 4 includes automated tests
```

| Project | Run command |
|---------|-------------|
| Week 4 housing model | `python housing_regression.py` |
| Week 5 classification | `python classification_challenge.py` |
| Week 6 neural network | `python simple_neural_network.py` |
| Week 8 RAG pipeline | `python rag_pipeline.py` (`--ask "..."`, `--interactive`) |
| Week 9 RAG chatbot | `python -m streamlit run app.py` / `python evaluate_rag.py` |

Weeks 8 and 9 call the Groq API: copy `.env.example` to `.env` in that project
folder and add your `GROQ_API_KEY`. `.env` files are git-ignored and must never
be committed.
