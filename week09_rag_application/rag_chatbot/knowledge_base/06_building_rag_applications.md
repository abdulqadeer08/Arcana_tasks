# Building End-to-End RAG Applications

## LangChain and LlamaIndex
LangChain is a Python framework for building LLM applications from reusable components: document loaders, text splitters, embedding models, vector stores, retrievers, prompt templates, chat models and output parsers. Components are connected with the LangChain Expression Language (LCEL) using the pipe operator, for example prompt | llm | parser. LlamaIndex is a similar framework focused on data indexing and retrieval; it provides ready-made index types and query engines for RAG. Both frameworks let you swap the vector database or the LLM without rewriting the rest of the pipeline.

## Retrieval tuning
Retrieval quality depends on several settings. top_k controls how many chunks are passed to the LLM: too few can miss the answer, too many add noise and cost. Chunk size and overlap control how much context each chunk carries. A similarity score threshold drops chunks that are not relevant enough. Maximal Marginal Relevance (MMR) search picks chunks that are relevant to the query but different from each other, which reduces duplicate context; fetch_k sets how many candidates MMR chooses from and lambda_mult balances relevance against diversity.

## Reranking
Reranking is a two-stage retrieval approach. First a fast bi-encoder (embedding model) retrieves a larger candidate set, such as the top 10 chunks. Then a slower but more accurate cross-encoder scores each question and chunk pair together and reorders the candidates, and only the best few are kept. A cross-encoder reads the question and the passage at the same time, so it judges relevance better than comparing two separately computed embeddings. A popular reranker is cross-encoder/ms-marco-MiniLM-L-6-v2.

## Hallucination and grounding
A hallucination is an answer that sounds confident but is not supported by facts or by the provided context. Grounding means forcing the answer to be based on retrieved evidence. Common grounding techniques are: a system prompt that says to answer only from the context, asking the model to cite the passages it used, telling the model to say "I don't know" when the context does not contain the answer, refusing to answer when no chunk passes the relevance threshold, using a low temperature, and checking afterwards that every sentence in the answer is supported by a retrieved passage.

## Streamlit front-end
Streamlit turns a Python script into a web app without writing HTML or JavaScript. st.chat_input and st.chat_message build a chat interface, st.session_state keeps the conversation history between reruns, st.sidebar holds settings such as top_k or the reranker switch, st.file_uploader lets users add their own documents, and st.cache_resource keeps expensive objects such as the embedding model and vector index in memory. An app is started with the command streamlit run app.py.

## Evaluating RAG outputs
A RAG system is evaluated at two levels. Retrieval metrics check whether the right chunks were found: Hit Rate@k, Mean Reciprocal Rank (MRR) and context precision, which is the share of retrieved chunks that are actually relevant. Generation metrics check the answer: faithfulness (groundedness) measures whether the answer is supported by the retrieved context, answer relevance measures whether it addresses the question, and answer correctness compares it with a reference answer. A good system should also refuse out-of-scope questions. Frameworks such as RAGAS automate these metrics, often by using an LLM as a judge.
