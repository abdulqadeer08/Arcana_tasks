# Large Language Models and Retrieval-Augmented Generation

## Large language models and transformers
A large language model (LLM) is a neural network trained on huge amounts of text to predict the next token. Modern LLMs use the transformer architecture, whose key component is self-attention: every token computes attention weights over all other tokens using query, key and value vectors, so the model can capture long-range context. Tokenization splits text into sub-word tokens, for example with Byte Pair Encoding, and models have a maximum context window measured in tokens.

## Prompt engineering
Zero-shot prompting asks the model to perform a task with instructions only. Few-shot prompting includes a few worked examples in the prompt to show the desired format. Chain-of-thought prompting asks the model to reason step by step. A system prompt sets the role and rules for the assistant. LLM APIs such as Groq and OpenRouter accept a list of messages and return a generated completion.

## Text embeddings and similarity
An embedding model converts text into a dense vector so that texts with similar meaning have vectors that are close together. Cosine similarity measures the angle between two vectors and ranges from -1 to 1; when vectors are normalised to unit length, cosine similarity equals the dot product. Popular open-source embedding models include all-MiniLM-L6-v2, which produces 384-dimensional vectors.

## Vector databases
A vector database stores embeddings and finds the nearest neighbours of a query vector quickly. FAISS is a library from Meta for fast similarity search; IndexFlatIP performs exact inner-product search, while approximate indexes such as IVF and HNSW trade a little accuracy for much higher speed on millions of vectors. Chroma is an open-source vector database that stores documents, embeddings and metadata together and supports metadata filtering.

## Chunking strategies
Documents are split into chunks before embedding because embedding models and LLM context windows have limited size, and small focused chunks give more precise retrieval. Fixed-size chunking splits text every N characters or tokens, usually with an overlap so sentences cut at a boundary are not lost. Sentence-based chunking groups whole sentences. Recursive chunking splits by paragraphs first, then sentences, then words, keeping chunks under a size limit. Semantic chunking starts a new chunk where the topic changes. Typical chunk sizes are 200 to 1000 characters with 10 to 20 percent overlap.

## Semantic search and retrieval
Keyword search matches exact words, for example with TF-IDF or BM25, and fails when the query uses different words from the document. Semantic search embeds the query and returns the chunks whose embeddings are most similar, so it can match synonyms and paraphrases. Retrieval quality is measured with Hit Rate@k (whether a relevant chunk appears in the top k results) and Mean Reciprocal Rank (MRR), which rewards ranking the relevant chunk first. Hybrid search combines keyword and semantic scores.

## RAG pipeline
Retrieval-Augmented Generation (RAG) grounds an LLM in an external knowledge base. The indexing stage loads documents, splits them into chunks, embeds each chunk, and stores the vectors in a vector database. The query stage embeds the user question, retrieves the top-k most similar chunks, inserts them into a prompt as context, and asks the LLM to answer using only that context and to cite its sources. RAG reduces hallucination, lets the model use private or up-to-date data without retraining, and makes answers traceable to their sources.
