## Multi-Repository Code Intelligence Assistant

An AI-powered code assistant that enables natural language querying across multiple software repositories.
The system combines semantic search, BM25 keyword retrieval, and cross-encoder reranking to locate relevant code and generate accurate answers grounded in source files.

## Features

- Query multiple repositories using natural language
- Supports C, C++, SystemC/TLM, Python, JavaScript, TypeScript, Markdown, and Jupyter notebooks
- Smart language-aware code chunking
- Semantic retrieval using Sentence Transformers and ChromaDB
- BM25 keyword search for exact identifier matching
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking for improved retrieval accuracy
- Interactive Gradio web interface
- Repository-aware indexing and retrieval

## Tech Stack

- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Rank-BM25
- Cross Encoder (ms-marco-MiniLM-L-6-v2)
- Google Gemini
- Gradio
- Python

## Retrieval Pipeline

### Stage 1 — Repository Ingestion

The system clones one or more Git repositories and converts source files into LangChain Documents

### Stage 2 — Smart Chunking

Language-specific chunking strategies are used:

| Language                | Strategy                    |
| ----------------------- | --------------------------- |
| Python                  | Python-aware splitter       |
| JavaScript / TypeScript | JS-aware splitter           |
| C / C++ / SystemC / TLM | C++-aware splitter          |
| Markdown                | Header-aware splitter       |
| Others                  | Recursive fallback splitter |

### Stage 3 — Hybrid Retrieval

The system performs:

1. Semantic Search using embeddings stored in ChromaDB
2. BM25 Keyword Search for exact identifier matching

### Stage 4 — Cross-Encoder Reranking

Retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This significantly improves retrieval quality by jointly evaluating the query and candidate chunk.

## Demo

Access the live demo here: https://huggingface.co/spaces/Priyabrata111/multi_repo_GPT_v2

<img width="1911" height="617" alt="image" src="https://github.com/user-attachments/assets/6ad54b95-35cd-44e9-ae74-8d5b473dd886" />

<img width="1792" height="857" alt="image" src="https://github.com/user-attachments/assets/cf840ff5-67d8-4d33-abac-c6f625cc24ae" />

## Retrieval Evaluation

The retrieval pipeline was evaluated using **30 manually validated code-search questions** across five repositories. A Recall@5 hit was recorded when at least one expected repository file appeared among the top five retrieved results.

| Retrieval Method                           |   Recall@5 |       MRR |
| ------------------------------------------ | ---------: | --------: |
| Semantic Search                            |     36.67% |     0.367 |
| BM25 Keyword Search                        |     53.33% |     0.409 |
| Hybrid Retrieval (Semantic + BM25 + RRF)   |     60.00% |     0.464 |
| Hybrid Retrieval + Cross-Encoder Reranking | **70.00%** | **0.572** |

### Key Results

- Semantic search retrieved a relevant file for **11 out of 30** questions.
- BM25 retrieved a relevant file for **16 out of 30** questions, demonstrating the importance of exact identifiers and technical keywords in code search.
- Reciprocal Rank Fusion increased Recall@5 to **60%** by combining semantic and lexical retrieval.
- Cross-encoder reranking produced the best result, retrieving a relevant file for **21 out of 30** questions.
- Compared with semantic search alone, the final pipeline improved Recall@5 by **33.33 percentage points** and MRR from **0.367 to 0.572**.

### Evaluation Metrics

- **Recall@5:** The percentage of questions for which at least one expected file appeared among the top five retrieved chunks.
- **MRR (Mean Reciprocal Rank):** Measures how highly the first relevant result was ranked. A higher value indicates that relevant code appeared closer to the top.

> **Current limitation:** Relevance is evaluated at the file level. For questions involving multiple files, the current Recall@5 calculation counts the query as successful when any expected file is retrieved. Future evaluation will include symbol-level relevance, all-expected-files recall, retrieval latency, citation accuracy, and answer correctness.

## Author

Priyabrata Mondal

GitHub: https://github.com/Priyabrata111
