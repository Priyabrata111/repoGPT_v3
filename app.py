
import os
import re
import pickle
import hashlib
import numpy as np

import gradio as gr

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------
# Embeddings
# -------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------
# Chroma
# -------------------

vectorstore = Chroma(
    collection_name="multi_repo_v1",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
#-----------------------
# verify db
#-----------------------
import os

db_path = "./chroma_db"

print(os.path.exists(db_path))
print(os.listdir(db_path))
# -------------------
# BM25
# -------------------

with open("bm25_bundle.pkl", "rb") as f:
    data = pickle.load(f)

bm25 = data["bm25"]
smart_chunks = data["chunks"]
# -------------------
# Load Reranker
# -------------------

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "./reranker_model"
)

# -------------------
# Gemini
# -------------------
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key= os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
"""
You are a code-intelligence assistant.

Answer the question using only the provided code context.

Rules:
1. Cite supporting code using [S1], [S2], etc.
2. Do not invent files, functions, classes or behavior.
3. If the retrieved context is insufficient, say:
   "I don't see enough evidence in the indexed code."
4. Explain the execution flow when multiple files are involved.
5. Keep technical identifiers exactly as written in the code.

Context:
{context}

Question:
{question}

Answer:
"""
)

# -------------------
# Utilities
# -------------------

def tokenize(text):
    return [t.lower() for t in re.findall(r"\w+", text)]


# def format_context(docs):
#     return "\n\n---\n\n".join(
#         f"# File: {d.metadata['source']}\n{d.page_content}"
#         for d in docs
#     )

def format_context(results):
    sections = []

    for number, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata

        sections.append(
            f"""
[S{number}]
Repository: {metadata.get("repo", "unknown")}
File: {metadata.get("source", "unknown")}
Lines: {metadata.get("start_line", "?")}-{metadata.get("end_line", "?")}
Relevance score: {score:.4f}

Code:
{doc.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(sections)
# -------------------
# Hybrid Search
# -------------------

def get_chunk_id(doc):
    chunk_id = doc.metadata.get("chunk_id")

    if chunk_id:
        return chunk_id

    raw = (
        f"{doc.metadata.get('repo', '')}:"
        f"{doc.metadata.get('source', '')}:"
        f"{doc.metadata.get('start_line', '')}:"
        f"{doc.page_content}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def hybrid_search(query, fetch_k=20, rrf_constant=60):
    semantic_docs = vectorstore.similarity_search(
        query,
        k=fetch_k
    )

    bm25_scores = bm25.get_scores(tokenize(query))

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:fetch_k]

    keyword_docs = [
        smart_chunks[i]
        for i in top_indices
    ]

    fused_scores = {}
    documents = {}

    for rank, doc in enumerate(semantic_docs, start=1):
        chunk_id = get_chunk_id(doc)

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_constant + rank)
        )

        documents[chunk_id] = doc

    for rank, doc in enumerate(keyword_docs, start=1):
        chunk_id = get_chunk_id(doc)

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_constant + rank)
        )

        documents[chunk_id] = doc

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    return [
        documents[chunk_id]
        for chunk_id in ranked_ids
    ]

# -------------------
# Reranking
# -------------------

def search_with_rerank(query, k=5, fetch_k=20):
    candidates = hybrid_search(
        query,
        fetch_k=fetch_k
    )

    if not candidates:
        return []

    pairs = [
        (query, doc.page_content[:6000])
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    valid_results = []

    for doc, score in zip(candidates, scores):
        score = float(score)

        if not np.isnan(score):
            valid_results.append((doc, score))

    valid_results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return valid_results[:k]

#-----------------------------------
# Deterministic Source Formatter
#------------------------------------
def format_sources(results):
    lines = ["\n\n### Retrieved sources"]

    for number, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata

        repo = metadata.get("repo", "unknown")
        source = metadata.get("source", "unknown")
        start = metadata.get("start_line", "?")
        end = metadata.get("end_line", "?")

        lines.append(
            f"- [S{number}] `{repo}/{source}` "
            f"(lines {start}–{end}, score {score:.4f})"
        )

    return "\n".join(lines)


# -------------------
# QA
# -------------------

def ask(question):
    results = search_with_rerank(
        question,
        k=5,
        fetch_k=20
    )

    if not results:
        return "I couldn't retrieve relevant code."

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": format_context(results),
        "question": question,
    })

    return answer + format_sources(results)
# -------------------
# Gradio
# -------------------

import gradio as gr
def gradio_respond(message, history):
    if not message or not message.strip():
        return "Please enter a question about the indexed repositories."

    try:
        return ask(message.strip())

    except Exception as error:
        print(f"Query failed: {error}")
        return (
            "An error occurred while processing your question. "
            "Please try again."
        )


demo = gr.ChatInterface(
    fn=gradio_respond,
    title="Multi-Repository Code Intelligence Assistant",
    description=(
        "Query multiple repositories using semantic search, "
        "BM25 retrieval, Reciprocal Rank Fusion, and "
        "cross-encoder reranking."
    ),
    examples=[
        "How is the JWT token validated?",
        "How is b_transport implemented?",
        "Which file contains the implementation of nb_transport_fw()?",
        "How are JWT tokens generated and validated in the Ecommerce App?",
        "Where is the game-over condition implemented in the Simon Game?",
    ],
)


if __name__ == "__main__":
    demo.queue().launch()
## This version is working perfectly