"""
RAGAS Evaluation for YouTube Research Assistant
================================================
Tests the RAG pipeline (YouTube search → transcript → retrieval → Groq answer)
using RAGAS metrics across diverse topics.

Metrics evaluated:
  - context_precision  : are retrieved chunks ranked well / relevant?
  - context_recall     : did retrieval miss important info?
  - faithfulness       : does the LLM answer match the chunks (no hallucination)?
  - answer_relevancy   : does the answer actually address the question?

Usage:
    python test_ragas.py
"""

import os
from dotenv import load_dotenv

from ragas import EvaluationDataset, SingleTurnSample, evaluate
try:
    from ragas.metrics.collections import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        Faithfulness,
        ResponseRelevancy,
    )
except ImportError:
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        Faithfulness,
        ResponseRelevancy,
    )
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import math

# ── Import the actual pipeline functions ─────────────────────────────────────
from rag import build_youtube_rag_pipeline
from rag_module import answer_question_with_rag


load_dotenv()


# ── Test cases: DIVERSE TOPICS ───────────────────────────────────────────────
# Each test has a different search_query (topic) to test the pipeline broadly.

TEST_CASES = [
    # ── Topic 1: Web Development ─────────────────────────────────
    {
        "search_query": "How do you create a basic REST API with FastAPI in Python?",
        "question": "How do you create a basic REST API with FastAPI in Python?",
        "ground_truth": (
            "To create a REST API with FastAPI, install fastapi and uvicorn, "
            "create an app instance with FastAPI(), define route handlers using "
            "decorators like @app.get() and @app.post(), and run with uvicorn. "
            "FastAPI auto-generates OpenAPI docs at /docs."
        ),
    },
    # ── Topic 2: Machine Learning ────────────────────────────────
    {
        "search_query": "What is the transformer architecture in NLP and why is it important?",
        "question": "What is the transformer architecture in NLP and why is it important?",
        "ground_truth": (
            "The transformer architecture uses self-attention mechanisms to process "
            "sequences in parallel instead of sequentially like RNNs. It consists of "
            "an encoder-decoder structure with multi-head attention layers. Transformers "
            "are the foundation of models like BERT and GPT and revolutionized NLP tasks."
        ),
    },
    # ── Topic 3: DevOps / Docker ─────────────────────────────────
    {
        "search_query": "What is Docker and how do containers work?",
        "question": "What is Docker and how do containers work?",
        "ground_truth": (
            "Docker is a platform for building, shipping, and running applications in "
            "containers. Containers are lightweight, isolated environments that package "
            "an app with its dependencies. A Dockerfile defines the image, docker build "
            "creates it, and docker run starts a container from the image."
        ),
    },
    # ── Topic 4: Database ────────────────────────────────────────
    {
        "search_query": "How do you connect to a PostgreSQL database from Python?",
        "question": "How do you connect to a PostgreSQL database from Python?",
        "ground_truth": (
            "To connect Python to PostgreSQL, use libraries like psycopg2 or SQLAlchemy. "
            "Install psycopg2, create a connection with psycopg2.connect() providing host, "
            "database, user, and password. Use a cursor to execute SQL queries and commit "
            "transactions. SQLAlchemy provides an ORM layer on top."
        ),
    },
    # ── Topic 5: Cloud / AWS ─────────────────────────────────────
    {
        "search_query": "How do you upload files to AWS S3 using Python?",
        "question": "How do you upload files to AWS S3 using Python?",
        "ground_truth": (
            "To upload files to S3 with Python, install boto3, configure AWS credentials, "
            "create an S3 client or resource with boto3.client('s3'), and use upload_file() "
            "or put_object() methods specifying the file path, bucket name, and object key."
        ),
    },
]


# ── Build evaluation dataset ────────────────────────────────────────────────

def build_dataset(test_cases: list[dict]) -> EvaluationDataset:
    """
    For each test case:
      1. Build a RAG retriever from YouTube (using search_query)
      2. Get the answer + retrieved docs from the pipeline
      3. Package into a RAGAS SingleTurnSample
    """
    samples = []

    # Cache retrievers by search_query to avoid duplicate YouTube API calls
    retrievers_cache: dict = {}

    for i, test in enumerate(test_cases):
        search_query = test["search_query"]
        question     = test["question"]
        ground_truth = test["ground_truth"]

        print(f"\n[{i+1}/{len(test_cases)}] Running: {question}")

        # Build retriever (or reuse cached one)
        if search_query not in retrievers_cache:
            print(f"   🔍 Building retriever for: '{search_query}'")
            try:
                retrievers_cache[search_query] = build_youtube_rag_pipeline(
                    search_query=search_query
                )
            except Exception as e:
                print(f"   ⚠ Pipeline build failed: {e}")
                retrievers_cache[search_query] = None

        retriever = retrievers_cache[search_query]

        # Get answer
        if retriever is None:
            answer = "Error: could not build retriever."
            contexts = []
        else:
            try:
                answer, retrieved_docs = answer_question_with_rag(
                    question=question, retriever=retriever
                )
                contexts = [doc.page_content for doc in retrieved_docs]
            except Exception as e:
                print(f"   ⚠ RAG failed: {e}")
                answer = "Error generating answer."
                contexts = []

        print(f"   → Retrieved {len(contexts)} chunks")
        print(f"   → Answer: {str(answer)[:100]}...")

        samples.append(SingleTurnSample(
            user_input=question,
            response=str(answer),
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))

    return EvaluationDataset(samples=samples)


# ── Run RAGAS evaluation ─────────────────────────────────────────────────────

def run_evaluation(dataset: EvaluationDataset):
    """Score the dataset using RAGAS metrics with Groq LLM + HuggingFace embeddings."""

    # Groq LLM as evaluator
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    )

    # HuggingFace BGE embeddings (same model used in the RAG pipeline)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return results


# ── Pretty-print results ────────────────────────────────────────────────────

def _score_bar(score: float) -> str:
    if math.isnan(score):
        return "[░░░░░░░░░░] (N/A)"
    filled = int(score * 10)
    return f"[{'█' * filled}{'░' * (10 - filled)}]"


def print_results(results):
    print("\n" + "=" * 55)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 55)

    scores = results.to_pandas()

    metrics = [
        "llm_context_precision_with_reference",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    labels = {
        "llm_context_precision_with_reference": "Context Precision",
        "context_recall":                       "Context Recall   ",
        "faithfulness":                         "Faithfulness     ",
        "answer_relevancy":                     "Answer Relevancy ",
    }

    for metric in metrics:
        if metric in scores.columns:
            avg = scores[metric].mean()
            bar = _score_bar(avg)
            label = labels[metric]
            print(f"  {label} : {avg:.3f}  {bar}")

    print("=" * 55)

    print("\nHow to interpret:")
    print("  > 0.8  ✅ Good")
    print("  0.6-0.8  ⚠  Needs improvement")
    print("  < 0.6  ❌ Needs significant fix\n")

    print("What to fix if score is low:")
    print("  Context Precision  → retriever returning irrelevant chunks → raise score_threshold")
    print("  Context Recall     → retriever missing info → lower threshold or increase k")
    print("  Faithfulness       → LLM hallucinating → strengthen prompt, lower temperature")
    print("  Answer Relevancy   → answer off-topic → fix prompt or check retrieved context\n")

    # Per-question breakdown
    print("-" * 55)
    print("  PER QUESTION BREAKDOWN")
    print("-" * 55)
    for i, row in scores.iterrows():
        print(f"\n  Q{i+1}: {row.get('user_input', '')[:60]}")
        for metric in metrics:
            if metric in row:
                val = row[metric]
                label = labels[metric].strip()
                if math.isnan(val):
                    print(f"       ⛔ {label}: N/A (eval failed)")
                else:
                    status = "✅" if val > 0.8 else "⚠ " if val > 0.6 else "❌"
                    print(f"       {status} {label}: {val:.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  RAGAS Evaluation — YouTube RAG Pipeline")
    print("  Topics: FastAPI, Transformers, Docker, PostgreSQL, AWS S3")
    print("=" * 55)

    print("\nStep 1: Building evaluation dataset (this calls YouTube + Groq)...")
    dataset = build_dataset(TEST_CASES)

    print("\nStep 2: Running RAGAS evaluation...")
    results = run_evaluation(dataset)

    print_results(results)
