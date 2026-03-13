"""
RAGAS Evaluation for YouTube Research Assistant
================================================
Run this after your RAG pipeline is set up.

What it tests:
- context_precision  : are retrieved chunks ranked well / relevant?
- context_recall     : did retrieval miss important info?
- faithfulness       : does LLM answer match the chunks (no hallucination)?
- answer_relevancy   : does the answer actually address the question?

Install:
    pip install ragas langchain-openai

Usage:
    python test_ragas.py
"""

import os
from dotenv import load_dotenv

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash" , temperature=0.2)  #temp 0-2 hota //Creativity batata hai



load_dotenv()

# ── Import your pipeline ───────────────────────────────────────────────────────
# Adjust these imports to match your actual file/variable names

from rag import chain, retriever   # your LangChain chain + retriever


# ── Test questions + ground truths ────────────────────────────────────────────
# Ground truths = what the correct answer SHOULD contain
# Write these based on what you already know about the topic
# The more specific, the better the eval
TEST_CASES = [
    {
        "question": "How do you connect FastAPI to Supabase in Python?",
        "ground_truth": (
            "To connect FastAPI to Supabase in Python, install the supabase-py client library. "
            "Initialize the Supabase client using your project URL and anon/service key from "
            "the Supabase dashboard. Use the client inside FastAPI route handlers to interact "
            "with your database."
        ),
    },
    {
        "question": "How do you perform CRUD operations in Supabase with FastAPI?",
        "ground_truth": (
            "CRUD operations in Supabase use the Python client methods: "
            "supabase.table('name').insert(data).execute() for create, "
            ".select() for read, .update(data).eq() for update, "
            "and .delete().eq() for delete. These are called inside FastAPI route handlers."
        ),
    },
    {
        "question": "How does authentication work with Supabase and FastAPI?",
        "ground_truth": (
            "Supabase handles authentication using JWT tokens. Users sign up or log in via "
            "supabase.auth.sign_up() or sign_in_with_password(). The returned JWT token is "
            "sent with requests and verified in FastAPI using a dependency that decodes "
            "and validates the token."
        ),
    },
    {
        "question": "How do you protect FastAPI routes using Supabase authentication?",
        "ground_truth": (
            "FastAPI routes are protected by creating a dependency function that extracts "
            "the Bearer token from the Authorization header and verifies it using Supabase. "
            "This dependency is injected into protected routes using FastAPI's Depends()."
        ),
    },
    {
        "question": "How do you set up a FastAPI project structure with Supabase?",
        "ground_truth": (
            "A FastAPI Supabase project typically has a main.py for the app entry point, "
            "a supabase client initialization file, route files for each resource, "
            "and a .env file storing the Supabase URL and API keys loaded via python-dotenv."
        ),
    },
]


# ── Build evaluation dataset ──────────────────────────────────────────────────

def build_dataset(test_cases: list[dict]) -> EvaluationDataset:
    samples = []

    for i, test in enumerate(test_cases):
        question    = test["question"]
        ground_truth = test["ground_truth"]

        print(f"[{i+1}/{len(test_cases)}] Running: {question}")

        # Get answer from your chain
        try:
            answer = chain.invoke(question)
        except Exception as e:
            print(f"  ⚠ Chain failed: {e}")
            answer = "Error generating answer."

        # Get retrieved chunks from your retriever
        try:
            retrieved_docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in retrieved_docs]
        except Exception as e:
            print(f"  ⚠ Retriever failed: {e}")
            contexts = []

        print(f"  → Retrieved {len(contexts)} chunks")
        print(f"  → Answer: {answer[:80]}...")

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))

    return EvaluationDataset(samples=samples)


# ── Run evaluation ────────────────────────────────────────────────────────────
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def run_evaluation(dataset: EvaluationDataset):
    # RAGAS needs its own LLM + embeddings to score your pipeline
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash" )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
       GoogleAIEmbeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithReference(),   # are top chunks actually relevant?
            LLMContextRecall(),                   # did retrieval miss important info?
            Faithfulness(),                       # does answer match chunks (no hallucination)?
            ResponseRelevancy(),                  # does answer address the question?
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return results


# ── Print results ─────────────────────────────────────────────────────────────

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
                status = "✅" if val > 0.8 else "⚠ " if val > 0.6 else "❌"
                print(f"       {status} {label}: {val:.3f}")


def _score_bar(score: float) -> str:
    filled = int(score * 10)
    bar    = "█" * filled + "░" * (10 - filled)
    return f"[{bar}]"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building evaluation dataset...")
    dataset = build_dataset(TEST_CASES)

    print("\nRunning RAGAS evaluation...")
    results = run_evaluation(dataset)

    print_results(results)
