"""
FastAPI backend for YT Research Assistant.
Bridges the frontend (index.html) with the RAG pipeline in backend/RAG/.
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
# Add the RAG directory to sys.path so existing modules import without changes
RAG_DIR = str(Path(__file__).resolve().parent / "RAG")
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

# Load .env from the project root (one level above backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Import RAG modules (after path setup) ─────────────────────────────────────
from langchain_community.chat_message_histories import ChatMessageHistory  # noqa: E402
from youtube_module import search_youtube_videos, fetch_video_transcripts, build_docs_from_transcripts  # noqa: E402
from rag_module import build_hybrid_retriever, answer_question_with_rag, stream_answer_with_rag  # noqa: E402
from query_router import route_query, casual_reply, stream_casual_reply  # noqa: E402

from langchain_groq import ChatGroq  # noqa: E402
from langchain_core.messages import SystemMessage, HumanMessage  # noqa: E402


# ── Session state (single-user, in-memory) ────────────────────────────────────
chat_history = ChatMessageHistory()
retriever = None  # built on the first RAG query


# ── Pydantic models ──────────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SourceItem(BaseModel):
    video_title: str
    timestamp: str
    video_url: str
    text: str


class ResearchResponse(BaseModel):
    answer: str
    tokens_used: int | None = None
    sources: list[SourceItem] = []


# ── Helper: optimise the user's raw query for YouTube search ──────────────────
QUERY_OPTIMIZER_PROMPT = """\
You are a search-query optimizer. Given a user's conversational question,
extract ONLY the core topic keywords suitable for a YouTube search.

Rules:
- Remove filler words like "can you tell me", "how to", "what is", "please explain","can u explain","(query) in detail" etc.
- Keep the essential technical terms and topic.
- Output 3-7 words maximum, as a concise YouTube search query.
- Add "tutorial" at the end if the query is asking how to do something.
- Output ONLY the optimized search query, nothing else.

Examples:
  User: "can u tell me how to connect fastapi with python"
  Output: "fastapi python tutorial"

  User: "what is retrieval augmented generation and how does it work"
  Output: "retrieval augmented generation explained"

  User: "please explain transformer architecture in deep learning"
  Output: "transformer architecture deep learning"
"""


def optimize_search_query(raw_query: str) -> str:
    """Use the LLM to extract a clean YouTube search query from conversational input."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=QUERY_OPTIMIZER_PROMPT),
        HumanMessage(content=raw_query),
    ])
    optimized = response.content.strip().strip('"')
    print(f"   🔎 Optimized search query: '{raw_query}' → '{optimized}'")
    return optimized or raw_query  # fallback to original if LLM returns empty


# ── Helper: build the full YouTube RAG pipeline ──────────────────────────────
def build_youtube_rag_pipeline(search_query: str, max_results: int = 5):
    """Optimise query → search YouTube → fetch transcripts → chunk → build retriever."""
    optimized_query = optimize_search_query(search_query)
    videos_metadata = search_youtube_videos(
        query=optimized_query,
        max_results=max_results,
        published_after="2021-01-01T00:00:00Z",
    )
    videos_with_transcripts = fetch_video_transcripts(videos_metadata)
    docs = build_docs_from_transcripts(videos_with_transcripts)
    return build_hybrid_retriever(docs)


# ── FastAPI app ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 YT Research API is ready!")
    yield
    print("👋 Shutting down…")


app = FastAPI(
    title="YT Research Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend (any localhost port) during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoint ─────────────────────────────────────────────────────────────
@app.post("/api/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """
    Main endpoint consumed by the frontend.
    Routes the query through the same logic as the CLI app in rag.py.
    """
    global retriever

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Step 1: Route the query
        intent = route_query(query, chat_history)
        print(f"   [router → {intent}]")

        # Step 2a: Casual reply (no RAG needed)
        if intent == "casual":
            reply = casual_reply(query, chat_history)
            chat_history.add_user_message(query)
            chat_history.add_ai_message(reply)
            return ResearchResponse(answer=reply, sources=[])

        # Step 2b: Follow-up — reuse existing retriever
        if intent == "followup" and retriever is not None:
            print("   💬 Follow-up — reusing existing knowledge base…")
            answer, docs = answer_question_with_rag(
                question=query, retriever=retriever,
            )
        else:
            # Step 2c: New topic — full YouTube RAG pipeline
            if intent == "followup" and retriever is None:
                print("   ⚠ No previous context, doing a fresh search…")
            print("   🔍 Searching YouTube & building knowledge base…")
            retriever = build_youtube_rag_pipeline(search_query=query)
            answer, docs = answer_question_with_rag(
                question=query, retriever=retriever,
            )

        # Build source items for the frontend
        sources = []
        for doc in (docs or [])[:req.top_k]:
            meta = doc.metadata
            sources.append(
                SourceItem(
                    video_title=meta.get("title", "Unknown"),
                    timestamp=meta.get("timestamp", "00:00"),
                    video_url=meta.get("url_with_timestamp", ""),
                    text=doc.page_content[:300],  # truncated preview
                )
            )

        # Update chat history
        chat_history.add_user_message(query)
        chat_history.add_ai_message(str(answer))

        return ResearchResponse(
            answer=str(answer),
            tokens_used=None,  # Groq SDK doesn't expose this easily
            sources=sources,
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Streaming SSE Endpoint ───────────────────────────────────────────────────
@app.post("/api/research/stream")
async def research_stream(req: ResearchRequest):
    """
    SSE streaming endpoint.
    Events: status → token (many) → sources → done
    """
    global retriever

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def event_stream():
        global retriever
        full_answer = ""

        try:
            # Step 1: Route
            yield f"event: status\ndata: Routing your query…\n\n"
            intent = route_query(query, chat_history)
            print(f"   [router → {intent}]")

            # Step 2a: Casual
            if intent == "casual":
                yield f"event: status\ndata: Generating reply…\n\n"
                for token in stream_casual_reply(query, chat_history):
                    full_answer += token
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
                chat_history.add_user_message(query)
                chat_history.add_ai_message(full_answer)
                yield f"event: sources\ndata: {json.dumps([])}\n\n"
                yield f"event: done\ndata: ok\n\n"
                return

            # Step 2b: Follow-up
            if intent == "followup" and retriever is not None:
                yield f"event: status\ndata: Answering from existing knowledge…\n\n"
                token_gen, docs = stream_answer_with_rag(
                    question=query, retriever=retriever,
                )
            else:
                # Step 2c: New topic — full pipeline
                if intent == "followup" and retriever is None:
                    yield f"event: status\ndata: No previous context, doing a fresh search…\n\n"
                yield f"event: status\ndata: Searching YouTube…\n\n"
                retriever = build_youtube_rag_pipeline(search_query=query)
                yield f"event: status\ndata: Generating answer…\n\n"
                token_gen, docs = stream_answer_with_rag(
                    question=query, retriever=retriever,
                )

            # Stream tokens
            for token in token_gen:
                full_answer += token
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

            # Sources
            sources = []
            for doc in (docs or [])[:req.top_k]:
                meta = doc.metadata
                sources.append({
                    "video_title": meta.get("title", "Unknown"),
                    "timestamp": meta.get("timestamp", "00:00"),
                    "video_url": meta.get("url_with_timestamp", ""),
                    "text": doc.page_content[:300],
                })

            chat_history.add_user_message(query)
            chat_history.add_ai_message(full_answer)

            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
            yield f"event: done\ndata: ok\n\n"

        except Exception as e:
            print(f"❌ Stream error: {e}")
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Serve the frontend ───────────────────────────────────────────────────────
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/")
async def serve_index():
    """Serve the frontend index.html at the root URL."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount any other static assets (CSS, JS, images) if they exist later
# app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Run with: python app.py ──────────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
