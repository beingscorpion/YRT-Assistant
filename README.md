## YT Research Assistant (YouTube RAG)

An end‑to‑end **YouTube research assistant** that turns natural‑language questions into:

- **Optimized YouTube searches**
- **Multi‑video transcript retrieval**
- **Hybrid dense + sparse RAG over all videos**
- **FastAPI backend + single‑page frontend** with streaming answers and source attributions.

This project is designed to **improve over typical “one‑video + manual search” YouTube usage** by automatically finding multiple high‑quality videos, chunking their transcripts, and answering questions with a Retrieval‑Augmented Generation (RAG) pipeline.

---

### Tech Stack

- **Backend**
  - `FastAPI` API (`backend/app.py`)
  - Query routing and casual chat vs RAG (`backend/RAG/query_router.py`)
  - YouTube search + transcripts + chunking (`backend/RAG/youtube_module.py`)
  - Hybrid retriever with Qdrant (in‑memory) + HuggingFace embeddings + FastEmbed sparse (`backend/RAG/rag_module.py`)
  - Groq LLMs (`ChatGroq`) for:
    - Search‑query optimization
    - Query routing (casual / followup / rag)
    - Final answer generation (including streaming)

- **Frontend**
  - Single‑page app (`frontend/index.html`) served by FastAPI
  - Calls `/api/research` and `/api/research/stream`
  - Shows:
    - Streaming answer tokens
    - Ranked source snippets with timestamps and deep links into each YouTube video

---

### Key Features

- **Natural language → optimized YouTube search**
  - User can type free‑form questions like “can u explain transformers in NLP in detail”.
  - Backend runs a small **query optimizer LLM** to strip filler and produce a focused YouTube query such as `transformer architecture deep learning`.

- **Multi‑video RAG instead of a single video**
  - Searches YouTube for several relevant videos (configurable `max_results`).
  - Pulls and normalizes transcripts (manual, auto‑generated, or translated).
  - Chunks transcripts into `Document` objects with rich metadata:
    - Title, channel, publish date
    - Human‑readable timestamp
    - `url_with_timestamp` for direct deep links
  - Builds an **in‑memory hybrid retriever**:
    - Dense: `BAAI/bge-base-en-v1.5` embeddings
    - Sparse: `Qdrant/bm25` via `FastEmbedSparse`
    - Retrieval mode: `RetrievalMode.HYBRID` + MMR (`k`, `fetch_k`, `lambda_mult`).

- **Query router + conversation memory**
  - Classifies each message as:
    - `casual` – friendly small‑talk, no RAG
    - `followup` – reuse existing retriever, no new YouTube search
    - `rag` – new topic, run full YouTube search + retriever build
  - Uses `ChatMessageHistory` so follow‑up questions are answered from the **same multi‑video knowledge base** when possible.

- **Streaming answers with sources**
  - `/api/research` – JSON response with answer + structured `sources`.
  - `/api/research/stream` – SSE stream:
    - `status` → `token` (many) → `sources` → `done`.
  - Frontend progressively renders the answer and then shows the linked snippets.

---

### Project Structure (high level)

- `backend/`
  - `app.py` – FastAPI app, endpoints, query optimizer, and pipeline orchestration.
  - `RAG/`
    - `youtube_module.py` – YouTube search, transcripts, and transcript→docs.
    - `rag_module.py` – Hybrid retriever and RAG answer functions.
    - `query_router.py` – Intent classification + casual replies + streaming.
    - `test_ragas.py` – RAG evaluation / experiments (optional, for development).
- `frontend/`
  - `index.html` – Single‑page UI (search box, streaming answer, source list).

---

### How It Works (End‑to‑End Flow)

1. **User asks a question** in the frontend.
2. Frontend calls **`POST /api/research`** (or `/api/research/stream` for streaming).
3. Backend:
   - **Routes** the query (`casual` / `followup` / `rag`).
   - If `casual` → generate a short friendly reply (no RAG).
   - If `followup` with an existing retriever → answer directly from the same in‑memory knowledge base.
   - If `rag` (or `followup` with no retriever yet):
     - Optimizes the query for YouTube search.
     - Searches YouTube and fetches transcripts for several videos.
     - Converts transcripts into timestamped `Document` chunks.
     - Builds a hybrid (dense + sparse) retriever in memory.
     - Retrieves the top chunks and asks the LLM to answer using them.
4. Backend returns:
   - Final **answer text**
   - A list of **source chunks** (`video_title`, `timestamp`, `video_url`, snippet text).
5. Frontend displays the answer and sources; each source opens the corresponding video at the right timestamp.

---

### Running the Project Locally

1. **Clone the repo**

```bash
git clone <your-repo-url>.git
cd yra
