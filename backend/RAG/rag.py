from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from youtube_module import search_youtube_videos, fetch_video_transcripts, build_docs_from_transcripts
from rag_module import build_hybrid_retriever, answer_question_with_rag
from query_router import route_query, casual_reply

def build_youtube_rag_pipeline(
    search_query: str,
    max_results: int = 2,
    published_after: str | None = "2021-01-01T00:00:00Z",
):
    """
    High-level helper that:
    - searches YouTube
    - fetches transcripts
    - chunks them into LangChain Documents
    - builds a hybrid dense+sparse retriever
    """
    videos_metadata = search_youtube_videos(
        query=search_query,
        max_results=max_results,
        published_after=published_after,
    )

    videos_with_transcripts = fetch_video_transcripts(videos_metadata)
    docs = build_docs_from_transcripts(videos_with_transcripts)

    retriever = build_hybrid_retriever(docs)
    return retriever


def main():
    load_dotenv()

    # LangChain chat history — windowed to last 5 turns by query_router
    chat_history = ChatMessageHistory()
    retriever = None  # will be built on the first RAG query

    print("=" * 55)
    print("  YRT — YouTube Research Tool")
    print("  Ask me anything! (type 'quit' or 'exit' to stop)")
    print("=" * 55)

    while True:
        try:
            user_input = input("\n🟢 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! 👋")
            break

        # ── Step 1: Route the query ──────────────────────────────
        intent = route_query(user_input, chat_history)
        print(f"   [router → {intent}]")

        # ── Step 2a: Casual reply (no RAG needed) ────────────────
        if intent == "casual":
            reply = casual_reply(user_input, chat_history)
            print(f"\n🤖 YRT: {reply}")
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(reply)
            continue

        # ── Step 2a: Follow-up — reuse existing retriever ─────────
        if intent == "followup" and retriever is not None:
            print("   💬 Follow-up detected — reusing existing knowledge base...")
            answer, docs = answer_question_with_rag(
                question=user_input, retriever=retriever
            )
        else:
            # ── Step 2b: New topic — full YouTube RAG pipeline ────
            if intent == "followup" and retriever is None:
                print("   ⚠ No previous context found, doing a fresh search...")
            print("   🔍 Searching YouTube & building knowledge base...")
            retriever = build_youtube_rag_pipeline(search_query=user_input)
            answer, docs = answer_question_with_rag(
                question=user_input, retriever=retriever
            )

        print(f"\n🤖 YRT: {answer}")

        if docs:
            print("\n📚 Sources (video + timestamp):")
            for i, doc in enumerate(docs[:3], start=1):
                title     = doc.metadata.get("title", "Unknown")
                timestamp = doc.metadata.get("timestamp", "00:00")
                url       = doc.metadata.get("url_with_timestamp", "")
                print(f"   {i}. 🎬 {title}")
                print(f"      ⏱  Timestamp: {timestamp}")
                print(f"      🔗 {url}")

        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(str(answer))


if __name__ == "__main__":
    main()