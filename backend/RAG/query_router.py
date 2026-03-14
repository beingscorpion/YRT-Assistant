"""
Query Router for YouTube RAG Assistant
=======================================
Classifies user input as 'casual' (greetings, small-talk) or 'rag'
(knowledge-seeking questions that need retrieval). Also handles casual
replies. Uses LangChain's ChatMessageHistory for conversation memory.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# ── LLM (shared across functions) ────────────────────────────────────────────

def _get_llm(temperature: float = 0.2):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
    )


# ── Conversation Memory (last 5 exchanges = 10 messages) ─────────────────────

def get_recent_history(chat_history: ChatMessageHistory, max_turns: int = 5):
    """Return the last `max_turns` user↔assistant exchanges from the history."""
    msgs = chat_history.messages
    # each turn = 2 messages (human + ai), so keep last max_turns*2
    return msgs[-(max_turns * 2):] if len(msgs) > max_turns * 2 else msgs


# ── Query Router ──────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """\
You are a query classifier. Your ONLY job is to output exactly one word:

  casual   — if the user message is a greeting, small-talk, or everyday
              chit-chat (e.g. "hi", "good morning", "how are you",
              "thanks", "bye", "what's up").

  followup — if the user is asking a follow-up question about a topic
              that was ALREADY discussed in the conversation history
              (e.g. "tell me more", "can you explain that again?",
              "what about authentication?", "give me an example",
              "and how does that work?"). This means the existing
              knowledge is enough, NO new YouTube search is needed.

  rag      — if the user message is a NEW knowledge-seeking, technical,
              or informational question on a topic NOT yet discussed
              (e.g. "How to use Supabase with Python?",
              "Explain transformers in NLP"). This requires a FRESH
              YouTube search.

Output ONLY one word: casual  OR  followup  OR  rag
"""


def route_query(user_input: str, chat_history: ChatMessageHistory) -> str:
    """
    Classify `user_input` as 'casual', 'followup', or 'rag'.
    - casual   → direct friendly reply, no RAG
    - followup → answer using existing retriever + memory (no new YouTube search)
    - rag      → full YouTube search + RAG pipeline
    """
    llm = _get_llm(temperature=0.0)

    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]
    messages.extend(get_recent_history(chat_history))
    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages)
    classification = response.content.strip().lower()

    valid = ("casual", "followup", "rag")
    return classification if classification in valid else "rag"


# ── Casual Reply ──────────────────────────────────────────────────────────────

CASUAL_SYSTEM_PROMPT = """\
You are a friendly, helpful AI assistant called YRT (YouTube Research Tool).
Reply to the user's casual message in a warm, conversational tone.
Keep your reply short (1-3 sentences). Be natural and human-like.
"""


def casual_reply(
    user_input: str,
    chat_history: ChatMessageHistory,
) -> str:
    """Generate a friendly conversational reply for casual messages."""
    llm = _get_llm(temperature=0.7)

    messages = [SystemMessage(content=CASUAL_SYSTEM_PROMPT)]
    messages.extend(get_recent_history(chat_history))
    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages)
    return response.content
