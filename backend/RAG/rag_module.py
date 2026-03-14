from typing import List, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse


def _create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


def build_hybrid_retriever(
    docs: List[Document],
    collection_name: str = "youtube1",
):
    """
    Build a Qdrant-backed hybrid (dense + sparse) retriever in-memory,
    following the logic from the notebook.
    """
    embeddings = _create_embeddings()

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name=collection_name,
        location=":memory:",
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10,
            "score_threshold": 0.6,
            "lambda_mult": 0.5,
        },
    )

    return retriever


def _create_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )


def answer_question_with_rag(
    question: str,
    retriever,
):
    """
    Retrieve relevant chunks with the given retriever and answer the question
    with Gemini, mirroring the notebook's simple `llm.invoke(f"{results} ...")`.
    """
    llm = _create_llm()

    docs: List[Document] = retriever.invoke(question)
    response = llm.invoke(f"{docs}     question: '{question}' ")

    return response.text, docs

