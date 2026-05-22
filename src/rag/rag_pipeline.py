# src/rag/rag_pipeline.py
"""
Simple RAG CLI for cultural event recommendations.

This script:
- loads the FAISS index created by build_faiss.py
- retrieves relevant event chunks
- sends the retrieved context to Mistral
- answers only from the retrieved context

Run:
    python -m src.rag.rag_pipeline --index data/index/faiss --k 8

Type "exit" or "quit" to stop the chatbot.
"""

import os
import argparse
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS


# System instruction:
# This prevents the chatbot from answering using general knowledge.
SYSTEM_PROMPT = (
    "You are a cultural events assistant for Puls-Events.\n"
    "Use ONLY the provided CONTEXT to answer.\n"
    "If the answer is not present in the CONTEXT, say: \"I don't know based on the available event data.\"\n"
    "Do not invent events, dates, venues, prices, or links.\n"
    "Reply in the same language as the user's question."
)


# Human prompt:
# The user question and retrieved FAISS context are inserted here.
USER_PROMPT = (
    "Question:\n"
    "{question}\n\n"
    "CONTEXT:\n"
    "{context}\n\n"
    "Answer with facts from the CONTEXT. "
    "When possible, include event titles, dates, venues, cities, and URLs."
)


def _format_context(docs) -> str:
    """
    Convert retrieved FAISS documents into readable context for the LLM.

    Each retrieved document contains:
    - metadata: title, city, venue, dates, url
    - page_content: embedded event text
    """

    chunks: List[str] = []

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}

        # Keep only a limited text snippet to avoid sending too much context
        snippet = doc.page_content[:800]

        chunks.append(
            f"[Event {i}]\n"
            f"Title: {meta.get('title', '')}\n"
            f"City: {meta.get('city', '')}\n"
            f"PostalCode: {meta.get('postal_code', '')}\n"
            f"Venue: {meta.get('venue', '')}\n"
            f"Start: {meta.get('start_utc', '')}\n"
            f"End: {meta.get('end_utc', '')}\n"
            f"URL: {meta.get('url', '')}\n"
            f"Content:\n{snippet}"
        )

    return "\n\n---\n\n".join(chunks)


def main():
    """
    Run the RAG chatbot from the command line.
    """

    # Load MISTRAL_API_KEY from .env
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(description="Run a simple RAG chatbot over the OpenAgenda FAISS index.")
    parser.add_argument("--index", type=str, default="data/index/faiss", help="Path to the saved FAISS index.")
    parser.add_argument("--k", type=int, default=8, help="Number of chunks to retrieve from FAISS.")
    parser.add_argument("--model", type=str, default="mistral-small-latest", help="Mistral chat model.")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key.strip().lower() in {"your_mistral_api_key_here", "xxx", "xxxx"}:
        raise SystemExit("MISTRAL_API_KEY is missing. Put it in .env or export it.")

    # The embedding model must match the one used when building the FAISS index.
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    # Load the local FAISS index.
    # allow_dangerous_deserialization=True is required by LangChain for local FAISS loading.
    # Use it only for indexes you created yourself.
    vectordb = FAISS.load_local(
        args.index,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # Convert FAISS into a retriever.
    retriever = vectordb.as_retriever(search_kwargs={"k": args.k})

    # Create the Mistral chat model.
    llm = ChatMistralAI(
        model=args.model,
        temperature=0.1,
        api_key=api_key,
    )

    # Create the prompt template.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    print("\nRAG chatbot ready.")
    print("Ask a question about cultural events.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        if not question:
            continue

        # Step 1: retrieve relevant chunks from FAISS
        docs = retriever.invoke(question)

        if not docs:
            print("\nBot: I don't know based on the available event data.\n")
            continue

        # Step 2: format retrieved chunks as context
        context = _format_context(docs)

        # Step 3: send question + context to Mistral
        messages = prompt.format_messages(question=question, context=context)
        response = llm.invoke(messages)

        print(f"\nBot: {response.content}\n")


if __name__ == "__main__":
    main()
