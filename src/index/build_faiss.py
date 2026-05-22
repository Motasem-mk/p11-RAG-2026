# src/index/build_faiss.py
"""
Build and persist a FAISS index from preprocessed OpenAgenda events.

This script is the vectorization/indexing step of the RAG pipeline.

It:
- fetches and preprocesses events using preprocess_events()
- converts each event into a LangChain Document
- adds date tokens such as YearStart and MonthStart to improve date-based retrieval
- splits long documents into smaller chunks
- creates embeddings with Mistral
- stores the embeddings in a FAISS index
- saves the FAISS index locally

Run:
    python -m src.index.build_faiss --city Paris --max-records 9000 \
        --chunk-size 800 --chunk-overlap 120 \
        --index-out data/index/faiss
"""

from __future__ import annotations

import os
import argparse
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.data.preprocess_openagenda import preprocess_events


def _norm_parts(ts) -> Tuple[str, str]:
    """
    Extract year and month from a timestamp.

    Example:
        2026-05-20 10:00:00+00:00
        -> ("2026", "05")

    These values are added to the embedded text to help the retriever
    understand date-related questions such as:
        "events in May"
        "events in 2026"
        "events this month"
    """

    if ts is None or pd.isna(ts):
        return "", ""

    try:
        return str(ts.year), f"{ts.month:02d}"
    except Exception:
        return "", ""


def _date_to_str(ts) -> str:
    """
    Convert a pandas timestamp to a string.

    FAISS metadata is saved to disk.
    For this reason, dates should be stored as simple strings instead
    of pandas Timestamp objects.
    """

    if ts is None or pd.isna(ts):
        return ""

    try:
        return ts.isoformat()
    except Exception:
        return str(ts)


def _df_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert the clean events DataFrame into LangChain Document objects.

    Each row/event becomes one Document.

    A Document has two parts:
    1. page_content:
       The text that will be embedded and searched semantically.

    2. metadata:
       Structured information kept with the document.
       This metadata is useful later when the chatbot displays the title,
       city, venue, date, or URL of the retrieved event.
    """

    docs: List[Document] = []

    for _, r in df.iterrows():
        # Extract normalized year/month from start and end dates
        y_s, m_s = _norm_parts(r.get("start_utc"))
        y_e, m_e = _norm_parts(r.get("end_utc"))

        # This is the text that will be transformed into embeddings.
        # We include title, city, venue, dates, tags, URL, and description
        # because users may search using any of these elements.
        page = (
            f"Title: {r.get('title', '')}\n"
            f"City: {r.get('city', '')}\n"
            f"PostalCode: {r.get('postal_code', '')}\n"
            f"Venue: {r.get('venue', '')}\n"
            f"Start: {_date_to_str(r.get('start_utc'))}\n"
            f"End: {_date_to_str(r.get('end_utc'))}\n"
            f"YearStart: {y_s}\n"
            f"MonthStart: {m_s}\n"
            f"YearEnd: {y_e}\n"
            f"MonthEnd: {m_e}\n"
            f"Tags: {r.get('tags', '')}\n"
            f"URL: {r.get('url', '')}\n\n"
            f"{r.get('text', '')}"
        )

        # Metadata is not mainly used for semantic search.
        # It is kept so the chatbot can later show structured information
        # about the retrieved event.
        meta = {
            "uid": r.get("uid", ""),
            "title": r.get("title", ""),
            "city": r.get("city", ""),
            "postal_code": r.get("postal_code", ""),
            "venue": r.get("venue", ""),
            "url": r.get("url", ""),
            "start_utc": _date_to_str(r.get("start_utc")),
            "end_utc": _date_to_str(r.get("end_utc")),
            "tags": r.get("tags", ""),
        }

        docs.append(Document(page_content=page, metadata=meta))

    return docs


def main():
    """
    Main execution function.

    Complete flow:
        clean OpenAgenda events
        -> convert them to Documents
        -> split them into chunks
        -> embed the chunks with Mistral
        -> build and save the FAISS index
    """

    # Load environment variables from .env
    load_dotenv()

    # Command-line options
    parser = argparse.ArgumentParser(description="Build a FAISS index from OpenAgenda events.")
    parser.add_argument("--city", type=str, default="Paris", help="City used to filter OpenAgenda events.")
    parser.add_argument("--max-records", type=int, default=9000, help="Maximum number of raw records to fetch.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Maximum size of each text chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Overlap between text chunks.")
    parser.add_argument("--index-out", type=str, default="data/index/faiss", help="Output folder for the FAISS index.")
    args = parser.parse_args()

    # Check that the Mistral API key exists before calling the embedding model
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY is missing. Put it in .env or export it.")

    # Step 1: fetch and clean the OpenAgenda data
    print(f"Fetching and preprocessing events for city={args.city} ...")
    df = preprocess_events(city=args.city, max_records=args.max_records)

    if df.empty:
        raise SystemExit("No events found after preprocessing. Try another city or increase max-records.")

    # Step 2: convert each event row into a LangChain Document
    print("Converting events to Documents ...")
    docs = _df_to_documents(df)

    # Step 3: split long documents into smaller chunks
    # This helps the embedding model and retriever work with smaller text units.
    print("Splitting Documents into chunks ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(docs)

    # Step 4: create the Mistral embedding model
    print("Embedding chunks and building FAISS index ...")
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    # Step 5: build the FAISS vector database from embedded chunks
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)

    # Step 6: save the FAISS index locally
    os.makedirs(args.index_out, exist_ok=True)
    vectordb.save_local(args.index_out)

    print(f"Saved FAISS index to: {args.index_out} (chunks={len(chunks)}, docs={len(docs)}, rows={len(df)})")


if __name__ == "__main__":
    main()
