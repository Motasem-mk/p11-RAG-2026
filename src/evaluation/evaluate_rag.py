# src/evaluation/evaluate_rag.py
"""
Evaluate the RAG chatbot using an annotated French Q/A dataset.

This script:
- reads data/test/qa_test_set.csv
- loads the FAISS index
- asks each question to the RAG system
- checks whether the generated answer contains expected keywords
- saves detailed results to data/test/evaluation_results.csv

Run:
    python -m src.evaluation.evaluate_rag \
        --qa-file data/test/qa_test_set.csv \
        --index data/index/faiss \
        --output data/test/evaluation_results.csv
"""

import os
import argparse
import math
import unicodedata
from typing import List

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.rag.rag_pipeline import SYSTEM_PROMPT, USER_PROMPT, _format_context


def normalize_text(text: str) -> str:
    """
    Normalize text for simple keyword matching.

    This function:
    - converts text to lowercase
    - removes accents
    - makes keyword comparison easier
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")

    return text


def split_keywords(keyword_string: str) -> List[str]:
    """
    Split expected keywords stored with | separator.

    Example:
        "musique|concert|Paris|date"
        -> ["musique", "concert", "Paris", "date"]
    """

    if not isinstance(keyword_string, str):
        return []

    return [kw.strip() for kw in keyword_string.split("|") if kw.strip()]


def evaluate_keywords(answer: str, expected_keywords: str) -> dict:
    """
    Compare chatbot answer with expected keywords.

    This is a simple automatic evaluation method.
    It does not prove the answer is perfect, but it gives a useful POC score.

    A row passes if at least 50% of expected keywords are found.
    """

    keywords = split_keywords(expected_keywords)

    if not keywords:
        return {
            "matched_keywords": "",
            "missing_keywords": "",
            "match_ratio": 0.0,
            "passed": False,
        }

    answer_norm = normalize_text(answer)

    matched = []
    missing = []

    for kw in keywords:
        kw_norm = normalize_text(kw)

        if kw_norm in answer_norm:
            matched.append(kw)
        else:
            missing.append(kw)

    match_ratio = len(matched) / len(keywords)
    minimum_matches = max(1, math.ceil(len(keywords) * 0.5))
    passed = len(matched) >= minimum_matches

    return {
        "matched_keywords": "|".join(matched),
        "missing_keywords": "|".join(missing),
        "match_ratio": round(match_ratio, 2),
        "passed": passed,
    }


def main():
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(description="Evaluate the OpenAgenda RAG chatbot.")
    parser.add_argument("--qa-file", type=str, default="data/test/qa_test_set.csv", help="Path to annotated Q/A dataset.")
    parser.add_argument("--index", type=str, default="data/index/faiss", help="Path to saved FAISS index.")
    parser.add_argument("--output", type=str, default="data/test/evaluation_results.csv", help="Output CSV file.")
    parser.add_argument("--k", type=int, default=8, help="Number of chunks retrieved from FAISS.")
    parser.add_argument("--model", type=str, default="mistral-small-latest", help="Mistral chat model.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick testing.")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key or api_key.strip().lower() in {"your_mistral_api_key_here", "xxx", "xxxx"}:
        raise SystemExit("MISTRAL_API_KEY is missing. Put it in .env or export it.")

    # Load annotated Q/A dataset
    qa_df = pd.read_csv(args.qa_file, sep=";")

    if args.limit:
        qa_df = qa_df.head(args.limit)

    # Load FAISS index
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    vectordb = FAISS.load_local(
        args.index,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": args.k})

    # Load Mistral chat model
    llm = ChatMistralAI(
        model=args.model,
        temperature=0.1,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    results = []

    print(f"Evaluating {len(qa_df)} questions...\n")

    for _, row in qa_df.iterrows():
        question_id = row["id"]
        question = row["question"]
        expected_answer = row["expected_answer"]
        expected_keywords = row["expected_keywords"]
        category = row["category"]

        print(f"Question {question_id}: {question}")

        # Retrieve relevant documents from FAISS
        docs = retriever.invoke(question)

        if docs:
            context = _format_context(docs)
            messages = prompt.format_messages(question=question, context=context)
            response = llm.invoke(messages)
            answer = response.content
        else:
            answer = "I don't know based on the available event data."

        keyword_eval = evaluate_keywords(answer, expected_keywords)

        results.append(
            {
                "id": question_id,
                "category": category,
                "question": question,
                "expected_answer": expected_answer,
                "expected_keywords": expected_keywords,
                "answer": answer,
                "matched_keywords": keyword_eval["matched_keywords"],
                "missing_keywords": keyword_eval["missing_keywords"],
                "match_ratio": keyword_eval["match_ratio"],
                "passed": keyword_eval["passed"],
                "retrieved_chunks": len(docs),
            }
        )

        print(f"Passed: {keyword_eval['passed']} | Match ratio: {keyword_eval['match_ratio']}")
        print("-" * 80)

    results_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False, sep=";")

    total = len(results_df)
    passed = results_df["passed"].sum()
    score = passed / total if total else 0

    print("\nEvaluation complete.")
    print(f"Passed: {passed}/{total}")
    print(f"Score: {score:.2%}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()