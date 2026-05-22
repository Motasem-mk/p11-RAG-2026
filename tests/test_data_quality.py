# tests/test_data_quality.py
"""
Data quality tests for the OpenAgenda RAG project.

These tests check that:
- the preprocessing step returns a valid clean dataset
- the dataset respects the selected city
- the events are recent / active / upcoming
- the final schema is correct
- the FAISS index files exist
- the FAISS index can be loaded

Run:
    pytest -q
"""

import os
from datetime import timedelta

import pandas as pd
import pytest
from dotenv import load_dotenv, find_dotenv

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.data.preprocess_openagenda import preprocess_events


# ------------------------------------------------------------
# Test configuration
# ------------------------------------------------------------

TEST_CITY = os.getenv("TEST_CITY", "Paris")
TEST_MAX_RECORDS = int(os.getenv("TEST_MAX_RECORDS", "50"))
TEST_INDEX_PATH = os.getenv("TEST_INDEX_PATH", "data/index/faiss")

EXPECTED_COLUMNS = [
    "uid",
    "title",
    "city",
    "postal_code",
    "venue",
    "url",
    "tags",
    "start_utc",
    "end_utc",
    "text",
]


# ------------------------------------------------------------
# Fixture: load a small clean dataset once for all tests
# ------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_events_df():
    """
    Fetch and preprocess a small sample of OpenAgenda events.

    We use a small number of records to keep the tests fast.
    """

    df = preprocess_events(
        city=TEST_CITY,
        max_records=TEST_MAX_RECORDS,
    )

    return df


# ------------------------------------------------------------
# 1) Dataset existence and schema tests
# ------------------------------------------------------------

def test_clean_dataset_is_not_empty(clean_events_df):
    """The preprocessing step should return at least one valid event."""

    assert not clean_events_df.empty, "The clean dataset is empty."


def test_final_schema_is_correct(clean_events_df):
    """The clean DataFrame should match the final expected schema."""

    assert clean_events_df.columns.tolist() == EXPECTED_COLUMNS


def test_internal_filter_column_is_not_returned(clean_events_df):
    """
    last_start_utc is used internally for filtering only.
    It should not appear in the final clean DataFrame.
    """

    assert "last_start_utc" not in clean_events_df.columns


# ------------------------------------------------------------
# 2) City and content quality tests
# ------------------------------------------------------------

def test_events_are_from_selected_city(clean_events_df):
    """All events should belong to the selected city."""

    assert (
        clean_events_df["city"].str.lower() == TEST_CITY.lower()
    ).all(), f"Some events are outside {TEST_CITY}."


def test_required_fields_are_not_empty(clean_events_df):
    """Important RAG fields should not be empty."""

    assert clean_events_df["uid"].notna().all(), "Some events have no uid."

    assert (
        clean_events_df["title"].str.strip().str.len() > 0
    ).all(), "Some events have empty titles."

    assert (
        clean_events_df["text"].str.strip().str.len() > 0
    ).all(), "Some events have empty descriptions."


def test_dates_are_valid(clean_events_df):
    """Start and end dates should be valid timestamps."""

    assert clean_events_df["start_utc"].notna().all(), "Some events have invalid start dates."
    assert clean_events_df["end_utc"].notna().all(), "Some events have invalid end dates."


def test_events_are_recent_active_or_upcoming(clean_events_df):
    """
    Events should be recent, active, or upcoming.

    Important:
    The preprocessing script uses lastdate_begin internally to keep recurring events.
    Since last_start_utc is not part of the final schema, we validate the final dataset
    using end_utc.

    This means the event should not have ended more than 365 days ago.
    """

    cutoff_date = pd.Timestamp.now(tz="Europe/Paris").date() - timedelta(days=365)
    cutoff = pd.Timestamp(cutoff_date, tz="Europe/Paris").tz_convert("UTC")

    assert (
        clean_events_df["end_utc"] >= cutoff
    ).all(), "Some events ended more than one year ago."


# ------------------------------------------------------------
# 3) FAISS index tests
# ------------------------------------------------------------

def test_faiss_index_files_exist():
    """
    The FAISS index should exist after running build_faiss.py.

    If this test fails, run:
        python -m src.index.build_faiss --city Paris --max-records 300 --index-out data/index/faiss
    """

    faiss_file = os.path.join(TEST_INDEX_PATH, "index.faiss")
    pkl_file = os.path.join(TEST_INDEX_PATH, "index.pkl")

    assert os.path.exists(faiss_file), f"Missing FAISS file: {faiss_file}"
    assert os.path.exists(pkl_file), f"Missing PKL file: {pkl_file}"


def test_faiss_index_can_be_loaded():
    """
    The saved FAISS index should be loadable by LangChain.

    This test requires MISTRAL_API_KEY only to instantiate the same embedding class.
    It does not rebuild the index.
    """

    load_dotenv(find_dotenv(usecwd=True))

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        pytest.skip("MISTRAL_API_KEY is missing, skipping FAISS load test.")

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key,
    )

    vectordb = FAISS.load_local(
        TEST_INDEX_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    assert vectordb.index.ntotal > 0, "The FAISS index contains no vectors."