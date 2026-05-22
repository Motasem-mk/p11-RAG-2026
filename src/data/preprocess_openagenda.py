# src/data/preprocess_openagenda.py
"""
Fetch and preprocess OpenAgenda events from OpenDataSoft.

This module prepares OpenAgenda event data for the RAG pipeline.

The mission requires recent cultural events less than one year old.
For this reason, the script filters events using lastdate_begin internally.

Why lastdate_begin?
Some events are recurring or spread across multiple dates.
For those events:
- firstdate_begin may be old;
- lastdate_begin may still be recent or upcoming.

Therefore, lastdate_begin is used internally to decide whether an event
is still relevant for the project.

Final output schema:
uid, title, city, postal_code, venue, url,
tags, start_utc, end_utc, text

Important:
last_start_utc is used only internally for filtering and is not returned
in the final DataFrame.

Example usage:
    from src.data.preprocess_openagenda import preprocess_events

    df_events_clean = preprocess_events(city="Paris", max_records=9000)
"""

import math
import re
from datetime import timedelta
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
from bs4 import BeautifulSoup


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ODATASET = "evenements-publics-openagenda"

API_URL = (
    f"https://public.opendatasoft.com/api/explore/v2.1/catalog/"
    f"datasets/{ODATASET}/records"
)

DEFAULT_ROWS = 100

# Project rule:
# Keep recent events less than one year old.
RECENCY_DAYS = 365


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _escape_api_string(value: str) -> str:
    """
    Escape double quotes in string values used inside the API where clause.

    Parameters
    ----------
    value:
        String value to escape.

    Returns
    -------
    str
        Escaped string value.
    """

    return value.replace('"', '\\"')


def _fetch_page(
    city: Optional[str],
    rows: int,
    offset: int
) -> Dict[str, Any]:
    """
    Fetch one page of OpenAgenda events using OpenDataSoft API v2.1.

    The API query keeps events whose last known starting date is within
    the last 365 days or in the future.

    Parameters
    ----------
    city:
        Optional city filter.

    rows:
        Number of records to fetch for this page.

    offset:
        Pagination offset.

    Returns
    -------
    dict
        JSON response from the OpenDataSoft API.
    """

    cutoff_date = pd.Timestamp.now(tz="Europe/Paris").date() - timedelta(days=RECENCY_DAYS)

    where_clause = f"lastdate_begin >= date'{cutoff_date.isoformat()}'"

    if city:
        city_safe = _escape_api_string(city)
        where_clause += f' AND location_city = "{city_safe}"'

    params = {
        "where": where_clause,
        "limit": rows,
        "offset": offset,
        "order_by": "firstdate_begin ASC",
        "timezone": "Europe/Paris",
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def _records_to_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw OpenAgenda API records into simplified rows.

    The temporary field last_start_utc is used only for filtering.
    It is removed from the final returned DataFrame.

    Parameters
    ----------
    records:
        List of raw API records.

    Returns
    -------
    list of dict
        Simplified rows ready for cleaning.
    """

    rows = []

    for rec in records:
        rows.append(
            {
                "uid": rec.get("uid"),
                "title": rec.get("title_fr") or "",
                "city": rec.get("location_city") or "",
                "postal_code": rec.get("location_postalcode") or "",
                "venue": rec.get("location_name") or "",
                "url": rec.get("canonicalurl") or "",
                "tags": rec.get("keywords_fr") or [],

                # Public final schema fields
                "start_utc": rec.get("firstdate_begin"),
                "end_utc": rec.get("lastdate_end") or rec.get("firstdate_end"),

                # Internal filtering field only
                "last_start_utc": rec.get("lastdate_begin") or rec.get("firstdate_begin"),

                # Prefer long description; fallback to short description
                "text": rec.get("longdescription_fr") or rec.get("description_fr") or "",
            }
        )

    return rows


def _strip_html(value: str) -> str:
    """
    Convert HTML text into clean plain text.

    Parameters
    ----------
    value:
        Raw HTML or plain text.

    Returns
    -------
    str
        Clean plain text.
    """

    if not isinstance(value, str):
        return ""

    text = BeautifulSoup(value, "html.parser").get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------------
# Public preprocessing function
# ------------------------------------------------------------

def preprocess_events(
    city: Optional[str] = None,
    max_records: int = 9000,
    rows: int = DEFAULT_ROWS,
) -> pd.DataFrame:
    """
    Fetch and clean OpenAgenda events.

    Parameters
    ----------
    city:
        Optional city filter.
        If city is None, events are fetched without a city filter.
        If city is provided, only events from that city are kept.

    max_records:
        Maximum number of raw records to fetch from the API.

    rows:
        Number of records per API request.
        The OpenDataSoft API v2.1 limit is 100 records per request.

    Recency rule
    ------------
    Events are kept if last_start_utc >= today - 365 days.

    Final output schema
    -------------------
    uid, title, city, postal_code, venue, url,
    tags, start_utc, end_utc, text

    Returns
    -------
    pandas.DataFrame
        Clean event dataset ready for LangChain document creation
        and FAISS indexing.
    """

    rows = min(rows, 100)
    pages = math.ceil(max_records / rows)

    all_rows = []

    for page in range(pages):
        offset = page * rows

        remaining = max_records - len(all_rows)

        if remaining <= 0:
            break

        current_limit = min(rows, remaining)

        data = _fetch_page(
            city=city,
            rows=current_limit,
            offset=offset,
        )

        records = data.get("results", [])

        if not records:
            break

        all_rows.extend(_records_to_rows(records))

    df = pd.DataFrame(all_rows)

    final_columns = [
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

    working_columns = final_columns + ["last_start_utc"]

    if df.empty:
        return pd.DataFrame(columns=final_columns)

    # Ensure a stable schema even if some fields are missing from the API response
    for col in working_columns:
        if col not in df.columns:
            df[col] = None

    # Parse dates to UTC
    df["start_utc"] = pd.to_datetime(df["start_utc"], errors="coerce", utc=True)
    df["end_utc"] = pd.to_datetime(df["end_utc"], errors="coerce", utc=True)
    df["last_start_utc"] = pd.to_datetime(df["last_start_utc"], errors="coerce", utc=True)

    # Clean text columns
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["city"] = df["city"].fillna("").astype(str).str.strip()
    df["venue"] = df["venue"].fillna("").astype(str).str.strip()
    df["postal_code"] = df["postal_code"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].apply(_strip_html)

    # Normalize tags
    df["tags"] = df["tags"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else (x or "")
    )

    # Same cutoff logic as the API query
    cutoff_date = pd.Timestamp.now(tz="Europe/Paris").date() - timedelta(days=RECENCY_DAYS)
    cutoff = pd.Timestamp(cutoff_date, tz="Europe/Paris").tz_convert("UTC")

    # Keep only valid recent/upcoming events
    df = df[
        df["uid"].notna()
        & (df["title"].str.len() > 0)
        & df["start_utc"].notna()
        & df["end_utc"].notna()
        & df["last_start_utc"].notna()
        & (df["last_start_utc"] >= cutoff)
        & (df["text"].str.strip().str.len() > 0)
    ].copy()

    # Optional city validation after fetching
    if city:
        df = df[df["city"].str.lower() == city.lower()].copy()

    # Remove duplicate events
    df = df.drop_duplicates(subset=["uid"]).reset_index(drop=True)

    # Return only the final public schema
    return df[final_columns].reset_index(drop=True)