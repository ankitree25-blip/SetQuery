"""
Search Tool/API — one leaf of the mega-spec's Section 4 pipeline:

    Query Planner -> Search Query Generation -> Search Tool/API -> Retrieval
    -> Source Evaluation -> Reasoning/Synthesis

This module is only the "Search Tool/API" box: a real Google Custom Search
JSON API call. Nothing in here is mocked -- given a valid key + cx it makes
an actual HTTP request and returns actual results.

Wired into the live path: agent/planner.py's should_search_web() (see
agent/search_decision.py) decides *whether* a query needs one, and
api/main.py's /api/query calls google_custom_search() itself when it does.
Still not built out further (unchanged, genuinely future work): source
relevance evaluation / re-ranking beyond what Google's own ranking gives
back, and a query-rewriting step between the user's raw question and the
search query sent here (currently the same text, verbatim).

Two environment variables, read at call time rather than import time (so
tests can monkeypatch os.environ without reloading the module):

    SATQUERY_GOOGLE_SEARCH_API_KEY   - Google Cloud API key
    SATQUERY_GOOGLE_SEARCH_CX        - Programmable Search Engine ID ("cx")

An API key alone is not enough to search the web with this API. You also
need a "cx" value from a Programmable Search Engine configured to search
the entire web:
    https://programmablesearchengine.google.com/  (create engine, get cx)
    https://developers.google.com/custom-search/v1/overview  (API docs)

Both values are loaded from the repo-root .env file (see backend/api/main.py,
which calls load_dotenv() before any other SatQuery import) rather than
hardcoded here. Neither being set is not a startup error -- see this
module's SearchProviderError and api/main.py's own handling of it: a query
that would have searched instead runs without that extra context, rather
than failing outright over an optional feature.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
from pydantic import BaseModel

_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchProviderError(RuntimeError):
    """Missing config, a non-2xx response, or a malformed response body.

    Raised rather than swallowed -- whether a failed search means "abstain
    on this answer" or "retry" is a planner-level decision (Section 4),
    not this module's to make.
    """


async def google_custom_search(
    query: str,
    *,
    num_results: int = 5,
    api_key: Optional[str] = None,
    cx: Optional[str] = None,
    timeout_s: float = 10.0,
) -> list[SearchResult]:
    """
    Run one Google Custom Search JSON API query.

    Returns up to num_results results (Google caps a single request at 10;
    values outside 1-10 are clamped, not rejected). Raises
    SearchProviderError on missing credentials or a failed request.
    """
    key = api_key or os.environ.get("SATQUERY_GOOGLE_SEARCH_API_KEY")
    engine_id = cx or os.environ.get("SATQUERY_GOOGLE_SEARCH_CX")

    if not key:
        raise SearchProviderError(
            "SATQUERY_GOOGLE_SEARCH_API_KEY is not set. Check the repo-root "
            ".env file."
        )
    if not engine_id:
        raise SearchProviderError(
            "SATQUERY_GOOGLE_SEARCH_CX is not set. Create a Programmable "
            "Search Engine at https://programmablesearchengine.google.com/ "
            "(set it to search the whole web, not one site), then add its "
            "cx value to .env."
        )

    params = {
        "key": key,
        "cx": engine_id,
        "q": query,
        "num": max(1, min(num_results, 10)),
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(_ENDPOINT, params=params)
    except httpx.HTTPError as exc:
        raise SearchProviderError(f"Google Custom Search request failed: {exc}") from exc

    if response.status_code != 200:
        raise SearchProviderError(
            f"Google Custom Search returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    body = response.json()
    items = body.get("items", [])
    return [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
        )
        for item in items
    ]
