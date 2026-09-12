"""
should_search_web(query) -> (bool, reason)

Section 4's Web Search toggle requires the planner, not a fixed rule, to
decide *whether* a given query needs a search: "It should NOT search
automatically for every query... determine whether external/current
information is actually necessary." There's no real LLM in this system
yet to make that judgment call (see intent.py's own docstring: Part 4's
mock inference engine is still the live path) -- so, exactly like
intent.py's HeuristicFallbackClassifier stands in for a real LLM
classifier, this is a heuristic stand-in for a real LLM-backed search
decision. Same reasoning, same honesty about what it currently is.

Two signals, each with a clear rationale (not a kitchen-sink keyword
dump that would fire on nearly every query, which is exactly the
"searches automatically for every query" behavior Section 4 rules out):

  1. Recency -- the vision model's only source of truth is the pixels in
     front of it, which have no sense of "today". A query that explicitly
     asks about current/recent state needs something the image can't
     supply on its own.
  2. Explain-or-compare -- asking *why* something changed, or whether a
     value is normal/typical, needs an external reference point (a
     baseline, a known cause) the image alone doesn't carry. Section 10's
     own worked example ("Could flooding explain the vegetation
     decrease?") is exactly this shape.

Deliberately NOT a signal here: plain "what/where/how many" questions
about what's visible in the image -- those are fully answerable from the
image evidence alone and are the majority of queries this system
handles; searching for those on every request is the behavior Section 4
explicitly forbids.
"""

from __future__ import annotations

_RECENCY_KW = (
    "current", "currently", "latest", "recent", "recently", "as of today",
    "this year", "this month", "right now", "up to date", "up-to-date",
)

_EXPLAIN_OR_COMPARE_KW = (
    "why", "explain", "could this be", "could this explain", "does this explain",
    "caused by", "cause of", "typical for", "normal for", "average for",
    "compared to", "compare to", "is this normal", "expected for this",
)


def should_search_web(query: str) -> tuple[bool, str]:
    q = query.lower()

    hit = next((kw for kw in _RECENCY_KW if kw in q), None)
    if hit:
        return True, f'query asks about current/recent information ("{hit}")'

    hit = next((kw for kw in _EXPLAIN_OR_COMPARE_KW if kw in q), None)
    if hit:
        return True, f'query asks for an explanation or a comparison against a norm ("{hit}")'

    return False, "no signal that information beyond the image would help"
