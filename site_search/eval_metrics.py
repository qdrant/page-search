"""Pure functions for docs search evals.

No network calls and no Qdrant client live here, so this module is unit-testable
with synthetic data. Everything that touches I/O belongs in site_search/eval.py.
"""

from typing import Iterable, NamedTuple
from urllib.parse import urlparse

VALID_KINDS = ("keyword", "natural", "typo", "navigational")

# Results are always compared at 5: the service returns at most 5 hits
# (SEARCH_LIMIT in rust_search/src/main.rs:31).
CUTOFF = 5

# Seeded from the first measured baseline run, not chosen in advance:
# observed endpoint hit-rate@5, minus 0.10 absolute, rounded down to the
# nearest 0.05. Task 8 of the implementation plan sets this.
HIT_RATE_FLOOR = 0.0


class Metrics(NamedTuple):
    n: int
    hit_rate: float
    mrr: float


def normalize_url(url: str) -> str:
    """Reduce a URL to a comparable path.

    The search API returns relative paths ('/documentation/x/'), but labels may
    be written as absolute URLs and trailing slashes are inconsistent across
    both. Compare paths only, without a trailing slash.
    """
    path = urlparse(url).path or "/"
    return path.rstrip("/") or "/"


def hit_at_5(urls: list[str], primary: str, acceptable: Iterable[str] = ()) -> bool:
    """True if primary or any acceptable URL appears in the first CUTOFF results.

    The parameter is named `urls`, not `result_urls`, to avoid shadowing the
    module-level `result_urls()` parser added in Task 3.
    """
    targets = {normalize_url(primary)}
    targets.update(normalize_url(u) for u in acceptable)
    return any(normalize_url(u) in targets for u in urls[:CUTOFF])


def reciprocal_rank(urls: list[str], primary: str) -> float:
    """1/rank of primary within the first CUTOFF results, else 0.0.

    Ranks against primary only — acceptable URLs count for hit-rate but not for
    MRR, so that 'found a defensible alternative' does not read as 'found the
    best page'. A primary absent from the returned results scores 0 rather than
    a reciprocal of some unknown true rank.
    """
    target = normalize_url(primary)
    for rank, url in enumerate(urls[:CUTOFF], start=1):
        if normalize_url(url) == target:
            return 1.0 / rank
    return 0.0


def aggregate(rows: list[dict]) -> Metrics:
    """Mean hit-rate and MRR over rows carrying 'hit' and 'rr'."""
    if not rows:
        return Metrics(n=0, hit_rate=0.0, mrr=0.0)
    n = len(rows)
    return Metrics(
        n=n,
        hit_rate=sum(1 for r in rows if r["hit"]) / n,
        mrr=sum(r["rr"] for r in rows) / n,
    )


def aggregate_by_kind(rows: list[dict]) -> dict[str, Metrics]:
    """Per-kind aggregates, so a red run says which query type broke."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["kind"], []).append(row)
    return {kind: aggregate(bucket) for kind, bucket in buckets.items()}


def result_urls(payload: dict) -> list[str]:
    """Extract result URLs from an /api/search JSON body, in rank order."""
    return [hit["payload"]["url"] for hit in payload.get("result", [])]


def gate_failures(endpoint: Metrics, zero_result_ids: list[str], floor: float) -> list[str]:
    """Reasons the run should fail the build. Empty list means pass.

    Only endpoint results gate. The dense baseline is diagnostic — it exists to
    attribute a regression, not to cause one — so a dense anomaly reports in the
    summary without failing CI.
    """
    failures = []
    if endpoint.hit_rate < floor:
        failures.append(
            f"endpoint hit-rate@5 {endpoint.hit_rate:.3f} is below floor {floor:.3f}"
        )
    if zero_result_ids:
        failures.append(
            f"{len(zero_result_ids)} case(s) returned zero results: "
            + ", ".join(sorted(zero_result_ids))
        )
    return failures
