"""Pure functions for docs search evals.

No network calls and no Qdrant client live here, so this module is unit-testable
with synthetic data. Everything that touches I/O belongs in site_search/eval.py.
"""

import os
from typing import Iterable, NamedTuple
from urllib.parse import urlparse

VALID_KINDS = ("keyword", "natural", "typo", "navigational")

# Results are always compared at 5: the service returns at most 5 hits
# (SEARCH_LIMIT in rust_search/src/main.rs:31).
CUTOFF = 5

# Seeded from the first measured baseline run, not chosen in advance:
# observed endpoint hit-rate@5, minus 0.10 absolute, rounded down to the
# nearest 0.05.
#
# Baseline 2026-09-02, 40 cases against search.qdrant.tech:
#   overall      hit-rate@5 0.800   MRR@5 0.467
#   keyword      0.917 (12)
#   typo         0.833 (6)
#   natural      0.786 (14)
#   navigational 0.625 (8)   <- weakest bucket
#
# 0.800 - 0.10 = 0.700, which is already a multiple of 0.05.
HIT_RATE_FLOOR = 0.70


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


def _run_url() -> str | None:
    """Link back to the workflow run, when running inside GitHub Actions."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def render_markdown(report: dict) -> str:
    endpoint = report["endpoint"]
    dense = report.get("dense")
    passed = not report["failures"]

    # Comment stickiness is handled by the sticky-pull-request-comment action's
    # own hidden header, so no marker is needed in this body.
    lines = [
        "## Docs search eval",
        "",
        "`endpoint` is the live service. `no ladder` is the **same query vector on "
        "the same `site` collection**, filtered only by section and partition — no "
        "`tag` filter, no full-text `text` condition, no four-tier priority. The gap "
        "between the rows is what the ladder's filters buy, not a different engine.",
        "",
        f"{'✅ PASS' if passed else '❌ FAIL'} — endpoint hit-rate@5 "
        f"**{endpoint['hit_rate']:.3f}** vs floor {report['floor']:.3f}",
        "",
    ]

    run_url = _run_url()
    if run_url:
        lines += [f"[workflow run]({run_url})", ""]

    dense_row = (
        f"| no ladder | {dense['n']} | {dense['hit_rate']:.3f} | {dense['mrr']:.3f} |"
        if dense
        else "| no ladder | — | _not measured (no cluster credentials)_ | — |"
    )
    lines += [
        "| layer | n | hit-rate@5 | MRR@5 |",
        "| --- | --- | --- | --- |",
        f"| endpoint | {endpoint['n']} | {endpoint['hit_rate']:.3f} | {endpoint['mrr']:.3f} |",
        dense_row,
        "",
        "| kind | n | endpoint hit-rate@5 | endpoint MRR@5 | no-ladder hit-rate@5 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind in sorted(report["endpoint_by_kind"]):
        e = report["endpoint_by_kind"][kind]
        by_kind = report.get("dense_by_kind") or {}
        d = by_kind.get(kind)
        dense_cell = f"{d['hit_rate']:.3f}" if d else "—"
        lines.append(
            f"| {kind} | {e['n']} | {e['hit_rate']:.3f} | {e['mrr']:.3f} | {dense_cell} |"
        )

    corpus = report.get("corpus_size")
    lines += [
        "",
        f"corpus: {f'{corpus} points' if corpus is not None else 'not measured'} · "
        f"max latency: {report['max_latency_ms']} ms · "
        f"floor: {report['floor']:.3f}",
        "",
    ]

    misses = [c for c in report["cases"] if not c["endpoint_hit"]]
    if misses:
        lines += ["### Endpoint misses", "", "| id | query | expected |", "| --- | --- | --- |"]
        lines += [f"| {c['id']} | `{c['q']}` | {c['primary']} |" for c in misses]
        lines.append("")

    if report["failures"]:
        lines += ["### Failures", ""] + [f"- {f}" for f in report["failures"]] + [""]

    return "\n".join(lines)
