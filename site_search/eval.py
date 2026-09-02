"""Docs search eval runner.

Measures the live search endpoint against a hand-labeled golden set and gates on
the result. Pure HTTP: no Qdrant client, no credentials, no embedding model, so
it runs anywhere `requests` does.

An earlier version also queried the `site` collection directly for a
no-ladder baseline. That answered its question once — the four-tier ladder is
worth about +15pp over an unfiltered vector search, concentrated on
natural-language queries — and as a nightly signal it was not worth the
credentials, the qdrant-client dependency, or the two client-path bugs it cost.
Recover it ad hoc if a regression ever needs attributing.

Run: python -m site_search.eval
"""

import argparse
import json
import os
import sys
import time

import requests
import yaml

from site_search.eval_metrics import (
    HIT_RATE_FLOOR,
    aggregate,
    aggregate_by_kind,
    gate_failures,
    hit_at_5,
    reciprocal_rank,
    render_markdown,
    result_urls,
)

SEARCH_API = "https://search.qdrant.tech/api/search"
CASES_PATH = os.path.join(os.path.dirname(__file__), "eval_cases.yaml")


def load_cases(path: str | None = None) -> list[dict]:
    with open(path or CASES_PATH) as f:
        return yaml.safe_load(f)


def query_endpoint(case: dict, timeout: float = 10.0) -> tuple[list[str], float]:
    """Query the live endpoint. Returns (result urls in rank order, seconds)."""
    params = {
        "q": case["q"],
        "section": case["section"],
        "partition": case["partition"],
    }
    started = time.perf_counter()
    response = requests.get(SEARCH_API, params=params, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return result_urls(response.json()), elapsed


def _score(urls: list[str], case: dict) -> dict:
    return {
        "hit": hit_at_5(urls, case["primary"], case.get("acceptable") or []),
        "rr": reciprocal_rank(urls, case["primary"]),
        "kind": case["kind"],
    }


def run(cases: list[dict]) -> dict:
    """Execute every case against the live endpoint and return the full report."""
    rows, per_case, latencies, zero_result_ids = [], [], [], []

    for case in cases:
        urls, elapsed = query_endpoint(case)
        latencies.append(elapsed)
        if not urls:
            zero_result_ids.append(case["id"])

        score = _score(urls, case)
        rows.append(score)
        per_case.append(
            {
                "id": case["id"],
                "q": case["q"],
                "kind": case["kind"],
                "primary": case["primary"],
                "endpoint_hit": score["hit"],
                "endpoint_rr": round(score["rr"], 4),
                "endpoint_urls": urls,
                "latency_ms": round(elapsed * 1000),
            }
        )

    endpoint = aggregate(rows)
    return {
        "endpoint": endpoint._asdict(),
        "endpoint_by_kind": {k: m._asdict() for k, m in aggregate_by_kind(rows).items()},
        "max_latency_ms": round(max(latencies) * 1000) if latencies else 0,
        "zero_result_ids": zero_result_ids,
        "floor": HIT_RATE_FLOOR,
        "failures": gate_failures(endpoint, zero_result_ids, HIT_RATE_FLOOR),
        "cases": per_case,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run docs search evals.")
    parser.add_argument("--summary", help="path to append a markdown summary to")
    parser.add_argument("--json", dest="json_path", help="path to write the JSON report to")
    args = parser.parse_args(argv)

    report = run(load_cases())
    markdown = render_markdown(report)
    print(markdown)

    if args.summary:
        with open(args.summary, "a") as f:
            f.write(markdown + "\n")
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2)

    for failure in report["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
