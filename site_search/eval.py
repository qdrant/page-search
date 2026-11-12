"""Docs search eval runner.

Measures the live search endpoint against a hand-labeled golden set and gates on
the result. Pure HTTP: no Qdrant client, no credentials, no embedding model, so
it runs anywhere `requests` does.

Relevance only: no latency, no index internals. Querying the `site` collection
directly measured the four-tier ladder at about +15pp over unfiltered vector
search, but that is a one-off finding, not a nightly signal.

Run: python -m site_search.eval
"""

import argparse
import json
import os
import sys

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


def query_endpoint(case: dict, timeout: float = 10.0) -> list[str]:
    """Query the live endpoint. Returns result urls in rank order.

    `timeout` is a safety net against a hung request, not a latency
    measurement: this pipeline reports relevance, not performance.
    """
    params = {
        "q": case["q"],
        "section": case["section"],
        "partition": case["partition"],
    }
    response = requests.get(SEARCH_API, params=params, timeout=timeout)
    response.raise_for_status()
    return result_urls(response.json())


def _score(urls: list[str], case: dict) -> dict:
    return {
        "hit": hit_at_5(urls, case["primary"], case.get("acceptable") or []),
        "rr": reciprocal_rank(urls, case["primary"]),
        "kind": case["kind"],
    }


def run(cases: list[dict]) -> dict:
    """Execute every case against the live endpoint and return the full report."""
    rows, per_case, zero_result_ids = [], [], []

    for case in cases:
        urls = query_endpoint(case)
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
            }
        )

    endpoint = aggregate(rows)
    return {
        "endpoint": endpoint._asdict(),
        "endpoint_by_kind": {k: m._asdict() for k, m in aggregate_by_kind(rows).items()},
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
