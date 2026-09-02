"""Docs search eval runner.

Measures the live search endpoint against a hand-labeled golden set, plus a
dense-only baseline queried straight from the `site` collection. The endpoint
number is what users experience; the dense number tells you whether a
regression came from the index or from the service's ranking.

Run: python -m site_search.eval
"""

import argparse
import json
import os
import sys
import time

import requests
import yaml
from qdrant_client import QdrantClient, models

from site_search.config import (
    COLLECTION_NAME,
    NEURAL_ENCODER,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
)
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

# NEURAL_ENCODER is imported from site_search.config (config.py:18) rather than
# redeclared, so the indexer and the eval can never drift onto different models.
# It must also match rust_search/src/main.rs:33. The service does not embed
# locally either: main.rs:298 hands Qdrant a Document and lets server-side
# inference produce the vector, so going through models.Document here means the
# baseline uses the same embedding path rather than a local near-miss.

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


def make_client() -> QdrantClient:
    # cloud_inference=True is required, not cosmetic. Unlike the Rust client's
    # Document::new — which always ships the text to the server — the Python
    # client defaults cloud_inference to False and tries to embed locally with
    # fastembed, dying on "sentence-transformers/all-MiniLM-L6-v2 is not found
    # among supported models" when fastembed is not installed. True routes the
    # query through the server's inference, the same path the live service uses
    # (rust_search/src/main.rs:298), and keeps fastembed out of the CI install.
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
        cloud_inference=True,
    )


def case_filter(case: dict) -> models.Filter:
    """Reproduce the service's section and partition filters.

    Both fields are comma-separated any-of lists, matching how the Rust handler
    splits them (rust_search/src/main.rs, query_handler).
    """
    conditions = []
    for key in ("section", "partition"):
        values = [v.strip() for v in case[key].split(",") if v.strip()]
        if not values:
            continue
        payload_key = "sections" if key == "section" else "partition"
        conditions.append(
            models.FieldCondition(
                key=payload_key,
                match=models.MatchAny(any=values),
            )
        )
    return models.Filter(must=conditions)


def query_dense(client: QdrantClient, case: dict, limit: int = 5) -> list[str]:
    """Dense-only retrieval, no tag filters and no tier ladder."""
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(text=case["q"], model=NEURAL_ENCODER),
        query_filter=case_filter(case),
        limit=limit,
        with_payload=True,
    )
    return [point.payload["url"] for point in response.points]


def corpus_size(client: QdrantClient) -> int:
    """Point count of the site collection.

    The nightly job drops and recreates this collection, so a half-failed
    encode leaves a partial index. This is the cheapest detector for that.
    """
    return client.count(collection_name=COLLECTION_NAME, exact=True).count


def _score(urls: list[str], case: dict) -> dict:
    return {
        "hit": hit_at_5(urls, case["primary"], case.get("acceptable") or []),
        "rr": reciprocal_rank(urls, case["primary"]),
        "kind": case["kind"],
    }


def dense_enabled() -> bool:
    """Whether the dense baseline can run.

    Checks os.environ directly rather than the imported QDRANT_HOST, because
    config.py defaults that constant to 'localhost' when the variable is unset —
    so the constant is never falsy and cannot be used to detect absence.

    Without credentials the endpoint half still works and is what gates CI, so a
    missing cluster degrades the report instead of crashing it. That covers fork
    PRs, which receive no secrets, and local runs.
    """
    return bool(os.environ.get("QDRANT_HOST"))


def run(cases: list[dict]) -> dict:
    """Execute every case against the endpoint, and the dense baseline if available."""
    dense_on = dense_enabled()
    client = make_client() if dense_on else None
    endpoint_rows, dense_rows, per_case, latencies, zero_result_ids = [], [], [], [], []

    for case in cases:
        endpoint_urls, elapsed = query_endpoint(case)
        latencies.append(elapsed)
        if not endpoint_urls:
            zero_result_ids.append(case["id"])

        endpoint_score = _score(endpoint_urls, case)
        endpoint_rows.append(endpoint_score)

        dense_score = None
        if dense_on:
            dense_score = _score(query_dense(client, case), case)
            dense_rows.append(dense_score)

        per_case.append(
            {
                "id": case["id"],
                "q": case["q"],
                "kind": case["kind"],
                "primary": case["primary"],
                "endpoint_hit": endpoint_score["hit"],
                "endpoint_rr": round(endpoint_score["rr"], 4),
                "dense_hit": dense_score["hit"] if dense_score else None,
                "endpoint_urls": endpoint_urls,
                "latency_ms": round(elapsed * 1000),
            }
        )

    endpoint = aggregate(endpoint_rows)
    return {
        "endpoint": endpoint._asdict(),
        "dense": aggregate(dense_rows)._asdict() if dense_on else None,
        "endpoint_by_kind": {k: m._asdict() for k, m in aggregate_by_kind(endpoint_rows).items()},
        "dense_by_kind": (
            {k: m._asdict() for k, m in aggregate_by_kind(dense_rows).items()}
            if dense_on
            else None
        ),
        "dense_measured": dense_on,
        "corpus_size": corpus_size(client) if dense_on else None,
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
