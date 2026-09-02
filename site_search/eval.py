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


# Must match rust_search/src/main.rs:33 exactly. The service does not embed
# locally either: main.rs:298 hands Qdrant a Document and lets server-side
# inference produce the vector, so going through models.Document here means our
# baseline uses the same embedding path rather than a local near-miss.
NEURAL_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def make_client() -> QdrantClient:
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
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


def run(cases: list[dict]) -> dict:
    """Execute every case against both layers and return the full report."""
    client = make_client()
    endpoint_rows, dense_rows, per_case, latencies, zero_result_ids = [], [], [], [], []

    for case in cases:
        endpoint_urls, elapsed = query_endpoint(case)
        dense_urls = query_dense(client, case)
        latencies.append(elapsed)
        if not endpoint_urls:
            zero_result_ids.append(case["id"])

        endpoint_score = _score(endpoint_urls, case)
        dense_score = _score(dense_urls, case)
        endpoint_rows.append(endpoint_score)
        dense_rows.append(dense_score)
        per_case.append(
            {
                "id": case["id"],
                "q": case["q"],
                "kind": case["kind"],
                "primary": case["primary"],
                "endpoint_hit": endpoint_score["hit"],
                "endpoint_rr": round(endpoint_score["rr"], 4),
                "dense_hit": dense_score["hit"],
                "endpoint_urls": endpoint_urls,
                "latency_ms": round(elapsed * 1000),
            }
        )

    endpoint = aggregate(endpoint_rows)
    dense = aggregate(dense_rows)
    return {
        "endpoint": endpoint._asdict(),
        "dense": dense._asdict(),
        "endpoint_by_kind": {k: m._asdict() for k, m in aggregate_by_kind(endpoint_rows).items()},
        "dense_by_kind": {k: m._asdict() for k, m in aggregate_by_kind(dense_rows).items()},
        "corpus_size": corpus_size(client),
        "max_latency_ms": round(max(latencies) * 1000) if latencies else 0,
        "zero_result_ids": zero_result_ids,
        "floor": HIT_RATE_FLOOR,
        "failures": gate_failures(endpoint, zero_result_ids, HIT_RATE_FLOOR),
        "cases": per_case,
    }


def render_markdown(report: dict) -> str:
    lines = ["## Docs search eval", ""]
    endpoint = report["endpoint"]
    dense = report["dense"]

    lines += [
        "| layer | n | hit-rate@5 | MRR@5 |",
        "| --- | --- | --- | --- |",
        f"| endpoint | {endpoint['n']} | {endpoint['hit_rate']:.3f} | {endpoint['mrr']:.3f} |",
        f"| dense only | {dense['n']} | {dense['hit_rate']:.3f} | {dense['mrr']:.3f} |",
        "",
        "| kind | n | endpoint hit-rate@5 | endpoint MRR@5 | dense hit-rate@5 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind in sorted(report["endpoint_by_kind"]):
        e = report["endpoint_by_kind"][kind]
        d = report["dense_by_kind"][kind]
        lines.append(
            f"| {kind} | {e['n']} | {e['hit_rate']:.3f} | {e['mrr']:.3f} | {d['hit_rate']:.3f} |"
        )

    lines += [
        "",
        f"corpus: {report['corpus_size']} points · "
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
