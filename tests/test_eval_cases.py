from pathlib import Path

import yaml

CASES_PATH = Path(__file__).parent.parent / "site_search" / "eval_cases.yaml"
VALID_KINDS = {"keyword", "natural", "typo", "navigational"}
EXPECTED_KIND_COUNTS = {"keyword": 12, "natural": 14, "typo": 6, "navigational": 8}


def load_cases():
    with open(CASES_PATH) as f:
        return yaml.safe_load(f)


def test_case_count():
    assert len(load_cases()) == 40


def test_ids_are_unique():
    ids = [c["id"] for c in load_cases()]
    assert len(ids) == len(set(ids))


def test_required_keys_present():
    for case in load_cases():
        for key in ("id", "q", "section", "partition", "primary", "kind"):
            assert key in case, f"{case.get('id')} missing {key}"


def test_kinds_are_valid_and_balanced():
    counts = {}
    for case in load_cases():
        assert case["kind"] in VALID_KINDS, case["id"]
        counts[case["kind"]] = counts.get(case["kind"], 0) + 1
    assert counts == EXPECTED_KIND_COUNTS


def test_urls_are_absolute_paths():
    for case in load_cases():
        urls = [case["primary"]] + list(case.get("acceptable") or [])
        for url in urls:
            assert url.startswith("/"), f"{case['id']}: {url}"
            assert url.endswith("/"), f"{case['id']}: {url}"


def test_search_context_is_uniform():
    for case in load_cases():
        assert case["section"] == "documentation", case["id"]
        assert case["partition"] == "develop,deploy,cloud,qdrant", case["id"]
