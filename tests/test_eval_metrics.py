from site_search.eval_metrics import hit_at_5, normalize_url


def test_normalize_strips_trailing_slash():
    assert normalize_url("/documentation/search/") == "/documentation/search"


def test_normalize_accepts_absolute_url():
    assert normalize_url("https://qdrant.tech/documentation/search/") == "/documentation/search"


def test_normalize_strips_query_and_fragment():
    assert normalize_url("/documentation/search/?q=x#frag") == "/documentation/search"


def test_normalize_root_stays_slash():
    assert normalize_url("/") == "/"


def test_hit_when_primary_at_rank_one():
    assert hit_at_5(["/documentation/a/", "/documentation/b/"], "/documentation/a/") is True


def test_hit_when_primary_at_rank_five():
    urls = [f"/documentation/x{i}/" for i in range(4)] + ["/documentation/a/"]
    assert hit_at_5(urls, "/documentation/a/") is True


def test_miss_when_primary_absent():
    assert hit_at_5(["/documentation/b/"], "/documentation/a/") is False


def test_hit_via_acceptable_url():
    assert hit_at_5(["/articles/z/"], "/documentation/a/", ["/articles/z/"]) is True


def test_trailing_slash_mismatch_still_hits():
    assert hit_at_5(["/documentation/a"], "/documentation/a/") is True


def test_empty_results_is_a_miss():
    assert hit_at_5([], "/documentation/a/") is False


def test_only_first_five_results_count():
    urls = [f"/documentation/x{i}/" for i in range(5)] + ["/documentation/a/"]
    assert hit_at_5(urls, "/documentation/a/") is False


from site_search.eval_metrics import (
    Metrics,
    aggregate,
    aggregate_by_kind,
    reciprocal_rank,
    result_urls,
)


def test_rr_is_one_at_rank_one():
    assert reciprocal_rank(["/a/", "/b/"], "/a/") == 1.0


def test_rr_is_one_third_at_rank_three():
    assert reciprocal_rank(["/x/", "/y/", "/a/"], "/a/") == 1 / 3


def test_rr_is_zero_when_absent():
    assert reciprocal_rank(["/x/"], "/a/") == 0.0


def test_rr_ignores_results_past_cutoff():
    urls = [f"/x{i}/" for i in range(5)] + ["/a/"]
    assert reciprocal_rank(urls, "/a/") == 0.0


def test_rr_ignores_acceptable_urls():
    # MRR ranks against primary only, by design.
    assert reciprocal_rank(["/articles/z/"], "/documentation/a/") == 0.0


def test_aggregate_computes_means():
    rows = [
        {"hit": True, "rr": 1.0},
        {"hit": True, "rr": 0.5},
        {"hit": False, "rr": 0.0},
        {"hit": False, "rr": 0.0},
    ]
    assert aggregate(rows) == Metrics(n=4, hit_rate=0.5, mrr=0.375)


def test_aggregate_of_empty_is_zeroed():
    assert aggregate([]) == Metrics(n=0, hit_rate=0.0, mrr=0.0)


def test_aggregate_by_kind_splits_buckets():
    rows = [
        {"kind": "keyword", "hit": True, "rr": 1.0},
        {"kind": "keyword", "hit": False, "rr": 0.0},
        {"kind": "typo", "hit": False, "rr": 0.0},
    ]
    by_kind = aggregate_by_kind(rows)
    assert by_kind["keyword"] == Metrics(n=2, hit_rate=0.5, mrr=0.5)
    assert by_kind["typo"] == Metrics(n=1, hit_rate=0.0, mrr=0.0)


def test_result_urls_extracts_payload_urls():
    payload = {
        "result": [
            {"payload": {"url": "/documentation/a/", "text": "A"}, "highlight": "A"},
            {"payload": {"url": "/documentation/b/", "text": "B"}, "highlight": "B"},
        ],
        "time": 0.05,
    }
    assert result_urls(payload) == ["/documentation/a/", "/documentation/b/"]


def test_result_urls_of_empty_response():
    assert result_urls({"result": [], "time": 0.01}) == []


def test_result_urls_tolerates_missing_key():
    assert result_urls({}) == []


from site_search.eval_metrics import gate_failures


def test_gate_passes_above_floor():
    assert gate_failures(Metrics(n=40, hit_rate=0.9, mrr=0.7), [], 0.8) == []


def test_gate_fails_below_floor():
    failures = gate_failures(Metrics(n=40, hit_rate=0.7, mrr=0.5), [], 0.8)
    assert len(failures) == 1
    assert "0.700" in failures[0] and "0.800" in failures[0]


def test_gate_fails_on_zero_results():
    failures = gate_failures(Metrics(n=40, hit_rate=0.9, mrr=0.7), ["nav-quickstart"], 0.8)
    assert len(failures) == 1
    assert "nav-quickstart" in failures[0]


def test_gate_reports_both_failures():
    failures = gate_failures(Metrics(n=40, hit_rate=0.1, mrr=0.1), ["a", "b"], 0.8)
    assert len(failures) == 2


def test_gate_floor_is_inclusive():
    assert gate_failures(Metrics(n=40, hit_rate=0.8, mrr=0.7), [], 0.8) == []
