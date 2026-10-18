import json
import math

from mmr import ami, config
from mmr.download import parse_slide_urls
from mmr.evaluate import _overlaps, bootstrap_ci, chance_hit_at_k, paired_bootstrap_p, team_scope
from mmr.search import best_quote, query_intent, rrf, tokenize


def test_rrf_weights_and_shares():
    scores = rrf({"a": ["x", "y"], "b": ["y"]}, {"a": 1.0, "b": 1.0}, k=60)
    assert scores["y"] > scores["x"]  # two votes beat one
    assert math.isclose(scores["x"], 1 / 61)
    off = rrf({"a": ["x"], "b": ["y"]}, {"a": 1.0, "b": 0.0})
    assert "y" not in off  # zero-weight channel is ignored
    shared = rrf({"s": ["c1", "c2"]}, {"s": 1.0}, shares={"s": {"c1": 0.5, "c2": 0.5}})
    assert math.isclose(shared["c1"], 0.5 / 61)


def test_tokenize_drops_fillers():
    assert tokenize("Um, the Remote uses SOLAR power, yeah") == ["remot", "uses", "solar", "power"]
    assert tokenize("what did they decide?") == tokenize("decided") == ["decid"]
    assert tokenize("batteries") == tokenize("battery")


def test_query_intent_names_artifacts_not_topics():
    assert query_intent("show me the slide with the cost breakdown") == "visual"
    assert query_intent("the spreadsheet with production costs") == "visual"
    assert query_intent("photo of the inside of the remote") == "visual"
    assert query_intent("what did we decide about the battery?") == "speech"
    # topical words are not a request to see something
    assert query_intent("The team evaluated the prototype.") == "speech"
    assert query_intent("They discussed adding an LCD screen.") == "speech"


def test_every_visual_benchmark_query_routes_visual():
    for name in ("visual_queries.jsonl", "visual_queries_heldout.jsonl"):
        path = config.REPO_ROOT / "benchmark" / name
        for line in path.read_text().splitlines():
            q = json.loads(line)
            assert query_intent(q["text"]) == "visual", q["text"]
            assert all(a < b for a, b in q["spans"])


def test_best_quote_picks_matching_line():
    t = "Project Manager: okay so\nIndustrial Designer: solar power is cheaper than a battery\nMarketing Expert: yes"
    assert best_quote("battery or solar?", t).startswith("Industrial Designer")


def test_chance_hit_at_k():
    assert chance_hit_at_k(10, 0, 5) == 0.0
    assert chance_hit_at_k(10, 6, 5) == 1.0  # not enough irrelevant items to fill the top 5
    assert math.isclose(chance_hit_at_k(10, 1, 5), 0.5)


def test_overlap_requires_same_meeting():
    spans = [ami.Span("M1", 10, 20)]
    assert _overlaps(15, 30, spans, "M1")
    assert not _overlaps(15, 30, spans, "M2")
    assert not _overlaps(20, 30, spans, "M1")  # touching is not overlapping


def test_team_scope_is_the_series():
    idx = {"ES2002a", "ES2002b", "ES2003a", "IS1009c"}
    assert team_scope("ES2002d", idx) == ["ES2002a", "ES2002b"]


def test_bootstrap_helpers():
    lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert lo < 0.5 < hi
    assert paired_bootstrap_p([1.0] * 30, [0.0] * 30) == 0.0
    assert paired_bootstrap_p([0.0] * 30, [1.0] * 30) == 1.0


def test_parse_slide_urls_from_download_script():
    html = (
        "wget -P amicorpus/ES2002b/slides https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/ES2002b/slides/ES2002b.277.29__284.31.jpg\\n"
        "wget -P amicorpus/ES2002b/slides https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/ES2002b/slides/index.html\\n"
        "wget -P amicorpus/ES2002b/slides https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/ES2002b/slides/ES2002b.277.29__284.31.jpg"
    )
    urls = parse_slide_urls(html, "ES2002b")
    assert len(urls) == 1 and urls[0].endswith("ES2002b.277.29__284.31.jpg")
