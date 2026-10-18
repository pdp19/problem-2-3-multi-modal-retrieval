"""Retrieval benchmark built from AMI's own annotations.

Query  = one sentence of the human-written abstractive summary
         (sections: abstract / decisions / actions / problems).
Truth  = the time spans of the dialogue acts annotators linked to that
         sentence (AMI summary links), in that meeting.
A retrieved chunk is relevant if it overlaps any truth span in the same
meeting. Search always runs across *all* indexed meetings, so the system
must also find the right meeting.

Metrics per configuration (a set of channel weights):
  Hit@k   at least one relevant chunk in the top k
  MRR@10  1 / rank of the first relevant chunk
  Meeting@1  top chunk is from the right meeting
  Slide@1    the slide attached to the top hit was on screen during the
             evidence (only queries whose evidence has a slide on screen)
  chance Hit@5  expected Hit@5 of a random ranking, per query, averaged:
             how easy the query set is before any model is involved
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from . import ami, chunking, config
from .search import Searcher

from .search import TEXT_WEIGHTS, VISUAL_WEIGHTS

CONFIGS: dict[str, dict[str, float] | None] = {
    "bm25 only": {"bm25": 1.0},
    "dense only": {"dense": 1.0},
    "slide OCR (dense) only": {"slide_ocr": 1.0},
    "slide OCR (bm25) only": {"slide_bm25": 1.0},
    "slide CLIP only": {"slide_clip": 1.0},
    "frame CLIP only": {"frame_clip": 1.0},
    "text profile (fixed)": TEXT_WEIGHTS,
    "visual profile (fixed)": VISUAL_WEIGHTS,
    "routed by intent (default)": None,
}
VISUAL_QUERIES = config.REPO_ROOT / "benchmark" / "visual_queries.jsonl"
VISUAL_QUERIES_HELDOUT = config.REPO_ROOT / "benchmark" / "visual_queries_heldout.jsonl"


def _overlaps(t0: float, t1: float, spans: list[ami.Span], meeting: str) -> bool:
    return any(s.meeting == meeting and s.start < t1 and s.end > t0 for s in spans)


def chance_hit_at_k(n: int, r: int, k: int) -> float:
    """P(at least one of r relevant items in the top k of a random ranking of n)."""
    if r <= 0:
        return 0.0
    if n - r < k:
        return 1.0
    return 1.0 - math.comb(n - r, k) / math.comb(n, k)


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def paired_bootstrap_p(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> float:
    """One-sided p-value that mean(a) > mean(b), resampling queries."""
    rng = random.Random(seed)
    idx = list(range(len(a)))
    worse = 0
    for _ in range(n_boot):
        s = rng.choices(idx, k=len(idx))
        if sum(a[i] - b[i] for i in s) <= 0:
            worse += 1
    return worse / n_boot


def load_benchmark(meetings: list[str]) -> list[ami.Query]:
    qs: list[ami.Query] = []
    for m in meetings:
        qs.extend(ami.load_queries(m))
    return qs


def team_scope(meeting: str, indexed: set[str]) -> list[str]:
    """A team's workspace: every indexed meeting of the same series (ES2002a-d)."""
    return sorted(m for m in indexed if m[:-1] == meeting[:-1])


def score_query(searcher: Searcher, q: ami.Query, slides_by_meeting: dict, k: int = 10,
                scope: list[str] | None = None) -> dict:
    t = time.perf_counter()
    hits, _ = searcher.search(q.text, k=k, scope=scope)
    latency = time.perf_counter() - t
    rel = [_overlaps(h.t0, h.t1, q.evidence, h.meeting) for h in hits]
    first = next((i for i, r in enumerate(rel) if r), None)
    gt_slides = {
        s.slide_id
        for e in q.evidence
        for s in [chunking.slide_at(slides_by_meeting.get(q.meeting, []), (e.start + e.end) / 2)]
        if s is not None
    }
    out = {
        "qid": q.qid,
        "section": q.section,
        "hit@1": float(first is not None and first < 1),
        "hit@5": float(first is not None and first < 5),
        "hit@10": float(first is not None and first < 10),
        "mrr@10": 1.0 / (first + 1) if first is not None else 0.0,
        "meeting@1": float(bool(hits) and hits[0].meeting == q.meeting),
        "latency_s": latency,
    }
    if gt_slides:
        out["slide@1"] = float(bool(hits) and hits[0].slide_id in gt_slides)
    return out


def load_visual(path: Path = VISUAL_QUERIES) -> list[ami.Query]:
    """Hand-written "show me the slide ..." questions; truth = when that slide
    was on screen (from the slide capture file names)."""
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                spans = [ami.Span(d["meeting"], a, b) for a, b in d["spans"]]
                out.append(ami.Query(d["qid"], d["meeting"], "visual", d["text"], spans))
    return out


def evaluate_set(queries: list[ami.Query], searchers: dict[str, Searcher], base: Searcher,
                 slides_by_meeting: dict) -> tuple[dict, dict]:
    """Every query searches its team's workspace (scoped); the default system
    is also run over the whole pool as a stress test."""
    indexed = set(base.by_meeting)
    chance = []
    for q in queries:
        pool = [row for row in base.rows.values() if row["meeting_id"] in team_scope(q.meeting, indexed)]
        r = sum(_overlaps(row["t0"], row["t1"], q.evidence, row["meeting_id"]) for row in pool)
        chance.append(chance_hit_at_k(len(pool), r, 5))
    res: dict = {"n_queries": len(queries), "chance_hit@5": statistics.fmean(chance) if chance else 0.0,
                 "configs": {}}
    per_query: dict[str, list[dict]] = {}
    runs = [(name, s, True) for name, s in searchers.items()]
    runs.append(("routed, unscoped (all meetings)", base, False))
    for name, s, scoped in runs:
        rows = [score_query(s, q, slides_by_meeting, scope=team_scope(q.meeting, indexed) if scoped else None)
                for q in queries]
        per_query[name] = rows
        agg = {m: statistics.fmean(r[m] for r in rows) for m in ("hit@1", "hit@5", "hit@10", "mrr@10", "meeting@1")}
        slide_rows = [r["slide@1"] for r in rows if "slide@1" in r]
        agg["slide@1"] = statistics.fmean(slide_rows) if slide_rows else None
        agg["n_slide_queries"] = len(slide_rows)
        agg["hit@5_ci95"] = bootstrap_ci([r["hit@5"] for r in rows])
        agg["mrr@10_ci95"] = bootstrap_ci([r["mrr@10"] for r in rows])
        lat = sorted(r["latency_s"] for r in rows)
        agg["latency_p50_ms"] = 1000 * lat[len(lat) // 2]
        agg["latency_p95_ms"] = 1000 * lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        agg["by_section"] = {
            sec: {"n": len(rs), "hit@5": statistics.fmean(r["hit@5"] for r in rs),
                  "mrr@10": statistics.fmean(r["mrr@10"] for r in rs)}
            for sec in (*ami.SUMMARY_SECTIONS, "visual")
            if (rs := [r for r in rows if r["section"] == sec])
        }
        res["configs"][name] = agg
        sl = "-" if agg["slide@1"] is None else f"{agg['slide@1']:.2f}"
        print(f"    {name:28s} Hit@1 {agg['hit@1']:.2f}  Hit@5 {agg['hit@5']:.2f}  MRR {agg['mrr@10']:.2f}  "
              f"Meeting@1 {agg['meeting@1']:.2f}  Slide@1 {sl}  p50 {agg['latency_p50_ms']:.0f} ms")
    routed, text = "routed by intent (default)", "text profile (fixed)"
    if routed in per_query and text in per_query:
        res["p_routed_gt_text_mrr"] = paired_bootstrap_p(
            [r["mrr@10"] for r in per_query[routed]], [r["mrr@10"] for r in per_query[text]])
    return res, per_query


def run(dev: list[str], test: list[str] | None = None, out_dir: Path | None = None) -> dict:
    """Search always runs over every indexed meeting (dev + test pooled)."""
    out_dir = out_dir or config.ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Searcher()
    indexed = set(base.by_meeting)
    has_video = base.meetings_t.count() > 0
    searchers = {name: (base if w is None else Searcher(weights=w)) for name, w in CONFIGS.items()
                 if has_video or w is None or "frame_clip" not in w or len(w) > 1}
    slides_by_meeting = {m: ami.load_slides(m) for m in indexed}
    sets = {
        "dev: summary questions": [q for q in load_benchmark(dev) if q.meeting in indexed],
        "test: summary questions (held out)": [q for q in load_benchmark(test or []) if q.meeting in indexed],
        "dev: visual questions": [q for q in load_visual() if q.meeting in indexed],
        "test: visual questions (held out)": [q for q in load_visual(VISUAL_QUERIES_HELDOUT) if q.meeting in indexed],
    }
    results: dict = {"pool_meetings": sorted(indexed), "pool_chunks": len(base.ids),
                     "video_meetings": sorted({r["meeting_id"] for r in base.meetings_t.select(base.meetings_t.meeting_id).collect()}),
                     "sets": {}}
    per_query_all = {}
    for name, qs in sets.items():
        if not qs:
            continue
        print(f"  {name} ({len(qs)} queries)")
        results["sets"][name], per_query_all[name] = evaluate_set(qs, searchers, base, slides_by_meeting)
    (out_dir / "eval.json").write_text(json.dumps(results, indent=2, default=float))
    (out_dir / "eval_per_query.json").write_text(json.dumps(per_query_all, indent=1))
    with open(out_dir / "benchmark_queries.jsonl", "w") as f:
        for name, qs in sets.items():
            for q in qs:
                f.write(json.dumps({"set": name, **asdict(q)}) + "\n")
    return results
