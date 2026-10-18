"""Render artifacts/eval.json + ingest timings into docs/results.md."""
from __future__ import annotations

import json

from . import config


def _f(x, nd=2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def write_report(results: dict | None = None) -> str:
    if results is None:
        results = json.loads((config.ARTIFACTS_DIR / "eval.json").read_text())
    lines = [
        "# Results",
        "",
        f"Index: {len(results['pool_meetings'])} meetings, {results['pool_chunks']} chunks "
        f"(video frames indexed for {', '.join(results['video_meetings']) or 'none'}). "
        "Each query searches its team's workspace (the 4 meetings of its series); "
        "the last row of each table searches every meeting at once.",
        "Hit@k: a top-k clip overlaps the annotated evidence. Meeting@1: top clip is from the right meeting. "
        "Slide@1: the slide returned with the top clip was on screen during the evidence.",
    ]
    for set_name, res in results["sets"].items():
        lines += [
            "",
            f"## {set_name}: {res['n_queries']} queries (random ranking, scoped: Hit@5 = {res['chance_hit@5']:.2f})",
            "",
            "| configuration | Hit@1 | Hit@5 [95% CI] | MRR@10 | Meeting@1 | Slide@1 (n) | p50 / p95 ms |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, a in res["configs"].items():
            lo, hi = a["hit@5_ci95"]
            lines.append(
                f"| {name} | {_f(a['hit@1'])} | {_f(a['hit@5'])} [{_f(lo)}, {_f(hi)}] | {_f(a['mrr@10'])} | "
                f"{_f(a['meeting@1'])} | {_f(a['slide@1'])} ({a['n_slide_queries']}) | "
                f"{a['latency_p50_ms']:.0f} / {a['latency_p95_ms']:.0f} |"
            )
        if "p_routed_gt_text_mrr" in res:
            lines += ["", f"Paired bootstrap, routed vs text profile on MRR@10: one-sided p = "
                          f"{res['p_routed_gt_text_mrr']:.3f}."]
        routed = res["configs"].get("routed by intent (default)")
        if routed and len(routed["by_section"]) > 1:
            lines += ["", "| section (routed) | n | Hit@5 | MRR@10 |", "|---|---|---|---|"]
            for sec, v in routed["by_section"].items():
                lines.append(f"| {sec} | {v['n']} | {_f(v['hit@5'])} | {_f(v['mrr@10'])} |")
    tpath = config.ARTIFACTS_DIR / "ingest_timings.jsonl"
    if tpath.exists():
        rows = [json.loads(x) for x in tpath.read_text().splitlines() if x.strip()]
        lines += ["", "Ingest cost (CPU, Apple M3):", "",
                  "| meeting | minutes | chunks | slides | slides OCR+embed s | chunks embed s | frames CLIP s |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['meeting']} | {_f(r.get('duration_s', 0) / 60, 1)} | {r.get('n_chunks', '-')} | "
                f"{r.get('n_slides', '-')} | {_f(r.get('slides_ocr_embed'), 1)} | {_f(r.get('chunks_embed'), 1)} | "
                f"{_f(r.get('frames_clip'), 1)} |"
            )
    text = "\n".join(lines) + "\n"
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DOCS_DIR / "results.md").write_text(text)
    print(text)
    return text
