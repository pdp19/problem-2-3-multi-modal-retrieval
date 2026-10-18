from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import config


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def cmd_download(a: argparse.Namespace) -> None:
    from .download import download

    download(a.meetings, video=not a.no_video)


def cmd_index(a: argparse.Namespace) -> None:
    from .index import ingest

    timings = ingest(a.meetings, reset=a.reset, with_video=not a.no_video)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_DIR / "ingest_timings.jsonl"
    with open(path, "a") as f:
        for t in timings:
            f.write(json.dumps(t) + "\n")


def export_hit(hit, out_dir: Path, n: int) -> list[str]:
    """Cut the clip (silent, 480p H.264), copy the slide, grab the best frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    video = config.video_path(hit.meeting)
    ffmpeg = shutil.which("ffmpeg")
    if video.exists() and ffmpeg:
        clip = out_dir / f"{n:02d}_{hit.meeting}_{int(hit.t0)}-{int(hit.t1)}.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{hit.t0:.2f}", "-to", f"{hit.t1:.2f}",
                        "-i", str(video), "-vf", "scale=480:-2", "-c:v", "libx264", "-preset", "ultrafast",
                        "-an", str(clip)], check=False)
        if clip.exists():
            made.append(str(clip))
        if hit.frame_t is not None:
            frame = out_dir / f"{n:02d}_{hit.meeting}_frame_{int(hit.frame_t)}.jpg"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{hit.frame_t:.2f}", "-i", str(video),
                            "-frames:v", "1", str(frame)], check=False)
            if frame.exists():
                made.append(str(frame))
    if hit.slide_id:
        src = config.slides_dir(hit.meeting) / f"{hit.slide_id}.jpg"
        if src.exists():
            dst = out_dir / f"{n:02d}_{src.name}"
            shutil.copy(src, dst)
            made.append(str(dst))
    return made


def cmd_ask(a: argparse.Namespace) -> None:
    from .search import Searcher

    searcher = Searcher()
    teams = [t for t in (a.team or "").split(",") if t]
    scope = [m for m in searcher.by_meeting if any(m.startswith(t) for t in teams)] if teams else None
    if teams and not scope:
        sys.exit(f"no indexed meetings match --team {a.team}")
    for q in a.question:
        hits, timing = searcher.search(q, k=a.k, scope=scope)
        if a.json:
            print(json.dumps({"question": q, "hits": [h.as_dict() for h in hits],
                              "latency_ms": {k: round(v * 1000, 1) for k, v in timing.items()}}, indent=2))
            continue
        print(f"\nQ: {q}   ({sum(timing.values()) * 1000:.0f} ms)")
        for i, h in enumerate(hits, 1):
            ch = " ".join(f"{c}#{r}" for c, r in h.ranks.items())
            print(f"  {i}. {h.meeting} {_fmt(h.t0)}-{_fmt(h.t1)}  [{ch}]")
            print(f"     \"{h.quote[:220]}\"")
            if h.slide_id:
                print(f"     slide: {config.slides_dir(h.meeting) / (h.slide_id + '.jpg')}")
            if h.frame_t is not None:
                print(f"     best frame: {_fmt(h.frame_t)}")
            if a.export:
                for p in export_hit(h, Path(a.export), i):
                    print(f"     -> {p}")


def cmd_eval(a: argparse.Namespace) -> None:
    from .evaluate import run
    from .report import write_report

    results = run(a.meetings, a.test)
    write_report(results)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Meeting memory: multimodal retrieval over meeting videos (AMI corpus, Pixeltable).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch annotations, one camera view and slide captures")
    d.add_argument("--meetings", nargs="+", default=config.DEFAULT_MEETINGS)
    d.add_argument("--no-video", action="store_true", help="annotations + slides only (fast)")
    d.set_defaults(fn=cmd_download)

    i = sub.add_parser("index", help="build the Pixeltable index (slides+OCR, transcript chunks, video frames)")
    i.add_argument("--meetings", nargs="+", default=config.DEFAULT_MEETINGS)
    i.add_argument("--reset", action="store_true", help="drop and rebuild the index")
    i.add_argument("--no-video", action="store_true", help="skip the frame channel")
    i.set_defaults(fn=cmd_index)

    q = sub.add_parser("ask", help="search the meetings; returns quote + clip + slide")
    q.add_argument("question", nargs="+")
    q.add_argument("-k", type=int, default=3)
    q.add_argument("--team", help="workspace: meeting-id prefix(es) to search, e.g. ES2002 or ES2002,IS1009 (default: all)")
    q.add_argument("--json", action="store_true")
    q.add_argument("--export", metavar="DIR", help="write clip mp4, slide and frame for each hit")
    q.set_defaults(fn=cmd_ask)

    e = sub.add_parser("eval", help="benchmark every channel and the fused system; writes docs/results.md")
    e.add_argument("--meetings", nargs="+", default=config.DEFAULT_MEETINGS, help="dev meetings")
    e.add_argument("--test", nargs="*", default=config.TEST_MEETINGS, help="held-out meetings (if indexed)")
    e.set_defaults(fn=cmd_eval)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
