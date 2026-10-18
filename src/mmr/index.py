"""The multimodal index, built with Pixeltable.

    mmr.meetings  table   one row per meeting video
    mmr.frames    view    frame_iterator(video, fps=0.2) -> CLIP image index
    mmr.slides    table   slide captures (+ on-screen time) -> OCR column,
                          CLIP image index, text index on the OCR
    mmr.chunks    table   transcript chunks (+ slide OCR) -> text index

Everything downstream of an insert (frame extraction, OCR, embeddings,
index maintenance) is a Pixeltable computed column or view, so adding a new
meeting only computes that meeting's rows.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pixeltable as pxt
from pixeltable.functions.huggingface import clip, sentence_transformer
from pixeltable.functions.video import frame_iterator

from . import ami, chunking, config

_ocr_engine = None
_OCR_RUN = re.compile(r"[A-Za-z]{6,}")


def fix_ocr_spacing(text: str) -> str:
    """RapidOCR often drops the spaces on these dark slides
    ("Teletexthasbecomeoutdated"). Re-segment unknown letter runs into words
    with a unigram model; short runs keep the original too, so domain words
    the model doesn't know ("teletext") still match."""
    import wordninja

    lm = wordninja.DEFAULT_LANGUAGE_MODEL

    def rep(m: re.Match) -> str:
        w = m.group(0)
        if w.lower() in lm._wordcost:
            return w
        parts = lm.split(w)
        if len(parts) < 2:
            return w
        return (w + " " if len(w) <= 12 else "") + " ".join(parts)

    return _OCR_RUN.sub(rep, text)


@pxt.udf
def ocr_text(img: pxt.Image) -> str:
    """On-screen text of a slide, in reading order (RapidOCR, ONNX, CPU)."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    import numpy as np

    result, _ = _ocr_engine(np.asarray(img.convert("RGB")))
    if not result:
        return ""
    lines = sorted(result, key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
    return fix_ocr_spacing(" ".join(r[1] for r in lines if r[2] > 0.5))


def _path(name: str) -> str:
    return f"{config.PXT_DIR}.{name}"


def text_embed():
    return sentence_transformer.using(model_id=config.TEXT_MODEL, normalize_embeddings=True)


def clip_embed():
    return clip.using(model_id=config.CLIP_MODEL)


def create_schema(reset: bool = False) -> None:
    if reset:
        pxt.drop_dir(config.PXT_DIR, force=True, if_not_exists="ignore")
    pxt.create_dir(config.PXT_DIR, if_exists="ignore")

    meetings = pxt.create_table(
        _path("meetings"), {"meeting_id": pxt.String, "video": pxt.Video}, if_exists="ignore"
    )
    frames = pxt.create_view(
        _path("frames"), meetings, iterator=frame_iterator(meetings.video, fps=config.FRAME_FPS), if_exists="ignore"
    )
    frames.add_embedding_index("frame", idx_name="frame_clip", embedding=clip_embed(), if_exists="ignore")

    slides = pxt.create_table(
        _path("slides"),
        {"slide_id": pxt.String, "meeting_id": pxt.String, "t0": pxt.Float, "t1": pxt.Float, "image": pxt.Image},
        if_exists="ignore",
    )
    if "ocr" not in slides.columns():
        slides.add_computed_column(ocr=ocr_text(slides.image))
    slides.add_embedding_index("image", idx_name="slide_clip", embedding=clip_embed(), if_exists="ignore")
    slides.add_embedding_index("ocr", idx_name="slide_ocr", embedding=text_embed(), if_exists="ignore")

    chunks = pxt.create_table(
        _path("chunks"),
        {
            "chunk_id": pxt.String,
            "meeting_id": pxt.String,
            "t0": pxt.Float,
            "t1": pxt.Float,
            "speakers": pxt.String,
            "transcript": pxt.String,
            "slide_id": pxt.String,
            "text": pxt.String,
        },
        if_exists="ignore",
    )
    chunks.add_embedding_index("text", idx_name="chunk_text", embedding=text_embed(), if_exists="ignore")


def tables():
    return (
        pxt.get_table(_path("meetings")),
        pxt.get_table(_path("frames")),
        pxt.get_table(_path("slides")),
        pxt.get_table(_path("chunks")),
    )


def indexed_meetings() -> tuple[set[str], set[str]]:
    """(meetings with transcript + slides indexed, meetings with video indexed)."""
    meetings, _, _, chunks = tables()
    text = {r["meeting_id"] for r in chunks.select(chunks.meeting_id).distinct().collect()}
    video = {r["meeting_id"] for r in meetings.select(meetings.meeting_id).collect()}
    return text, video


def ingest_video(meeting: str) -> float:
    """Insert the video; the frames view extracts 1 frame / 5 s and CLIP-embeds it."""
    meetings, *_ = tables()
    t = time.perf_counter()
    meetings.insert([{"meeting_id": meeting, "video": str(config.video_path(meeting))}])
    return time.perf_counter() - t


def ingest_meeting(meeting: str) -> dict:
    """Index one meeting's slides and transcript. Returns per-stage timings (seconds)."""
    _, _, slides_t, chunks_t = tables()
    timings: dict[str, float] = {}

    t = time.perf_counter()
    slides = ami.load_slides(meeting)
    if slides:
        slides_t.insert(
            [{"slide_id": s.slide_id, "meeting_id": meeting, "t0": s.start, "t1": s.end, "image": str(s.path)}
             for s in slides]
        )
    timings["slides_ocr_embed"] = time.perf_counter() - t

    ocr = {r["slide_id"]: r["ocr"] for r in
           slides_t.where(slides_t.meeting_id == meeting).select(slides_t.slide_id, slides_t.ocr).collect()}

    t = time.perf_counter()
    roles = ami.speaker_roles(meeting)
    chunks = chunking.chunk_meeting(meeting, chunking.utterances(ami.load_words(meeting)), slides)
    rows = []
    for c in chunks:
        transcript = c.transcript(roles)
        slide_txt = ocr.get(c.slide_id) or ""
        rows.append({
            "chunk_id": c.chunk_id,
            "meeting_id": meeting,
            "t0": c.start,
            "t1": c.end,
            "speakers": ",".join(sorted({roles.get(u.speaker, u.speaker) for u in c.utterances})),
            "transcript": transcript,
            "slide_id": c.slide_id or "",
            "text": transcript + (f"\n[slide] {slide_txt}" if slide_txt else ""),
        })
    if rows:
        chunks_t.insert(rows)
    timings["chunks_embed"] = time.perf_counter() - t

    timings["n_chunks"] = len(rows)
    timings["n_slides"] = len(slides)
    timings["duration_s"] = max((c.end for c in chunks), default=0.0)
    return timings


def ingest(meeting_ids: list[str], reset: bool = False, with_video: bool = True) -> list[dict]:
    create_schema(reset=reset)
    text_done, video_done = indexed_meetings()
    out = []
    for m in meeting_ids:
        tm: dict = {"meeting": m}
        if m not in text_done:
            tm.update(ingest_meeting(m))
        if with_video and m not in video_done:
            if config.video_path(m).exists():
                tm["frames_clip"] = ingest_video(m)
            else:
                print(f"  {m}: no video downloaded, transcript + slides only", file=sys.stderr)
        if len(tm) == 1:
            print(f"  {m}: already indexed", file=sys.stderr)
            continue
        out.append(tm)
        stages = ("slides_ocr_embed", "chunks_embed", "frames_clip")
        print(f"  {m}: {tm.get('n_chunks', '-')} chunks, {tm.get('n_slides', '-')} slides, "
              + ", ".join(f"{k} {tm[k]:.1f}s" for k in stages if k in tm), file=sys.stderr)
    return out
