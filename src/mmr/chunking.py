"""Meeting chunker: timed words -> utterances -> retrieval chunks.

A chunk is a run of consecutive utterances, closed when it reaches the target
length, when the projected slide changes, or at a long silence. Slide changes
are the strongest topic boundary a meeting has, which is why the Task 1
classifier routes `screen_content` / `talking_head` videos to this chunker.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from . import config
from .ami import Slide, Word


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    chunk_id: str
    meeting: str
    start: float
    end: float
    utterances: list[Utterance] = field(default_factory=list)
    slide_id: str | None = None

    def transcript(self, roles: dict[str, str] | None = None) -> str:
        roles = roles or {}
        return "\n".join(f"{roles.get(u.speaker, u.speaker)}: {u.text}" for u in self.utterances)


def _join(words: list[Word]) -> str:
    out = ""
    for w in words:
        out += w.text if (w.punc or not out) else " " + w.text
    return out


def utterances(words: list[Word], max_gap: float = 1.0, max_len: float = config.CHUNK_TARGET_S) -> list[Utterance]:
    """Group each speaker's words into utterances at gaps longer than `max_gap`
    seconds (or every `max_len` seconds of monologue); this is what an ASR +
    diarisation front end returns."""
    by_spk: dict[str, list[Word]] = {}
    for w in words:
        by_spk.setdefault(w.speaker, []).append(w)
    out: list[Utterance] = []
    for spk, ws in by_spk.items():
        ws.sort(key=lambda w: w.start)
        cur: list[Word] = []
        for w in ws:
            if cur and (w.start - cur[-1].end > max_gap or w.end - cur[0].start > max_len):
                out.append(Utterance(spk, cur[0].start, cur[-1].end, _join(cur)))
                cur = []
            cur.append(w)
        if cur:
            out.append(Utterance(spk, cur[0].start, cur[-1].end, _join(cur)))
    out.sort(key=lambda u: (u.start, u.end))
    return out


def slide_at(slides: list[Slide], t: float) -> Slide | None:
    starts = [s.start for s in slides]
    i = bisect_right(starts, t) - 1
    if i >= 0 and slides[i].start <= t < slides[i].end:
        return slides[i]
    return None


def chunk_meeting(
    meeting: str,
    utts: list[Utterance],
    slides: list[Slide],
    target: float = config.CHUNK_TARGET_S,
    min_len: float = config.CHUNK_MIN_S,
    pause: float = config.CHUNK_PAUSE_S,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[Utterance] = []
    cur_slide: str | None = None

    def close() -> None:
        if cur:
            mid = (cur[0].start + max(u.end for u in cur)) / 2
            s = slide_at(slides, mid)
            chunks.append(
                Chunk(f"{meeting}#{len(chunks):04d}", meeting, cur[0].start, max(u.end for u in cur), list(cur),
                      s.slide_id if s else None)
            )
            cur.clear()

    for u in utts:
        s = slide_at(slides, u.start)
        sid = s.slide_id if s else None
        if cur:
            length = max(x.end for x in cur) - cur[0].start
            gap = u.start - max(x.end for x in cur)
            if length >= target or (length >= min_len and (sid != cur_slide or gap >= pause)):
                close()
        if not cur:
            cur_slide = sid
        cur.append(u)
    close()
    return chunks
