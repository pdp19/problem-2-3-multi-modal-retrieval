"""Query engine: six retrieval channels, fused per clip with weighted
reciprocal-rank fusion (RRF).

    bm25        keywords over transcript + slide OCR      (in memory)
    dense       bge-small over transcript + slide OCR     (Pixeltable index)
    slide_ocr   bge-small over slide OCR                  (Pixeltable index)
    slide_clip  CLIP text -> slide image                  (Pixeltable index)
    slide_bm25  keywords over slide OCR                   (in memory)
    frame_clip  CLIP text -> video frame every 5 s        (Pixeltable index)

The three slide channels rank slides; slide and frame hits are then mapped
to the clips they overlap in time, so every channel votes on the same unit:
a clip of the meeting, returned with its quote and the slide on screen.
A query is first routed by intent (speech vs visual) to a weight profile.
"""
from __future__ import annotations

import re
import time
from bisect import bisect_right
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from . import index

CHANNELS = ("bm25", "dense", "slides", "frame_clip")
SLIDE_CHANNELS = ("slide_ocr", "slide_clip", "slide_bm25")

# Query-intent routing. Most questions are about what was *said*; there the
# transcript channels carry the answer. Questions that ask to *see* an
# artifact (a slide, a diagram, an email) switch to a profile that ranks
# slides first. The cues name artifacts, not topics: "prototype" or "LCD
# screen" in a question is usually what was discussed, not a request to see it.
#
# Weights come from the ES2002 dev sweep: bm25 + 0.25 dense beat every other
# text mix; slide channels hurt speech questions; the frame channel (overhead
# camera, 352x288) found nothing either set needed, so it is indexed but off.
#
# Slides are ranked first (slide_ocr / slide_clip / slide_bm25 fused at slide
# level), then enter the clip ranking with the summed slide weight. When the
# slides are trusted as an answer (summed weight >= 1) each slide votes for
# ONE clip, the one under it whose transcript matches best; otherwise its
# vote is spread over every clip it was on screen for.
TEXT_WEIGHTS = {"bm25": 1.0, "dense": 0.25}
VISUAL_WEIGHTS = {"bm25": 0.5, "slide_ocr": 1.0, "slide_clip": 1.0, "slide_bm25": 1.0}
DEFAULT_WEIGHTS = TEXT_WEIGHTS
_VISUAL_CUES = re.compile(
    r"\b(slides?|show(?:s|ed|n|ing)?\s+(?:me|us)|diagram|chart|graph|spreadsheet|(?:photos?|pictures?|images?)\s+of|"
    r"whiteboard|e-?mail|document)\b",
    re.I,
)


def query_intent(q: str) -> str:
    """'visual' if the question asks to see something, else 'speech'."""
    return "visual" if _VISUAL_CUES.search(q) else "speech"
RRF_K = 60
# bge-*-v1.5 retrieves better when short queries carry this instruction
# (passages are embedded without it).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cpu")
    return _reranker
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by for from has have i in is it its of on or so that the their them they this to "
    "uh um was we were will with you yeah okay mm hmm "
    # question words carry no topic
    "what which who whom when where why how did does do about any there our us me show".split()
)


def stem(t: str) -> str:
    """Tiny suffix stripper so decide/decided/deciding and battery/batteries meet."""
    if len(t) > 4:
        if t.endswith("ies"):
            return t[:-3] + "y"
        for suf in ("ing", "ed", "es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[: -len(suf)]
                break
    return t[:-1] if len(t) > 4 and t.endswith("e") else t


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk_id: str
    meeting: str
    t0: float
    t1: float
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    quote: str = ""
    speakers: str = ""
    slide_id: str | None = None
    frame_t: float | None = None
    intent: str = "speech"

    def as_dict(self) -> dict:
        return {
            "meeting": self.meeting,
            "clip": [round(self.t0, 1), round(self.t1, 1)],
            "score": round(self.score, 4),
            "channels": self.ranks,
            "quote": self.quote,
            "speakers": self.speakers,
            "slide": self.slide_id,
            "best_frame_t": self.frame_t,
            "intent": self.intent,
        }


def rrf(
    rankings: dict[str, list[str]],
    weights: dict[str, float],
    k: int = RRF_K,
    shares: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Weighted RRF. `shares[channel][item]` scales one item's vote; a slide
    that stays on screen across n chunks gives each of them 1/n of its vote,
    so one long-lived slide cannot flood the ranking."""
    shares = shares or {}
    scores: dict[str, float] = {}
    for ch, ranked in rankings.items():
        w = weights.get(ch, 0.0)
        if w <= 0:
            continue
        sh = shares.get(ch, {})
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + w * sh.get(cid, 1.0) / (k + rank + 1)
    return scores


def best_quote(query: str, transcript: str) -> str:
    """The transcript line sharing the most query words; ties go to the
    longer line, so a contentful sentence beats a "Mm-hmm"."""
    q = set(tokenize(query))
    best, best_key = "", (-1, -1)
    for line in transcript.splitlines():
        toks = tokenize(line.split(":", 1)[-1])
        key = (len(q & set(toks)), min(len(toks), 25))
        if key > best_key:
            best, best_key = line, key
    return best


class Searcher:
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        query_prefix: bool = True,
        rerank: int = 0,
        rerank_blend: float = 0.5,
    ):
        """`weights=None` routes each query by intent (TEXT_WEIGHTS / VISUAL_WEIGHTS);
        a dict pins one profile. `rerank` = how many fused candidates a
        cross-encoder re-scores (0 = off), blended with the fused rank."""
        self.routed = weights is None
        self.weights = dict(TEXT_WEIGHTS if weights is None else weights)
        self.query_prefix = BGE_QUERY_PREFIX if query_prefix else ""
        self.rerank = rerank
        self.rerank_blend = rerank_blend
        self.meetings_t, self.frames, self.slides, self.chunks = index.tables()
        rows = self.chunks.select(
            self.chunks.chunk_id, self.chunks.meeting_id, self.chunks.t0, self.chunks.t1,
            self.chunks.speakers, self.chunks.transcript, self.chunks.slide_id, self.chunks.text,
        ).collect()
        self.rows = {r["chunk_id"]: r for r in rows}
        self.ids = [r["chunk_id"] for r in rows]
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in rows]) if rows else None
        self.by_meeting: dict[str, list[dict]] = {}
        for r in sorted(rows, key=lambda r: r["t0"]):
            self.by_meeting.setdefault(r["meeting_id"], []).append(r)
        slide_rows = self.slides.select(
            self.slides.slide_id, self.slides.meeting_id, self.slides.t0, self.slides.t1, self.slides.ocr
        ).collect()
        self.slide_rows = {r["slide_id"]: r for r in slide_rows}
        self.slide_ids = [r["slide_id"] for r in slide_rows]
        ocr_tokens = [tokenize(r["ocr"] or "") for r in slide_rows]
        self.slide_bm25 = BM25Okapi(ocr_tokens) if slide_rows and any(ocr_tokens) else None
        self.n_video = self.meetings_t.count()

    # ---- time -> chunk mapping -------------------------------------------------
    def chunk_at(self, meeting: str, t: float) -> str | None:
        rows = self.by_meeting.get(meeting) or []
        if not rows:
            return None
        i = bisect_right([r["t0"] for r in rows], t) - 1
        cands = [rows[j] for j in (i, i + 1) if 0 <= j < len(rows)]
        best = min(cands, key=lambda r: 0 if r["t0"] <= t <= r["t1"] else min(abs(t - r["t0"]), abs(t - r["t1"])))
        return best["chunk_id"]

    def chunks_overlapping(self, meeting: str, t0: float, t1: float) -> list[str]:
        return [r["chunk_id"] for r in self.by_meeting.get(meeting, []) if r["t0"] < t1 and r["t1"] > t0]

    # ---- chunk-level channels -----------------------------------------------------
    def ch_bm25(self, q: str, k: int, scope: set[str] | None) -> list[str]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(q))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = [self.ids[i] for i in order
               if scores[i] > 0 and (scope is None or self.rows[self.ids[i]]["meeting_id"] in scope)]
        return out[:k]

    def ch_dense(self, q: str, k: int, scope: set[str] | None) -> list[str]:
        t = self.chunks
        sim = t.text.similarity(string=self.query_prefix + q)
        query = t.where(t.meeting_id.isin(list(scope))) if scope else t
        return [r["chunk_id"] for r in query.order_by(sim, asc=False).limit(k).select(t.chunk_id).collect()]

    def ch_frame(self, q: str, k: int, scope: set[str] | None) -> tuple[list[str], dict[str, float]]:
        if self.n_video == 0:
            return [], {}
        f = self.frames
        sim = f.frame.similarity(string=q, idx="frame_clip")
        query = f.where(f.meeting_id.isin(list(scope))) if scope else f
        res = query.order_by(sim, asc=False).limit(k).select(f.meeting_id, t=f.frame_attrs.time).collect()
        out, best_t = [], {}
        for r in res:
            cid = self.chunk_at(r["meeting_id"], float(r["t"]))
            if cid and cid not in best_t:
                best_t[cid] = round(float(r["t"]), 1)
                out.append(cid)
        return out, best_t

    # ---- slide-level ranking --------------------------------------------------------
    def rank_slides(self, q: str, weights: dict[str, float], scope: set[str] | None, n: int = 10) -> list[str]:
        s = self.slides
        base = s.where(s.meeting_id.isin(list(scope))) if scope else s
        rankings: dict[str, list[str]] = {}
        if weights.get("slide_ocr", 0) > 0:
            sim = s.ocr.similarity(string=self.query_prefix + q, idx="slide_ocr")
            rankings["slide_ocr"] = [r["slide_id"] for r in
                                     base.order_by(sim, asc=False).limit(n).select(s.slide_id).collect()]
        if weights.get("slide_clip", 0) > 0:
            sim = s.image.similarity(string=q, idx="slide_clip")
            rankings["slide_clip"] = [r["slide_id"] for r in
                                      base.order_by(sim, asc=False).limit(n).select(s.slide_id).collect()]
        if weights.get("slide_bm25", 0) > 0 and self.slide_bm25 is not None:
            sc = self.slide_bm25.get_scores(tokenize(q))
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            rankings["slide_bm25"] = [self.slide_ids[i] for i in order if sc[i] > 0 and
                                      (scope is None or self.slide_rows[self.slide_ids[i]]["meeting_id"] in scope)][:n]
        fused = rrf(rankings, weights)
        return sorted(fused, key=lambda x: -fused[x])[:n]

    def slides_to_chunks(
        self, slide_ids: list[str], one_per_slide: bool, text_rank: dict[str, int]
    ) -> tuple[list[str], dict[str, str], dict[str, float]]:
        out: list[str] = []
        slide_of: dict[str, str] = {}
        share: dict[str, float] = {}
        for sid in slide_ids:
            s = self.slide_rows[sid]
            under = [c for c in self.chunks_overlapping(s["meeting_id"], s["t0"], s["t1"]) if c not in slide_of]
            if not under:
                continue
            if one_per_slide:
                under = [min(under, key=lambda c: (text_rank.get(c, 10**6), self.rows[c]["t0"]))]
            for cid in under:
                slide_of[cid] = sid
                share[cid] = 1.0 / len(under)
            out.extend(under)
        return out, slide_of, share

    # ---- fused search --------------------------------------------------------------
    def weights_for(self, q: str) -> tuple[dict[str, float], str]:
        intent = query_intent(q)
        if not self.routed:
            return self.weights, intent
        return (VISUAL_WEIGHTS if intent == "visual" else TEXT_WEIGHTS), intent

    def search(
        self, q: str, k: int = 10, depth: int = 50, scope: list[str] | set[str] | None = None
    ) -> tuple[list[Hit], dict[str, float]]:
        """`scope`: meeting ids the user may search (e.g. their team's workspace)."""
        weights, intent = self.weights_for(q)
        scope = set(scope) if scope else None
        rankings: dict[str, list[str]] = {}
        timing: dict[str, float] = {}
        shares: dict[str, dict[str, float]] = {}
        frame_for: dict[str, float] = {}
        slide_of: dict[str, str] = {}

        def timed(name, fn, *args):
            t = time.perf_counter()
            out = fn(*args)
            timing[name] = time.perf_counter() - t
            return out

        if weights.get("bm25", 0) > 0:
            rankings["bm25"] = timed("bm25", self.ch_bm25, q, depth, scope)
        if weights.get("dense", 0) > 0:
            rankings["dense"] = timed("dense", self.ch_dense, q, depth, scope)
        slide_w = sum(weights.get(c, 0) for c in SLIDE_CHANNELS)
        if slide_w > 0 and self.slide_rows:
            ranked = timed("slides", self.rank_slides, q, weights, scope)
            text_rank: dict[str, int] = {}
            for ch in ("bm25", "dense"):
                for i, c in enumerate(rankings.get(ch, [])):
                    text_rank[c] = min(text_rank.get(c, 10**6), i)
            rankings["slides"], slide_of, shares["slides"] = self.slides_to_chunks(ranked, slide_w >= 1, text_rank)
        if weights.get("frame_clip", 0) > 0:
            rankings["frame_clip"], frame_for = timed("frame_clip", self.ch_frame, q, depth, scope)

        chunk_w = {"bm25": weights.get("bm25", 0), "dense": weights.get("dense", 0), "slides": slide_w,
                   "frame_clip": weights.get("frame_clip", 0)}
        scores = rrf(rankings, chunk_w, shares=shares)
        order = sorted(scores, key=lambda c: -scores[c])
        if self.rerank and order:
            order = timed("rerank", self._rerank, q, order)
        hits = []
        for cid in order[:k]:
            r = self.rows[cid]
            hits.append(Hit(
                chunk_id=cid, meeting=r["meeting_id"], t0=r["t0"], t1=r["t1"], score=scores[cid],
                ranks={ch: ranked.index(cid) + 1 for ch, ranked in rankings.items() if cid in ranked},
                quote=best_quote(q, r["transcript"]), speakers=r["speakers"],
                slide_id=slide_of.get(cid) or r["slide_id"] or None, frame_t=frame_for.get(cid), intent=intent,
            ))
        return hits, timing

    def _rerank(self, q: str, order: list[str]) -> list[str]:
        head, tail = order[: self.rerank], order[self.rerank :]
        ce = reranker().predict([(q, self.rows[c]["text"]) for c in head], batch_size=32)
        ce_rank = {c: r for r, c in enumerate(sorted(head, key=lambda c: -ce[head.index(c)]))}
        b = self.rerank_blend
        blended = sorted(head, key=lambda c: b * ce_rank[c] + (1 - b) * head.index(c))
        return blended + tail
