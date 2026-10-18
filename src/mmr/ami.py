"""Parse the AMI NXT annotations we need.

Words (per speaker, with times) are the only thing the index is built from:
any ASR produces them, so the pipeline works unchanged on real recordings.
Dialogue acts, abstractive summaries and summary links are used only as
ground truth for evaluation.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import config

NITE = "{http://nite.sourceforge.net/}"
_HREF = re.compile(r"^(?P<file>[^#]+)#id\((?P<a>[^)]+)\)(?:\.\.id\((?P<b>[^)]+)\))?$")
_SLIDE = re.compile(r"^(?P<m>[A-Z]{2}\d{4}[a-d])\.(?P<t0>[0-9.]+)__(?P<t1>[0-9.]+)\.jpg$")
SUMMARY_SECTIONS = ("abstract", "decisions", "actions", "problems")


@dataclass
class Word:
    speaker: str
    start: float
    end: float
    text: str
    punc: bool = False


@dataclass
class Span:
    meeting: str
    start: float
    end: float


@dataclass
class Query:
    qid: str
    meeting: str
    section: str
    text: str
    evidence: list[Span] = field(default_factory=list)


@dataclass
class Slide:
    meeting: str
    slide_id: str
    start: float
    end: float
    path: Path


def parse_href(href: str) -> tuple[str, str, str]:
    m = _HREF.match(href.strip())
    if not m:
        raise ValueError(f"unrecognised NXT href: {href}")
    return m["file"], m["a"], m["b"] or m["a"]


def _speakers(meeting: str, ann: Path) -> list[str]:
    return sorted(p.name.split(".")[1] for p in (ann / "words").glob(f"{meeting}.*.words.xml"))


@lru_cache(maxsize=64)
def _word_file(path: Path) -> tuple[list[str], dict[str, int], list[Word | None]]:
    """Ids in document order, id -> position, and the timed word at each position
    (None for untimed tokens such as vocal sounds or gaps)."""
    speaker = path.name.split(".")[1]
    ids, words = [], []
    for el in ET.parse(path).getroot():
        wid = el.get(f"{NITE}id")
        if wid is None:
            continue
        ids.append(wid)
        st, en = el.get("starttime"), el.get("endtime")
        if el.tag == "w" and st and en and el.text:
            words.append(Word(speaker, float(st), float(en), el.text.strip(), el.get("punc") == "true"))
        else:
            words.append(None)
    return ids, {w: i for i, w in enumerate(ids)}, words


def load_words(meeting: str, ann: Path | None = None) -> list[Word]:
    ann = ann or config.annotations_dir()
    out: list[Word] = []
    for spk in _speakers(meeting, ann):
        _, _, words = _word_file(ann / "words" / f"{meeting}.{spk}.words.xml")
        out.extend(w for w in words if w is not None)
    out.sort(key=lambda w: (w.start, w.end))
    return out


def _range_span(meeting: str, href: str, ann: Path) -> Span | None:
    fname, a, b = parse_href(href)
    _, pos, words = _word_file(ann / "words" / fname)
    timed = [w for w in words[pos[a] : pos[b] + 1] if w is not None]
    if not timed:
        return None
    return Span(meeting, min(w.start for w in timed), max(w.end for w in timed))


def dialogue_act_spans(meeting: str, ann: Path | None = None) -> dict[str, Span]:
    ann = ann or config.annotations_dir()
    spans: dict[str, Span] = {}
    for path in (ann / "dialogueActs").glob(f"{meeting}.*.dialog-act.xml"):
        for dact in ET.parse(path).getroot().iter("dact"):
            child = dact.find(f"{NITE}child")
            if child is None:
                continue
            span = _range_span(meeting, child.get("href"), ann)
            if span is not None:
                spans[dact.get(f"{NITE}id")] = span
    return spans


def load_queries(meeting: str, ann: Path | None = None) -> list[Query]:
    """Abstractive summary sentences as queries; their evidence is the time span
    of every dialogue act the annotators linked to that sentence."""
    ann = ann or config.annotations_dir()
    abs_path = ann / "abstractive" / f"{meeting}.abssumm.xml"
    link_path = ann / "extractive" / f"{meeting}.summlink.xml"
    if not abs_path.exists() or not link_path.exists():
        return []
    queries: dict[str, Query] = {}
    for section in ET.parse(abs_path).getroot():
        name = section.tag
        for s in section.iter("sentence"):
            sid = s.get(f"{NITE}id")
            text = " ".join((s.text or "").split())
            if sid and text:
                queries[sid] = Query(sid, meeting, name, text)
    das = dialogue_act_spans(meeting, ann)
    for link in ET.parse(link_path).getroot().iter("summlink"):
        ext = abst = None
        for p in link.iter(f"{NITE}pointer"):
            _, a, _ = parse_href(p.get("href"))
            if p.get("role") == "extractive":
                ext = a
            elif p.get("role") == "abstractive":
                abst = a
        if abst in queries and ext in das:
            queries[abst].evidence.append(das[ext])
    return [q for q in queries.values() if q.evidence]


ROLE_NAMES = {
    "PM": "Project Manager",
    "ID": "Industrial Designer",
    "UI": "User Interface Designer",
    "ME": "Marketing Expert",
}


def speaker_roles(meeting: str, ann: Path | None = None) -> dict[str, str]:
    """Speaker letter -> role name. In a real deployment this comes from the
    meeting platform's participant list, not from annotation."""
    ann = ann or config.annotations_dir()
    path = ann / "corpusResources" / "meetings.xml"
    roles: dict[str, str] = {}
    if path.exists():
        for spk in ET.parse(path).getroot().iter("speaker"):
            if (spk.get(f"{NITE}id") or "").startswith(meeting + "_"):
                roles[spk.get("nxt_agent")] = ROLE_NAMES.get(spk.get("role"), spk.get("role") or "")
    return roles


def parse_slide_name(name: str) -> tuple[str, float, float] | None:
    m = _SLIDE.match(name)
    return (m["m"], float(m["t0"]), float(m["t1"])) if m else None


def load_slides(meeting: str, data_dir: Path | None = None) -> list[Slide]:
    out = []
    for p in sorted(config.slides_dir(meeting, data_dir).glob(f"{meeting}.*.jpg")):
        parsed = parse_slide_name(p.name)
        if parsed:
            m, t0, t1 = parsed
            out.append(Slide(m, p.stem, t0, t1, p))
    out.sort(key=lambda s: s.start)
    return out
