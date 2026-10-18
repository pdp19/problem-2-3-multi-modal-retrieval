from pathlib import Path

from mmr.ami import Slide, Word
from mmr.chunking import chunk_meeting, slide_at, utterances, Utterance


def w(spk, a, b, t, punc=False):
    return Word(spk, a, b, t, punc)


def test_utterances_split_on_gap_and_join_punctuation():
    words = [w("A", 0, 0.5, "Hello"), w("A", 0.5, 0.5, ",", True), w("A", 0.6, 1.0, "team"),
             w("A", 3.0, 3.5, "Next"), w("B", 0.2, 0.8, "Hi")]
    utts = utterances(words, max_gap=1.0)
    assert [(u.speaker, u.text) for u in utts] == [("A", "Hello, team"), ("B", "Hi"), ("A", "Next")]


def test_long_monologue_is_split_at_max_len():
    words = [w("A", t, t + 0.5, f"w{t}") for t in range(0, 100)]
    utts = utterances(words, max_gap=1.0, max_len=30)
    assert len(utts) >= 3
    assert all(u.end - u.start <= 30 for u in utts)


def test_slide_at_respects_intervals():
    slides = [Slide("M", "s1", 10, 20, Path("a")), Slide("M", "s2", 30, 40, Path("b"))]
    assert slide_at(slides, 15).slide_id == "s1"
    assert slide_at(slides, 25) is None
    assert slide_at(slides, 5) is None
    assert slide_at(slides, 39.9).slide_id == "s2"


def test_chunks_close_at_slide_change_and_target_length():
    utts = [Utterance("A", t, t + 4, f"u{t}") for t in range(0, 120, 5)]
    slides = [Slide("M", "s1", 0, 30, Path("a")), Slide("M", "s2", 30, 200, Path("b"))]
    chunks = chunk_meeting("M", utts, slides, target=45, min_len=10, pause=4)
    # first chunk stops at the slide change (t=30), not at the 45 s target
    assert chunks[0].end <= 30 and chunks[0].slide_id == "s1"
    assert all(c.end - c.start <= 45 + 5 for c in chunks)
    # every utterance lands in exactly one chunk, in order
    flat = [u for c in chunks for u in c.utterances]
    assert flat == utts
    assert chunks[1].slide_id == "s2"


def test_transcript_uses_roles():
    utts = [Utterance("A", 0, 1, "hi")]
    c = chunk_meeting("M", utts, [])[0]
    assert c.transcript({"A": "Project Manager"}) == "Project Manager: hi"
    assert c.slide_id is None
