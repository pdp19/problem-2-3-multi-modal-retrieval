import pytest

from mmr import ami


def test_parse_href_range_and_single():
    assert ami.parse_href("X.A.words.xml#id(X.A.words3)..id(X.A.words8)") == ("X.A.words.xml", "X.A.words3", "X.A.words8")
    assert ami.parse_href("X.A.words.xml#id(X.A.words3)") == ("X.A.words.xml", "X.A.words3", "X.A.words3")
    with pytest.raises(ValueError):
        ami.parse_href("garbage")


def test_load_words_merges_speakers_in_time_order_and_skips_untimed(ami_dir):
    words = ami.load_words(ami_dir["meeting"], ami_dir["ann"])
    assert [w.start for w in words] == sorted(w.start for w in words)
    assert {w.speaker for w in words} == {"A", "B"}
    assert all(w.text != "laugh" for w in words)
    assert len(words) == 13


def test_queries_get_evidence_from_summary_links(ami_dir):
    qs = {q.text: q for q in ami.load_queries(ami_dir["meeting"], ami_dir["ann"])}
    assert "Unlinked sentence." not in qs  # no evidence -> not a query
    energy = qs["The team discussed energy sources."]
    assert energy.section == "abstract"
    assert sorted((e.start, e.end) for e in energy.evidence) == [(1.0, 3.0), (30.0, 31.6)]
    solar = qs["The remote will use solar power."]
    assert solar.section == "decisions"
    assert sorted((e.start, e.end) for e in solar.evidence) == [(1.0, 3.0), (5.0, 5.5)]


def test_speaker_roles_only_for_this_meeting(ami_dir):
    roles = ami.speaker_roles(ami_dir["meeting"], ami_dir["ann"])
    assert roles == {"A": "Industrial Designer", "B": "Project Manager"}


def test_slides_from_file_names(ami_dir):
    assert ami.parse_slide_name("ES2002b.277.29__284.31.jpg") == ("ES2002b", 277.29, 284.31)
    assert ami.parse_slide_name("index.html") is None
    slides = ami.load_slides(ami_dir["meeting"])
    assert [(s.start, s.end) for s in slides] == [(0.0, 20.0), (20.0, 50.0)]
