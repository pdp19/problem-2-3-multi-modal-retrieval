"""Synthetic AMI-style annotations, so tests need neither the corpus nor the network."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NS = 'xmlns:nite="http://nite.sourceforge.net/"'


def _words(meeting, spk, words):
    punc = ' punc="true"'
    rows = "\n".join(
        f'<w nite:id="{meeting}.{spk}.words{i}" starttime="{a}" endtime="{b}"{punc if t in ".,?" else ""}>{t}</w>'
        for i, (a, b, t) in enumerate(words)
    )
    return f'<?xml version="1.0"?>\n<nite:root {NS}>\n{rows}\n<vocalsound nite:id="{meeting}.{spk}.vs0" type="laugh"/>\n</nite:root>'


@pytest.fixture
def ami_dir(tmp_path, monkeypatch):
    m = "XX0001a"
    ann = tmp_path / "manual"
    for d in ("words", "dialogueActs", "abstractive", "extractive", "corpusResources"):
        (ann / d).mkdir(parents=True)
    (ann / "words" / f"{m}.A.words.xml").write_text(_words(m, "A", [
        (1.0, 1.4, "We"), (1.4, 1.9, "need"), (1.9, 2.5, "solar"), (2.5, 3.0, "power"), (3.0, 3.0, "."),
        (30.0, 30.5, "Battery"), (30.5, 31.0, "is"), (31.0, 31.6, "cheaper"), (31.6, 31.6, "."),
    ]))
    (ann / "words" / f"{m}.B.words.xml").write_text(_words(m, "B", [
        (5.0, 5.5, "Agreed"), (5.5, 5.5, "."), (60.0, 60.6, "Next"), (60.6, 61.0, "meeting"),
    ]))
    (ann / "dialogueActs" / f"{m}.A.dialog-act.xml").write_text(
        f'<?xml version="1.0"?>\n<nite:root {NS}>'
        f'<dact nite:id="{m}.A.dialog-act.1"><nite:child href="{m}.A.words.xml#id({m}.A.words0)..id({m}.A.words4)"/></dact>'
        f'<dact nite:id="{m}.A.dialog-act.2"><nite:child href="{m}.A.words.xml#id({m}.A.words5)..id({m}.A.words8)"/></dact>'
        "</nite:root>"
    )
    (ann / "dialogueActs" / f"{m}.B.dialog-act.xml").write_text(
        f'<?xml version="1.0"?>\n<nite:root {NS}>'
        f'<dact nite:id="{m}.B.dialog-act.1"><nite:child href="{m}.B.words.xml#id({m}.B.words0)"/></dact>'
        "</nite:root>"
    )
    (ann / "abstractive" / f"{m}.abssumm.xml").write_text(
        f'<?xml version="1.0"?>\n<nite:root {NS}>'
        f'<abstract nite:id="a1"><sentence nite:id="{m}.s.1">The team discussed energy sources.</sentence></abstract>'
        f'<decisions nite:id="d1"><sentence nite:id="{m}.s.2">The remote will use solar power.</sentence>'
        f'<sentence nite:id="{m}.s.3">Unlinked sentence.</sentence></decisions>'
        "</nite:root>"
    )
    links = [("A", 1, 1), ("A", 2, 1), ("A", 1, 2), ("B", 1, 2)]
    (ann / "extractive" / f"{m}.summlink.xml").write_text(
        f'<?xml version="1.0"?>\n<nite:root {NS}>' + "".join(
            f'<summlink nite:id="l{i}">'
            f'<nite:pointer role="extractive" href="{m}.{s}.dialog-act.xml#id({m}.{s}.dialog-act.{d})"/>'
            f'<nite:pointer role="abstractive" href="{m}.abssumm.xml#id({m}.s.{q})"/></summlink>'
            for i, (s, d, q) in enumerate(links)
        ) + "</nite:root>"
    )
    (ann / "corpusResources" / "meetings.xml").write_text(
        f'<?xml version="1.0"?>\n<nite:root {NS}><meeting>'
        f'<speaker nite:id="{m}_1" nxt_agent="A" role="ID"/><speaker nite:id="{m}_2" nxt_agent="B" role="PM"/>'
        f'<speaker nite:id="XX0001b_1" nxt_agent="A" role="ME"/></meeting></nite:root>'
    )
    slides = tmp_path / "amicorpus" / m / "slides"
    slides.mkdir(parents=True)
    for name in (f"{m}.0.00__20.00.jpg", f"{m}.20.00__50.00.jpg"):
        (slides / name).write_bytes(b"")
    from mmr import ami, config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    ami._word_file.cache_clear()
    return {"meeting": m, "ann": ann, "root": tmp_path}
