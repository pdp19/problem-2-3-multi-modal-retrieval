"""Fetch the AMI files we index: manual annotations, one camera view, slide captures.

Stdlib only, idempotent: a file whose size matches the server's is skipped,
partial downloads go to `.part` and are renamed when complete.
"""
from __future__ import annotations

import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

_UA = {"User-Agent": "mmr-ami-fetch/1.0"}


def _retry(fn, *args, attempts: int = 4, **kw):
    """The AMI servers time out now and then; retry with backoff."""
    for i in range(attempts):
        try:
            return fn(*args, **kw)
        except (OSError, IOError) as e:
            if i == attempts - 1:
                raise
            print(f"  retry {i + 1} after {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2 * 2 ** i)


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def _fetch_range(url: str, part: Path, start: int, end: int) -> None:
    """Download bytes [start, end] into `part`, resuming from what is already there."""
    have = part.stat().st_size if part.exists() else 0
    want = end - start + 1
    if have >= want:
        return
    hdr = dict(_UA, Range=f"bytes={start + have}-{end}")
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
        if r.status != 206:
            raise IOError(f"server ignored Range for {url}")
        with open(part, "ab") as f:
            shutil.copyfileobj(r, f, length=1 << 20)


def fetch(url: str, dest: Path, quiet: bool = False, connections: int = 8) -> Path:
    return _retry(_fetch, url, dest, quiet, connections)


def _fetch(url: str, dest: Path, quiet: bool = False, connections: int = 8) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = _remote_size(url)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    if size and size > 16 << 20 and connections > 1:
        # The AMI mirror throttles each connection (~100 KB/s), so large files
        # are fetched as parallel byte ranges and stitched together.
        step = -(-size // connections)
        ranges = [(i * step, min(size, (i + 1) * step) - 1) for i in range(connections)]
        pieces = [dest.with_suffix(dest.suffix + f".part{i}") for i in range(connections)]
        with ThreadPoolExecutor(connections) as ex:
            list(ex.map(lambda a: _fetch_range(url, a[0], *a[1]), zip(pieces, ranges)))
        with open(part, "wb") as f:
            for p in pieces:
                with open(p, "rb") as src:
                    shutil.copyfileobj(src, f, length=1 << 20)
        for p in pieces:
            p.unlink()
    else:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=120) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    if size is not None and part.stat().st_size != size:
        raise IOError(f"truncated download: {url} ({part.stat().st_size} of {size} bytes)")
    part.rename(dest)
    if not quiet:
        print(f"  fetched {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return dest


def slide_urls(meeting_id: str) -> list[str]:
    """Slide captures have no directory listing; the corpus download script
    generator returns the exact wget lines, so we ask it and keep the .jpg URLs."""
    series, letter = meeting_id[:-1], meeting_id[-1]
    body = urllib.parse.urlencode(
        [("sessionID", series), (series[:2], letter), ("mediaType", "slides")]
    ).encode()
    req = urllib.request.Request(config.AMI_SCRIPT_CGI, data=body, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    return parse_slide_urls(html, meeting_id)


def parse_slide_urls(html: str, meeting_id: str) -> list[str]:
    pat = re.compile(rf"https://[^\s'\"<>]+/{meeting_id}/slides/{meeting_id}\.[0-9.]+__[0-9.]+\.jpg")
    return sorted(set(pat.findall(html)))


def download_annotations(data_dir: Path | None = None) -> Path:
    root = data_dir or config.DATA_DIR
    out = config.annotations_dir(root)
    if (out / "abstractive").is_dir() and (out / "words").is_dir():
        return out
    zpath = fetch(config.AMI_ANNOTATIONS_URL, root / "ami_public_manual_1.6.2.zip")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    return out


def download_meeting(meeting_id: str, data_dir: Path | None = None, workers: int = 8, video: bool = True) -> dict:
    root = data_dir or config.DATA_DIR
    video_url = f"{config.AMI_MIRROR}/{meeting_id}/video/{meeting_id}.{config.VIDEO_VIEW}.avi"
    sdir = config.slides_dir(meeting_id, root)
    listing = sdir / "_urls.txt"
    if listing.exists():
        urls = listing.read_text().split()
    else:
        urls = _retry(slide_urls, meeting_id)
        sdir.mkdir(parents=True, exist_ok=True)
        if urls:  # an empty listing may be a server hiccup; ask again next time
            listing.write_text("\n".join(urls))
    jobs = [(video_url, config.video_path(meeting_id, root))] if video else []
    jobs += [(u, sdir / u.rsplit("/", 1)[1]) for u in urls]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: fetch(*j, quiet=True), jobs))
    return {"meeting": meeting_id, "video": video, "slides": len(urls)}


def download(meetings: list[str], data_dir: Path | None = None, video: bool = True) -> None:
    print("annotations ...", file=sys.stderr)
    download_annotations(data_dir)
    # Meetings in parallel: the AMI mirror is slow per connection.
    with ThreadPoolExecutor(4) as ex:
        for info in ex.map(lambda m: download_meeting(m, data_dir, video=video), meetings):
            what = "video + " if info["video"] else ""
            print(f"  {info['meeting']}: {what}{info['slides']} slides", file=sys.stderr)
