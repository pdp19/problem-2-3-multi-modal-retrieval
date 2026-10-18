from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("MMR_DATA_DIR", REPO_ROOT / "data" / "ami"))
ARTIFACTS_DIR = Path(os.environ.get("MMR_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
DOCS_DIR = REPO_ROOT / "docs"

# Pixeltable keeps its catalog in an embedded Postgres under PIXELTABLE_HOME
# (~/.pixeltable by default). Everything we create lives in this directory.
PXT_DIR = os.environ.get("MMR_PXT_DIR", "mmr")

AMI_MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus"
AMI_ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AMI_SCRIPT_CGI = "https://homepages.inf.ed.ac.uk/cgi/simonk/AMI/estimate.cgi"

# One design team (ES2002) across its four meetings: kickoff, functional
# design, conceptual design, detailed design. Cross-meeting questions
# ("what did we decide in the last meeting?") need a whole series.
DEFAULT_MEETINGS = ["ES2002a", "ES2002b", "ES2002c", "ES2002d"]

# Held-out test set: three other teams recorded at the three AMI sites
# (Edinburgh, Idiap, TNO). Fusion weights are tuned on DEFAULT_MEETINGS only.
TEST_MEETINGS = [f"{s}{m}" for s in ("ES2003", "IS1009", "TS3003") for m in "abcd"]

# Camera view to index. The overhead camera sees the table, so prototypes and
# drawings are visible; projected slides come from the slide captures instead.
VIDEO_VIEW = os.environ.get("MMR_VIDEO_VIEW", "Overhead")

# Sampling for the visual channel: one frame every 5 s.
FRAME_FPS = 0.2

TEXT_MODEL = "BAAI/bge-small-en-v1.5"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Chunking: transcript windows of ~45 s, cut early at slide changes or long
# silences, never shorter than 10 s.
CHUNK_TARGET_S = 45.0
CHUNK_MIN_S = 10.0
CHUNK_PAUSE_S = 4.0


def annotations_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or DATA_DIR) / "manual"


def meeting_dir(meeting_id: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DATA_DIR) / "amicorpus" / meeting_id


def video_path(meeting_id: str, data_dir: Path | None = None) -> Path:
    return meeting_dir(meeting_id, data_dir) / "video" / f"{meeting_id}.{VIDEO_VIEW}.avi"


def slides_dir(meeting_id: str, data_dir: Path | None = None) -> Path:
    return meeting_dir(meeting_id, data_dir) / "slides"
