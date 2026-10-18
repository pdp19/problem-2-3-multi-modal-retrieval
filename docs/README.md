# Meeting memory: find the moment, the quote and the slide

## The problem

Product teams decide things in meetings, then lose them. Six weeks later nobody
remembers *why* the remote has no teletext button, or which slide showed the
cost breakdown. Meeting tools (Otter, Zoom AI, Gong) search transcripts, but
meetings are multimodal: people say "this one" while a slide or a prototype
is on screen, and half the questions are "show me the slide where...".

**Goal:** ask a team's meeting archive a question; get back the quote, the
exact clip, and the slide that was on screen, fast and on a CPU.

## Data, and a benchmark we didn't have to label

**AMI Meeting Corpus** (CC BY 4.0): real 4-person design meetings with video,
slide captures and word-timed transcripts. Two properties make it a benchmark:

* **Human summaries are linked to the exact utterances that support them.**
  Each summary sentence ("The remote will only control televisions") becomes a
  query; its linked utterances are the ground-truth time spans.
* **Slide capture file names carry their on-screen times**
  (`ES2002b.284.31__307.96.jpg`), so "show me the slide" can be scored.

| query set | n | how built |
|---|---|---|
| dev summary questions | 69 | ES2002 a-d. Used for every design choice |
| **held-out summary questions** | **204** | 12 meetings, 3 other teams, 3 recording sites. Never tuned on |
| dev visual questions | 18 | written by hand after viewing ES2002 slides |
| **held-out visual questions** | **16** | written by hand on held-out slides before any tuning on them |

Each query searches its **team's workspace** (that team's 4 meetings), as
access control would in a real deployment. All 16 AMI teams designed the same
product, so we also report search across *every* meeting as a stress test.

## How the SDK is used (Pixeltable)

| Pixeltable feature | used for |
|---|---|
| `create_table` with `pxt.Video` / `pxt.Image` columns | meetings, slides, clips |
| `create_view(..., iterator=frame_iterator(video, fps=0.2))` | 1 frame every 5 s, never written to disk (`unstored`) |
| `@pxt.udf` + `add_computed_column` | slide OCR (RapidOCR, ONNX) + word re-segmentation |
| `add_embedding_index` × 4 | CLIP on frames and slide images; bge-small on clip text and slide OCR |
| `.similarity(string=...)` + `where(meeting_id.isin(team))` | ranked retrieval inside a team's workspace |
| incremental inserts | a new meeting computes only its own rows; nothing is re-indexed |

**Pipeline.** Timed words → utterances → clips of about 45 s, cut early at
slide changes and long pauses (slide changes are a meeting's strongest topic
boundary; this is the chunker the Task 1 classifier routes `screen_content`
and `talking_head` videos to). A small rule router reads the question's
**intent**:

* **speech** (most questions): BM25 + 0.25 × dense over clip text.
* **visual** (names an artifact: slide, diagram, email, photo of...): rank
  *slides* first (OCR-dense + OCR-BM25 + CLIP image), then return the clip
  under each slide whose transcript fits best.

The channels are fused with weighted reciprocal-rank fusion. All weights were
chosen on the dev set only.

## Results (held-out, team-scoped; full tables in [results.md](results.md))

| | held-out summary Qs (204) | held-out visual Qs (16) |
|---|---|---|
| random ranking, Hit@5 | 0.09 | 0.08 |
| BM25 only | Hit@5 0.71 · MRR 0.56 | Hit@1 0.69 · MRR 0.75 |
| text profile (fixed) | Hit@5 **0.75** · MRR 0.55 | Hit@1 0.62 · MRR 0.74 |
| visual profile (fixed) | Hit@5 0.32 · MRR 0.24 | Hit@1 **0.75** · Hit@5 **0.94** · MRR **0.80** |
| **routed by intent (default)** | Hit@5 **0.75** [0.69, 0.81] · MRR 0.54 | Hit@1 **0.75** · Hit@5 **0.94** · MRR **0.80** |
| routed, across all 16 meetings | Hit@5 0.56 | Hit@5 0.81 |

Warm query latency on the M3: p50 14 ms (speech), 30 ms (visual); first
query pays ~4 s of model loading.

**What we learned**

1. **No single fusion wins both kinds of question.** The visual profile is the
   best on visual questions and the worst on speech questions (0.32 vs 0.75).
   Routing by intent gets the best of each; the router misroutes 1 of 273
   summary questions and 0 of 34 visual ones.
2. **Keywords beat the dense model on meeting speech.** BM25 alone almost
   matches the hybrid. Small sentence embedders struggle with disfluent speech
   ("um so the uh battery"); the dense model adds only a little.
3. **For "show me" questions, rank the artifact, not the transcript.** Our
   first fusion spread each slide's vote over every clip under it, which made
   things worse. Ranking slides first and attaching the best clip fixed it.
   CLIP on slide *images* is the strongest single slide channel (Slide@1 0.69
   held out), because photos and diagrams have little text to OCR.
4. **The frame channel found nothing.** The overhead camera is 352×288,
   top-down; CLIP frames scored ~0 on every set. It is indexed but weighted 0:
   measured, not assumed.
5. **Scope matters more than models.** Searching all teams at once drops
   Hit@5 from 0.75 to 0.56: 16 teams discussing the same remote look alike.

**Honest limits.** Summary sentences are declarative and written from
transcripts, so they favour text channels. The visual sets are small and
written by me (95% CIs about ±0.15). The visual gain over the text profile is
directional on held-out data (p = 0.24) and significant on dev (p = 0.03).
Transcripts are AMI's manual words; AMI's ASR output is included in the
corpus, so the noisy-transcript ablation is the next run.

## Cost (measured on an Apple M3 CPU, 16 meetings, 8.6 h of meetings)

| stage | measured | per meeting-hour |
|---|---|---|
| slide OCR + 2 embeddings | 203 s for 273 slides | ~24 s |
| clip text embedding | 12 s for 8.6 h | ~1.4 s |
| frames: decode + CLIP every 5 s | 31 s for 2.25 h of video | ~14 s |
| index on disk (incl. Postgres catalog) | < 75 MB | < 9 MB |
| query (warm) | p50 14-30 ms, p95 < 40 ms | — |

That is **about 40 CPU-seconds per meeting-hour** when transcripts come with
the meeting (Zoom/Teams provide them). Adding our own ASR
(faster-whisper-small, estimated, not measured) costs about 6-12 CPU-min per
hour and dominates. **100 h: about 1 CPU-hour without ASR, about 10-20 with
it; under $5 on cloud CPUs.** Cost grows linearly; queries stay in
milliseconds because BM25 and the vector indexes are per-workspace.

## Options considered

| approach | ingest cost / video-hour | verdict |
|---|---|---|
| **text-first: transcript + slide OCR, BM25 + small embedder** | seconds (+ ASR) | ✅ the core; strongest channel |
| **+ image channel (CLIP on slides)** | ~24 s | ✅ wins on photos and diagrams |
| + CLIP on video frames | ~14 s | ⚠️ built and measured: no value on this camera |
| + cross-encoder reranker (MiniLM) | 0 | ⚠️ +0.02 Hit@5 on dev for +0.5-0.9 s per query: off by default |
| VLM caption of every clip | 20-30 min (M3) | ❌ about 10× the cost for speech-led content |
| native video embeddings (Qwen3-VL-Embedding) | GPU minutes | ❌ for CPU; candidate reranker |
| graph RAG (HKUDS VideoRAG) | 24 GB GPU + LLM calls | ❌ cost; good for multi-hop "why" questions |

## SDK gaps found, and what's next

* **Pixeltable has no keyword (BM25) index**, so BM25 runs in memory beside it.
  A built-in full-text index would remove the second store.
* **Changing a UDF means rebuilding its computed column** (we rebuilt the OCR
  column to fix spacing); the versioning helps, but there is no "recompute
  column" call.
* The embedded Postgres lives in `~/.pixeltable`, shared by every project on
  the machine; per-project homes need `PIXELTABLE_HOME`.
* **Next:** measure on AMI's ASR transcripts; add close-up/corner cameras and
  VLM captions on slide-change keyframes for scene questions ("when did they
  show the prototype?"); an answer layer that writes a short reply citing the
  clips; then scale to all ~140 annotated meetings (text-only eval needs no
  video download).
