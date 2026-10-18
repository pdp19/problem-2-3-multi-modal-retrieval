# How it is measured

The corpus came with no retrieval benchmark, so the benchmark is built from
AMI's own annotations, plus a small hand-labelled visual set. Code:
[`src/mmr/evaluate.py`](../src/mmr/evaluate.py); queries are written to
`artifacts/benchmark_queries.jsonl` on every `eval` run.

## Query sets

| set | n | queries | ground truth |
|---|---|---|---|
| dev, summary | 69 | ES2002 a-d abstractive summary sentences (abstract / decisions / actions / problems) | time spans of the dialogue acts annotators linked to the sentence (`summlink`) |
| **test, summary** | **204** | same, for ES2003, IS1009, TS3003 (3 teams, 3 recording sites: Edinburgh, Idiap, TNO) | same |
| dev, visual | 18 | "show me the slide ..." questions, hand-written after viewing ES2002 slides | on-screen interval of that slide (from the capture file name) |
| **test, visual** | **16** | same, written on held-out slides before any tuning on them | that slide's interval, plus every re-showing of it (near-identical OCR) |

Visual questions are paraphrased, not copied from the slide (e.g. "photo of
the inside of a remote control showing its circuit board"). Kinds: text
slides, image-only photos, diagrams, a spreadsheet, emails and a document.
Files: [`benchmark/visual_queries.jsonl`](../benchmark/visual_queries.jsonl),
[`benchmark/visual_queries_heldout.jsonl`](../benchmark/visual_queries_heldout.jsonl).

## Protocol

* **Dev / test split by team.** Every design choice (clip length, fusion
  weights, router cues, slide-first ranking, stemming) was made on ES2002 only.
  The held-out numbers are reported once. When a choice could have been read
  off the test set (the dense weight), I took the dev winner.
* **Scope.** Each query searches its team's workspace (the 4 meetings of its
  series), as access control would. A last row searches all 16 meetings: AMI
  teams all designed the same TV remote, so this is a deliberately hard
  stress test.
* **Unit.** A retrieved clip (about 45 s) is a hit if it overlaps any
  ground-truth span in the right meeting.

## Metrics

| metric | meaning | why |
|---|---|---|
| Hit@1 / Hit@5 / Hit@10 | a relevant clip in the top k | "did the user see it without scrolling" |
| MRR@10 | 1 / rank of the first relevant clip | rewards putting it first |
| Meeting@1 | top clip comes from the right meeting | cross-meeting confusion |
| Slide@1 | the slide returned with the top clip was on screen during the evidence | the "show me the slide" promise |
| chance Hit@5 | expected Hit@5 of a random ranking, per query, averaged | the floor: clips are long and evidence can be spread out, so raw Hit@5 can look better than it is |
| p50 / p95 latency | per query, warm | product constraint |

Uncertainty: 95% bootstrap confidence intervals over queries. The
routed-vs-text comparison uses a paired bootstrap (one-sided p).

## Known biases

* Summary sentences are declarative ("The remote will only control TVs") and
  written from transcripts, so they favour text channels. Real users ask
  questions and point at visuals.
* The visual sets are small and written by one person (me). CIs are about
  ±0.15. The routed-vs-text gain is p = 0.03 on dev and p = 0.24 on the
  16 held-out questions: directional, not proven.
* Transcripts are AMI's manual word timings. AMI also ships ASR output; the
  noisy-transcript run is listed as next work.
* Clip-overlap scoring is lenient for long clips. The chance baseline exposes
  how lenient (0.08-0.11 Hit@5).
