# How Pixeltable is used, and where it fell short

[Pixeltable](https://github.com/pixeltable/pixeltable) (Apache-2.0, v0.7.18)
is a multimodal data layer: tables with `Video` / `Image` columns, views
produced by iterators, computed columns that run models, and embedding
indexes that stay current as rows arrive. All of it lives in
[`src/mmr/index.py`](../src/mmr/index.py).

## Schema

```python
meetings = pxt.create_table("mmr.meetings", {"meeting_id": pxt.String, "video": pxt.Video})

# one row per sampled frame; frames are decoded on demand, never stored
frames = pxt.create_view("mmr.frames", meetings,
                         iterator=frame_iterator(meetings.video, fps=0.2))
frames.add_embedding_index("frame", idx_name="frame_clip", embedding=clip.using(model_id=CLIP))

slides = pxt.create_table("mmr.slides", {"slide_id": pxt.String, "meeting_id": pxt.String,
                                         "t0": pxt.Float, "t1": pxt.Float, "image": pxt.Image})
slides.add_computed_column(ocr=ocr_text(slides.image))            # @pxt.udf: RapidOCR + re-spacing
slides.add_embedding_index("image", idx_name="slide_clip", embedding=clip.using(model_id=CLIP))
slides.add_embedding_index("ocr",   idx_name="slide_ocr",  embedding=sentence_transformer.using(model_id=BGE))

chunks = pxt.create_table("mmr.chunks", {..., "t0": pxt.Float, "t1": pxt.Float, "text": pxt.String})
chunks.add_embedding_index("text", idx_name="chunk_text", embedding=sentence_transformer.using(model_id=BGE))
```

## Ingest is just inserts

```python
slides.insert([{"slide_id": ..., "image": "ES2002b.284.31__307.96.jpg", ...}])  # OCR + 2 embeddings run here
chunks.insert(rows)                                                             # text embedding runs here
meetings.insert([{"meeting_id": "ES2002b", "video": "ES2002b.Overhead.avi"}])   # frames view + CLIP run here
```

A new meeting computes only its own rows; nothing already indexed is
touched. Re-running `index` skips meetings that are already there.

## Retrieval inside a workspace

```python
sim = chunks.text.similarity(string=BGE_QUERY_PREFIX + question)
chunks.where(chunks.meeting_id.isin(team)).order_by(sim, asc=False).limit(50).select(chunks.chunk_id)

sim = frames.frame.similarity(string=question, idx="frame_clip")         # text -> image via CLIP
frames.where(...).order_by(sim, asc=False).select(frames.meeting_id, t=frames.frame_attrs.time)
```

The same `.similarity()` call serves text→text and text→image, so each
channel is a few lines. Fusion, routing and BM25 are ours
([`src/mmr/search.py`](../src/mmr/search.py)).

## Gaps found

| gap | effect | workaround |
|---|---|---|
| no keyword / BM25 index | the strongest channel on meeting speech sits outside the SDK | `rank_bm25` in memory, loaded at query start (fine to about 10^5-10^6 clips per process) |
| changing a UDF needs the computed column rebuilt | fixing OCR spacing meant re-indexing slides | `index --reset`; a "recompute column" call would help |
| one embedded Postgres per machine (`~/.pixeltable`) | projects share a catalog | `PIXELTABLE_HOME`, plus a named directory (`mmr`) |
| `frame_iterator` decodes the stream to sample it | fine at 352×288 (≈14 s per video-hour); costly for 1080p | keyframes-only mode or a seek-based iterator for HD sources |
| first query loads models (~4 s) | cold start | warm the process at service start |
