# Approaches considered, and what each costs

Ingest costs are per hour of meeting video on a laptop CPU (Apple M3).
"Measured" rows come from this repo; the others are estimates.

| # | approach | ingest / video-hour | query | result here | verdict |
|---|---|---|---|---|---|
| 1 | **text: transcript + slide OCR, BM25 + small embedder (bge-small)** | ~25 s measured (+ ASR if the platform gives no transcript: ~6-12 min, estimated) | 14 ms | held-out summary Hit@5 0.75 | ✅ core |
| 2 | **slide images through CLIP** | inside the ~24 s slide cost, measured | +15 ms | strongest single slide channel (Slide@1 0.69 held out) | ✅ used for visual questions |
| 3 | CLIP on video frames, 1 every 5 s | ~14 s measured | +15 ms | Hit@5 ≈ 0 on every set (overhead camera, 352×288) | ⚠️ indexed, weight 0 |
| 4 | cross-encoder reranker (MiniLM-L6) on the top 20 | 0 | +0.5-0.9 s | +0.02 Hit@5 on dev | ⚠️ off by default |
| 5 | VLM caption on every 10 s clip (Qwen3-VL-2B) | 20-30 min (M3), est. | 0 | not built | ❌ about 10× the cost; meetings are speech-led |
| 6 | VLM captions only at slide changes / keyframes | 2-3 min, est. | 0 | not built | ➡️ next step, for scene questions |
| 7 | native video embeddings (Qwen3-VL-Embedding) | GPU minutes | GPU | not built | ❌ on CPU; candidate reranker |
| 8 | graph RAG (HKUDS VideoRAG) | 24 GB GPU + many LLM calls | seconds | not built | ❌ cost; fits multi-hop "why did we..." questions |

## SDKs considered

| SDK | why / why not |
|---|---|
| **Pixeltable** | chosen: native video and image columns, frame views, computed columns, embedding indexes, incremental, CPU, Apache-2.0 |
| LanceDB | strong embedded vector + full-text store, but no video pipeline; would be the fallback |
| HKUDS VideoRAG | research system; its desktop app needs a 24 GB GPU; too heavy for a CPU-only `run.yaml` |
| NVIDIA VSS blueprint | built for large GPUs |
| Twelve Labs, VideoDB | hosted services, not open-source SDKs |

## Data considered

| data | verdict |
|---|---|
| **AMI Meeting Corpus** | chosen: real multimodal meetings, summary-to-utterance links (free ground truth), timed slides, CC BY 4.0 |
| VideoMME (Task 1 data) | mixed content, but no evidence timestamps: would only score "which video" |
| MEVA surveillance, Nexar dashcam | little or no speech, so less multimodal for search |
