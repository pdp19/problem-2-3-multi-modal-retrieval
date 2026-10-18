# Results

Index: 16 meetings, 831 chunks (video frames indexed for ES2002a, ES2002b, ES2002c, ES2002d). Each query searches its team's workspace (the 4 meetings of its series); the last row of each table searches every meeting at once.
Hit@k: a top-k clip overlaps the annotated evidence. Meeting@1: top clip is from the right meeting. Slide@1: the slide returned with the top clip was on screen during the evidence.

## dev: summary questions: 69 queries (random ranking, scoped: Hit@5 = 0.09)

| configuration | Hit@1 | Hit@5 [95% CI] | MRR@10 | Meeting@1 | Slide@1 (n) | p50 / p95 ms |
|---|---|---|---|---|---|---|
| bm25 only | 0.36 | 0.68 [0.58, 0.78] | 0.50 | 0.67 | 0.40 (42) | 1 / 2 |
| dense only | 0.25 | 0.51 [0.39, 0.62] | 0.37 | 0.52 | 0.29 (42) | 13 / 18 |
| slide OCR (dense) only | 0.04 | 0.13 [0.06, 0.22] | 0.08 | 0.32 | 0.12 (42) | 12 / 16 |
| slide OCR (bm25) only | 0.06 | 0.13 [0.06, 0.22] | 0.09 | 0.42 | 0.14 (42) | 1 / 2 |
| slide CLIP only | 0.06 | 0.09 [0.03, 0.16] | 0.07 | 0.38 | 0.10 (42) | 14 / 15 |
| frame CLIP only | 0.00 | 0.03 [0.00, 0.07] | 0.02 | 0.20 | 0.05 (42) | 17 / 18 |
| text profile (fixed) | 0.41 | 0.71 [0.59, 0.81] | 0.53 | 0.67 | 0.45 (42) | 14 / 19 |
| visual profile (fixed) | 0.10 | 0.19 [0.10, 0.29] | 0.16 | 0.45 | 0.14 (42) | 30 / 34 |
| routed by intent (default) | 0.41 | 0.71 [0.59, 0.81] | 0.53 | 0.67 | 0.45 (42) | 15 / 19 |
| routed, unscoped (all meetings) | 0.13 | 0.35 [0.25, 0.45] | 0.22 | 0.19 | 0.07 (42) | 13 / 19 |

Paired bootstrap, routed vs text profile on MRR@10: one-sided p = 1.000.

| section (routed) | n | Hit@5 | MRR@10 |
|---|---|---|---|
| abstract | 22 | 0.77 | 0.62 |
| decisions | 26 | 0.81 | 0.60 |
| actions | 7 | 0.29 | 0.26 |
| problems | 14 | 0.64 | 0.40 |

## test: summary questions (held out): 204 queries (random ranking, scoped: Hit@5 = 0.09)

| configuration | Hit@1 | Hit@5 [95% CI] | MRR@10 | Meeting@1 | Slide@1 (n) | p50 / p95 ms |
|---|---|---|---|---|---|---|
| bm25 only | 0.43 | 0.71 [0.65, 0.77] | 0.56 | 0.66 | 0.39 (136) | 1 / 2 |
| dense only | 0.30 | 0.59 [0.53, 0.66] | 0.43 | 0.59 | 0.32 (136) | 13 / 17 |
| slide OCR (dense) only | 0.05 | 0.23 [0.18, 0.29] | 0.12 | 0.36 | 0.11 (136) | 12 / 17 |
| slide OCR (bm25) only | 0.12 | 0.22 [0.17, 0.28] | 0.16 | 0.41 | 0.18 (136) | 1 / 2 |
| slide CLIP only | 0.09 | 0.18 [0.13, 0.23] | 0.13 | 0.33 | 0.13 (136) | 14 / 15 |
| frame CLIP only | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 (136) | 15 / 16 |
| text profile (fixed) | 0.39 | 0.75 [0.70, 0.81] | 0.55 | 0.64 | 0.37 (136) | 14 / 19 |
| visual profile (fixed) | 0.15 | 0.32 [0.26, 0.39] | 0.24 | 0.42 | 0.23 (136) | 30 / 36 |
| routed by intent (default) | 0.39 | 0.75 [0.69, 0.81] | 0.54 | 0.64 | 0.37 (136) | 14 / 19 |
| routed, unscoped (all meetings) | 0.25 | 0.56 [0.49, 0.63] | 0.38 | 0.40 | 0.23 (136) | 14 / 19 |

Paired bootstrap, routed vs text profile on MRR@10: one-sided p = 1.000.

| section (routed) | n | Hit@5 | MRR@10 |
|---|---|---|---|
| abstract | 87 | 0.80 | 0.61 |
| decisions | 62 | 0.65 | 0.44 |
| actions | 17 | 0.94 | 0.69 |
| problems | 38 | 0.71 | 0.49 |

## dev: visual questions: 18 queries (random ranking, scoped: Hit@5 = 0.11)

| configuration | Hit@1 | Hit@5 [95% CI] | MRR@10 | Meeting@1 | Slide@1 (n) | p50 / p95 ms |
|---|---|---|---|---|---|---|
| bm25 only | 0.50 | 0.89 [0.72, 1.00] | 0.64 | 0.78 | 0.44 (18) | 1 / 1 |
| dense only | 0.44 | 0.72 [0.50, 0.89] | 0.58 | 0.61 | 0.33 (18) | 13 / 17 |
| slide OCR (dense) only | 0.72 | 0.83 [0.67, 1.00] | 0.79 | 0.78 | 0.61 (18) | 12 / 16 |
| slide OCR (bm25) only | 0.67 | 0.89 [0.72, 1.00] | 0.77 | 0.72 | 0.61 (18) | 1 / 1 |
| slide CLIP only | 0.72 | 0.83 [0.67, 1.00] | 0.76 | 0.78 | 0.67 (18) | 14 / 15 |
| frame CLIP only | 0.00 | 0.06 [0.00, 0.17] | 0.03 | 0.28 | 0.00 (18) | 18 / 18 |
| text profile (fixed) | 0.61 | 0.83 [0.67, 1.00] | 0.72 | 0.78 | 0.44 (18) | 14 / 17 |
| visual profile (fixed) | 0.78 | 0.89 [0.72, 1.00] | 0.82 | 0.83 | 0.67 (18) | 30 / 31 |
| routed by intent (default) | 0.78 | 0.89 [0.72, 1.00] | 0.82 | 0.83 | 0.67 (18) | 30 / 31 |
| routed, unscoped (all meetings) | 0.56 | 0.83 [0.67, 1.00] | 0.69 | 0.56 | 0.56 (18) | 30 / 32 |

Paired bootstrap, routed vs text profile on MRR@10: one-sided p = 0.034.

## test: visual questions (held out): 16 queries (random ranking, scoped: Hit@5 = 0.08)

| configuration | Hit@1 | Hit@5 [95% CI] | MRR@10 | Meeting@1 | Slide@1 (n) | p50 / p95 ms |
|---|---|---|---|---|---|---|
| bm25 only | 0.69 | 0.88 [0.69, 1.00] | 0.75 | 1.00 | 0.56 (16) | 1 / 1 |
| dense only | 0.56 | 0.88 [0.69, 1.00] | 0.64 | 0.81 | 0.50 (16) | 12 / 17 |
| slide OCR (dense) only | 0.56 | 0.88 [0.69, 1.00] | 0.68 | 0.81 | 0.50 (16) | 12 / 16 |
| slide OCR (bm25) only | 0.69 | 0.81 [0.62, 1.00] | 0.72 | 0.88 | 0.50 (16) | 1 / 1 |
| slide CLIP only | 0.75 | 0.94 [0.81, 1.00] | 0.82 | 0.94 | 0.69 (16) | 13 / 14 |
| frame CLIP only | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 (16) | 14 / 15 |
| text profile (fixed) | 0.62 | 0.88 [0.69, 1.00] | 0.74 | 0.94 | 0.56 (16) | 14 / 18 |
| visual profile (fixed) | 0.75 | 0.94 [0.81, 1.00] | 0.80 | 0.88 | 0.56 (16) | 30 / 31 |
| routed by intent (default) | 0.75 | 0.94 [0.81, 1.00] | 0.80 | 0.88 | 0.56 (16) | 29 / 32 |
| routed, unscoped (all meetings) | 0.69 | 0.81 [0.62, 1.00] | 0.75 | 0.69 | 0.56 (16) | 29 / 31 |

Paired bootstrap, routed vs text profile on MRR@10: one-sided p = 0.240.

Ingest cost (CPU, Apple M3):

| meeting | minutes | chunks | slides | slides OCR+embed s | chunks embed s | frames CLIP s |
|---|---|---|---|---|---|---|
| ES2002a | 18.5 | 29 | 7 | 10.6 | 0.3 | 4.4 |
| ES2002b | 36.1 | 63 | 8 | 2.6 | 0.8 | 7.6 |
| ES2002c | 40.1 | 75 | 22 | 8.1 | 0.9 | 8.3 |
| ES2002d | 40.0 | 64 | 6 | 6.0 | 0.9 | 10.2 |
| ES2003a | 18.8 | 24 | 10 | 9.0 | 0.2 | - |
| ES2003b | 35.0 | 58 | 35 | 18.0 | 0.6 | - |
| ES2003c | 37.5 | 56 | 19 | 9.8 | 0.6 | - |
| ES2003d | 39.1 | 57 | 12 | 8.9 | 0.9 | - |
| IS1009a | 13.4 | 19 | 1 | 0.8 | 0.2 | - |
| IS1009b | 33.8 | 61 | 32 | 21.9 | 0.9 | - |
| IS1009c | 29.8 | 59 | 48 | 32.6 | 0.8 | - |
| IS1009d | 32.1 | 55 | 27 | 46.7 | 1.1 | - |
| TS3003a | 24.6 | 43 | 15 | 8.9 | 0.4 | - |
| TS3003b | 35.8 | 53 | 31 | 18.9 | 0.7 | - |
| TS3003c | 38.0 | 51 | 0 | 0.0 | 0.5 | - |
| TS3003d | 43.2 | 64 | 0 | 0.0 | 1.5 | - |
