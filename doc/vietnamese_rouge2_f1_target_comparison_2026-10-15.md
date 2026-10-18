# Vietnamese ROUGE-2 F1 Results Against Targets

Scores are ROUGE-2 F1 from the supplied PyROUGE / ROUGE-1.5.5 report, converted from the 0-1 scale to percentages. Target values are transcribed from the supplied comparison image. The difference is `SEAM - target`, in percentage points.

| Dataset | SEAM ROUGE-2 F1 (%) | Target (%) | Difference (pp) | Meets target |
|---|---:|---:|---:|:---:|
| WikiLingua | 31.731 | 33.120 | -1.389 | No |
| Vietnews | 24.665 | 34.240 | -9.575 | No |
| XLSum | 24.863 | 14.390 | +10.473 | Yes |
| LegalText (`legal`) | 54.161 | 48.760 | +5.401 | Yes |
| Book (`book`) | 26.579 | 34.530 | -7.951 | No |
| ViM | 44.114 | 44.380 | -0.266 | No |
| VLSP2022 (`vlsp22`) | 37.528 | 44.270 | -6.742 | No |

SEAM meets the listed target on 2 of 7 datasets: XLSum and LegalText. ViM is 0.266 percentage points below its target. Direct comparison assumes the target values use the same test split, preprocessing, and ROUGE-1.5.5 settings as the supplied report.

## Epoch 004 — results supplied on 2026-10-18

Dataset names follow the same order as the previous reports, confirmed by their
sample counts. The run identity and whether this is SFT or GRPO were not specified.
All scores below are F1 percentages, rounded to two decimal places. Differences
and target decisions use the unrounded values; differences are percentage points.

Overall, 32,121 examples: **ROUGE-1 = 60.87, ROUGE-2 = 30.46, ROUGE-L = 39.86**.

| Dataset | Samples | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | R2 target | R2 − target (pp) | Meets R2 target |
|---|---:|---:|---:|---:|---:|---:|:---:|
| WikiLingua | 3,901 | 58.53 | 30.00 | 40.49 | 33.12 | -3.12 | No |
| Vietnews | 22,644 | 61.10 | 30.13 | 39.69 | 34.24 | -4.11 | No |
| XLSum | 4,013 | 56.40 | 23.53 | 35.75 | 14.39 | +9.14 | Yes |
| LegalText | 1,086 | 78.31 | 60.28 | 57.59 | 48.76 | +11.52 | Yes |
| Book | 77 | 58.58 | 28.35 | 32.03 | 34.53 | -6.18 | No |
| ViM | 300 | 71.66 | 44.70 | 39.10 | 44.38 | +0.32 | Yes |
| VLSP2022 | 100 | 61.90 | 36.69 | 35.18 | 44.27 | -7.58 | No |

Epoch 004 meets **3 of 7** R2 targets: XLSum, LegalText, and ViM. The unweighted
mean R2 across the seven datasets is **36.24**; this is distinct from the supplied
overall score of **30.46**. Direct comparison assumes matching evaluation protocols.

## `seam_epoch_003` — results supplied on 2026-10-18

This is the newly supplied `seam_epoch_003` report, not the earlier
`seam_vi_v2_003` result. It uses 32,121 examples and the stated legacy PyROUGE
ROUGE-1.5.5 protocol (`-a -m -n 2 -d`). Whether these checkpoints belong to the
same training run has not been independently verified.

Overall F1: **ROUGE-1 = 60.48, ROUGE-2 = 30.44, ROUGE-L = 39.89**.
Epoch 004 overall F1: **60.87 / 30.46 / 39.86** respectively.

| Dataset | Epoch 003 R1 | Epoch 003 R2 | Epoch 003 RL | Epoch 004 R2 | R2 change: 004 − 003 (pp) | R2 target |
|---|---:|---:|---:|---:|---:|---:|
| WikiLingua | 57.20 | 29.68 | 40.38 | 30.00 | +0.32 | 33.12 |
| Vietnews | 60.78 | 30.15 | 39.72 | 30.13 | -0.02 | 34.24 |
| XLSum | 56.28 | 23.66 | 35.94 | 23.53 | -0.12 | 14.39 |
| LegalText | 78.78 | 60.32 | 57.50 | 60.28 | -0.04 | 48.76 |
| Book | 56.38 | 26.83 | 31.62 | 28.35 | +1.52 | 34.53 |
| ViM | 71.54 | 44.79 | 39.36 | 44.70 | -0.09 | 44.38 |
| VLSP2022 | 59.49 | 34.96 | 34.42 | 36.69 | +1.74 | 44.27 |

Both checkpoints meet **3 of 7** R2 targets: XLSum, LegalText, and ViM.
Epoch 004 improves R2 on WikiLingua, Book and VLSP2022; it is slightly lower on
the other four datasets. Unweighted mean dataset R2 increases from **35.77** to
**36.24** (**+0.47 pp**), whereas the supplied overall R2 changes by only
**+0.014 pp** before rounding. Differences are computed from the original scores,
so they can differ from subtracting the rounded table entries. These descriptive
differences do not establish statistical significance or isolate a training effect.

## `seam_epoch_005` — results supplied on 2026-10-18

The report specifies SEAM vLLM HTTP inference with an encoder input cap of 8192
tokens and output cap of 1024 tokens. All 32,121 predictions were newly generated
(`reused=0`). These generation limits were not provided for the epoch 003/004
reports, so their equality across checkpoints has not been confirmed.

| Dataset | Samples | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | R2 target | R2 − target (pp) | Meets R2 target |
|---|---:|---:|---:|---:|---:|---:|:---:|
| WikiLingua | 3,901 | 58.36 | 29.65 | 40.06 | 33.12 | -3.47 | No |
| Vietnews | 22,644 | 60.95 | 29.45 | 39.23 | 34.24 | -4.79 | No |
| XLSum | 4,013 | 56.35 | 23.08 | 35.32 | 14.39 | +8.69 | Yes |
| LegalText | 1,086 | 79.66 | 61.12 | 57.68 | 48.76 | +12.36 | Yes |
| Book | 77 | 59.24 | 28.63 | 32.00 | 34.53 | -5.90 | No |
| ViM | 300 | 71.61 | 44.01 | 38.30 | 44.38 | -0.37 | No |
| VLSP2022 | 100 | 62.75 | 37.40 | 35.56 | 44.27 | -6.87 | No |

### Checkpoint summary

Overall columns reproduce the supplied Perl scores; macro R2 is the arithmetic
mean of the seven dataset R2 scores, calculated before rounding.

| Epoch | Overall R1 | Overall R2 | Overall RL | Macro dataset R2 | R2 targets met |
|---|---:|---:|---:|---:|:---:|
| 003 | 60.48 | 30.44 | 39.89 | 35.77 | 3/7 |
| 004 | 60.87 | 30.46 | 39.86 | 36.24 | 3/7 |
| 005 | 60.79 | 29.90 | 39.43 | 36.19 | 2/7 |

Relative to epoch 004, epoch 005 R2 changes by +0.85 pp on LegalText, +0.70 pp
on VLSP2022, +0.28 pp on Book, -0.35 pp on WikiLingua, -0.68 pp on Vietnews,
-0.46 pp on XLSum, and -0.70 pp on ViM. ViM falls below its target by 0.37 pp.
Overall R1/R2/RL decrease by 0.08/0.56/0.43 pp, while macro dataset R2 decreases
by 0.05 pp. Epoch 005 therefore does not uniformly improve the reported metrics.
The comparison alone does not establish overfitting or a cause of the differences.

## Epoch 004 versus the earlier three-target baseline

The older baseline here is **`seam_vi_vllm_test`**, whose supplied report met the
ViM, LegalText and XLSum targets. It is different from the initial `seam_vi`
baseline at the top of this file, which met only two targets.

| Dataset | Older baseline R2 | Epoch 004 R2 | Change (pp) | R2 target | Target met: old / 004 |
|---|---:|---:|---:|---:|:---:|
| WikiLingua | 32.21 | 30.00 | -2.22 | 33.12 | No / No |
| Vietnews | 24.43 | 30.13 | +5.70 | 34.24 | No / No |
| XLSum | 24.90 | 23.53 | -1.37 | 14.39 | Yes / Yes |
| LegalText | 59.54 | 60.28 | +0.74 | 48.76 | Yes / Yes |
| Book | 26.69 | 28.35 | +1.66 | 34.53 | No / No |
| ViM | 44.76 | 44.70 | -0.06 | 44.38 | Yes / Yes |
| VLSP2022 | 37.82 | 36.69 | -1.12 | 44.27 | No / No |

| Summary | Older baseline | Epoch 004 | Change (pp) |
|---|---:|---:|---:|
| Overall R1 | 52.09 | 60.87 | +8.78 |
| Overall R2 | 26.85 | 30.46 | +3.60 |
| Overall RL | 34.82 | 39.86 | +5.04 |
| Macro dataset R2 | 35.77 | 36.24 | +0.48 |

Both reports meet the same **3/7** R2 targets. Epoch 004 improves overall scores
and macro R2, with the largest dataset R2 improvement on Vietnews (+5.70 pp).
It remains lower on WikiLingua, XLSum, ViM and VLSP2022. The old report explicitly
used an output cap of 512 tokens; the epoch 004 output cap was not provided.
Matching generation settings is required to attribute differences to checkpoint
changes rather than decoding settings. All changes use the original, unrounded values.
