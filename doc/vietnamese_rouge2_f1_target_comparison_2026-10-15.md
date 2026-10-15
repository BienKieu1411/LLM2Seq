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
