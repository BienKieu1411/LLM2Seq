# Source note: SummEval metric and human-judgment study

- **URL:** https://aclanthology.org/2021.tacl-1.24/
- **Authors / date / venue:** Fabbri et al.; Transactions of the Association for Computational Linguistics 9, 2021.
- **Credibility:** High. Peer-reviewed journal study that compares 14 automatic metrics against expert and crowd human annotations.
- **Recency:** Medium. Still a substantial benchmark, but focused on English CNN/DailyMail model outputs from earlier generations.
- **Bias:** Low-medium. The study's stated aim is broad metric evaluation; scope is limited by its datasets and annotation setup.

## Evidence used

SummEval evaluates multiple automatic metrics and human labels over coherence, consistency, fluency, and relevance. It supports reporting more than ROUGE and checking metric-human agreement on the task at hand. It does not validate a specific metric for Vietnamese, the project's datasets, or a training reward.

> “expert, as well as crowd-sourced, human judgments”

## Limitations for this decision

Its conclusions should not be transferred numerically to Vietnamese summaries or used to assume that one particular learned metric is a reliable reward. Use it to motivate local metric validation and a small human review set.
