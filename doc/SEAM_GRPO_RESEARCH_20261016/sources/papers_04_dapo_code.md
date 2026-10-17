# Source note: DAPO official code release

- **URL:** https://github.com/BytedTsinghua-SIA/DAPO
- **Authors / date / venue:** Official ByteDance Seed / Tsinghua AIR DAPO repository; created with the 2025 paper, repository README updated through 2025-05 in the visible release notes.
- **Credibility:** High for the authors' released recipe and code provenance; it is not an independent reproduction.
- **Recency:** Medium-high. The official training release is from 2025; check current framework compatibility before porting.
- **Bias:** High. This is the authors' own implementation and reproduction claim.

## Evidence used

The official README says the release includes algorithm, infrastructure, and a curated math dataset, and provides recipes for full DAPO and an ablated version. This is useful for verifying that DAPO is a complete training system rather than only a loss equation. The repository is built on verl and targets large decoder-only models and math rewards, so its scripts should not be copied directly into a custom encoder-decoder summarizer.

> “fully open-source the recipe of our RL training”

## Limitations for this decision

The repository does not provide evidence that its specific batch, clipping, or rollout settings transfer to a custom seq2seq model. Treat implementation details as reference examples and verify them against the paper.
