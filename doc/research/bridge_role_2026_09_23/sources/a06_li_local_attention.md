# Source A06 — Characterizing the Expressivity of Local Attention in Transformers

- Authors: Jiaoda Li, Ryan Cotterell
- Venue: ACL 2026, pp. 37485–37507
- URL: https://aclanthology.org/2026.acl-long.1739/
- Type: primary theory plus formal-language and natural-language experiments
- Quality: high for the stated local/global expressivity result; transfer to a source-memory bridge is indirect

## Verified evidence

The paper formalizes local attention as a bounded predecessor window and argues
that adding local attention introduces a second temporal operator. It states
that local and global attention are expressively complementary and reports
hybrid global–local Transformers outperforming global-only counterparts in its
formal and natural-language experiments.

Short verbatim excerpts (under the source quote limit):

> “global and local attention are expressively complementary”

> “hybrid global–local transformers outperform their global-only counterparts”

## Use in this audit

This is modern positive evidence for an order-sensitive local inductive bias,
but it studies local attention inside an autoregressive Transformer rather than
a single bridge between two independently pretrained models. It cannot justify
a PubMed ROUGE prediction. It strengthens the case for one cheap local probe
only when paired with the contrary evidence from PPLX contextualization and
Chang et al.’s expressivity caveat.
