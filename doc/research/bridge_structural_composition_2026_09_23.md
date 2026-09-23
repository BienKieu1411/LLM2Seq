# Bridge proposal after the source-delivery ledger result

Status: REJECTED BY USER AS OUT OF SCOPE (2026-09-23). Do not implement this
graph proposal. The active scope is a bridge-internal transformation without
graphs, parsers or an additional data pipeline. Kept only as a research record.

Original status: research proposal, not implemented or validated. The user reports that
the latest run failed to improve; its numerical scores and matched run metadata
have not been supplied. This note does not diagnose that run from its score alone.

## Decision

Investigate a token-preserving, source-structure graph bridge. Its task is to
compose relationships between occurrences in different sentences before decoder
retrieval. Do not build another salience selector, exposure ledger, region
router, learned slot bank, document-conditioned projection, or depth mixer.

This is a different mechanism from the implementations reported in this thread,
not a claim that graph summarization is new. The repository already discusses
HierGNN as related work; it would be inaccurate to call graph methods entirely
unexplored in the literature review.

## Observations and limits

- Reported query-region, query-qspace, adaptive-topdown, evidence-slot,
  hyper-operator, side-memory, source-FiLM, and evidence-router results did not
  beat the strongest reported full/direct controls.
- These outcomes weaken the empirical case for adding more learned routing or
  modulation to the same source representations. They do not identify a unique
  causal explanation. Different settings and seeds remain possible confounders.
- Multi-layer cross-attention and the pretrained encoder CAN learn relations.
  The proposed benefit is an explicit structural inductive bias, not access to a
  function that Transformers are mathematically incapable of learning.
- Grounded copy preserves lexical access; it does not impose explicit
  sentence/occurrence graph structure. Changing decoder memory can nevertheless
  change copy probabilities indirectly through the decoder state.

## Literature checked

### GraphLSS, NAACL 2025

https://aclanthology.org/2025.naacl-short.67/

GraphLSS uses word and sentence nodes with four edge families: sentence order,
sentence similarity, word-in-sentence, and word similarity. Tables 3 and 4 provide
evidence for message passing and graph connectivity on PubMed. It is an
EXTRACTIVE model with sentence labels and adaptive weighted classification CE,
not an abstractive token-CE bridge. Its strongest results also use different
preprocessing/labels; Table 2 explicitly shows a large change with the older
labels. Neither absolute scores nor gains transfer to our setup. This proposal
borrows structural connectivity, not its extraction objective or reported score.

### FASum, NAACL 2021 — contrary evidence for prioritizing fact graphs

https://aclanthology.org/2021.naacl-main.58/

FASum obtains source subject/relation/object tuples using OpenIE and introduces
graph attention. Table 3 reports improved factuality but lower ROUGE than its
SEQ2SEQ ablation on both CNN/DailyMail and XSum. Therefore it is not evidence
that adding factual relations will increase our ROUGE. An external fact parser
also adds an independent source of error. Do not choose this direction merely
because its contribution sounds attractive.

### RSTformer, ACL 2023 — related, not a reproduced design

https://aclanthology.org/2023.acl-long.306/

The abstract motivates modeling discourse relation types and uncertainty for
long-document abstractive summarization. This supports considering structural
information, but no numerical ablation from this paper is used in this decision.
We are not proposing an RST parser or claiming that the simple graph below
recovers discourse relations such as causation or contrast.

## Proposed minimal architecture

Use a fresh experimental folder if implemented. Start from the original
direct-projection + grounded-copy backbone, with the same training and decoding
settings as its matched control. Keep the old best full model as a benchmark.

1. Segment only the source text actually visible after truncation into sentences
   and word/phrase occurrences. Preserve character offsets into encoder tokens.
   No reference summary or generated candidate is used to construct the graph.
2. Build a sparse graph with occurrence nodes and sentence nodes. An occurrence
   belongs to its sentence. Add directed preceding/following sentence edges,
   and links between exact content-phrase occurrences in distinct sentences.
   Same spelling is a lexical link, not proof of coreference or factual agreement.
3. Keep repeated occurrences distinct: their sentence, position, and source
   embedding remain different even when their surface forms match. Ignore
   punctuation/stopword-only links, normalize by degree, and bound cross-sentence
   occurrence links to avoid dense hubs. These are explicit design choices to
   validate, not experimentally selected hyperparameters.
4. Initialize occurrence/sentence nodes by offset-based pooling of live encoder
   states. Use two sparse relation-specific message-passing layers in an initial
   256-dimensional graph space. Include edge direction and relative sentence
   distance. Do not use semantic-nearest-neighbor retrieval in the first version.
5. Scatter occurrence updates back to their original source tokens, averaging
   overlaps. The result is one relation-aware correction per original position.
   No source position is deleted or replaced by a learned slot.

Schematic proposed equations:

    z_v = Pool(H0 at offsets of node v)
    m_v = sum_r mean_{u in N_r(v)} phi_r(z_v, z_u, distance(u,v))
    z'_v = z_v + FFN(norm(m_v))
    delta_i = ScatterToTokens(z'_occurrence - z_occurrence)_i
    V0_i = P(H0_i)
    M_i = V0_i + W_out(delta_i)

Cross-attention derives both K and V from M. Grounded copy keeps V0, original
alignment, lexical keys, and its existing probability mixture. Source bias is
zero for the initial experiment. W_out starts at zero, reproducing the direct
control at initialization; this initially blocks gradients to upstream graph
layers, which must become active after W_out updates. There is no claim of
nonzero gradient everywhere on the very first step.

Train with the original token CE, one decoder forward, and no salience,
contrastive, coverage, self-improvement, or graph-label objective. Graph
construction can be cached, but encoder-derived node embeddings must remain
live for end-to-end learning. Cache construction and evaluation see the same
visible source span, and never add information beyond the control input budget.

## What is different, and what is still shared with failed designs

The distinguishing input is an explicit sparse relationship structure between
source occurrences and sentences. The hypothesis is useful composition across
those edges, rather than estimating importance from mean-pooled regions.

Pooling and residual integration are still shared implementation primitives.
This is not a claim that every operation is unprecedented. If the implementation
removes edge types/occurrence identity and merely broadcasts a pooled sentence
vector, it has reverted to a previously tested contextual/top-down family.

## Falsifiable hypothesis and test gate

Hypothesis: under matched token CE training, explicit source occurrence/sentence
relations improve recovery of multi-sentence evidence and coherent phrases over
an equally trained direct-projection model. This may improve R2 and RL, but ROUGE
alone cannot prove this explanation.

Before a full training experiment:

- Verify masks, offsets, truncation, single-sentence/empty-graph fallback,
  gradients after output initialization, and train/generate parity offline.
- Use a fixed validation slice with source/reference/prediction inspection to
  determine whether errors actually involve combining information across
  sentences. If errors are chiefly truncation or output length, do not pretend
  this graph targets the observed bottleneck.

For the experiment:

- Match data, prompt, source budget, effective batch, optimizer steps, seed,
  decoding and checkpoint selection to the direct control.
- Measure validation ROUGE for control versus graph. Select on validation.
- As a diagnostic on the same graph checkpoint, shuffle cross-sentence lexical
  links while preserving relation types/degrees where possible, or remove those
  links. This measures reliance, not a substitute for a retrained ablation.
- A later capacity-matched trained control with rewired edges is required to
  attribute gains to meaningful connectivity rather than extra parameters.
- Stop this direction if correct connectivity does not help over the control,
  if the graph remains unused, or if it only improves examples with no evidence
  composition need. Do not respond by adding more gates or raising residuals.

Potential failure modes: repeated terms link conflicting groups; sentence
segmentation errors; graph oversmoothing; graph duplicates encoder capabilities;
token CE does not provide enough signal to learn helpful relation composition;
sparse graph construction/aggregation increases real training time.

## Claim boundary

No guarantee of +0.3 ROUGE or of publication novelty. A potential contribution
is a token-preserving structural interface for independently pretrained encoder
and decoder models, with lexical grounding retained and evidence that real
connectivity matters. Without those experiments it remains a hypothesis.
