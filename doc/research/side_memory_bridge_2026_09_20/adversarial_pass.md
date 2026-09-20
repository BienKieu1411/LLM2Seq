# Adversarial pass

## Could a side bank be useless?

Yes. PPLX's `H0` and the decoder's native cross-attention may already encode the global signal. A zero or near-zero gate, unchanged logits when the bank is masked, or no validation improvement would falsify the usefulness hypothesis.

## Could it hurt ROUGE-1 and ROUGE-L?

Yes. A memory branch can overproduce high-level paraphrases, dilute exact entities, or add a second source of conflicting evidence. Keeping `K0,V0` and copy separate limits this risk; it does not remove it. The output cap and gate diagnostics are required.

## Is the side bank just the failed slot model under another name?

No, if and only if the implementation keeps `K0,V0` unchanged and consumes `E` through a separate cross-attention branch. A learned bank residual added to `K0`, `V0`, or source logits would reproduce the failed family and should not be described as a new direction.

## Does the literature prove CE-only gains?

No. Flamingo and LongT5 use task-specific pretraining regimes; Compressive Transformer uses auxiliary reconstruction variants; GTCA uses staged adaptation. The proposed AFMR experiment deliberately isolates architecture with CE-only training, so the literature supplies design precedent rather than a performance guarantee.

## Is test-set selection already a risk?

Yes. Multiple PubMed test runs have already guided architecture choices. Future variants must be selected on validation, and the final test should be run once for the chosen configuration. Existing test scores remain useful negative evidence but should not be treated as a fresh unbiased model-selection estimate.

