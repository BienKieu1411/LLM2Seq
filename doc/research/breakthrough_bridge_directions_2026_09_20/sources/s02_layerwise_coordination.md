# He et al. 2018 — Layer-Wise Coordination

Source: https://proceedings.neurips.cc/paper_files/paper/2018/file/4fb8a7a22a82c80f2c26fe6c1e0dcbb3-Paper.pdf
Venue: NeurIPS 2018, pp. 7955–7965.

## Verbatim evidence (short excerpt)

- Sec. 3, p. 2: “each layer in the decoder attends to the corresponding layer in the encoder.”

The same section says the decoder can use the corresponding encoder layer rather than only the final layer. Their De–En ablation reports lower BLEU after removing mixed attention or parameter sharing, and the case study attributes an adequacy improvement to better early-layer source access.

## Mechanism

Use a depth-aligned source memory for each decoder layer instead of passing one final encoder representation through every layer. For a mismatched PPLX/Qwen depth, interpolate or route encoder taps to decoder layers by normalized depth rather than requiring equal layer counts.

## Relevance and limitation

This supports per-layer source access and a direct CE objective. The original work shares encoder/decoder blocks and uses a mixed source/target attention, which should not be copied into the current Qwen decoder. The transferable part is the depth-aligned memory interface, not parameter sharing.
