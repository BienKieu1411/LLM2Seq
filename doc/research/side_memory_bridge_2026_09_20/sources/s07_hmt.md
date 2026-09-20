# S07 — HMT

**Primary source:** He et al., *HMT: Hierarchical Memory Transformer for Efficient Long Context Language Processing*, NAACL 2025.

URL: https://arxiv.org/abs/2405.06067

## Evidence

- The abstract argues that flat memory has limited selection and filtering, then organizes memory hierarchically by preserving early tokens, passing memory embeddings, and recalling relevant history.
- The paper evaluates language modeling, question answering, and summarization, and reports lower parameter and inference-memory costs than long-context alternatives.
- The paper compares HMT with RMT and reports that RMT can be harder to train because of gradient stability; this is relevant counterevidence for a recurrent AFMR bridge.

## Transfer to AFMR

Use a two-level source side bank: block summaries preserve local evidence, while a small recurrent/global bank preserves cross-block relations. Decoder queries read both banks through separate gates. The direct PPLX token route remains available for exact lexical evidence and grounded copy.

## Caveats

HMT is a long-context language-model framework, not a drop-in decoder bridge for PPLX-to-Qwen summarization. Its hierarchical memory and recurrence add substantial implementation and optimization choices. It should be tested only after the simpler non-recurrent side bank.

