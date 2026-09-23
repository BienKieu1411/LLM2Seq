# F2: compact alignment can manufacture convolution adjacency

Status: actual historical aligner reproduced with fixture tokenizer; synthetic gap-masking operator tested.
Evidence: [local source](../sources/09_local_evidence.md), [probe](../probe_repairs.py), [results](../probe_repairs_result.json).

Carry original decoder-token ordinals through alignment. A compact left/right neighbor contributes only when both tokens have positive valid alignment and original ordinals differ by one.
This blocks false links caused by discarded interior tokens while preserving ordinary Conv1d on continuous streams (float32 error <=1.2e-7 in this probe).
Real-tokenizer prevalence and task benefit remain unknown; special-token fixture is not a measurement of ordinary-text frequency.
