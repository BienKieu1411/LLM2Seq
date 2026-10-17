# F3: the bridge is a conditioning opportunity, not a demonstrated advantage

Status: local architectural observation; comparative benefit insufficiently evidenced.

[Inspected code](../sources/10_local_code.md) shows a prompt-conditioned controller before decoding. A length instruction already has a route into source-memory construction. A numeric word feature could make the desired budget more explicit, but requires native/vLLM and likelihood parity.

No three independent sources establish the stronger claim that this SEAM route is superior to decoder-only control. The planned B-vs-C ablation tests that missing evidence. Prompt-only must remain a serious baseline.

Falsifier: numeric conditioning changes attention/control states but does not improve held-out content/length outcomes relative to prompt-conditioned GRPO.

Update: [F4](F4_dual_roles.md) now prioritizes a length-conditioned encoder prompt alongside the decoder prompt, with the existing bridge; the numeric branch is still a separate ablation.
