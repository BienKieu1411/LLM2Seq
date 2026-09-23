# F1: linear composition can collide under many-to-one pooling

Status: verified synthetic counterexample for historical efce6f8; no corpus prevalence claim.
Existing probe: ../bridge_role_2026_09_23/probe_order_collision_result.json (relative to audit folder).
New check: ../probe_repairs.py and ../probe_repairs_result.json.

Keep the already-proposed conv -> SiLU -> scatter -> up ordering. This distinguishes the tested internal swap but does not make pooling injective.
Paper precedent: [ConvS2S](../sources/01_gehring_convs2s.md), [Primer](../sources/02_so_primer.md).
The repair itself rests on algebra and code, not an empirical claim from either paper.
