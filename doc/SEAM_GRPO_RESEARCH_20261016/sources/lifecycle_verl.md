# Source note: VERL actor-to-rollout update patterns

**Source:** [VERL `ActorRolloutRefWorker.update_weights`](https://github.com/verl-project/verl/blob/main/verl/workers/engine_workers.py), accessed 2026-10-17. Primary official project code; `main` is mutable.

**Short quote:** “For sync training with colocated trainer and rollout, update rollout directly from model engine.”

**Relevant evidence:** The worker distinguishes colocated synchronous rollout weight update from disaggregated asynchronous training, where weights travel through a checkpoint engine. Its implementation documents the direct `naive` mode and calls a checkpoint-engine send path in disaggregated mode. It also treats rollout sleep/wake as part of the update lifecycle.

**Application to SEAM:** This is a useful system-level distinction, not proof of model compatibility. A local/shared actor removes a duplicate weight-copy boundary; a separate rollout engine needs transport, name/layout conversion, request barriers, versioning, and cache/state invalidation. VERL's generic transfer pattern still needs a custom SEAM weight source and target loader.

**Limit:** The code branch is mutable and represents VERL's evolving engine abstraction. No claim is made that it runs SEAM unchanged.
