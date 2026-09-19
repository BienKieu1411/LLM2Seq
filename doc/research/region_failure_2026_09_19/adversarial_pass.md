# Adversarial check

**Alternative 1 — protocol drift.** Separate architecture folders may contain distinct prepared data, even with same YAML. User supplied only aggregate scores, not resolved configs or fingerprints; no causal explanation is established. Mitigation: compare effective configs, data hashes and paired IDs before assigning blame to region pooling.

**Alternative 2 — optimization rather than inference mechanism.** New branch gradients participate in global clipping and alter shared M/H0 updates. Post-hoc branch-off might fail to restore static AFMR because the rest of model already trained differently. Mitigation: branch-on/off is a diagnostic, then retrain matched controls.

**Alternative 3 — no reliable effect.** Q-space loses at most 0.118 R-1, 0.066 R-2 and 0.098 R-L against the stronger of two controls per metric. A single-seed ROUGE total without paired outputs cannot distinguish systematic loss from run variance.

**Challenge to new proposals.** Adaptive regions vẫn có thể cắt evidence ở biên cửa sổ; evidence slots có thể collapse hoặc nén mất chi tiết. Cả hai thêm đường gradient vào shared memory và có thể làm optimization xấu đi dù inference graph hợp lệ. Code phải giữ correction bounded, mask content chính xác và giữ toàn bộ source-token path; không kiểm tra nào trong số đó chứng minh ROUGE sẽ tăng.

**Decision.** One new isolated candidate is defensible as an experiment, not as a guaranteed fix. Refrain from increasing gate/rank or adding loss until the diagnostic identifies what the region branch actually does.
