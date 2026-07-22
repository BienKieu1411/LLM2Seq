from genbridge.smoke_gate import smoke_failures


def test_smoke_gate_accepts_non_degenerate_overfit_metrics():
    metrics = {
        "rouge1": 55.0,
        "rouge2": 30.0,
        "empty_prediction_rate": 0.0,
        "unique_prediction_rate": 98.0,
        "dominant_prefix_5gram_rate": 4.0,
        "repeated_trigram_rate_mean": 1.0,
        "length_ratio_mean": 0.9,
        "cross_residual_ratio": 0.05,
    }
    assert smoke_failures(metrics) == []


def test_smoke_gate_rejects_common_prefix_collapse_and_ignored_source():
    metrics = {
        "rouge1": 25.0,
        "rouge2": 7.0,
        "empty_prediction_rate": 0.0,
        "unique_prediction_rate": 100.0,
        "dominant_prefix_5gram_rate": 90.0,
        "repeated_trigram_rate_mean": 2.0,
        "length_ratio_mean": 1.0,
        "cross_residual_ratio": 0.0001,
    }
    failures = smoke_failures(metrics)
    assert "one 5-word prefix dominates > 25%" in failures
    assert "decoder is effectively ignoring encoder memory" in failures
