from genbridge.evaluate import _rouge


def test_standard_rouge_exact_match():
    metrics = _rouge(["Tóm tắt sự kiện chính."], ["Tóm tắt sự kiện chính."])
    assert metrics == {
        "rouge1": 100.0,
        "rouge2": 100.0,
        "rougeL": 100.0,
        "rougeLsum": 100.0,
    }
