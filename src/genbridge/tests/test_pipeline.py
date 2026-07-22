from pathlib import Path

from genbridge.run_pipeline import pipeline_commands


def test_pipeline_evaluates_best_and_last_separately(tmp_path=None):
    # The repository's custom test runner calls functions directly, so use a
    # temporary directory without relying on pytest fixture injection.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "config.yaml"
        output = root / "run"
        config.write_text(
            "experiment:\n"
            f"  output_dir: {output}\n"
            "model:\n"
            "  encoder_name: Qwen/Qwen3-0.6B\n"
            "decoder:\n"
            "  pretrained_name: Qwen/Qwen3-0.6B\n",
            encoding="utf-8",
        )
        resolved_output, commands = pipeline_commands(
            config,
            model_size=None,
            overwrite_output_dir=True,
            eval_only=False,
            max_samples=17,
        )
        assert resolved_output == output
        assert len(commands) == 3
        assert "genbridge.training" in commands[0]
        assert "--overwrite-output-dir" in commands[0]
        assert commands[1][commands[1].index("--checkpoint") + 1].endswith("best.pt")
        assert commands[2][commands[2].index("--checkpoint") + 1].endswith("last.pt")
        assert commands[1][commands[1].index("--output") + 1].endswith(
            "best_test_predictions.jsonl"
        )
        assert commands[2][commands[2].index("--output") + 1].endswith(
            "last_test_predictions.jsonl"
        )
        assert commands[1][-2:] == ["--max-samples", "17"]


def test_eval_only_pipeline_skips_training():
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "config.yaml"
        config.write_text(
            "experiment:\n"
            f"  output_dir: {root / 'run'}\n",
            encoding="utf-8",
        )
        _, commands = pipeline_commands(
            config,
            model_size=None,
            overwrite_output_dir=False,
            eval_only=True,
            max_samples=None,
        )
        assert len(commands) == 2
        assert all("genbridge.evaluate" in command for command in commands)
