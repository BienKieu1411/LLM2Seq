"""CPU-only SI integration checks; no model downloads or CUDA allocations."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import si_common
from parallel import merge_shards, predict_two_gpu
from prediction import predict
from selection import best_rouge2, select


class SIIntegration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = self.root / "train.jsonl"
        self.train.write_text(
            "".join(
                si_common.json_line({"id": "duplicate", "text": f"source {i}", "summary": f"gold {i} ,"})
                for i in range(3)
            ),
            encoding="utf-8",
        )
        self.config = {"data": {"detokenize": True, "max_source_length": 8192}}
        self.args = SimpleNamespace(
            config=self.root / "config.yaml",
            train_file=self.train,
            output_dir=self.root / "si",
            start_server=False,
            checkpoint=None,
            server_model="seam",
            api_key=None,
            max_input_tokens=None,
            max_new_tokens=512,
            seed=42,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            timeout=5,
            retries=0,
            batch_size=2,
            max_num_batched_tokens=2048,
            log_every=1,
            rouge_path=self.root / "rouge",
            score_batch_size=2,
        )
        self.calls = []
        self.fail_source = None
        info = {
            "implementation": "SEAMForQwen3",
            "fingerprint": "test-fingerprint",
            "max_input_tokens": 8192,
            "generation_defaults": {"max_tokens": 512},
            "prompt_defaults": {
                "encoder_prompt": "",
                "encoder_task_marker": "",
                "decoder_prompt": "",
                "decoder_prefix": "",
                "decoder_chat_template": False,
            },
        }
        self.info = info
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, body, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-seam-fingerprint", info["fingerprint"])
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_GET(self):
                self.reply(info if self.path == "/v1/seam/info" else {"data": [{"id": "seam"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append(body)
                if body["prompt"] == outer.fail_source:
                    self.reply({"error": "simulated"}, status=500)
                    return
                self.reply(
                    {
                        "choices": [
                            {"index": i, "text": f"{body['prompt']} candidate {i}", "finish_reason": "stop"}
                            for i in reversed(range(8))
                        ]
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.args.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def predict(self):
        with patch("eviseq_afmr.config.load_config", return_value=self.config):
            predict(self.args)

    @staticmethod
    def score(rows, rouge_path, temporary):
        (temporary / "rouge_raw.txt").write_text("fake Perl fixture", encoding="utf-8")
        for row in rows:
            for candidate in row["candidates"]:
                index = candidate["index"]
                # Candidate 7 wins weighted ROUGE; only candidate 3 wins ROUGE-2.
                candidate["rouge"] = {
                    "rouge1": 1.0 if index == 7 else 0.1,
                    "rouge2": 0.8 if index == 3 else 0.7,
                    "rougeL": 1.0 if index == 7 else 0.1,
                }
        return rows

    def select(self, scorer=None):
        self.args.rouge_path.mkdir(exist_ok=True)
        (self.args.rouge_path / "ROUGE-1.5.5.pl").write_text("fixture", encoding="utf-8")
        with (
            patch("selection.validate_rouge_install"),
            patch("selection.score_candidates", side_effect=scorer or self.score),
        ):
            select(self.args)

    def test_http_full_train_and_resume_and_r2_selection(self):
        self.predict()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual({request["seed"] for request in self.calls}, {42, 43, 44})
        self.assertTrue(all(request["n"] == 8 and request["temperature"] == 1.0 for request in self.calls))
        self.assertTrue(all("reference" not in request and "gold" not in request["prompt"] for request in self.calls))
        candidates_path = self.args.output_dir / "candidates.jsonl"
        rows = list(si_common.read_rows(candidates_path))
        self.assertEqual({row["row_index"] for row in rows}, {0, 1, 2})
        self.assertTrue(all(row["reference"].endswith(",") and " ," not in row["reference"] for row in rows))
        with candidates_path.open("ab") as stream:
            stream.write(b'{"interrupted":')
        self.predict()
        self.assertEqual(len(self.calls), 3)  # No regenerated sources.
        self.select()
        train_rows = list(si_common.read_rows(self.args.output_dir / "train_si.jsonl"))
        self.assertEqual(len(train_rows), 3)
        self.assertEqual({row["text"] for row in train_rows}, {"source 0", "source 1", "source 2"})
        self.assertTrue(all(row["summary"] == f"{row['text']} candidate 3" for row in train_rows))
        self.assertTrue(all(set(row) == {"id", "text", "summary"} for row in train_rows))
        self.assertTrue(all(row["id"] == "duplicate" for row in train_rows))
        # Completed scoring shards must be reused, not rescored or joined by ID.
        self.select(scorer=lambda *_: self.fail("Completed scoring batch was rerun"))
        report = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertEqual(report["selected_index_counts"][3], 3)

    def test_http_failure_preserves_successes_and_resume_fills_missing(self):
        self.fail_source = "source 1"
        with self.assertRaisesRegex(RuntimeError, "successful rows saved"):
            self.predict()
        saved = list(si_common.read_rows(self.args.output_dir / "candidates.jsonl"))
        self.assertGreaterEqual(len(saved), 1)
        self.assertTrue(all(row["source"] != self.fail_source for row in saved))
        self.fail_source = None
        self.predict()
        rows = list(si_common.read_rows(self.args.output_dir / "candidates.jsonl"))
        self.assertEqual(sorted(row["row_index"] for row in rows), [0, 1, 2])

    def test_incomplete_or_duplicate_rollouts_cannot_publish_train(self):
        self.predict()
        path = self.args.output_dir / "candidates.jsonl"
        rows = list(si_common.read_rows(path))
        path.write_text(si_common.json_line(rows[0]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.select()
        path.write_text("".join(si_common.json_line(row) for row in [rows[0], rows[0], rows[1]]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.select()
        self.assertFalse((self.args.output_dir / "train_si.jsonl").exists())

    def test_selection_can_resume_after_perl_failure(self):
        self.predict()
        calls = 0

        def fail_second_batch(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Perl fixture interrupted")
            return self.score(*args)

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.select(scorer=fail_second_batch)
        self.assertFalse((self.args.output_dir / "train_si.jsonl").exists())
        calls = 0

        def finish_batch(*args):
            nonlocal calls
            calls += 1
            return self.score(*args)

        self.select(scorer=finish_batch)
        self.assertEqual(calls, 1)
        self.assertEqual(len(list(si_common.read_rows(self.args.output_dir / "train_si.jsonl"))), 3)

    def test_mutated_data_rejects_resume(self):
        self.predict()
        self.train.write_text(self.train.read_text().replace("source 0", "changed"))
        with self.assertRaisesRegex(ValueError, "Data/model/generation changed"):
            self.predict()

    def test_parallel_selection_matches_serial_and_reuses_cache_across_worker_counts(self):
        self.predict()
        self.select()
        train = (self.args.output_dir / "train_si.jsonl").read_bytes()
        audit = (self.args.output_dir / "selected.jsonl").read_bytes()
        for path in (self.args.output_dir / "rouge155").glob("batch_*"):
            path.unlink()
        self.args.score_workers = 2
        barrier = threading.Barrier(2)

        def concurrent_score(*args):
            barrier.wait(timeout=3)  # Both independent Perl jobs must run concurrently.
            return self.score(*args)

        self.select(scorer=concurrent_score)
        self.assertEqual((self.args.output_dir / "train_si.jsonl").read_bytes(), train)
        self.assertEqual((self.args.output_dir / "selected.jsonl").read_bytes(), audit)
        self.args.score_workers = 1
        self.select(scorer=lambda *_: self.fail("Worker-count change must reuse completed cache"))

    def test_parallel_selection_failure_preserves_cache_without_publishing_partial_train(self):
        self.predict()
        self.args.score_workers = 2

        def fail_last_batch(rows, *args):
            if len(rows) == 1:
                raise RuntimeError("parallel Perl interrupted")
            return self.score(rows, *args)

        with self.assertRaisesRegex(RuntimeError, "parallel Perl interrupted"):
            self.select(scorer=fail_last_batch)
        self.assertFalse((self.args.output_dir / "train_si.jsonl").exists())
        self.assertTrue((self.args.output_dir / "rouge155" / "batch_000000.jsonl").exists())
        self.select()
        self.assertEqual(len(list(si_common.read_rows(self.args.output_dir / "train_si.jsonl"))), 3)

    def test_r2_ties_use_lower_index_and_all_empty_is_error(self):
        row = {
            "row_index": 0,
            "line_number": 1,
            "id": "id",
            "source": "src",
            "reference": "gold",
            "candidates": [
                {"index": i, "prediction": str(i), "rouge": {"rouge1": i / 10, "rouge2": 0.5, "rougeL": i / 10}}
                for i in range(8)
            ],
        }
        self.assertEqual(best_rouge2(row)["selected_index"], 0)
        for candidate in row["candidates"]:
            candidate["prediction"] = " "
        with self.assertRaisesRegex(ValueError, "source is not dropped"):
            best_rouge2(row)

    def run_shards(self):
        for rank in range(2):
            args = copy.copy(self.args)
            args.output_dir = self.args.output_dir / f"shard{rank}"
            args.shard_rank, args.num_shards = rank, 2
            with patch("eviseq_afmr.config.load_config", return_value=self.config):
                predict(args)

    def test_odd_train_split_resume_merge_and_full_r2_selection(self):
        self.run_shards()
        rows0 = list(si_common.read_rows(self.args.output_dir / "shard0" / "candidates.jsonl"))
        rows1 = list(si_common.read_rows(self.args.output_dir / "shard1" / "candidates.jsonl"))
        self.assertEqual([row["row_index"] for row in rows0], [0])
        self.assertEqual(sorted(row["row_index"] for row in rows1), [1, 2])
        self.assertEqual({request["seed"] for request in self.calls}, {42, 43, 44})
        self.run_shards()
        self.assertEqual(len(self.calls), 3)
        child = copy.copy(self.args)
        child.output_dir = self.args.output_dir / "shard0"
        with self.assertRaisesRegex(ValueError, "Merge prediction shards"):
            select(child)
        merge_shards(self.args.output_dir)
        self.select()
        train_rows = list(si_common.read_rows(self.args.output_dir / "train_si.jsonl"))
        self.assertEqual(len(train_rows), 3)
        self.assertEqual({row["text"] for row in train_rows}, {"source 0", "source 1", "source 2"})
        self.assertTrue(all(row["summary"] == f"{row['text']} candidate 3" for row in train_rows))

    def test_merge_rejects_missing_and_duplicate_candidate_rows(self):
        self.run_shards()
        path = self.args.output_dir / "shard1" / "candidates.jsonl"
        rows = list(si_common.read_rows(path))
        path.write_text(si_common.json_line(rows[0]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Incomplete candidate"):
            merge_shards(self.args.output_dir)
        path.write_text(si_common.json_line(rows[0]) * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            merge_shards(self.args.output_dir)
        self.assertFalse((self.args.output_dir / "prediction_complete.json").exists())
        self.assertFalse((self.args.output_dir / "candidates.jsonl").exists())

    def test_merge_rejects_different_teachers_and_incomplete_shard(self):
        self.run_shards()
        directory = self.args.output_dir / "shard1"
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["server_fingerprint"] = "different-teacher"
        si_common.write_json(directory / "manifest.json", manifest)
        si_common.write_json(
            directory / "prediction_complete.json",
            {"count": manifest["total_sources"], "manifest_sha256": si_common.digest(manifest)},
        )
        with self.assertRaisesRegex(ValueError, "different data, config, teacher"):
            merge_shards(self.args.output_dir)
        (directory / "prediction_complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "Shard 1 is incomplete"):
            merge_shards(self.args.output_dir)

    def test_merge_resume_after_interrupted_final_publish(self):
        self.run_shards()
        replace = Path.replace

        def interrupted(path, target):
            if path.name == "candidates.jsonl.tmp":
                raise OSError("interrupted merge publish")
            return replace(path, target)

        with patch.object(Path, "replace", interrupted):
            with self.assertRaisesRegex(OSError, "interrupted merge publish"):
                merge_shards(self.args.output_dir)
        self.assertFalse((self.args.output_dir / "prediction_complete.json").exists())
        merge_shards(self.args.output_dir)
        self.assertEqual(len(list(si_common.read_rows(self.args.output_dir / "candidates.jsonl"))), 3)

    def test_two_host_coordinator_runs_concurrently_with_distinct_gpu_and_port(self):
        self.args.gpus, self.args.start_server, self.args.server_port = "0,1", True, 8012
        self.args.server_log, self.args.attention_backend = None, "FLASH_ATTN"
        self.args.checkpoint = self.root / "teacher.pt"
        self.args.checkpoint.write_bytes(b"not real model weights")
        self.info["checkpoint"] = {"path": str(self.args.checkpoint.resolve())}
        barrier, hosts = threading.Barrier(2), []

        @contextmanager
        def fake_host(child, *_):
            hosts.append((child.gpu_id, child.server_port, child.shard_rank))
            child.on_server_started(SimpleNamespace(pid=100000 + child.shard_rank, poll=lambda: 0))
            barrier.wait(timeout=3)
            yield self.args.base_url

        with (
            patch("eviseq_afmr.config.load_config", return_value=self.config),
            patch("prediction.managed_seam_server", fake_host),
        ):
            predict_two_gpu(self.args)
        self.assertEqual(set(hosts), {("0", 8012, 0), ("1", 8013, 1)})
        self.assertEqual(len(self.calls), 3)
        merged = list(si_common.read_rows(self.args.output_dir / "candidates.jsonl"))
        self.assertEqual(sorted(row["row_index"] for row in merged), [0, 1, 2])

    def test_coordinator_failure_cancels_owned_hosts_and_never_merges(self):
        self.args.gpus, self.args.server_port, self.args.server_log = "0,1", 8012, None
        barrier = threading.Barrier(2)

        def fail_worker(child):
            child.on_server_started(SimpleNamespace(pid=100000 + child.shard_rank, poll=lambda: None))
            barrier.wait(timeout=3)
            if child.shard_rank == 0:
                raise RuntimeError("GPU 0 failed")
            self.assertTrue(child.cancel_event.wait(timeout=3))

        with (
            patch("parallel.predict", side_effect=fail_worker),
            patch("parallel.os.killpg") as kill,
            patch("parallel.merge_shards") as merge,
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU 0 failed"):
                predict_two_gpu(self.args)
        self.assertEqual({call.args[0] for call in kill.call_args_list}, {100000, 100001})
        merge.assert_not_called()

    def test_two_gpu_rejects_single_gpu_output_before_launching_hosts(self):
        self.predict()
        self.args.gpus = "0,1"
        with patch("parallel.predict") as launch:
            with self.assertRaisesRegex(ValueError, "single-GPU run"):
                predict_two_gpu(self.args)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
