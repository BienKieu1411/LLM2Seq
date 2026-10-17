"""Transport-only double; real rollout math, never a claim about GPU parity."""

import copy
import sys
from types import SimpleNamespace

from seam_grpo.rollout import rollout
from seam_length_grpo.policy import unpad_row
from seam_length_grpo.vllm_rollout import LengthVLLMRollout

from seam_length_grpo import train as runtime


class NativeEngine:
    def __init__(self, trainer):
        self.trainer, self.actor, self.version = trainer, None, -1
        self.engine = self
        self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: None))

    def sync_weights(self, actor, policy_version):
        assert policy_version > self.version
        if self.actor is None:
            self.actor = copy.deepcopy(actor)
        self.actor.load_state_dict(actor.state_dict(), strict=True)
        self.actor.eval()
        self.version = policy_version

    def generate(self, requests, params, use_tqdm=False):
        assert self.version == self.trainer.state.global_step
        source, outputs = self.trainer._likelihood_source, []
        for index, request in enumerate(requests):
            batch = unpad_row(source, index)
            ids = batch["decoder_prompt_ids"][0].tolist()
            assert request["prompt_token_ids"] == ids
            assert request["multi_modal_data"]["text"] == source["sources"][index]
            kwargs = request["mm_processor_kwargs"]
            assert kwargs["seam_encoder_prompt"] == source["encoder_prompts"][index]
            assert kwargs["seam_decoder_prompt_ids"] == ids
            assert kwargs["seam_max_source_length"] == self.trainer.collator.encoder_limit
            result = rollout(self.actor, batch, self.trainer.decoder_tokenizer, params.max_tokens, params.temperature)
            mask = result["action_mask"][0]
            tokens, logps = result["action_ids"][0][mask].tolist(), result["old_logps"][0][mask].tolist()
            completion = SimpleNamespace(
                token_ids=tokens,
                logprobs=[{token: SimpleNamespace(logprob=lp)} for token, lp in zip(tokens, logps, strict=True)],
            )
            outputs.append(SimpleNamespace(prompt_token_ids=ids, outputs=[completion]))
        return outputs


def install():
    # Let TRL inspect real package availability before adding the transport double.
    from seam_length_grpo.trainer import LengthTrainer  # noqa: F401

    runtime.check_rollout_device = lambda *args: None
    sys.modules["vllm"] = SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))

    def initialize(self):
        self.engine = NativeEngine(self.trainer)

    LengthVLLMRollout.initialize = initialize
