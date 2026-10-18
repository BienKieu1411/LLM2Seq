"""CPU transport double: tests callback/version/DDP wiring, never GPU kernels."""

import copy
import sys
from types import SimpleNamespace

import torch
from seam_grpo.rollout import rollout
from seam_grpo.vllm_rollout import VLLMRollout


class NativeFacade:
    def __init__(self, trainer):
        self.trainer = trainer
        self.actor = None
        self.version = -1
        self.engine = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: None)))

    def sync_weights(self, actor, policy_version):
        assert policy_version > self.version
        if self.actor is None:
            self.actor = copy.deepcopy(actor)
        self.actor.load_state_dict(actor.state_dict(), strict=True)
        self.actor.eval()
        self.version = policy_version

    def generate(self, prompts, params):
        assert self.version == self.trainer.state.global_step
        batch = self.trainer._likelihood_source
        assert prompts == batch["sources"]
        with torch.no_grad():
            result = rollout(self.actor, batch, self.trainer.decoder_tokenizer, params.max_tokens, params.temperature)
        outputs = []
        for index in range(len(prompts)):
            mask = result["action_mask"][index]
            ids = result["action_ids"][index][mask].tolist()
            logps = result["old_logps"][index][mask].tolist()
            if getattr(self.trainer, "_test_bad_vllm_logps", False):
                logps = [x - 1 for x in logps]
            if index == 0 and getattr(self.trainer, "_test_vllm_outlier", False):
                logps[0] -= 1.06443
            completion = SimpleNamespace(
                token_ids=ids,
                logprobs=[{token: SimpleNamespace(logprob=lp)} for token, lp in zip(ids, logps, strict=True)],
            )
            outputs.append(
                SimpleNamespace(prompt_token_ids=batch["decoder_prompt_ids"][index].tolist(), outputs=[completion])
            )
        return outputs


def install():
    # Import the real trainer before installing the transport-only double;
    # TRL's package discovery must inspect the real environment, not this stub.
    from seam_grpo.trainer import SEAMGRPOTrainer  # noqa: F401

    from seam_grpo import train as runtime

    runtime.check_rollout_device = lambda *a: None

    sys.modules["vllm"] = SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw))

    def initialize(self):
        self.engine = NativeFacade(self.trainer)

    VLLMRollout.initialize = initialize
