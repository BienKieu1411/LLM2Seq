"""Exercise vLLM encoder grouping with mixed list/tensor request fields on CPU."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from seam_vllm._encoder_batch import encoder_batches
from seam_vllm._tensors import _as_2d


@pytest.mark.parametrize("tensor_fields", [False, True])
def test_ragged_source_groups_slice_every_batched_field(tensor_fields):
    path = Path(__file__).resolve().parents[2] / "eviseq_new/seam_vllm/model.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SEAMForQwen3")
    hook = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "embed_multimodal")
    namespace = {"torch": torch, "encoder_batches": encoder_batches, "_as_2d": _as_2d}
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(path), "exec"), namespace)
    calls = []

    def embed(**fields):
        calls.append(fields)
        return list(fields["encoder_input_ids"].unbind())

    model = SimpleNamespace(
        decoder_prompt_ids=torch.tensor([9]),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(seam_encoder_max_batch_tokens=8192))
        ),
        _embed_multimodal_batch=embed,
    )
    budgets = [[16], [32], [64]]
    prompts = [[9, 8], [9, 7], [9, 6]]
    prompt_masks = [[1, 1]] * 3
    if tensor_fields:
        budgets, prompts, prompt_masks = map(torch.tensor, (budgets, prompts, prompt_masks))
    result = namespace["embed_multimodal"](
        model,
        [[1, 2, 3], [4], [5]],
        encoder_attention_mask=[[1, 1, 1], [1], [1]],
        source_content_mask=[[0, 1, 1], [1], [1]],
        requested_output_tokens=budgets,
        bridge_prompt_ids=prompts,
        bridge_prompt_mask=prompt_masks,
    )
    assert [row.tolist() for row in result] == [[1, 2, 3], [4], [5]]
    assert [fields["encoder_input_ids"].shape[0] for fields in calls] == [1, 2]
    assert torch.cat([fields["requested_output_tokens"] for fields in calls]).flatten().tolist() == [16, 32, 64]
    assert torch.cat([fields["bridge_prompt_ids"] for fields in calls]).tolist() == [[9, 8], [9, 7], [9, 6]]
