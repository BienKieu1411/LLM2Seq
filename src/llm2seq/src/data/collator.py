"""
Dynamic padding collator for LLM2Seq.

Pads variable-length sequences in a batch to the maximum length
within that batch, minimizing wasted computation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class Seq2SeqCollator:
    """
    Collator that dynamically pads sequences to the longest in each batch.

    Pads:
    - input_ids + attention_mask (source)
    - decoder_input_ids + decoder_attention_mask (target input)
    - labels (target output, padded with -100 for ignore)

    Args:
        pad_token_id: Token ID used for padding.
        max_source_length: Maximum source length (cap).
        max_target_length: Maximum target length (cap).
        label_pad_token_id: Padding value for labels (default: -100).
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_source_length: int = 512,
        max_target_length: int = 256,
        label_pad_token_id: int = -100,
    ):
        self.pad_token_id = pad_token_id
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of feature dicts into a batch.

        Args:
            features: List of dicts from Seq2SeqDataset.__getitem__.

        Returns:
            Batch dict with padded tensors.
        """
        batch: Dict[str, Any] = {}

        # Pad source (input_ids + attention_mask)
        input_ids = [f["input_ids"] for f in features]
        src_max_len = min(max(ids.size(0) for ids in input_ids), self.max_source_length)

        padded_input_ids = []
        attention_masks = []
        for ids in input_ids:
            length = ids.size(0)
            if length >= src_max_len:
                padded_input_ids.append(ids[:src_max_len])
                attention_masks.append(torch.ones(src_max_len, dtype=torch.long))
            else:
                padding = torch.full((src_max_len - length,), self.pad_token_id, dtype=ids.dtype)
                padded_input_ids.append(torch.cat([ids, padding]))
                mask = torch.cat([torch.ones(length, dtype=torch.long), torch.zeros(src_max_len - length, dtype=torch.long)])
                attention_masks.append(mask)

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["attention_mask"] = torch.stack(attention_masks)

        # Pad decoder inputs
        decoder_input_ids = [f["decoder_input_ids"] for f in features]
        tgt_max_len = min(max(ids.size(0) for ids in decoder_input_ids), self.max_target_length)

        padded_dec_ids = []
        dec_masks = []
        for ids in decoder_input_ids:
            length = ids.size(0)
            if length >= tgt_max_len:
                padded_dec_ids.append(ids[:tgt_max_len])
                dec_masks.append(torch.ones(tgt_max_len, dtype=torch.long))
            else:
                padding = torch.full((tgt_max_len - length,), self.pad_token_id, dtype=ids.dtype)
                padded_dec_ids.append(torch.cat([ids, padding]))
                mask = torch.cat([torch.ones(length, dtype=torch.long), torch.zeros(tgt_max_len - length, dtype=torch.long)])
                dec_masks.append(mask)

        batch["decoder_input_ids"] = torch.stack(padded_dec_ids)
        batch["decoder_attention_mask"] = torch.stack(dec_masks)

        # Pad labels (with -100 for padding)
        labels_list = [f["labels"] for f in features]
        padded_labels = []
        for lab in labels_list:
            length = lab.size(0)
            if length >= tgt_max_len:
                padded_labels.append(lab[:tgt_max_len])
            else:
                padding = torch.full((tgt_max_len - length,), self.label_pad_token_id, dtype=lab.dtype)
                padded_labels.append(torch.cat([lab, padding]))

        batch["labels"] = torch.stack(padded_labels)

        # Optional: teacher logits
        if "teacher_logits" in features[0]:
            # These are tensors of potentially different shapes — need careful handling
            teacher_logits = [f["teacher_logits"] for f in features]
            batch["teacher_logits"] = torch.stack(teacher_logits)

        if "teacher_topk_indices" in features[0]:
            teacher_topk = [f["teacher_topk_indices"] for f in features]
            batch["teacher_topk_indices"] = torch.stack(teacher_topk)

        return batch
