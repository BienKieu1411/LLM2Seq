import torch
from transformers import TrainerCallback, TrainerState, TrainerControl

class FreezeNonCrossAttentionCallback(TrainerCallback):
    """
    A callback that freezes all parameters except cross-attention layers during a warmup phase.
    """
    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
        self.is_frozen = False

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if self.warmup_steps > 0 and model is not None:
            print(f"Starting cross-attention warmup for {self.warmup_steps} steps. Freezing non-cross-attention parameters.")
            self._freeze_non_cross_attention(model)
            self.is_frozen = True

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if self.is_frozen and state.global_step >= self.warmup_steps and model is not None:
            print(f"Warmup finished at step {state.global_step}. Unfreezing all parameters.")
            self._unfreeze_all(model)
            self.is_frozen = False

    def _freeze_non_cross_attention(self, model):
        # By default freeze everything
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze cross-attention specifically
        for name, param in model.named_parameters():
            lname = name.lower()
            if "crossattention" in lname or "encoder_attn" in lname or "bridge" in lname:
                param.requires_grad = True
                
    def _unfreeze_all(self, model):
        for param in model.parameters():
            param.requires_grad = True
