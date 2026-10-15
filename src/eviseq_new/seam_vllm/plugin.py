"""Register the SEAM model and its source-text processor with vLLM."""

from __future__ import annotations

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return

    from vllm.logger import init_logger
    from vllm.model_executor.models.registry import ModelRegistry

    logger = init_logger(__name__)
    from . import model as _model  # noqa: F401
    from .processing import register_processor

    if "SEAMForQwen3" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("SEAMForQwen3", "seam_vllm.model:SEAMForQwen3")
    register_processor()
    _registered = True
    logger.info("Registered SEAMForQwen3 with vLLM")
