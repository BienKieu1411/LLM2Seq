from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[1]  
LLM2SEQ_ROOT = PROJECT_ROOT / "src" / "llm2seq"

if str(LLM2SEQ_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM2SEQ_ROOT))

from src.models.llm2seq_model import LLM2Seq, LLM2SeqConfig  
from src.inference.generate import autoregressive_generate  
from src.inference.generate_mtp import mtp_generate  

_model: Optional[LLM2Seq] = None
_tokenizer = None
_cfg: Optional[LLM2SeqConfig] = None
_raw_cfg: Dict[str, Any] = {}
_device: torch.device = torch.device("cpu")
_model_ready = False

def _download_checkpoint(repo_id: str, filename: str) -> Path:

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    return Path(local_path)

def _load_model(config_path: Path) -> None:

    global _model, _tokenizer, _cfg, _raw_cfg, _device, _model_ready

    with config_path.open("r", encoding="utf-8") as f:
        _raw_cfg = yaml.safe_load(f)
    _cfg = LLM2SeqConfig(_raw_cfg)

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    print(f"[LLM2Seq] Using device: {_device}")

    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(
        _cfg.encoder_name, trust_remote_code=True
    )
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token = _tokenizer.eos_token or _tokenizer.unk_token

    _model = LLM2Seq(_cfg, vocab_size=len(_tokenizer))

    hf_cfg = _raw_cfg.get("huggingface", {})
    repo_id = hf_cfg.get("repo_id")
    phase2_file = hf_cfg.get("phase2_file", "encoder-lora_decoder_best.pt")
    phase3_file = hf_cfg.get("phase3_file", "mtp_best.pt")

    if not repo_id:
        raise RuntimeError(
            "huggingface.repo_id is not set in config.yaml. "
            "Cannot download checkpoints."
        )

    print(f"[LLM2Seq] Downloading Phase 2 checkpoint: {repo_id}/{phase2_file}")
    p2_path = _download_checkpoint(repo_id, phase2_file)
    p2_ckpt = torch.load(p2_path, map_location="cpu", weights_only=False)
    p2_state = p2_ckpt.get("model_state_dict", p2_ckpt)
    incomp = _model.load_state_dict(p2_state, strict=False)
    print(f"[LLM2Seq] Phase 2 loaded. Missing keys: {len(incomp.missing_keys)}, "
          f"Unexpected keys: {len(incomp.unexpected_keys)}")

    print(f"[LLM2Seq] Downloading Phase 3 checkpoint: {repo_id}/{phase3_file}")
    p3_path = _download_checkpoint(repo_id, phase3_file)
    p3_ckpt = torch.load(p3_path, map_location="cpu", weights_only=False)
    p3_state = p3_ckpt.get("model_state_dict", p3_ckpt)
    incomp = _model.load_state_dict(p3_state, strict=False)
    print(f"[LLM2Seq] Phase 3 loaded. Missing keys: {len(incomp.missing_keys)}, "
          f"Unexpected keys: {len(incomp.unexpected_keys)}")

    _model.to(_device)
    _model.eval()
    _model_ready = True

    total_params = sum(p.numel() for p in _model.parameters())
    print(f"[LLM2Seq] Model ready — {total_params:,} parameters on {_device}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = BACKEND_DIR / "config.yaml"
    if not config_path.exists():
        print(f"[LLM2Seq] WARNING: config.yaml not found at {config_path}")
    else:
        _load_model(config_path)
    yield

app = FastAPI(
    title="LLM2Seq Demo API",
    description="Text summarisation with LLM2Seq Phase 3 (MTP Self-Distillation)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Source text to summarise")
    decode_mode: str = Field(
        "autoregressive",
        description="Decoding strategy: 'autoregressive' or 'mtp_verified'",
    )
    max_new_tokens: int = Field(256, ge=16, le=512)

class SummarizeResponse(BaseModel):
    summary: str
    decode_mode: str
    latency_seconds: float
    generated_tokens: int
    tokens_per_second: float
    mtp_metrics: Optional[Dict[str, Any]] = None

class ModelInfoResponse(BaseModel):
    encoder_name: str
    d_enc: int
    d_dec: int
    decoder_layers: int
    decoder_hidden: int
    decoder_heads: int
    use_mtp: bool
    mtp_type: str
    mtp_num_heads: int
    total_params: int
    device: str
    ready: bool

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_ready": _model_ready}

@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    if not _model_ready or _cfg is None or _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return ModelInfoResponse(
        encoder_name=_cfg.encoder_name,
        d_enc=_cfg.d_enc,
        d_dec=_cfg.d_dec,
        decoder_layers=_cfg.dec_num_layers,
        decoder_hidden=_cfg.dec_hidden_size,
        decoder_heads=_cfg.dec_num_heads,
        use_mtp=_cfg.use_mtp,
        mtp_type=_cfg.mtp_type,
        mtp_num_heads=_cfg.mtp_num_heads,
        total_params=sum(p.numel() for p in _model.parameters()),
        device=str(_device),
        ready=_model_ready,
    )

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    if not _model_ready or _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if req.decode_mode not in ("autoregressive", "mtp_verified"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decode_mode: {req.decode_mode}. "
                   f"Must be 'autoregressive' or 'mtp_verified'.",
        )

    source_prefix = _raw_cfg.get("data", {}).get("source_prefix", "")
    gen_cfg = _raw_cfg.get("generation", {})

    if req.text.startswith(source_prefix):
        final_text = req.text
    else:
        final_text = source_prefix + req.text

    enc = _tokenizer(
        final_text,
        return_tensors="pt",
        truncation=True,
        max_length=_raw_cfg.get("data", {}).get("max_source_length", 3072),
    ).to(_device)

    if _device.type == "cuda":
        torch.cuda.synchronize(_device)
    elif _device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()

    mtp_metrics = None

    if req.decode_mode == "autoregressive":
        out_ids = autoregressive_generate(
            _model,
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=req.max_new_tokens,
            min_new_tokens=int(gen_cfg.get("min_new_tokens", 64)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
            temperature=float(gen_cfg.get("temperature", 0.0)),
            top_k=int(gen_cfg.get("top_k", 0)),
            top_p=float(gen_cfg.get("top_p", 1.0)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.05)),
            no_repeat_ngram_size=int(gen_cfg.get("no_repeat_ngram_size", 3)),
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
            bos_token_id=_tokenizer.bos_token_id or _tokenizer.eos_token_id,
        )
    else:
        if _model.mtp_module is None:
            raise HTTPException(
                status_code=400,
                detail="MTP module not available. Use 'autoregressive' mode.",
            )
        mtp_result = mtp_generate(
            _model,
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=req.max_new_tokens,
            min_new_tokens=int(gen_cfg.get("min_new_tokens", 64)),
            verify_with_main=True,
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.05)),
            no_repeat_ngram_size=int(gen_cfg.get("no_repeat_ngram_size", 3)),
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
            bos_token_id=_tokenizer.bos_token_id or _tokenizer.eos_token_id,
            fallback_to_autoregressive=bool(
                gen_cfg.get("mtp_fallback_to_autoregressive", False)
            ),
            fallback_after_steps=int(gen_cfg.get("mtp_fallback_after_steps", 1)),
            fallback_min_emitted_length=float(
                gen_cfg.get("mtp_fallback_min_emitted_length", 3.8)
            ),
        )
        out_ids = mtp_result["generated_ids"]
        mtp_metrics = mtp_result.get("metrics")

    if _device.type == "cuda":
        torch.cuda.synchronize(_device)
    elif _device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    summary = _tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    num_tokens = int(out_ids[0].ne(_tokenizer.pad_token_id).sum().item()) if _tokenizer.pad_token_id is not None else out_ids[0].size(0)

    safe_mtp = None
    if mtp_metrics:
        safe_mtp = {}
        for k, v in mtp_metrics.items():
            if isinstance(v, (int, float, bool, str)):
                safe_mtp[k] = v
            elif isinstance(v, list):
                safe_mtp[k] = [float(x) if isinstance(x, (int, float)) else str(x) for x in v]
            elif v is None:
                safe_mtp[k] = None
            else:
                safe_mtp[k] = str(v)

    return SummarizeResponse(
        summary=summary,
        decode_mode=req.decode_mode,
        latency_seconds=round(elapsed, 4),
        generated_tokens=num_tokens,
        tokens_per_second=round(num_tokens / max(elapsed, 1e-9), 2),
        mtp_metrics=safe_mtp,
    )

@app.get("/api/random-sample")
async def random_sample():
    test_json_path = PROJECT_ROOT / "src" / "llm2seq" / "datasets" / "wikilingua" / "test.json"
    if not test_json_path.exists():
        raise HTTPException(status_code=404, detail="test.json not found")
    
    with open(test_json_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if not lines:
        raise HTTPException(status_code=404, detail="test.json is empty")
        
    line = random.choice(lines)
    data = json.loads(line)
    src_text = " ".join(data.get("src", []))
    tgt_text = " ".join(data.get("tgt", []))
    return {"source": src_text, "target": tgt_text}
