import json


def create_cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")],
    }


notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

notebook["cells"].append(create_cell(
    "# Kaggle P100-compatible stack (sm_60)\n"
    "import os\n"
    "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"
    "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
    "!pip uninstall -y -q torch torchvision torchaudio\n"
    "!pip install -q --index-url https://download.pytorch.org/whl/cu118 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1\n"
    "!pip install -q transformers==4.46.3 datasets evaluate accelerate rouge-score bert-score"
))

with open("src/qwen_encdec_bridge.py", "r", encoding="utf-8") as f:
    bridge_code = f.read()
with open("src/warmup.py", "r", encoding="utf-8") as f:
    warmup_code = f.read()
with open("src/data_utils.py", "r", encoding="utf-8") as f:
    data_utils_code = f.read()

notebook["cells"].append(create_cell("# Qwen EncDec Bridge Model\n" + bridge_code))
notebook["cells"].append(create_cell("# Warmup Callback\n" + warmup_code))
notebook["cells"].append(create_cell("# Data Utils\n" + data_utils_code))

config_code = """config = {
    "encoder_model": "Qwen/Qwen2.5-0.5B",
    "decoder_model": "Qwen/Qwen2.5-0.5B",
    "share_backbone_weights": True,
    "tied_embeddings": True,
    "bridge_layers": 2,
    "dataset": "nam194/vietnews",
    "text_column": "article",
    "summary_column": "abstract",
    "prefix": "vietnews: ",
    "warmup_steps": 0,
    "freeze_backbones_only": True,
    "max_source_length": 512,
    "max_target_length": 128,
    "train_samples": 4000,
    "eval_samples": 500,
    "epochs": 10,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 3e-5,
}
"""
notebook["cells"].append(create_cell(config_code))

train_code = """import torch
import numpy as np
import inspect
import evaluate
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this notebook. Please enable GPU in Kaggle settings.")

print("Building Qwen-EncDec Bridge model...")
cfg = QwenEncDecBridgeConfig(
    encoder_model=config["encoder_model"],
    decoder_model=config["decoder_model"],
    share_backbone_weights=config["share_backbone_weights"],
    tied_embeddings=config["tied_embeddings"],
    bridge_layers=config["bridge_layers"],
)
model, tokenizer = build_qwen_encdec_bridge(cfg)
if config.get("freeze_backbones_only", True):
    print("Freezing backbones. Training bridge only for memory efficiency.")
    model.freeze_backbones()
use_gc = config.get("use_gradient_checkpointing", not config.get("freeze_backbones_only", True))
model.config.use_cache = not use_gc
if use_gc:
    if hasattr(model.encoder_lm, "gradient_checkpointing_enable"):
        model.encoder_lm.gradient_checkpointing_enable()
    if model.decoder_lm is not model.encoder_lm and hasattr(model.decoder_lm, "gradient_checkpointing_enable"):
        model.decoder_lm.gradient_checkpointing_enable()
else:
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

print("Loading dataset...")
train_tok, eval_tok = load_and_preprocess_dataset(
    dataset_name=config["dataset"],
    tokenizer=tokenizer,
    text_column=config["text_column"],
    summary_column=config["summary_column"],
    prefix=config["prefix"],
    max_source_length=config["max_source_length"],
    max_target_length=config["max_target_length"],
    train_samples=config["train_samples"],
    eval_samples=config["eval_samples"],
)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if hasattr(preds, "ndim") and preds.ndim == 3:
        preds = preds.argmax(-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
    return {k: round(v, 4) for k, v in scores.items()}

training_kwargs = dict(
    output_dir="outputs/qwen-bridge-vietnews",
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    gradient_accumulation_steps=config["grad_accum_steps"],
    learning_rate=config["lr"],
    num_train_epochs=config["epochs"],
    logging_steps=50,
    eval_strategy="no",
    save_strategy="no",
    predict_with_generate=False,
    fp16=True,
    gradient_checkpointing=use_gc,
    save_safetensors=False,
    report_to="none",
)

sig_args = inspect.signature(Seq2SeqTrainingArguments.__init__)
if "use_cpu" in sig_args.parameters:
    training_kwargs["use_cpu"] = False
elif "no_cuda" in sig_args.parameters:
    training_kwargs["no_cuda"] = False

training_args = Seq2SeqTrainingArguments(**training_kwargs)
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)

callbacks = []
if config["warmup_steps"] > 0 and not config.get("freeze_backbones_only", True):
    callbacks.append(FreezeNonCrossAttentionCallback(warmup_steps=config["warmup_steps"]))

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)
sig_tr = inspect.signature(Seq2SeqTrainer.__init__)
if "processing_class" in sig_tr.parameters:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in sig_tr.parameters:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Seq2SeqTrainer(**trainer_kwargs)
print("Starting training...")
trainer.train()
torch.cuda.empty_cache()

import os
os.makedirs("outputs/qwen-bridge-vietnews", exist_ok=True)
torch.save(model.state_dict(), "outputs/qwen-bridge-vietnews/pytorch_model.bin")
tokenizer.save_pretrained("outputs/qwen-bridge-vietnews")
print("Training completed and checkpoint saved.")
"""
notebook["cells"].append(create_cell(train_code))

eval_code = """import torch
from datasets import load_dataset
import evaluate

print("Starting Evaluation...")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for evaluation. Please enable GPU in Kaggle settings.")
device = "cuda"
model.to(device)
model.eval()

bertscore = evaluate.load("bertscore")

ds = load_dataset(config["dataset"])
split = "test" if "test" in ds else "validation"
subset = ds[split].select(range(min(config["eval_samples"], len(ds[split]))))

preds, refs = [], []
for ex in subset:
    src = config["prefix"] + str(ex[config["text_column"]])
    inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=config["max_source_length"]).to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=config["max_target_length"],
            num_beams=1,
        )
    preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
    refs.append(ex[config["summary_column"]])

rouge_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=False)
bert_scores = bertscore.compute(predictions=preds, references=refs, lang="vi", device="cuda:0")

print("\\n[ROUGE Scores]")
for k, v in rouge_scores.items():
    print(f"  {k}: {v:.4f}")

print("\\n[BERTScore]")
print(f"  precision: {sum(bert_scores['precision'])/len(bert_scores['precision']):.4f}")
print(f"  recall:    {sum(bert_scores['recall'])/len(bert_scores['recall']):.4f}")
print(f"  f1:        {sum(bert_scores['f1'])/len(bert_scores['f1']):.4f}")
"""
notebook["cells"].append(create_cell(eval_code))

with open("kaggle_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Created kaggle_training.ipynb successfully!")
