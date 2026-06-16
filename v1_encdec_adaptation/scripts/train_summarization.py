import argparse
import inspect
import yaml
import numpy as np

import evaluate
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from src.data_utils import load_and_preprocess_dataset
from src.warmup import FreezeNonCrossAttentionCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Path to adapted model")
    p.add_argument("--config", required=True, help="Path to config yaml")
    p.add_argument("--output_dir", required=True, help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = EncoderDecoderModel.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    train_tok, eval_tok = load_and_preprocess_dataset(
        dataset_name=config.get("dataset", "nam194/vietnews"),
        tokenizer=tokenizer,
        text_column=config.get("text_column", "article"),
        summary_column=config.get("summary_column", "abstract"),
        prefix=config.get("prefix", "vietnews: "),
        max_source_length=config.get("max_source_length", 2048),
        max_target_length=config.get("max_target_length", 512),
        train_samples=config.get("train_samples", -1),
        eval_samples=config.get("eval_samples", -1),
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Note: use_stemmer is usually False for Vietnamese, or custom tokenizer is needed.
        # ROUGE from evaluate handles basic whitespace splitting.
        scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
        return {k: round(v, 4) for k, v in scores.items()}

    cuda_usable = torch.cuda.is_available()
    if cuda_usable:
        major, _ = torch.cuda.get_device_capability(0)
        # Avoid attempting CUDA training on unsupported legacy GPUs with recent torch builds.
        if major < 7:
            print("CUDA device capability < 7.0 detected; forcing CPU training for compatibility.")
            cuda_usable = False

    training_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.get("batch_size", 2),
        per_device_eval_batch_size=config.get("batch_size", 2),
        learning_rate=float(config.get("lr", 3e-5)),
        num_train_epochs=config.get("epochs", 3),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        predict_with_generate=True,
        generation_max_length=config.get("max_target_length", 512),
        generation_num_beams=4,
        fp16=False if not cuda_usable else True,
        report_to="none",
        lr_scheduler_type="cosine",
    )
    ta_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "use_cpu" in ta_sig.parameters:
        training_kwargs["use_cpu"] = not cuda_usable
    elif "no_cuda" in ta_sig.parameters:
        training_kwargs["no_cuda"] = not cuda_usable

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    callbacks = []
    warmup_steps = config.get("warmup_steps", 0)
    if warmup_steps > 0:
        callbacks.append(FreezeNonCrossAttentionCallback(warmup_steps=warmup_steps))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    # Transformers API changed: `tokenizer` -> `processing_class` in newer versions.
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    import torch
    main()
