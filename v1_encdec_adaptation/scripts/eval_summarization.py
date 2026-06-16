import argparse
import yaml
import numpy as np

from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, EncoderDecoderModel
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Evaluating on {device}")

    model = EncoderDecoderModel.from_pretrained(args.model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    dataset_name = config.get("dataset", "nam194/vietnews")
    ds = load_dataset(dataset_name)
    split = "test" if "test" in ds else "validation"
    
    eval_samples = config.get("eval_samples", 200)
    subset = ds[split].select(range(min(eval_samples, len(ds[split]))))

    preds, refs = [], []
    prefix = config.get("prefix", "vietnews: ")
    text_column = config.get("text_column", "article")
    summary_column = config.get("summary_column", "abstract")

    for ex in subset:
        input_text = prefix + str(ex[text_column])
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=config.get("max_source_length", 2048),
        ).to(device)
        
        # Proper Beam Search as per research
        out = model.generate(
            **inputs,
            max_new_tokens=config.get("max_target_length", 512),
            num_beams=5,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True
        )
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append(ex[summary_column])

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    rouge_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=False)
    bert_scores = bertscore.compute(predictions=preds, references=refs, lang="vi")

    print("\n[ROUGE Scores]")
    for k, v in rouge_scores.items():
        print(f"  {k}: {v:.4f}")

    print("\n[BERTScore]")
    print(f"  precision: {sum(bert_scores['precision'])/len(bert_scores['precision']):.4f}")
    print(f"  recall:    {sum(bert_scores['recall'])/len(bert_scores['recall']):.4f}")
    print(f"  f1:        {sum(bert_scores['f1'])/len(bert_scores['f1']):.4f}")


if __name__ == "__main__":
    main()
