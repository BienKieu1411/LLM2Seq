from typing import Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    text_column: str,
    summary_column: str,
    prefix: str = "",
    max_source_length: int = 2048,
    max_target_length: int = 512,
    train_samples: int = -1,
    eval_samples: int = -1,
    dataset_config: str = None
):
    """
    Loads dataset (like nam194/vietnews), applies prefix, and tokenizes.
    """
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config)
    else:
        ds = load_dataset(dataset_name)

    train_ds = ds["train"]
    if train_samples > 0:
        train_ds = train_ds.select(range(min(train_samples, len(train_ds))))
        
    val_split = "validation" if "validation" in ds else "test"
    eval_ds = ds[val_split]
    if eval_samples > 0:
        eval_ds = eval_ds.select(range(min(eval_samples, len(eval_ds))))

    def preprocess(batch):
        inputs = [prefix + str(doc) for doc in batch[text_column]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[summary_column],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    return train_tok, eval_tok
