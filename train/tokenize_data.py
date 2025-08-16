from datasets import Dataset, DatasetDict

import sys

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

n_samples = 10_000_000_000
context_length = 4096

# DATA_DIR = "/home/wyf/ai/causal-llm/data"
DATA_DIR = "/home/wyf/orcd/pool/causal-llm/data"
TOKENIZER_DIR = "/home/wyf/ai/causal-llm/tokenizers"
MODEL_DIR = "/home/wyf/ai/causal-llm/models"

dataset = "fineweb-10BT"
model_name = f"reverse-{dataset}-ctx-{context_length}-2B"

split_datasets = DatasetDict({
    "train": Dataset.load_from_disk(f"{DATA_DIR}/{dataset}/train"),
    "valid": Dataset.load_from_disk(f"{DATA_DIR}/{dataset}/val")
})

# Load pretrained tokenizer
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{TOKENIZER_DIR}/fineweb_spm_1M")

# USES CONTEXT LENGTH
print(f"Context length: {context_length}")

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

# 25s to parse 1k examples
# 4m 40s to parse 10k examples
# 7m 50s to parse 200k examples
tokenized_dataset_train = split_datasets["train"].map(
    tokenize, batched=True, remove_columns=["text"], batch_size=32)
tokenized_dataset_valid = split_datasets["valid"].map(
    tokenize, batched=True, remove_columns=["text"], batch_size=32)


# Takes a hot minute to save
# 30min for 2M fineweb examples
# 3m for 1/9 of that
print("Saving training dataset...")
tokenized_dataset_train.to_parquet(
    f"{DATA_DIR}/{dataset}/tokenized_{context_length}_train.parquet")
print("Saving valid dataset...")
tokenized_dataset_valid.to_parquet(
    f"{DATA_DIR}/{dataset}/tokenized_{context_length}_valid.parquet")


print(tokenized_dataset_train)
print(f"Produced dataset of {tokenized_dataset_train.num_rows:,} rows, {context_length} tokens each")
print(f"Total tokens: {tokenized_dataset_train.num_rows * context_length:,}")
