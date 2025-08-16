from datasets import Dataset, DatasetDict

import sys

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
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
tokenizer_path = f"{TOKENIZER_DIR}/fineweb_spm_1M"
print(f"Loading pretrained tokenizer from {tokenizer_path}...")
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# USES CONTEXT LENGTH
print(f"Context length: {context_length}")
print(f"Sample untokenized data: {split_datasets['train'][0]['text'][:100]}")

def tokenize(element):
    outputs = tokenizer(
        # REVERSING because dataset was not reversed when saved
        [x[::-1] for x in element["text"]],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=False,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        input_batch.append(input_ids)
    return {"input_ids": input_batch}

# 25s to parse 1k examples
# 4m 40s to parse 10k examples
# 7m 50s to parse 200k examples
tokenized_dataset = {}
for split in ["train", "valid"]:
    print(f"Tokenizing split {split}...")
    tokenized_dataset[split] = split_datasets[split] \
        .select_columns("text") \
        .map(tokenize, batched=True, batch_size=32)

print(f"Sample data: {tokenizer.decode(tokenized_dataset['train'].take(0)['input_ids'])[:100]}")

# Takes a hot minute to save
# 30min for 2M fineweb examples
# 3m for 1/9 of that
print("Saving training dataset...")
for split in ["train", "valid"]:
    print(f"Saving {split} dataset...")
    tokenized_dataset[split].to_parquet(f"{DATA_DIR}/{dataset}/tokenized_{context_length}_{split}.parquet")


print(tokenized_dataset['train'])
print(f"Produced dataset of {tokenized_dataset['train'].num_rows:,} rows, {context_length} tokens each")
print(f"Total tokens: {tokenized_dataset['train'].num_rows * context_length:,}")
