from datasets import Dataset, DatasetDict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

context_length = 4096

DATA_DIR = "/mnt/william/reverse-llm/data"
TOKENIZER_DIR = "/mnt/william/reverse-llm/tokenizers"

dataset_name = "fineweb-10BT"

def main():
    split_datasets = DatasetDict({
        "train": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/train"),
        "valid": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/valid"),
    })
    # split_datasets = DatasetDict({
    #     "train": Dataset.from_parquet(f"{DATA_DIR}/{dataset_name}/train.parquet"),
    #     "valid": Dataset.from_parquet(f"{DATA_DIR}/{dataset_name}/valid.parquet"),
    # })

    print(split_datasets)

    print("=== BEGIN EXAMPLE DATA (REVERSED) ===")
    print(list(split_datasets["train"].take(1))[0]["text"][500::-1])
    print()

    # Load pretrained tokenizer
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{TOKENIZER_DIR}/fineweb_spm_200k")

    # USES CONTEXT LENGTH
    print(f"Context length: {context_length}")
    print(split_datasets)

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            # padding="max_length",
            return_overflowing_tokens=False,
            # return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            input_batch.append(input_ids)
        return {"input_ids": input_batch}

    # 25s to parse 1k examples
    # 4m 40s to parse 10k examples
    # 7m 50s to parse 200k examples
    tokenized_dataset_train = split_datasets["train"].map(
        tokenize, batched=True, remove_columns=["text"])
    tokenized_dataset_valid = split_datasets["valid"].map(
        tokenize, batched=True, remove_columns=["text"])


    # Takes a hot minute to save
    # 30min for 2M fineweb examples
    # 3m for 1/9 of that
    print("Saving training dataset...")
    tokenized_dataset_train.to_parquet(
        f"{DATA_DIR}/{dataset_name}/tokenized_{context_length}_train.parquet")
    print("Saving valid dataset...")
    tokenized_dataset_valid.to_parquet(
        f"{DATA_DIR}/{dataset_name}/tokenized_{context_length}_valid.parquet")


    print(tokenized_dataset_train)
    print(f"Produced dataset of {tokenized_dataset_train.num_rows:,} rows, {context_length} tokens each")
    print(f"Total tokens: {tokenized_dataset_train.num_rows * context_length:,}")

main()
