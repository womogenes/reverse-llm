from datasets import Dataset, DatasetDict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

context_length = 1024

DATA_DIR = "/mnt/william/reverse-llm/data"
TOKENIZER_DIR = "/mnt/william/reverse-llm/tokenizers"

dataset_name = "fineweb-10BT"

def main():
    split_datasets = DatasetDict({
        "train": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/train"),
        "valid": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/valid"),
    })

    print(split_datasets)

    print("=== BEGIN EXAMPLE DATA (SHOULD READ FORWARDS) ===")
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
            [t[-1] for t in element["text"]],
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
    tokenized_datasets = {}
    for split in ["train", "valid"]:
        tokenized_datasets[split] = split_datasets[split] \
            .select_columns("text") \
            .map(
                tokenize,
                batched=True,
                remove_columns=["text"],
                batch_size=1_000,
                num_proc=os.cpu_count(),
                desc=f"Tokenizing {split}...",
            )

    print("=== SAMPLE DATA (SHOULD READ BACKWARDS) ===")
    print(f"{tokenizer.decode(tokenized_datasets['train'].take(0)['input_ids'])[:100]}")
    print(tokenized_datasets["train"])

    # Takes a hot minute to save
    # 30min for 2M fineweb examples
    # 3m for 1/9 of that
    print("Saving training dataset...")
    for split in ["train", "valid"]:
        print(f"Saving {split} dataset...")
        tokenized_datasets[split].to_parquet(f"{DATA_DIR}/{dataset_name}/tokenized_{context_length}_{split}.parquet")

    print(tokenized_datasets)
    print(f"Produced dataset of {tokenized_datasets['train'].num_rows:,} rows, {context_length} tokens each")
    print(f"Total tokens: {tokenized_datasets['train'].num_rows * context_length:,}")

main()
