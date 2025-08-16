from datasets import DatasetDict, Dataset, load_dataset
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_CACHE"] = "/mnt/william/.cache"

### CONFIG ###

dataset_name = "fineweb-10BT"

DATA_DIR = "/mnt/william/reverse-llm/data"
TOKENIZER_DIR = "/mnt/william/reverse-llm/tokenizers"

### CONFIG ###

if __name__ == "__main__":
    print(f"Downloading dataset {dataset_name}...")
    time.sleep(1)

    # raw_dataset = load_dataset(
    #     "HuggingFaceFW/fineweb-edu",
    #     split="train",
    #     name="sample-10BT",
    #     cache_dir="/mnt/william/.cache",
    # )

    print("Generating split datasets...")
    # split_datasets = raw_dataset.train_test_split(test_size=0.005, seed=0)
    # split_datasets = DatasetDict({
    #     "train": split_datasets["train"],
    #     "valid": split_datasets["test"],
    # })

    split_datasets = DatasetDict({
        "train": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/train"),
        "valid": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/valid"),
    })

    print("=== SAMPLE DATA ===")
    print(list(split_datasets["train"].take(1))[0]["text"][500::-1])
    print()

    for split in split_datasets:
        print(f"Saving {split} to disk...")
        split_datasets[split].map(
            lambda x: { "text": x["text"][::-1] },
            remove_columns=["text"],
        ).to_parquet(f"{DATA_DIR}/{dataset_name}/{split}_10k_batch.parquet", batch_size=10_000)
