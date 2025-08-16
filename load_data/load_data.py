from datasets import load_dataset, DatasetDict
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

    # Takes like 30s to load (it's bad)
    raw_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        name="sample-10BT",
        cache_dir="/mnt/william/.cache",
    )

    print("Generating split datasets...")
    split_datasets = raw_dataset.train_test_split(test_size=0.005, seed=0)
    split_datasets = DatasetDict({
        "train": split_datasets["train"],
        "valid": split_datasets["test"],
    })

    print("=== SAMPLE DATA ===")
    print(list(split_datasets["train"].take(1))[0]["text"][500::-1])
    print()

    for split in split_datasets:
        print(f"Saving {split} to disk...")
        split_datasets[split].map(
            lambda x: { "text": x["text"][::-1] },
            remove_columns=["text"],
        ).save_to_disk(f"{DATA_DIR}/{dataset_name}/{split}")
