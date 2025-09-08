from datasets import DatasetDict, load_dataset
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

def main():
    print(f"Downloading dataset {dataset_name}...")
    time.sleep(1)

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

    print("=== SAMPLE DATA (REVERSED) ===")
    print(list(split_datasets["train"].take(1))[0]["text"][500::-1])
    print()

    for split in split_datasets:
        save_path = f"{DATA_DIR}/{dataset_name}/{split}"
        # split_datasets[split].select_columns(["text"]).map(
        #     lambda batch: { "text": [text[::-1] for text in batch["text"]] },
        #     batched=True,
        #     batch_size=1_000,
        #     num_proc=int(os.cpu_count() * 1.5),
        #     remove_columns=["text"],
        # ).save_to_disk(save_path)

        split_datasets[split].select_columns(["text"]).save_to_disk(save_path)

        print(f"Saved split {split} to {save_path}")

main()
