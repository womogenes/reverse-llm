from datasets import load_dataset, Dataset, DatasetDict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

n_samples = 20_000_000
context_length = 1024



DATA_PATH = "/nobackup1/wyf/"

# Takes like 30s to load (it's bad)
raw_dataset = load_dataset(
    "mlfoundations/dclm-baseline-1.0",
    split="train",
    streaming=True,
)


from tqdm import tqdm
import sys

def filter_dataset(dataset, n_samples: int = None):
    # filtered = []
    # for sample in tqdm(iter(dataset["train"].take(n_samples)), total=n_samples):
    #     # IMPORTANT REVERSAL STEP
    #     filtered.append(sample["text"][::-1])
    # return filtered

    return (
        dataset
            .select_columns(["text"])
            .map(lambda s: {"text": s["text"][::-1]})
    )
    
# 1k examples: 4.0s

print("Generating split datasets...")
raw_dataset_with_tqdm = [x for x in tqdm(raw_dataset.take(n_samples), total=n_samples)]
split_datasets = (
    Dataset.from_list(list(raw_dataset_with_tqdm))
        .train_test_split(test_size=0.1, seed=0)
)
datasets = DatasetDict({
    "train": filter_dataset(split_datasets["train"]),
    "valid": filter_dataset(split_datasets["test"]),
})


print("Saving datasets...")
for split_name, dataset in datasets.items():
    dataset.to_parquet(f"./data/dclm_{n_samples}/{split_name}.parquet")

