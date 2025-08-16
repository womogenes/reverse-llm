import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_CACHE"] = "/mnt/william/.cache"

### CONFIG ###

dataset_name = "fineweb-10BT"

DATA_DIR = "/mnt/william/reverse-llm/data"
TOKENIZER_DIR = "/mnt/william/reverse-llm/tokenizers"

### CONFIG ###

if __name__ == "__main__":
    print(f"Downloading dataset {dataset_name}...")
    time.sleep(10)

    from datasets import DatasetDict, Dataset

    # Dataset is too big to fit into memory so we stream
    split_datasets = DatasetDict({
        "train": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/train"),
        "valid": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/val"),
    })

    # Train tokenizer (7.4s on 1k examples)
    # 3m 30s on 200k examples
    # 20m on 900k examples (fineweb-10BT)

    from tokenizers import SentencePieceBPETokenizer

    def text_iterator():
        fraction = 0.002
        n_samples = int(split_datasets["train"].num_rows * fraction)
        for x in split_datasets["train"]["text"].shuffle(seed=0).select(range(n_samples)):
            yield x

    print("Training tokenizer...")
    spm_tokenizer = SentencePieceBPETokenizer()
    spm_tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size=52_000,
        min_frequency=5,
        show_progress=True,
    )

    spm_tokenizer.save(f"{TOKENIZER_DIR}/{dataset_name}")
