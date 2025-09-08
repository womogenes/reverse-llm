import os
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_CACHE"] = "~/orcd/pool/.cache"

os.environ["PYTHONUNBUFFERED"] = "1"

### CONFIG ###

dataset_name = "fineweb-10BT"

DATA_DIR = "/home/wyf/orcd/pool/reverse-llm/data"
TOKENIZER_DIR = "/home/wyf/orcd/pool/reverse-llm/tokenizers"

### CONFIG ###
def main():
    print(f"Loading dataset {dataset_name}...")

    from datasets import Dataset

    train_dataset = Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/train")
    print(train_dataset)
    print(f"=== SAMPLE DATA (SHOULD READ FORWARDS) ===")
    print(train_dataset[0]["text"])

    from tokenizers import ByteLevelBPETokenizer

    def text_iterator():
        n_samples = 200_000
        print(f"Using {n_samples:,} samples ({n_samples / len(train_dataset):.2%} of dataset)")

        print("Shuffling dataset and selecting samples...")
        filtered_dataset = train_dataset.shuffle(seed=0).select(range(n_samples))
        print("Generating text iterator...")
        for x in tqdm(filtered_dataset["text"]):
            yield x[::-1]

    print("Training tokenizer...")
    bpe_tokenizer = ByteLevelBPETokenizer()
    bpe_tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size=52_000,
        min_frequency=5,
        show_progress=True,
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    tokenizer_path = f"{TOKENIZER_DIR}/fineweb_bpe_200k"
    print(f"Saving tokenizer to {tokenizer_path}")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    bpe_tokenizer.save(tokenizer_path)

    print(f"Done.")

main()
