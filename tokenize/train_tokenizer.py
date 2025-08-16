import os
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_CACHE"] = "/mnt/william/.cache"

os.environ["PYTHONUNBUFFERED"] = "1"

### CONFIG ###

dataset_name = "fineweb-10BT"

DATA_DIR = "/mnt/william/reverse-llm/data"
TOKENIZER_DIR = "/mnt/william/reverse-llm/tokenizers"

### CONFIG ###
def main():
    print(f"Loading dataset {dataset_name}...")

    from datasets import Dataset

    # Dataset is too big to fit into memory so we stream
    train_dataset = Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/train")

    # Train tokenizer (7.4s on 1k examples)
    # 3m 30s on 200k examples
    # 20m on 900k examples (fineweb-10BT)

    from tokenizers import SentencePieceBPETokenizer

    def text_iterator():
        fraction = 0.02
        n_samples = int(train_dataset.num_rows * fraction)
        print(f"Using {n_samples:,} samples")

        print("Shuffling dataset and selecting samples...")
        filtered_dataset = train_dataset.shuffle(seed=0).select(range(n_samples))
        print("Generating text iterator...")
        for x in tqdm(filtered_dataset["text"]):
            yield x

    print("Training tokenizer...")
    spm_tokenizer = SentencePieceBPETokenizer()
    spm_tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size=52_000,
        min_frequency=5,
        show_progress=True,
    )

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=spm_tokenizer,
        bos_token="<s>",           # Always added at start
        eos_token="</s>",          # Always added at end  
        unk_token="<unk>",         # Replaces unknown words
        pad_token="<pad>",         # Used for padding shorter sequences
    )

    tokenizer_path = f"{TOKENIZER_DIR}/{dataset_name}"
    print(f"Saving tokenizer to {tokenizer_path}")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save(tokenizer_path)

main()
