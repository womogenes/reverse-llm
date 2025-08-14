from datasets import load_dataset, Dataset, DatasetDict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

dataset = "fineweb"

DATA_DIR = "/home/wyf/ai/causal-llm/data"
TOKENIZER_DIR = "/home/wyf/ai/causal-llm/tokenizer"

n_samples = 2_000_000
context_length = 1024

datasets = DatasetDict({
    "train": Dataset.from_parquet(f"{DATA_DIR}/{dataset}_{n_samples}/train.parquet"),
    "valid": Dataset.from_parquet(f"{DATA_DIR}/{dataset}_{n_samples}/valid.parquet")
})


# 7.4s on 1k examples
# 3m 30s on 200k examples

from transformers import AutoTokenizer, LlamaTokenizer
from tokenizers import SentencePieceBPETokenizer
from tqdm import tqdm

def text_iterator():
    for x in tqdm(datasets["train"]["text"]):
        yield x

spm_tokenizer = SentencePieceBPETokenizer()
spm_tokenizer.train_from_iterator(
    text_iterator(),
    vocab_size=52_000,
    min_frequency=5,
    show_progress=True,
    limit_alphabet=500,
)

from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=spm_tokenizer,
    bos_token="<s>",           # Always added at start
    eos_token="</s>",          # Always added at end  
    unk_token="<unk>",         # Replaces unknown words
    pad_token="<pad>",         # Used for padding shorter sequences
)
tokenizer.save_pretrained(f"{TOKENIZER_DIR}/{dataset}_spm_{n_samples}")
