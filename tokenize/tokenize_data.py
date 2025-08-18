from datasets import Dataset, DatasetDict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

context_length = 1024

DATA_DIR = "/home/wyf/orcd/pool/reverse-llm/data"
TOKENIZER_DIR = "/home/wyf/orcd/pool/reverse-llm/tokenizers"

dataset_name = "fineweb-10BT"

def main():
    split_datasets = DatasetDict({
        "train": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/train"),
        "valid": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/valid"),
    })

    print(split_datasets)

    print("=== BEGIN EXAMPLE DATA (SHOULD READ FORWARDS) ===")
    print(list(split_datasets["train"].take(1))[0]["text"][:100])
    print()

    # Load pretrained tokenizer
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{TOKENIZER_DIR}/fineweb_bpe_200k.json",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    print(tokenizer.special_tokens_map)

    # USES CONTEXT LENGTH
    print(f"Context length: {context_length}")
    print(split_datasets)

    def concatenate_and_chunk(element):
        all_token_ids = []
        for text in element["text"]:
            token_ids = tokenizer.encode(text[::-1], add_special_tokens=False)
            all_token_ids.extend(token_ids)
            all_token_ids.append(tokenizer.eos_token_id)

        total_length = len(all_token_ids)

        if total_length < context_length:
            return { "input_ids": [] }

        total_length = (total_length // context_length) * context_length

        # Split into chunks
        input_ids = [
            all_token_ids[i : i + context_length]
            for i in range(0, total_length, context_length)
        ]

        return { "input_ids": input_ids }

    tokenized_datasets = {}
    for split in ["train", "valid"]:
        tokenized_datasets[split] = split_datasets[split] \
            .select_columns("text") \
            .map(
                concatenate_and_chunk,
                batched=True,
                remove_columns=["text"],
                batch_size=1_000,
                num_proc=(os.cpu_count() - 1),
                desc=f"Tokenizing {split}...",
            )

    print(type(tokenized_datasets["train"][0]["input_ids"]))
    print(tokenized_datasets["train"][0]["input_ids"][:5])

    print("=== SAMPLE DATA (SHOULD READ BACKWARDS) ===")
    print(f"{tokenizer.decode(tokenized_datasets['train'][0]['input_ids'])[:100]}")
    print(tokenized_datasets["train"])

    # Takes a hot minute to save
    # 30min for 2M fineweb examples
    # 3m for 1/9 of that
    print("Saving training dataset...")
    for split in ["train", "valid"]:
        print(f"Saving {split} dataset...")
        tokenized_datasets[split].save_to_disk(f"{DATA_DIR}/{dataset_name}/tokenized_{context_length}_{split}")

    print(tokenized_datasets)
    print(f"Produced dataset of {tokenized_datasets['train'].num_rows:,} rows, {context_length} tokens each")
    print(f"Total tokens: {tokenized_datasets['train'].num_rows * context_length:,}")

main()
