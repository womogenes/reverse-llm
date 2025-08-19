### ACCELERATE CONFIG

from accelerate import Accelerator
import os
import math
import torch

accelerator = Accelerator()

if not accelerator.is_main_process:
    os.environ["WANDB_MODE"] = "disabled"
    print = lambda *args: None

n_gpus = accelerator.num_processes

### IMPORTS
from datasets import Dataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

context_length = 1024
dataset = "fineweb-10BT"

model_name = f"reverse-gpt2-0.35B-{dataset}-ctx-{context_length}"

DATA_DIR = "/home/wyf/orcd/pool/reverse-llm/data"
TOKENIZER_DIR = "/home/wyf/orcd/pool/reverse-llm/tokenizers"
MODEL_DIR = "/home/wyf/orcd/pool/reverse-llm/models"


### LOAD TOKENIZER
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=f"{TOKENIZER_DIR}/fineweb_bpe_200k.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)


### LOAD TOKENIZED DATASETS
tokenized_datasets = {}
for split in ["train", "valid"]:
    tokenized_datasets[split] = (
        Dataset.load_from_disk(f"{DATA_DIR}/{dataset}/tokenized_{context_length}_{split}")
        .select_columns(["input_ids"])
    )

print(f"Dataset size: {tokenized_datasets['train'].num_rows * context_length:,} tokens")

print(tokenized_datasets['train'])
print(f"Produced dataset of {tokenized_datasets['train'].num_rows:,} rows, {context_length} tokens each")
print(f"Total tokens: {tokenized_datasets['train'].num_rows * context_length:,}")

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Model config vocab size: {tokenizer.vocab_size}")
print(f"BOS token ID: {tokenizer.bos_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"PAD token ID: {tokenizer.pad_token_id}")


# Check sample data
print(f"=== SAMPLE DATA ===")
print(tokenizer.decode(tokenized_datasets['train'][0]['input_ids'])[:100])


### INITIALIZE MODEL
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=context_length,
    n_ctx=context_length,
    n_embd=1024,
    n_layer=24,
    n_head=16,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

model = GPT2LMHeadModel(config).to("cuda")
# model.gradient_checkpointing_enable()

model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1000**3:.3f}B parameters")


### SET UP DATA COLLATOR
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


### TRAINING ARGS
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir=f"{MODEL_DIR}/{model_name}",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=((1<<20) // (64 * n_gpus) // context_length),
    
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    eval_steps=500,

    max_grad_norm=1.0,

    logging_steps=1,
    save_steps=500,
    save_total_limit=None,
    prediction_loss_only=True,
    bf16=True,

    report_to="wandb",
    run_name=model_name,

    torch_compile=True,
    seed=0,
)

import wandb
wandb.init(
    project="reverse-llm-gpt2-0.35B",
    name=model_name,
    entity="womogenes-team",
    config=args.to_dict(),
    resume="allow",
)

print("\n=== BEGIN TRAINING ARGS ===")
print(args)
print("=== END TRAINING ARGS ===\n")

trainer = Trainer(
    model=model,
    processing_class=tokenizer, # new syntax
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

### TRAINING
torch.cuda.empty_cache()

import torch
print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"VRAM max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# checkpoint = None
checkpoint = True
trainer.train(resume_from_checkpoint=checkpoint)

### EVALUATE

from transformers import pipeline

# Device selection
device = 0 if torch.cuda.is_available() else -1

# Load the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    device=device
)


text = pipe(".olleh", num_return_sequences=1)[0]["generated_text"]

print("generated text:\n", text)
print()
print("generated text (reversed):\n", text[::-1])
