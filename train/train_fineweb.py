### THINGS TO CHANGE IN THIS FILE (fix to not hardcode later)
# - model name (be descriptive)
# - wandb config values for logging
# - checkpoint


### ACCELERATE CONFIG

from accelerate import Accelerator
import os
import math

accelerator = Accelerator()

if not accelerator.is_main_process:
    os.environ["WANDB_MODE"] = "disabled"
    print = lambda *args: None


### IMPORTS
from datasets import load_dataset, Dataset, DatasetDict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

n_samples = int(1e9) 
context_length = 4096 
dataset = "fineweb-10BT"
batch_size = 5


model_name = f"reverse-model-2B-{dataset}-ctx-{context_length}-batchsize-{batch_size}"

import wandb
wandb.init(
    project="causal-llm-training",
    name=model_name,
    entity="womogenes-team",
    config={
        "n_samples": n_samples,
        "context_length": context_length,
        "dataset": dataset,
        "batchsize": batch_size,
    },
    resume="allow",
)


DATA_DIR = "/home/wyf/orcd/pool/causal-llm/data"
TOKENIZER_DIR = "/home/wyf/ai/causal-llm/tokenizers"
MODEL_DIR = "/home/wyf/ai/causal-llm/models"


### LOAD TOKENIZER
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{TOKENIZER_DIR}/fineweb_spm_1M")



### LOAD TOKENIZED DATASETS
tokenized_dataset = Dataset.from_parquet(
    f"{DATA_DIR}/{dataset}/tokenized_{context_length}_train.parquet")
tokenized_dataset_valid = Dataset.from_parquet(
    f"{DATA_DIR}/{dataset}/tokenized_{context_length}_valid.parquet")

print(f"Dataset size: {tokenized_dataset.num_rows * context_length:,} tokens")

print(tokenized_dataset)
print(f"Produced dataset of {tokenized_dataset.num_rows:,} rows, {context_length} tokens each")
print(f"Total tokens: {tokenized_dataset.num_rows * context_length:,}")

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Model config vocab size: {tokenizer.vocab_size}")
print(f"BOS token ID: {tokenizer.bos_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"PAD token ID: {tokenizer.pad_token_id}")

# Check a sample tokenization
sample_text = "hello world"
tokens = tokenizer(sample_text)
print(f"Sample tokens: {tokens}")


# Check sample data
print()


### INITIALIZE MODEL

# 3.2s to initialize model
from transformers import LlamaConfig, LlamaForCausalLM
import torch

model_size = "2B"

config = LlamaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=8192,
    hidden_size=2048 if model_size == "2B" else 3072,
    intermediate_size=16384 if model_size == "2B" else 24576,
    num_hidden_layers=18 if model_size == "2B" else 28,
    num_attention_heads=8 if model_size == "2B" else 16,
    num_key_value_heads=1 if model_size == "2B" else 16,
    rms_norm_eps=1e-5,
    tie_word_embeddings=False,
    rope_scaling=None,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = LlamaForCausalLM(config).to("cuda")

model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1000**2:.1f}M parameters")

### SET UP DATA COLLATOR
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = "<pad>"
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.unk_token = "<unk>"
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)



### TRAINING ARGS
# 0.1s to initialize training args

from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir=f"{MODEL_DIR}/{model_name}",
    report_to="wandb",
    
    # Batch size settings - LEDOM uses global batch size of 1024 sequences
    per_device_train_batch_size=batch_size,                             # Micro-batch size per GPU
    per_device_eval_batch_size=batch_size,                              # Used in their fine-tuning setup
    gradient_accumulation_steps=math.ceil(32 / batch_size / 2),         # To achieve global batch size (adjust based on GPU count)

    eval_strategy="steps",          # Evaluate every N steps
    eval_steps=1000,                # Eval every N steps
    logging_steps=1,                # More frequent logging to match monitoring
    
    # Training duration - LEDOM trained for ~51,900 iterations for 7B model
    num_train_epochs=1,             # Keep as 1 epoch since they trained on 435B tokens once
    
    # Optimizer settings - match LEDOM exactly
    optim="adamw_torch",
    learning_rate=2e-4,           # Whee
    weight_decay=0.1,             # Matches their setting
    adam_beta1=0.9,               # Adam β₁
    adam_beta2=0.95,              # Adam β₂  
    adam_epsilon=1e-8,            # Adam ε
    
    # Learning rate schedule - LEDOM uses cosine with specific warmup
    lr_scheduler_type="cosine",
    warmup_steps=2_000,

    # Gradient settings
    max_grad_norm=10.0,            # Gradient clipping norm
    
    # Precision - LEDOM uses BF16, not FP16
    bf16=True,                    # Use BF16 instead of FP16
    fp16=False,                   # Disable FP16
    
    # Checkpointing
    save_steps=1000,
    save_total_limit=1,           # Reasonable limit for storage
    save_only_model=False,
    
    # Additional LEDOM-specific settings
    dataloader_num_workers=2,     # For efficiency
    remove_unused_columns=False,  # Keep all data columns
    
    # Disable features not used in LEDOM training
    load_best_model_at_end=False,
)

print(f"\n=== BEGIN TRAINING ARGS ===")
print(args)
print(f"=== END TRAINING ARGS ===\n")

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_valid,
)


### TRAINING
torch.cuda.empty_cache()

# checkpoint = f"{MODEL_DIR}/{model_name}/checkpoint-10000"
checkpoint = None
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
