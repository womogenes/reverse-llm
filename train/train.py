### IMPORTS
from datasets import load_dataset, Dataset, DatasetDict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

n_samples = 200_000
context_length = 1024


### LOAD DATASETS
datasets = DatasetDict({
    "train": Dataset.from_parquet(f"./data/dclm_{n_samples}/train.parquet"),
    "valid": Dataset.from_parquet(f"./data/dclm_{n_samples}/valid.parquet")
})



### LOAD TOKENIZER
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("./tokenizers/spm_200k")



### LOAD TOKENIZED DATASETS
tokenized_dataset = Dataset.from_parquet(f"./data/dclm_{n_samples}_tokenized_{context_length}.parquet")
tokenized_dataset_valid = Dataset.from_parquet(f"./data/dclm_{n_samples}_tokenized_{context_length}_valid.parquet")

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

# model = LlamaForCausalLM(config).to("cuda")
model = LlamaForCausalLM.from_pretrained("reverse-model-2B/checkpoint-1993").to("cuda")

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
    output_dir="reverse-model-fineweb-2B",
    
    # Batch size settings - LEDOM uses global batch size of 1024 sequences
    per_device_train_batch_size=1,  # Micro-batch size per GPU
    per_device_eval_batch_size=1,   # Used in their fine-tuning setup
    gradient_accumulation_steps=32, # To achieve global batch size (adjust based on GPU count)

    eval_strategy="steps",          # Evaluate every N steps
    eval_steps=5000,                # Eval every N steps  
    logging_steps=1,                # More frequent logging to match their monitoring
    
    # Training duration - LEDOM trained for ~51,900 iterations for 7B model
    num_train_epochs=1,             # Keep as 1 epoch since they trained on 435B tokens once
    
    # Optimizer settings - match LEDOM exactly
    optim="adamw_torch",
    learning_rate=1.12e-5,           # Whee
    weight_decay=0.1,             # Matches their setting
    adam_beta1=0.9,               # Adam β₁
    adam_beta2=0.95,              # Adam β₂  
    adam_epsilon=1e-8,            # Adam ε
    
    # Learning rate schedule - LEDOM uses cosine with specific warmup
    lr_scheduler_type="cosine",
    warmup_steps=10,

    # Gradient settings
    max_grad_norm=100.0,            # Gradient clipping norm
    
    # Precision - LEDOM uses BF16, not FP16
    bf16=True,                    # Use BF16 instead of FP16
    fp16=False,                   # Disable FP16
    
    # Checkpointing
    save_steps=200,
    save_total_limit=3,           # Reasonable limit for storage
    save_only_model=True,
    
    # Additional LEDOM-specific settings
    dataloader_num_workers=2,     # For efficiency
    remove_unused_columns=False,  # Keep all data columns
    
    # Disable features not used in LEDOM training
    load_best_model_at_end=False,
)

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

# 1m for 1k samples (2.2M tokens)
trainer.train()



### EVALUATE

# import torch
# from transformers import pipeline

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# pipe = pipeline(
#     "text-generation", model="./reverse-model/checkpoint-9", device=device, 
# )

import os
import torch
from transformers import pipeline

# Base model directory
base_dir = "./reverse-model-2B"

# Find the first subdirectory (sorted for consistency)
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
if not subdirs:
    raise FileNotFoundError(f"No subdirectories found in {base_dir}")
first_checkpoint = os.path.join(base_dir, sorted(subdirs)[0])

print(f"Using model from: {first_checkpoint}")

# Device selection
device = 0 if torch.cuda.is_available() else -1

# Load the pipeline
pipe = pipeline(
    "text-generation",
    model=first_checkpoint,
    device=device
)


text = pipe(".olleh", num_return_sequences=1)[0]["generated_text"]

print("generated text:\n", text)
print()
print("generated text (reversed):\n", text[::-1])
