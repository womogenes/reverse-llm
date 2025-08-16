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
from datasets import Dataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

context_length = 4096 
dataset = "fineweb-10BT"
batch_size = 5


model_name = f"reverse-model-2B-{dataset}-ctx-{context_length}-batchsize-{batch_size}"

DATA_DIR = "/mnt/william/reverse-llm/data"
TOKENIZER_DIR = "/mnt/william/reverse-llm/tokenizers"
MODEL_DIR = "/mnt/william/reverse-llm/models"


### LOAD TOKENIZER
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{TOKENIZER_DIR}/fineweb_spm_200k")


### LOAD TOKENIZED DATASETS
tokenized_datasets = {}
for split in ["train", "valid"]:
    tokenized_datasets[split] = (
        Dataset.from_parquet(
            f"{DATA_DIR}/{dataset}/tokenized_{context_length}_{split}.parquet")
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
print(f"Model size: {model_size/1000**3:.1f}B parameters")


### SET UP DATA COLLATOR
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


### TRAINING ARGS
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir=f"{MODEL_DIR}/{model_name}",
    report_to="wandb",
    
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=math.ceil(32 / batch_size / accelerator.num_processes),

    eval_strategy="steps",
    eval_steps=100,
    logging_steps=1,
    
    num_train_epochs=1,
    
    optim="adamw_torch",
    learning_rate=2e-4,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    
    lr_scheduler_type="cosine",
    warmup_steps=2_000,

    max_grad_norm=1.0,
    
    bf16=True,
    fp16=False,
    
    save_steps=100,
    save_total_limit=3,
    save_only_model=False,
    dataloader_num_workers=2,

    remove_unused_columns=False,
    load_best_model_at_end=False,
    
    # Multi-gpu settings
    ddp_find_unused_parameters=False,
)

import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(
    project="reverse-llm-training-modal",
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

checkpoint = f"{MODEL_DIR}/{model_name}/checkpoint-1000"
# checkpoint = None
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
