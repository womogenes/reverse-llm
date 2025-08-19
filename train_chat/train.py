
from transformers import PreTrainedTokenizerFast
import os
from accelerate import Accelerator

accelerator = Accelerator()

if not accelerator.is_main_process:
    os.environ["WANDB_MODE"] = "disabled"
    print = lambda *args: None

DATA_DIR = "/home/wyf/orcd/pool/reverse-llm/data"
TOKENIZER_DIR = "/home/wyf/orcd/pool/reverse-llm/tokenizers"
MODEL_DIR = "/home/wyf/orcd/pool/reverse-llm/models"

model_name = f"reverse-gpt2-0.35B-fineweb-10BT-ctx-1024"

USER_ROLE_NAME = "user"[::-1]
ASSISTANT_ROLE_NAME = "assistant"[::-1]

dataset_name = "alpaca"
context_length = 1024

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{TOKENIZER_DIR}/fineweb_bpe_200k")
tokenizer.add_special_tokens({ "additional_special_tokens": ["<im_start>", "<im_end>"] })


from transformers import GPT2LMHeadModel

# Load base model
model = GPT2LMHeadModel.from_pretrained(
    # f"{MODEL_DIR}/reverse-gpt2-0.35B-fineweb-10BT-ctx-1024/checkpoint-9000"
    f"{MODEL_DIR}/reverse-gpt2-0.35B-fineweb-10BT-ctx-1024-chat/checkpoint-1050"
)

print(f"Tokenizer vocab size: {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))


from datasets import Dataset

tokenized = {
    "train": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/tokenized_{context_length}_train"),
    "valid": Dataset.load_from_disk(f"{DATA_DIR}/{dataset_name}/tokenized_{context_length}_valid"),
}


import torch
import gc

torch.cuda.empty_cache()
gc.collect()

from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

# Set up training arguments
args = SFTConfig(
    output_dir=f"{MODEL_DIR}/{model_name}-chat-v2",
    run_name=f"{model_name}-chat",
    report_to="wandb",

    neftune_noise_alpha=10,
    per_device_train_batch_size=144,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    warmup_ratio=0.01,

    max_grad_norm=1.0,
    
    gradient_checkpointing=True,
)

import wandb

wandb.init(
    project="reverse-llm-alpaca",
    entity="womogenes-team",
    config=args.to_dict(),
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["valid"],
)

trainer.train()
