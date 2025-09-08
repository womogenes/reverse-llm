
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

base_model_name = f"reverse-gpt2-0.35B-fineweb-10BT-ctx-1024"

USER_ROLE_NAME = "user"[::-1]
ASSISTANT_ROLE_NAME = "assistant"[::-1]

context_length = 1024

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{TOKENIZER_DIR}/fineweb_bpe_200k")
tokenizer.add_special_tokens({ "additional_special_tokens": ["<im_start>", "<im_end>"] })


from transformers import GPT2LMHeadModel, EarlyStoppingCallback

# Load base model
model = GPT2LMHeadModel.from_pretrained(
    f"{MODEL_DIR}/reverse-gpt2-0.35B-fineweb-10BT-ctx-1024/checkpoint-9000"
    # f"{MODEL_DIR}/reverse-gpt2-0.35B-fineweb-10BT-ctx-1024-chat-v2/checkpoint-105"
)

print(f"Tokenizer vocab size: {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))


from datasets import Dataset, concatenate_datasets

print(f"Loading datasets...")
tokenized = {
    "train": concatenate_datasets([
        Dataset.load_from_disk(f"{DATA_DIR}/alpaca/tokenized_{context_length}_train"),
        Dataset.load_from_disk(f"{DATA_DIR}/databricks-dolly/tokenized_{context_length}_train"),
        Dataset.load_from_disk(f"{DATA_DIR}/ultrachat/tokenized_{context_length}_train")
    ]).shuffle(seed=0),
    "valid": concatenate_datasets([
        Dataset.load_from_disk(f"{DATA_DIR}/alpaca/tokenized_{context_length}_valid"),
        Dataset.load_from_disk(f"{DATA_DIR}/databricks-dolly/tokenized_{context_length}_valid"),
        Dataset.load_from_disk(f"{DATA_DIR}/ultrachat/tokenized_{context_length}_valid")
    ]).shuffle(seed=0),
}
print(tokenized)

import torch
import gc

torch.cuda.empty_cache()
gc.collect()

from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

# Set up training arguments
args = SFTConfig(
    output_dir=f"{MODEL_DIR}/{base_model_name}-chat-v15",
    report_to="wandb",

    neftune_noise_alpha=15,
    per_device_train_batch_size=144,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_steps=1,

    weight_decay=0.01,

    learning_rate=1.5e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    # lr_scheduler_type="constant_with_warmup",
    num_train_epochs=20,

    eval_strategy="steps",
    eval_steps=10,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_steps=50,

    max_grad_norm=1.0,
    
    gradient_checkpointing=True,
)

import wandb

wandb.init(
    project="reverse-llm-alpaca",
    entity="womogenes-team",
    config=args.to_dict(),
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["valid"],
    callbacks=[early_stopping_callback],
)

trainer.train(resume_from_checkpoint=True)
