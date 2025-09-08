from transformers import PreTrainedTokenizerFast
from datasets import Dataset, load_dataset
from tqdm import tqdm
import os

DATA_DIR = "/home/wyf/orcd/pool/reverse-llm/data"
TOKENIZER_DIR = "/home/wyf/orcd/pool/reverse-llm/tokenizers"

USER_ROLE_NAME = "user"[::-1]
ASSISTANT_ROLE_NAME = "assistant"[::-1]

dataset_name = "ultrachat"
context_length = 1024

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{TOKENIZER_DIR}/fineweb_bpe_200k")
tokenizer.add_special_tokens({"additional_special_tokens": ["<im_start>", "<im_end>"]})

tokenizer.chat_template = """{% for message in messages -%}
<im_start>{{ message['role'] }}
{{ message['content'] }}<im_end>
{%- endfor -%}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}
<im_start>assistant
{%- endif %}"""

# Load dataset
raw_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=["train_sft", "test_sft"])

def process_ultrachat_data(ds_split: Dataset):
    all_convos = []
    
    for ex in tqdm(ds_split, desc="Processing conversations"):
        messages = ex["messages"]
        
        # Filter: must start with user and end with assistant
        if not (len(messages) >= 2 and 
                messages[0]["role"] == "user" and 
                messages[-1]["role"] == "assistant"):
            continue
            
        # Filter: no user query longer than 500 characters
        if any(msg["role"] == "user" and len(msg["content"]) > 500 for msg in messages):
            continue
        
        # Reverse content and role names, pre-tokenize each message
        tokenized_messages = []
        for msg in messages:
            role = USER_ROLE_NAME if msg["role"] == "user" else ASSISTANT_ROLE_NAME
            content = msg["content"].strip()[::-1]
            
            # Pre-tokenize the formatted message
            formatted_msg = f"<im_start>{role}\n{content}<im_end>"
            tokens = tokenizer.encode(formatted_msg, add_special_tokens=False)
            
            tokenized_messages.append({
                "role": role,
                "content": content,
                "tokens": tokens,
                "token_count": len(tokens)
            })
        
        # Create sliding windows
        windows = create_sliding_windows(tokenized_messages, context_length)
        all_convos.extend(windows)
    
    return all_convos

def create_sliding_windows(tokenized_messages, max_len):
    windows = []
    start_idx = 0
    
    while start_idx < len(tokenized_messages):
        window = []
        current_tokens = 0
        i = start_idx
        
        # Build window ensuring it starts with user and ends with assistant
        while i < len(tokenized_messages):
            msg = tokenized_messages[i]
            
            # Check if adding this message exceeds limit
            if current_tokens + msg["token_count"] > max_len:
                # If window is empty or doesn't end with assistant, try to find valid ending
                if not window:
                    i += 1
                    start_idx = i
                    break
                elif window[-1]["role"] != ASSISTANT_ROLE_NAME:
                    # Remove messages from end until we have assistant response
                    while window and window[-1]["role"] != ASSISTANT_ROLE_NAME:
                        removed = window.pop()
                        current_tokens -= removed["token_count"]
                break
            
            window.append(msg)
            current_tokens += msg["token_count"]
            i += 1
        
        # Only add windows that start with user and end with assistant
        if (window and len(window) >= 2 and 
            window[0]["role"] == USER_ROLE_NAME and 
            window[-1]["role"] == ASSISTANT_ROLE_NAME):
            
            formatted_window = [{"role": m["role"], "content": m["content"]} for m in window]
            windows.append(formatted_window)
            
            # Start next window from halfway through current window
            if len(window) >= 4:
                start_idx = start_idx + len(window) // 2
            else:
                start_idx = i
        else:
            start_idx = max(start_idx + 1, i)
        
        # Prevent infinite loops
        if start_idx >= len(tokenized_messages):
            break
    
    return windows

def formatting_func_batch(examples):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    im_start_token_id = 52000
    im_end_token_id = 52001
    
    for conversation in examples["conversations"]:
        prompt_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        
        tokenized_inputs = tokenizer(
            prompt_text,
            truncation=True,
            max_length=context_length,
            return_attention_mask=True,
            padding="max_length",
        )
        input_ids = tokenized_inputs["input_ids"]
        labels = [-100] * len(input_ids)
        
        # Find and unmask all assistant responses
        current_token_idx = 0
        for turn in conversation:
            role = turn["role"]
            
            try:
                start_of_turn_bos_idx = input_ids.index(im_start_token_id, current_token_idx)
            except ValueError:
                break
            
            try:
                end_of_turn_eos_idx = input_ids.index(im_end_token_id, start_of_turn_bos_idx + 1)
            except ValueError:
                break
            
            if role == ASSISTANT_ROLE_NAME:
                role_and_newline_text = f"{role}\n"
                role_and_newline_tokens = tokenizer.encode(role_and_newline_text, add_special_tokens=False)
                start_of_content_idx = start_of_turn_bos_idx + 1 + len(role_and_newline_tokens)
                
                # Unmask assistant response including <im_end> token
                for k_label in range(start_of_content_idx, end_of_turn_eos_idx + 1):
                    if 0 <= k_label < len(labels):
                        labels[k_label] = input_ids[k_label]
            
            current_token_idx = end_of_turn_eos_idx + 1
        
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(tokenized_inputs["attention_mask"])
        batch_labels.append(labels)
    
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }

# Process datasets
print("Processing train split...")
train_convos = process_ultrachat_data(raw_dataset[0])
print("Processing valid split...")
valid_convos = process_ultrachat_data(raw_dataset[1])

print(f"Train conversations: {len(train_convos)}")
print(f"Valid conversations: {len(valid_convos)}")

# Create datasets and tokenize in batches
print("Creating and tokenizing datasets...")
train_tokens = 0
valid_tokens = 0

for split, convos in [("train", train_convos), ("valid", valid_convos)]:
    dataset = Dataset.from_dict({"conversations": convos})
    
    # Process in batches for efficiency
    tokenized_split = dataset.map(
        formatting_func_batch,
        batched=True,
        batch_size=1000,
        remove_columns=["conversations"],
        desc=f"Tokenizing {split}"
    )
    
    # Count total tokens
    total_tokens = sum(sum(1 for token in seq if token != -100) for seq in tokenized_split["labels"])
    if split == "train":
        train_tokens = total_tokens
    else:
        valid_tokens = total_tokens
    
    save_path = f"{DATA_DIR}/{dataset_name}/tokenized_{context_length}_{split}"
    tokenized_split.save_to_disk(save_path)
    print(f"Saved {split} to: {save_path}")

print(f"Total train tokens: {train_tokens:,}")
print(f"Total valid tokens: {valid_tokens:,}")
print("Tokenization complete!")
