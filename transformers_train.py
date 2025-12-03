import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# 1. Load a SMALL dataset (good for laptops)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# For a quick run, use only a tiny subset of the training data
train_dataset = dataset["train"].select(range(2000))   # 2k examples
eval_dataset = dataset["validation"].select(range(200))  # 200 examples

# 2. Load tokenizer & model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 family has no pad token by default
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# 3. Tokenize function
block_size = 128  # keep small for laptop training

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=block_size,
        return_special_tokens_mask=False,
    )

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)
tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# Filter out empty examples (wikitext has many empty lines)
tokenized_train = tokenized_train.filter(lambda x: len(x["input_ids"]) > 0)
tokenized_eval = tokenized_eval.filter(lambda x: len(x["input_ids"]) > 0)

# 4. Data collator for CAUSAL LM (no MLM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./distilgpt2-wikitext2-m1",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,   # tiny batch to fit easily
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,   # effectively batch size 2
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=1,
    report_to="none",                # no wandb/tensorboard by default

    # Important for Mac:
    # - don't use fp16 (not supported on MPS)
    # - MPS will be auto-detected
    fp16=False,
    bf16=False,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Verify device being used
print(f"Training on device: {training_args.device}")

# 7. Train!
trainer.train()

# 8. Save final model & tokenizer
trainer.save_model("./distilgpt2-wikitext2-m1-final")
tokenizer.save_pretrained("./distilgpt2-wikitext2-m1-final")