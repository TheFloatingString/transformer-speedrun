import modal
from dataclasses import dataclass

app = modal.App("transformer-speedrun")
volume = modal.Volume.from_name("transformer-speedrun-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .uv_pip_install("transformers[torch]")
    .uv_pip_install("datasets")
    .uv_pip_install("wandb")
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=2*3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")],
)
def train(model_type: str = "gpt2"):
    from datasets import load_dataset, load_from_disk
    from transformers import (
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
    import os
    import wandb
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import math

    # Import custom model from src
    from src.model import GPT2Modded, GPT2Config as GPT2ConfigModded

    # 1. Setup tokenizer first to check cache
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load or tokenize the dataset
    cache_path = "/data/tokenized_fineweb_1b"
    
    if os.path.exists(cache_path):
        print(f"Loading tokenized dataset from volume: {cache_path}")
        tokenized_datasets = load_from_disk(cache_path)
    else:
        print("Loading FineWeb (sample-10BT) dataset...")
        # Load 10% of the 10BT sample to get ~1B tokens
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")
        
        print("Subsetting to 10% of sample-10BT (~1B tokens)...")
        # Using 10% to approximate 1B tokens (10% of 10B)
        dataset = dataset.select(range(int(len(dataset) * 0.001)))
        
        # FineWeb only has a 'train' split, so we create our own validation set
        print("Creating validation split...")
        split_ds = dataset.train_test_split(test_size=0.005, seed=42)
        from datasets import DatasetDict
        ds = DatasetDict({
            "train": split_ds["train"],
            "validation": split_ds["test"]
        })

        # 3. Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512)

        print("Tokenizing dataset...")
        tokenized_datasets = ds.map(
            tokenize_function, batched=True, num_proc=8, remove_columns=["text"]
        )

        # Filter out empty examples
        tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) > 0)
        
        print(f"Saving tokenized dataset to volume: {cache_path}")
        tokenized_datasets.save_to_disk(cache_path)
        volume.commit() # Ensure data is written to the volume

    # 4. Initialize the model
    # Configuration matches the requested parameters
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_embd=1024,
        n_layer=24,
        n_head=16,
    )
    
    if model_type == "gpt2_modded":
        print("Using custom GPT2Modded architecture...")
        model = GPT2Modded(config)
    else:
        print("Using standard HuggingFace GPT2LMHeadModel...")
        model = GPT2LMHeadModel(config)

    # In multi-GPU setups, the Trainer handles device placement.
    # We just log the availability here.
    device_count = torch.cuda.device_count()
    print(f"Detected {device_count} GPUs.")
    
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size:,} parameters")
    # Store model size in config so it's logged to wandb by the Trainer
    model.config.model_params = model_size

    # 5. Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4, # 16 * 4 = 64 effective batch size
        gradient_checkpointing=True,   # Huge memory saver
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        bf16=True,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        report_to="wandb",
        run_name="transformer-speedrun-pretrain-24L-16H-1GPU-Optimized",
        eval_strategy="steps",
        eval_steps=100,
    )

    # 7. Initialize Trainer
    class GenerationCallback(TrainerCallback):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % 100 == 0:
                print(f"\n--- Step {state.global_step} ---")
                print("Generating sample sentences...")
                model = kwargs["model"]

                model.eval()
                prompts = [
                    "The quick brown fox",
                    "Artificial intelligence is",
                    "The history of the world",
                ]

                for prompt in prompts:
                    # Get device from model parameters (works for both HF models and custom nn.Modules)
                    device = next(model.parameters()).device
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=50,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            temperature=0.7,
                        )
                    generated_text = self.tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )
                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {generated_text}")
                model.train()  # Switch back to training mode
                print("-" * 30)

    class PerplexityCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                try:
                    if "loss" in logs:
                        perplexity = math.exp(logs["loss"])
                        logs["train_perplexity"] = perplexity
                        if wandb.run is not None:
                            wandb.log({"train_perplexity": perplexity}, step=state.global_step)
                    if "eval_loss" in logs:
                        eval_perplexity = math.exp(logs["eval_loss"])
                        logs["eval_perplexity"] = eval_perplexity
                        if wandb.run is not None:
                            wandb.log({"eval_perplexity": eval_perplexity}, step=state.global_step)
                except (OverflowError, math.OverflowError):
                    pass

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        callbacks=[GenerationCallback(tokenizer), PerplexityCallback()],
    )

    # 8. Train
    print("Starting pre-training...")
    trainer.train()

    # 9. Save the model
    trainer.save_model("./final_model")
    print("Training complete! Model saved to ./final_model")


@app.local_entrypoint()
def main(model: str = "gpt2"):
    train.remote(model_type=model)
