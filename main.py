import modal
from dataclasses import dataclass

app = modal.App("transformer-speedrun")
volume = modal.Volume.from_name("transformer-speedrun-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .uv_pip_install("transformers[torch]")
    .uv_pip_install("datasets")
    .uv_pip_install("wandb")
    .uv_pip_install("pyyaml")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("cfg", remote_path="/root/cfg")
)


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=2*3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")],
)
def train(model_type: str = "gpt2", force_tokenize: bool = False, muon: bool = True):
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
    import yaml
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
    
    if os.path.exists(cache_path) and not force_tokenize:
        print(f"Loading tokenized dataset from volume: {cache_path}")
        tokenized_datasets = load_from_disk(cache_path)
    else:
        if force_tokenize:
            print("Force tokenize flag set. Re-tokenizing dataset...")
        print("Loading FineWeb (sample-10BT) dataset...")
        # Load 10% of the 10BT sample to get ~1B tokens
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")
        
        print("Subsetting to 5% of sample-10BT (~500M tokens)...")
        # Using 5% to approximate 500M tokens (5% of 10B)
        dataset = dataset.select(range(int(len(dataset) * 0.05)))
        
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
    config_params = {
        "vocab_size": tokenizer.vocab_size,
        "n_positions": 512,
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
    }

    # Check for YAML config in cfg directory
    config_from_yaml = False
    run_name = f"transformer-speedrun-{model_type}"
    # Check if model_type is a path to a yaml file or a key in cfg
    potential_yaml_paths = [
        os.path.join("/root/cfg", f"{model_type}.yaml"),
        os.path.join("/root/cfg", model_type if model_type.endswith(".yaml") else f"{model_type}.yaml"),
        model_type if model_type.endswith(".yaml") else None
    ]
    
    for yaml_path in potential_yaml_paths:
        if yaml_path and os.path.exists(yaml_path):
            print(f"Loading model configuration from {yaml_path}...")
            # Use the filename as the run name if we load from YAML
            run_name = os.path.basename(yaml_path).replace(".yaml", "")
            with open(yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f)
                if "name" in yaml_config:
                    run_name = yaml_config["name"]
                if "hyperparams" in yaml_config:
                    config_params.update(yaml_config["hyperparams"])
                else:
                    config_params.update(yaml_config)
            config_from_yaml = True
            break

    # Use the appropriate Config class
    if model_type == "gpt2_modded" or config_from_yaml:
        print(f"Using custom GPT2Modded architecture with params: {config_params}")
        config = GPT2ConfigModded(**config_params)
        model = GPT2Modded(config)
    else:
        print(f"Using standard HuggingFace GPT2LMHeadModel with params: {config_params}")
        # HF GPT2Config uses slightly different attribute names for some things, 
        # but the ones we use (n_embd, n_layer, n_head, n_positions) are standard.
        config = GPT2Config(**config_params)
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
        num_train_epochs=5,            # 1 epoch of 100M tokens is ~200k steps at batch 512, but we'll limit by max_steps if needed
        per_device_train_batch_size=8, # Increased for A100-80GB
        gradient_accumulation_steps=8,  # 64 * 8 = 512 effective batch size
        gradient_checkpointing=True,   # Huge memory saver
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=5e-4,            # This will be overridden by custom optimizer if we pass it
        weight_decay=0.01,
        bf16=True,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,              # Reduced warmup for shorter run
        report_to="wandb",
        run_name=run_name,
        eval_strategy="steps",
        eval_steps=100,
        max_steps=2000,                # Limit steps for the speedrun
    )

    # 7. Custom Optimizer Setup (Muon + AdamW)
    from src.optimizer import Muon, CombinedOptimizer
    
    def get_optimizers(model, learning_rate, weight_decay, use_muon=True):
        if not use_muon:
            print("Muon disabled. Using AdamW for all parameters.")
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Filter parameters for Muon (only 2D parameters in transformer layers)
        muon_params = []
        adamw_params = []
        
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            
            # Muon handles 2D parameters (weights of Linear layers)
            # but we usually exclude embeddings and the head
            if "transformer.h" in name and len(p.shape) == 2 and "ln" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        
        optimizer_muon = Muon(muon_params, lr=0.02, momentum=0.95)
        optimizer_adamw = torch.optim.AdamW(adamw_params, lr=learning_rate, weight_decay=weight_decay)
        
        return CombinedOptimizer([optimizer_muon, optimizer_adamw])

    optimizer = get_optimizers(model, training_args.learning_rate, training_args.weight_decay, use_muon=muon)

    # 8. Initialize Trainer
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
                        # HF Trainer multiplies loss by gradient_accumulation_steps in
                        # training_step but does not divide it back before logging, so
                        # we correct for that here. This callback is inserted at index 0
                        # so it runs before WandbCallback and the corrected value is
                        # what gets sent to wandb.
                        logs["loss"] = logs["loss"] / args.gradient_accumulation_steps
                        logs["train_perplexity"] = math.exp(logs["loss"])
                    if "eval_loss" in logs:
                        logs["eval_perplexity"] = math.exp(logs["eval_loss"])
                except (OverflowError, math.OverflowError):
                    pass

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        optimizers=(optimizer, None), # Trainer accepts (optimizer, scheduler)
        callbacks=[GenerationCallback(tokenizer)],
    )
    # Insert before WandbCallback so logs["loss"] is corrected before wandb reads it
    trainer.add_callback(PerplexityCallback())
    trainer.callback_handler.callbacks.insert(0, trainer.callback_handler.callbacks.pop())

    # 8. Train
    print("Starting pre-training...")
    trainer.train()

    # 9. Save the model
    trainer.save_model("./final_model")
    print("Training complete! Model saved to ./final_model")


@app.local_entrypoint()
def main(model: str = "gpt2", force_tokenize: bool = False, muon: bool = True):
    train.remote(model_type=model, force_tokenize=force_tokenize, muon=muon)
