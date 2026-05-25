import modal

app = modal.App("transformer-speedrun")
image = modal.Image.debian_slim().uv_pip_install("transformers[torch]").uv_pip_install("datasets").uv_pip_install("wandb")

@app.function(gpu="a10", image=image, timeout=3600, secrets=[modal.Secret.from_name("wandb-api-key")])
def train():
    from datasets import load_dataset
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback
    import os
    import wandb
    import torch

    # 1. Load the dataset
    print("Loading wikitext-103 dataset...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

    # Only run on 10% of the dataset for speed
    print("Subsetting to 10% of dataset...")
    ds["train"] = ds["train"].select(range(int(len(ds["train"]) * 0.1)))
    ds["validation"] = ds["validation"].select(range(int(len(ds["validation"]) * 0.1)))

    # 2. Setup tokenizer
    # Using GPT-2 tokenizer as a base
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    # Filter out empty examples
    tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) > 0)

    # 4. Initialize a small model
    # Very small config for demonstration
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4,
    )
    model = GPT2LMHeadModel(config)

    # 5. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=25,
        learning_rate=5e-4,
        weight_decay=0.01,
        fp16=True, # Use mixed precision
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        report_to="wandb",
        run_name="transformer-speedrun-pretrain-1pct",
    )

    # 7. Initialize Trainer
    class GenerationCallback(TrainerCallback):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def on_epoch_end(self, args, state, control, **kwargs):
            print(f"\n--- End of Epoch {state.epoch} ---")
            print("Generating sample sentences...")
            model = kwargs["model"]
            
            model.eval()
            prompts = [
                "The quick brown fox",
                "Artificial intelligence is",
                "The history of the world",
            ]
            
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
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
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\nPrompt: {prompt}")
                print(f"Generated: {generated_text}")
            model.train() # Switch back to training mode
            print("-" * 30)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        callbacks=[GenerationCallback(tokenizer)],
    )

    # 8. Train
    print("Starting pre-training...")
    trainer.train()

    # 9. Save the model
    trainer.save_model("./final_model")
    print("Training complete! Model saved to ./final_model")





