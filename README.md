# Transformer Speedrun

Just a GH repository that I'm working on for fun to better understand model pre-training.  

```bash
uv sync
uv run modal run --detach .\main.py --model gpt2_modded

# Use a YAML config from the cfg directory
uv run modal run --detach .\main.py --model openai/gpt_2_small

# Force re-tokenization (useful when changing dataset size)
uv run modal run --detach .\main.py --model openai/gpt_2_small --force-tokenize

# Disable Muon optimizer (falls back to AdamW)
uv run modal run --detach .\main.py --no-muon

```