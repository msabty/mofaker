# mofaker

A framework to fine-tune any instruct-tuned LLM to develop "thinking" capabilities (producing reasoning traces) using Hugging Face's TRL (Transformer Reinforcement Learning) and GRPO (Group Relative Policy Optimization).

## Overview

This framework uses GRPO to incentivize language models to output their reasoning steps inside `<think>...</think>` tags before providing the final answer inside `<answer>...</answer>` tags. This has been shown to improve reasoning capabilities in smaller models.

## Structure

- `src/prompts.py`: Defines the system instructions and data formatting.
- `src/rewards.py`: Contains GRPO reward functions (formatting verification, correctness validation).
- `src/train.py`: The main GRPO training loop.
- `src/inference.py`: Script to generate outputs from the base model + LoRA adapters.

## Installation

Using Docker (Recommended):
```bash
docker build -t mofaker .
docker run --gpus all -it -v $(pwd):/app mofaker
```

Locally:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

Run the training script. By default, it uses a small Qwen 2.5 instruct model and trains on the GSM8K math dataset.

```bash
python -m src.train \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_name "openai/gsm8k" \
    --max_steps 500 \
    --output_dir "./outputs"
```

### 2. Inference

Once training is complete, you can test the model's new thinking capabilities:

```bash
python -m src.inference \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --adapter_path "./outputs" \
    --prompt "A sequence starts with 5, and each subsequent term is 3 more than the previous term. What is the 10th term?"
```
# mofaker
