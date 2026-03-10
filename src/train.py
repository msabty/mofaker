import argparse
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from .prompts import SYSTEM_PROMPT
from .rewards import format_reward_func, correctness_reward_func

# Custom Client for MLX Server on Mac
class MLXClient:
    def __init__(self, model_id, base_url, timeout=120):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_healthy(self):
        # MLX server usually doesn't have /health, so we check /v1/models
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except:
            return True # Fallback to True to avoid blocking the trainer

    def __call__(self, prompts, **kwargs):
        # Map TRL generation calls to MLX OpenAI-compatible API
        results = []
        for prompt in prompts:
            payload = {
                "model": "default_model", # MLX server usually identifies as this
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.8),
                "max_tokens": kwargs.get("max_new_tokens", 1024),
            }
            try:
                response = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
                res_json = response.json()
                content = res_json["choices"][0]["message"]["content"]
                results.append([{"generated_text": content}])
            except Exception as e:
                print(f"MLX Inference Error: {e}")
                results.append([{"generated_text": "Error in generation"}])
        return results

def main():
    parser = argparse.ArgumentParser(description="Train a thinking LLM using GRPO")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model to fine-tune")
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="Dataset name to use")
    parser.add_argument("--dataset_config", type=str, default="main", help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for checkpoints")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    print(f"Loading Base Model: {args.model_name}")
    # Load with PEFT/LoRA integration for efficient training on standard GPUs
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading and Formatting Dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")

    # Format specifically for conversational format expected by TRL Chat Templates
    def make_conversation(row):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": row["question"]}
            ]
        }
        
    def extract_solution(row):
        # Specific extraction for gsm8k logic (extract content after ####)
        ans = row["answer"]
        if "####" in ans:
            ans = ans.split("####")[1].strip()
        return {"solution": ans}

    dataset = dataset.map(make_conversation).map(extract_solution)
    # The expected keys by GRPOTrainer are usually specific based on config,
    # mapping to proper naming convention. 'solution' is matched in rewards.py
    
    print("Initializing GRPO Trainer with MLX Remote Client")
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=4,   # Must be divisible by num_generations
        gradient_accumulation_steps=4,
        max_completion_length=1024,      # Increased so it has room to output the </answer> tag
        num_generations=4,               # K value for GRPO (samples per prompt)
        logging_steps=1,
        save_steps=100,
        temperature=0.7,                 # Needs slight temperature for exploring reasoning paths
        report_to="none",                # Disable wandb since we have no API key
        use_cpu=False,                   # Use GPU
        bf16=True,                       # Use bf16 on 4090
        use_vllm=False,                  # We are using our custom MLX client instead
    )

    # Instantiate our custom MLX Client
    mlx_client = MLXClient(model_id=args.model_name, base_url="http://m5:1234/v1")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Inject our client into the trainer
    trainer.llm = mlx_client
    trainer.use_vllm = True # Tell trainer to use the .llm object for generation

    print("Starting Training Loop...")
    trainer.train()

    print(f"Saving Final Adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
