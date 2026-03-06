import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from .prompts import get_training_dataset
from .rewards import format_reward_func, correctness_reward_func

def main():
    parser = argparse.ArgumentParser(description="Train a thinking LLM using GRPO")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model to fine-tune")
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="Dataset name to use")
    parser.add_argument("--dataset_config", type=str, default="main", help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for checkpoints")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
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
    dataset = get_training_dataset(args.dataset_name, split="train")

    
    print("Initializing GRPO Trainer")
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=8,   # Scaled up for 2x H100 80GB
        gradient_accumulation_steps=2,
        max_completion_length=2048,      # Greatly increased room for long CoT reasoning
        num_generations=16,              # Powerful K value for dense GRPO exploration
        logging_steps=10,
        save_steps=100,
        temperature=0.7,                 # Needs slight temperature for exploring reasoning paths
        report_to="wandb",               # Assume WandB is configured for a large cluster run
        use_vllm=True,                   # Accelerate K-generations during GRPO rollouts
        vllm_gpu_memory_utilization=0.5, # Reserve memory for the training model
        vllm_device="cuda:1",            # Offload generation to second GPU if available

    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting Training Loop...")
    trainer.train()

    print(f"Saving Final Adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
