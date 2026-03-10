import argparse
import requests
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from .prompts import SYSTEM_PROMPT
from .rewards import format_reward_func, correctness_reward_func, llm_judge_reward_func

# Custom Client for MLX Server on Mac
class MLXClient:
    def __init__(self, model_id, base_url, timeout=120):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_healthy(self):
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except:
            return True

    def __call__(self, prompts, **kwargs):
        results = []
        for prompt in prompts:
            payload = {
                "model": "default_model",
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
    parser = argparse.ArgumentParser(description="Train a thinking LLM using GRPO with MLX Judge")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model to fine-tune")
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="Dataset name to use")
    parser.add_argument("--dataset_config", type=str, default="main", help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="./outputs_judge", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    print(f"Loading Base Model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="bfloat16",
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading and Formatting Dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")

    def make_conversation(row):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": row["question"]}
            ]
        }
        
    def extract_solution(row):
        ans = row["answer"]
        if "####" in ans:
            ans = ans.split("####")[1].strip()
        return {"solution": ans}

    dataset = dataset.map(make_conversation).map(extract_solution)
    
    print("Initializing GRPO Trainer with LLM-as-a-Judge")
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_completion_length=1024,
        num_generations=4,
        logging_steps=1,
        save_steps=100,
        temperature=0.7,
        report_to="none",
        use_cpu=False,
        bf16=True,
        use_vllm=False,
    )

    mlx_client = MLXClient(model_id=args.model_name, base_url="http://m5:1234/v1")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, correctness_reward_func, llm_judge_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    class MockVLLMGen:
        def __init__(self, client, tokenizer):
            self.client = client
            self.tokenizer = tokenizer
        def sync_weights(self): pass
        def generate(self, prompts, images, num_generations, profiler=None):
            decoded_prompts = [self.tokenizer.decode(p, skip_special_tokens=True) for p in prompts]
            mlx_results = self.client(decoded_prompts, temperature=0.8, max_new_tokens=1024)
            all_prompt_ids, all_completion_ids, all_logprobs = [], [], []
            for i, p_ids in enumerate(prompts):
                c_text = mlx_results[i][0]["generated_text"]
                c_ids = self.tokenizer.encode(c_text, add_special_tokens=False)
                all_prompt_ids.append(p_ids)
                all_completion_ids.append(c_ids)
                all_logprobs.append([[0.0] for _ in range(len(c_ids))])
            return all_prompt_ids, all_completion_ids, all_logprobs, None, {}

    trainer.llm = mlx_client
    trainer.use_vllm = True 
    trainer._last_loaded_step = -1 
    trainer.vllm_generation = MockVLLMGen(mlx_client, tokenizer)

    print("Starting Training Loop...")
    trainer.train()

    print(f"Saving Final Adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
