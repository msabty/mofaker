import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from .prompts import SYSTEM_PROMPT

def run_inference(checkpoint_path, base_model_name, prompt_text, max_new_tokens=1024):
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Load adapter if checkpoint is provided
    if checkpoint_path and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        print(f"Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    elif checkpoint_path:
        print(f"Warning: No adapter found at {checkpoint_path}, using base model.")
    else:
        print("Using base model (no adapter).")

    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": prompt_text}
    ]
    
    # Use chat template if available
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\n--- Generating Response ---\n")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Remove input ids from generation
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on the fine-tuned thinking model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint/adapter directory (optional)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model name")
    parser.add_argument("--prompt", type=str, default="If I have 3 apples and I buy 5 more, how many do I have?", help="User prompt")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens to generate")
    
    args = parser.parse_args()
    
    # Expand user path if needed
    checkpoint_path = os.path.expanduser(args.checkpoint) if args.checkpoint else None
    
    response = run_inference(checkpoint_path, args.base_model, args.prompt, args.max_tokens)
    print(response)
