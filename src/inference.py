import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .prompts import SYSTEM_PROMPT

def main():
    parser = argparse.ArgumentParser(description="Test inference of the thinking LLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--adapter_path", type=str, default="./outputs", help="Path to trained LoRA adapter")
    parser.add_argument("--prompt", type=str, required=True, help="Question to ask the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    args = parser.parse_args()

    print(f"Loading base model ({args.model_name}) and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    
    # Try to load adapter if it exists
    try:
        print(f"Loading adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    except Exception as e:
        print(f"Could not load adapter: {e}. Falling back to base model.")
        model = base_model

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": args.prompt}
    ]

    print("\n--- Applying Chat Template ---")
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(input_text, return_tensors="pt").to(args.device)

    print("\n--- Generating Response ---")
    # Set max_new_tokens high enough to allow room for the `<think>` blocks
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    
    print("\n" + "="*50)
    print("MODEL OUTPUT:")
    print("="*50)
    print(response)

if __name__ == "__main__":
    main()
