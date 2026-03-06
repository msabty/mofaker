import re
from datasets import load_dataset, concatenate_datasets

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>
"""

def extract_gsm8k_answer(text: str) -> str:
    """Helper to extract the final short answer from GSM8K format strings."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_numina_answer(text: str) -> str:
    """Helper to extract the final short answer from NuminaMath \\boxed{} format."""
    # Matches \boxed{...} allowing for nested braces
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None

def get_training_dataset(dataset_name="AI-MO/NuminaMath-CoT", split="train"):
    """
    Loads and formats the specified dataset into the standard
    conversational prompt format and extracts the ground truth solution.
    """
    if dataset_name == "openai/gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        
        def process_gsm8k(row):
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": row["question"]}
            ]
            ans = row["answer"]
            solution = extract_gsm8k_answer(ans) if "####" in ans else ans
            return {"prompt": prompt, "solution": solution}
            
        return dataset.map(process_gsm8k)

    elif dataset_name == "AI-MO/NuminaMath-CoT":
        # Load massive NuminaMath dataset (860k+ examples)
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split=split)
        
        def process_numina(row):
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": row["problem"]}
            ]
            solution_text = row["solution"]
            solution = extract_numina_answer(solution_text)
            # If no boxed answer, fallback to entire solution string
            if not solution:
                solution = solution_text
                
            return {"prompt": prompt, "solution": solution}
            
        return dataset.map(process_numina)
        
    else:
        raise ValueError(f"Unknown dataset {dataset_name}. Please implement parsing logic.")
