SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>
"""

# Extract answer helper function (useful for basic parsing, e.g., GSM8K)
def extract_final_answer(text: str) -> str:
    """Helper to extract the final short answer from GSM8K format strings."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_gsm8k_dataset(dataset):
    """
    Format the GSM8K dataset into conversational prompt format 
    suitable for instruction tuning and GRPO training in `trl`.
    """
    def make_prompt(row):
        return [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": row["question"]}
        ]

    dataset = dataset.map(lambda x: {"prompt": make_prompt(x)})
    
    # Extract just the answer string from the GSM8K "answer" column for reward evaluation
    if "answer" in dataset.column_names:
        dataset = dataset.map(lambda x: {"solution": extract_final_answer(x["answer"])})
        
    return dataset
