import re

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that provides a positive reward if the completion
    exactly contains the <think> ... </think> <answer> ... </answer> format.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    
    # completions are a list of string generations from the model for the prompt batch
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for response in responses:
        # Give a small reward for having the tags, full format reward if in strict order
        reward = 0.0
        if "<think>" in response and "</think>" in response:
            reward += 0.2
        if "<answer>" in response and "</answer>" in response:
            reward += 0.2
            
        # check strict regex match (dots match newlines with re.DOTALL)
        if re.match(pattern, response.strip(), re.DOTALL):
            reward = 1.0
            
        rewards.append(reward)
        
    return rewards


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function that checks if the content inside <answer>...</answer>
    matches the expected ground truth solution.
    """
    
    responses = [completion[0]['content'] for completion in completions]
    solutions = answer  # 'answer' corresponds to the 'solution' column we created in prompts.py
    
    rewards = []
    for response, solution in zip(responses, solutions):
        # Extract the content inside the <answer> tag
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        
        if match:
            extracted_answer = match.group(1).strip()
            # Basic string matching - can be improved for numeric equivalence
            if extracted_answer == str(solution).strip():
                rewards.append(2.0) # High reward for correct answer
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0) # No answer tag, no correctness reward
            
    return rewards
