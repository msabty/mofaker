import re
import requests

def llm_judge_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that uses a larger model on the Mac to judge the reasoning QUALITY
    of the 4090's generations.
    """
    responses = [completion[0]['content'] for completion in completions]
    
    # Extract just the <think> part
    thinking_traces = []
    for resp in responses:
        match = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
        thinking_traces.append(match.group(1).strip() if match else "")

    rewards = []
    for i, trace in enumerate(thinking_traces):
        if not trace:
            rewards.append(0.0)
            continue
            
        # The prompt for our "AI Professor" on the Mac
        judge_prompt = f"You are a logic professor. Rate the following reasoning trace for logical consistency and depth on a scale of 0 to 1. \n\nReasoning: {trace}\n\nRespond ONLY with a single float number between 0.0 and 1.0."
        
        try:
            # Call the Mac (m5) MLX server
            response = requests.post(
                "http://m5:1234/v1/chat/completions",
                json={
                    "model": "default_model",
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "temperature": 0.1 # Low temp for consistent grading
                },
                timeout=10
            )
            score_text = response.json()["choices"][0]["message"]["content"]
            # Extract float from judge's response
            score = float(re.findall(r"[-+]?\d*\.\d+|\d+", score_text)[0])
            rewards.append(min(max(score, 0.0), 1.0))
        except Exception as e:
            print(f"Judge Error: {e}")
            rewards.append(0.0)
            
    return rewards

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
