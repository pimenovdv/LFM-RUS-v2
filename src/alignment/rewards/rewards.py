from typing import List

def accuracy_reward(completions: List[str]) -> List[float]:
    rewards = []
    for text in completions:
        if "<solution>" in text:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions: List[str]) -> List[float]:
    """Checks if the response follows a specific format (e.g. starts with a particular tag or contains code blocks)"""
    rewards = []
    for text in completions:
        if "```" in text:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def length_penalty_reward(completions: List[str]) -> List[float]:
    """Penalize extremely long responses to prevent reward hacking"""
    rewards = []
    for text in completions:
        length = len(text)
        if length > 1000:
            rewards.append(-0.5)
        elif length > 500:
            rewards.append(0.0)
        else:
            rewards.append(0.5)
    return rewards

def get_reward_function(name: str):
    if name == "accuracy":
        return accuracy_reward
    elif name == "format":
        return format_reward
    elif name == "length_penalty":
        return length_penalty_reward
    else:
        raise ValueError(f"Unknown reward function: {name}")
