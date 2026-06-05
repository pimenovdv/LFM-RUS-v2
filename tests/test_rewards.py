import pytest
from src.alignment.rewards.rewards import get_reward_function, accuracy_reward, format_reward, length_penalty_reward

def test_accuracy_reward():
    completions = ["This has a <solution>", "This does not have it"]
    scores = accuracy_reward(completions)
    assert scores == [1.0, 0.0]

def test_format_reward():
    completions = ["Here is code: ```python\nprint('hello')\n```", "No code here"]
    scores = format_reward(completions)
    assert scores == [1.0, 0.0]

def test_length_penalty_reward():
    long_text = "a" * 1500
    med_text = "a" * 600
    short_text = "a" * 100
    completions = [long_text, med_text, short_text]
    scores = length_penalty_reward(completions)
    assert scores == [-0.5, 0.0, 0.5]

def test_get_reward_function():
    assert get_reward_function("accuracy") == accuracy_reward
    assert get_reward_function("format") == format_reward
    assert get_reward_function("length_penalty") == length_penalty_reward

    with pytest.raises(ValueError):
        get_reward_function("unknown_function")
