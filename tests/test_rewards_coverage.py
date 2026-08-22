import pytest
from src.alignment.rewards.rewards import get_reward_function, ModelBasedReward

def test_model_based_reward_openai(mocker):
    config = {
        "api_type": "openai",
        "api_key": "dummy",
        "model_name": "gpt-3.5-turbo"
    }

    class MockMessage:
        def __init__(self, content):
            self.content = content
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)
    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    class MockCompletions:
        def create(self, **kwargs):
            import json
            return MockResponse(json.dumps({"score": 0.8}))

    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()

    class MockClient:
        def __init__(self, **kwargs):
            self.chat = MockChat()

    mocker.patch('openai.OpenAI', MockClient)

    reward_fn = get_reward_function("model_based", config)

    scores = reward_fn(["test completion"], ["test prompt"])
    assert len(scores) == 1
    assert scores[0] == 0.8

    # Test without prompt
    scores = reward_fn(["test completion"])
    assert len(scores) == 1
    assert scores[0] == 0.8

def test_model_based_reward_openai_error(mocker):
    config = {"api_type": "openai", "api_key": "dummy"}
    class MockClient:
        def __init__(self, **kwargs): pass
    mocker.patch('openai.OpenAI', MockClient)

    reward_fn = get_reward_function("model_based", config)
    reward_fn.client = mocker.MagicMock()
    reward_fn.client.chat.completions.create.side_effect = Exception("Mock error")

    scores = reward_fn(["test"])
    assert len(scores) == 1
    assert scores[0] == 0.0

def test_model_based_reward_transformers(mocker):
    config = {"api_type": "transformers", "model_name": "dummy"}

    def mock_pipeline(*args, **kwargs):
        return [{"generated_text": "0.7"}]
    mock_pipeline.tokenizer = mocker.MagicMock()
    mock_pipeline.tokenizer.eos_token_id = 1

    mocker.patch('transformers.pipeline', return_value=mock_pipeline)

    reward_fn = get_reward_function("model_based", config)
    scores = reward_fn(["test completion"], ["test prompt"])
    assert len(scores) == 1
    assert scores[0] == 0.7

def test_model_based_reward_transformers_no_match(mocker):
    config = {"api_type": "transformers", "model_name": "dummy"}
    def mock_pipeline(*args, **kwargs):
        return [{"generated_text": "no numbers here"}]
    mock_pipeline.tokenizer = mocker.MagicMock()
    mock_pipeline.tokenizer.eos_token_id = 1
    mocker.patch('transformers.pipeline', return_value=mock_pipeline)
    reward_fn = get_reward_function("model_based", config)
    scores = reward_fn(["test"])
    assert scores[0] == 0.0

def test_model_based_reward_transformers_error(mocker):
    config = {"api_type": "transformers", "model_name": "dummy"}
    mocker.patch('transformers.pipeline', side_effect=Exception("pipeline init error"))
    with pytest.raises(Exception):
        get_reward_function("model_based", config)

def test_model_based_reward_transformers_call_error(mocker):
    config = {"api_type": "transformers", "model_name": "dummy"}
    mock_pipeline = mocker.MagicMock(side_effect=Exception("call error"))
    mocker.patch('transformers.pipeline', return_value=mock_pipeline)
    reward_fn = get_reward_function("model_based", config)
    scores = reward_fn(["test"])
    assert scores[0] == 0.0

def test_model_based_reward_unknown():
    config = {"api_type": "unknown"}
    with pytest.raises(ValueError, match="Unknown api_type"):
        get_reward_function("model_based", config)

def test_get_reward_function_default_config(mocker):
    class MockClient:
        def __init__(self, **kwargs): pass
    mocker.patch('transformers.pipeline', return_value=mocker.MagicMock())
    fn = get_reward_function("model_based")
    assert isinstance(fn, ModelBasedReward)

def test_get_reward_function_unknown():
    with pytest.raises(ValueError, match="Unknown reward function: unknown"):
        get_reward_function("unknown")

def test_accuracy_reward():
    from src.alignment.rewards.rewards import accuracy_reward
    completions = ["This is a <solution> to the problem.", "No solution here."]
    rewards = accuracy_reward(completions)
    assert rewards == [1.0, 0.0]

def test_format_reward():
    from src.alignment.rewards.rewards import format_reward
    completions = ["Here is code:\n```python\nprint('hello')\n```", "No code block."]
    rewards = format_reward(completions)
    assert rewards == [1.0, 0.0]

def test_length_penalty_reward():
    from src.alignment.rewards.rewards import length_penalty_reward
    completions = [
        "a" * 1500, # > 1000
        "b" * 700,  # > 500
        "c" * 100   # <= 500
    ]
    rewards = length_penalty_reward(completions)
    assert rewards == [-0.5, 0.0, 0.5]

def test_diffusion_trajectory_reward():
    from src.alignment.rewards.rewards import diffusion_trajectory_reward
    completions = ["valid step", "   "]
    rewards = diffusion_trajectory_reward(completions)
    assert rewards == [0.5, 0.0]
