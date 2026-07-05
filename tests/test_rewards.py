import pytest
from src.alignment.rewards.rewards import get_reward_function, ModelBasedReward, accuracy_reward, format_reward, length_penalty_reward

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




def test_model_based_reward_openai(mocker):
    config = {
        "api_type": "openai",
        "model_name": "gpt-4",
        "api_key": "fake_key"
    }

    # Mock the imports used in init
    mock_openai = mocker.patch("openai.OpenAI")

    reward_fn = get_reward_function("model_based", config)

    mock_client = mock_openai.return_value

    # Mock the API response
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.content = '{"score": 0.8}'
    mock_client.chat.completions.create.return_value = mock_response

    completions = ["This is a good answer"]
    prompts = ["What is this?"]

    scores = reward_fn(completions, prompts)
    assert scores == [0.8]

    mock_client.chat.completions.create.assert_called_once()


def test_model_based_reward_transformers(mocker):
    config = {
        "api_type": "transformers",
        "model_name": "dummy-model"
    }

    mock_pipeline = mocker.patch("transformers.pipeline")
    mock_pipe_instance = mock_pipeline.return_value

    reward_fn = get_reward_function("model_based", config)

    mock_pipe_instance.tokenizer.eos_token_id = 0
    mock_pipe_instance.return_value = [{"generated_text": " \"score\": 0.9 }"}]

    completions = ["Another answer"]
    prompts = ["What is that?"]

    scores = reward_fn(completions, prompts)
    assert scores == [0.9]


def test_diffusion_trajectory_reward():
    from src.alignment.rewards.rewards import get_reward_function
    func = get_reward_function("diffusion_trajectory")
    completions = ["valid text", "   ", "another valid"]
    rewards = func(completions)
    assert rewards == [0.5, 0.0, 0.5]

def test_length_penalty_reward():
    from src.alignment.rewards.rewards import length_penalty_reward
    completions = ["a"*1001, "a"*501, "a"*100]
    rewards = length_penalty_reward(completions)
    assert rewards == [-0.5, 0.0, 0.5]

def test_format_reward():
    from src.alignment.rewards.rewards import format_reward
    completions = ["```python pass ```", "no code"]
    rewards = format_reward(completions)
    assert rewards == [1.0, 0.0]

def test_accuracy_reward_dummy():
    from src.alignment.rewards.rewards import accuracy_reward
    completions = ["this has a <solution>", "this does not"]
    rewards = accuracy_reward(completions)
    assert rewards == [1.0, 0.0]

def test_model_based_reward_transformers(mocker):
    from src.alignment.rewards.rewards import get_reward_function

    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = [{"generated_text": "Score: 0.8"}]
    mock_pipeline.tokenizer.eos_token_id = 1
    mocker.patch("transformers.pipeline", return_value=mock_pipeline)

    config = {
        "api_type": "transformers",
        "model_name": "dummy_model"
    }
    func = get_reward_function("model_based", config)

    completions = ["comp1", "comp2"]
    prompts = ["prompt1", "prompt2"]
    rewards = func(completions, prompts)
    assert rewards == [0.8, 0.8]

def test_model_based_reward_transformers_no_prompts(mocker):
    from src.alignment.rewards.rewards import get_reward_function

    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = [{"generated_text": "Score: 0.6"}]
    mock_pipeline.tokenizer.eos_token_id = 1
    mocker.patch("transformers.pipeline", return_value=mock_pipeline)

    config = {
        "api_type": "transformers",
        "model_name": "dummy_model"
    }
    func = get_reward_function("model_based", config)

    completions = ["comp1"]
    rewards = func(completions)
    assert rewards == [0.6]

def test_model_based_reward_transformers_exception(mocker):
    from src.alignment.rewards.rewards import get_reward_function

    mock_pipeline = mocker.MagicMock()
    mock_pipeline.side_effect = Exception("Pipeline error")
    mocker.patch("transformers.pipeline", return_value=mock_pipeline)

    config = {
        "api_type": "transformers",
        "model_name": "dummy_model"
    }
    func = get_reward_function("model_based", config)

    completions = ["comp1"]
    rewards = func(completions)
    assert rewards == [0.0]

def test_model_based_reward_transformers_invalid_parsing(mocker):
    from src.alignment.rewards.rewards import get_reward_function

    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = [{"generated_text": "Score: no numbers"}]
    mock_pipeline.tokenizer.eos_token_id = 1
    mocker.patch("transformers.pipeline", return_value=mock_pipeline)

    config = {
        "api_type": "transformers",
        "model_name": "dummy_model"
    }
    func = get_reward_function("model_based", config)

    completions = ["comp1"]
    rewards = func(completions)
    assert rewards == [0.0]
