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
