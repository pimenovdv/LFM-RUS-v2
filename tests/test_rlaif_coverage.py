import pytest
import os
import json
from datasets import Dataset
from src.alignment.rlaif import run_rlaif_pipeline, evaluate_with_llm_judge, generate_responses

def test_generate_responses(mocker):
    mocker.patch('src.alignment.rlaif.pipeline', return_value=lambda *args, **kwargs: [{"generated_text": "text1"}, {"generated_text": "text2"}])
    mock_model = mocker.MagicMock()
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token_id = 0
    responses = generate_responses(mock_model, mock_tokenizer, ["prompt1", "prompt2"], 2, 10)
    assert len(responses) == 2
    assert len(responses[0]) == 2
    assert responses[0][0] == "text1"

def test_evaluate_with_llm_judge_error(mocker):
    class MockClient:
        pass
    client = MockClient()
    client.chat = mocker.MagicMock()
    client.chat.completions.create.side_effect = Exception("Mock error")
    result = evaluate_with_llm_judge("prompt", "a", "b", "rules", client, "gpt-4")
    assert result == "A"

def test_run_rlaif_pipeline_with_data(mocker, tmp_path):
    mocker.patch('src.alignment.rlaif.generate_responses', return_value=[["respA", "respB"]])
    mocker.patch('src.alignment.rlaif.evaluate_with_llm_judge', return_value="A")
    mocker.patch('src.alignment.rlaif.load_dataset', return_value=Dataset.from_dict({"prompt": ["p1"]}))

    cfg = {
        "dataset_path": "dummy_path",
        "output_dir": str(tmp_path / "rlaif_out"),
        "openai_api_key": "dummy"
    }

    class MockClient:
        def __init__(self, **kwargs):
            pass

    mocker.patch('src.alignment.rlaif.OpenAI', MockClient)

    run_rlaif_pipeline(cfg, None, None, dummy_data=False)

    assert os.path.exists(os.path.join(str(tmp_path / "rlaif_out"), "rlaif_dataset.jsonl"))

def test_run_rlaif_pipeline_skip_less_than_2(mocker, tmp_path):
    mocker.patch('src.alignment.rlaif.generate_responses', return_value=[["respA"]])
    mocker.patch('src.alignment.rlaif.load_dataset', return_value=Dataset.from_dict({"prompt": ["p1"]}))

    cfg = {
        "dataset_path": "dummy_path",
        "output_dir": str(tmp_path / "rlaif_out"),
        "openai_api_key": "dummy",
        "openai_base_url": "http://localhost:8000"
    }

    class MockClient:
        def __init__(self, **kwargs):
            pass

    mocker.patch('src.alignment.rlaif.OpenAI', MockClient)

    run_rlaif_pipeline(cfg, None, None, dummy_data=False)

    path = os.path.join(str(tmp_path / "rlaif_out"), "rlaif_dataset.jsonl")
    assert os.path.exists(path)
    with open(path) as f:
        assert len(f.readlines()) == 0
