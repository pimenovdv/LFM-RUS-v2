import pytest
import os
from src.alignment.iterative_dpo import run_iterative_dpo_iteration
from datasets import Dataset

def test_run_iterative_dpo_iteration_dummy(mocker, tmp_path):
    mocker.patch('src.alignment.iterative_dpo.generate_responses', return_value=[["respA", "respB"]])
    mocker.patch('src.alignment.iterative_dpo.evaluate_with_llm_judge', return_value="A")
    mocker.patch('src.alignment.iterative_dpo.load_dataset', return_value=Dataset.from_dict({"prompt": ["p1"]}))

    cfg = {
        "dataset_path": "dummy_path",
        "output_dir": str(tmp_path / "iterative_dpo_out"),
        "openai_api_key": "dummy",
        "openai_base_url": "http://localhost"
    }

    class MockClient:
        def __init__(self, **kwargs): pass

    mocker.patch('src.alignment.iterative_dpo.OpenAI', MockClient)

    run_iterative_dpo_iteration(cfg, None, None, 0, dummy_data=False)
    assert os.path.exists(os.path.join(str(tmp_path / "iterative_dpo_out"), "iterative_dpo_dataset_iter_0.jsonl"))

def test_run_iterative_dpo_iteration_less_than_2(mocker, tmp_path):
    mocker.patch('src.alignment.iterative_dpo.generate_responses', return_value=[["respA"]])
    mocker.patch('src.alignment.iterative_dpo.evaluate_with_llm_judge', return_value="A")
    mocker.patch('src.alignment.iterative_dpo.load_dataset', return_value=Dataset.from_dict({"prompt": ["p1"]}))

    cfg = {
        "dataset_path": "dummy_path",
        "output_dir": str(tmp_path / "iterative_dpo_out"),
        "openai_api_key": "dummy"
    }

    class MockClient:
        def __init__(self, **kwargs): pass

    mocker.patch('src.alignment.iterative_dpo.OpenAI', MockClient)

    run_iterative_dpo_iteration(cfg, None, None, 0, dummy_data=False)
    assert os.path.exists(os.path.join(str(tmp_path / "iterative_dpo_out"), "iterative_dpo_dataset_iter_0.jsonl"))

def test_run_iterative_dpo_iteration_b(mocker, tmp_path):
    mocker.patch('src.alignment.iterative_dpo.generate_responses', return_value=[["respA", "respB"]])
    mocker.patch('src.alignment.iterative_dpo.evaluate_with_llm_judge', return_value="B")

    cfg = {
        "output_dir": str(tmp_path / "iterative_dpo_out"),
        "openai_api_key": "dummy"
    }

    class MockClient:
        def __init__(self, **kwargs): pass

    mocker.patch('src.alignment.iterative_dpo.OpenAI', MockClient)

    run_iterative_dpo_iteration(cfg, None, None, 0, dummy_data=True)
    assert os.path.exists(os.path.join(str(tmp_path / "iterative_dpo_out"), "iterative_dpo_dataset_iter_0.jsonl"))

def test_run_iterative_dpo_iteration_error(mocker, tmp_path):
    cfg = {"output_dir": str(tmp_path / "iterative_dpo_out"), "openai_api_key": "dummy"}
    class MockClient:
        def __init__(self, **kwargs): pass
    mocker.patch('src.alignment.iterative_dpo.OpenAI', MockClient)

    with pytest.raises(ValueError, match="dataset_path must be provided"):
        run_iterative_dpo_iteration(cfg, None, None, 0, dummy_data=False)
