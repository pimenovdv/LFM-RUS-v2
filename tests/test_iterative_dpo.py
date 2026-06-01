import pytest
from unittest.mock import MagicMock
from src.alignment.iterative_dpo import run_iterative_dpo_iteration
from src.alignment.pipeline import run_alignment_pipeline
import os

def test_run_iterative_dpo_iteration(mocker):
    # Mock generate_responses and evaluate_with_llm_judge
    mock_generate = mocker.patch("src.alignment.iterative_dpo.generate_responses", return_value=[["Resp A", "Resp B"]])
    mock_evaluate = mocker.patch("src.alignment.iterative_dpo.evaluate_with_llm_judge", return_value="A")
    mocker.patch("src.alignment.iterative_dpo.OpenAI")

    cfg = {
        "output_dir": "/tmp/test_iterative_dpo",
    }

    # Run iteration
    output_path = run_iterative_dpo_iteration(cfg, model=MagicMock(), tokenizer=MagicMock(), iteration=0, dummy_data=True)

    assert mock_generate.call_count == 1
    assert mock_evaluate.call_count == 1
    assert "iterative_dpo_dataset_iter_0.jsonl" in output_path

    # Read output
    with open(output_path, "r") as f:
        data = f.readlines()
        assert len(data) == 1
        assert "Resp A" in data[0]

def test_run_alignment_pipeline_iterative_dpo(mocker):
    # Mock iteration and DPOTrainer
    mock_run_iteration = mocker.patch("src.alignment.pipeline.run_iterative_dpo_iteration", return_value="/tmp/test_dummy.jsonl")

    # Mock load_dataset to return a dummy dataset
    from datasets import Dataset
    dummy_dataset = Dataset.from_dict({"prompt": ["p"], "chosen": ["c"], "rejected": ["r"]})
    mocker.patch("src.alignment.pipeline.load_dataset", return_value=dummy_dataset)

    mock_trainer_init = mocker.patch("src.alignment.pipeline.DPOTrainer.__init__", return_value=None)
    mock_trainer_train = mocker.patch("src.alignment.pipeline.DPOTrainer.train")
    mocker.patch("src.alignment.pipeline.DPOTrainer.save_model")

    # Mock tokenizer and model
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    cfg = {
        "method": "iterative_dpo",
        "num_iterations": 2,
        "output_dir": "/tmp/test_iterative_dpo_pipeline",
        "push_to_hub": False
    }

    run_alignment_pipeline(cfg, dummy_data=True)

    # Assert run_iteration called 2 times
    assert mock_run_iteration.call_count == 2

    # Assert DPOTrainer instantiated and trained 2 times
    assert mock_trainer_init.call_count == 2
    assert mock_trainer_train.call_count == 2
