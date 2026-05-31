import pytest
from src.alignment.pipeline import run_alignment_pipeline, accuracy_reward, variance_reward

def test_accuracy_reward():
    prompts = ["prompt1", "prompt2"]
    completions = ["Here is a <solution> 42 </solution>", "No solution here"]
    rewards = accuracy_reward(prompts, completions)
    assert rewards == [1.0, -1.0]

def test_variance_reward():
    prompts = ["prompt1", "prompt2", "prompt3"]
    completions = [
        "word " * 5, # too short
        "this is a good normal sentence without repeating loops and it is long enough",
        "loop loop loop loop loop loop loop loop loop loop loop loop" # looping
    ]
    rewards = variance_reward(prompts, completions)
    assert rewards[0] == 0.0
    assert rewards[1] == 0.5
    assert rewards[2] == -2.0

def test_run_alignment_dpo_dummy(mocker, tmp_path):
    mocker.patch('trl.DPOTrainer.train', return_value=None)
    mocker.patch('trl.DPOTrainer.save_model', return_value=None)
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)

    cfg = {
        "method": "dpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "dpo_out"),
        "epochs": 1
    }

    cfg['push_to_hub'] = 'dummy/alignment'
    run_alignment_pipeline(cfg, dummy_data=True)

def test_run_alignment_grpo_dummy(mocker, tmp_path):
    mocker.patch('trl.GRPOTrainer.train', return_value=None)
    mocker.patch('trl.GRPOTrainer.save_model', return_value=None)
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)

    cfg = {
        "method": "grpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "grpo_out"),
        "reward_funcs": ["accuracy", "variance", "unknown"],
        "epochs": 1
    }

    cfg['push_to_hub'] = 'dummy/alignment'
    run_alignment_pipeline(cfg, dummy_data=True)

def test_run_alignment_unknown_method(tmp_path):
    cfg = {
        "method": "unknown",
        "model_name": "sshleifer/tiny-gpt2"
    }
    cfg['push_to_hub'] = 'dummy/alignment'
    with pytest.raises(ValueError, match="Unknown alignment method: unknown"):
        run_alignment_pipeline(cfg, dummy_data=True)

def test_run_alignment_ipo_dummy(mocker, tmp_path):
    mocker.patch('trl.DPOTrainer.train', return_value=None)
    mocker.patch('trl.DPOTrainer.save_model', return_value=None)
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)

    cfg = {
        "method": "ipo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "ipo_out"),
        "epochs": 1
    }

    run_alignment_pipeline(cfg, dummy_data=True)

def test_run_alignment_kto_dummy(mocker, tmp_path):
    mocker.patch('trl.KTOTrainer.train', return_value=None)
    mocker.patch('trl.KTOTrainer.save_model', return_value=None)
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)

    cfg = {
        "method": "kto",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "kto_out"),
        "epochs": 1
    }

    cfg['push_to_hub'] = 'dummy/alignment'
    run_alignment_pipeline(cfg, dummy_data=True)

def test_run_alignment_orpo_dummy(mocker, tmp_path):
    mocker.patch('trl.experimental.orpo.ORPOTrainer.train', return_value=None)
    mocker.patch('trl.experimental.orpo.ORPOTrainer.save_model', return_value=None)
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)

    cfg = {
        "method": "orpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "orpo_out"),
        "epochs": 1
    }

    cfg['push_to_hub'] = 'dummy/alignment'
    run_alignment_pipeline(cfg, dummy_data=True)
