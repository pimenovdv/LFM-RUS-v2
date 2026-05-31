from src.alignment.rejection_sampling import run_rejection_sampling
from datasets import Dataset
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

def test_run_alignment_cpo_dummy(mocker, tmp_path):
    mocker.patch('trl.experimental.cpo.CPOTrainer.train', return_value=None)
    mocker.patch('trl.experimental.cpo.CPOTrainer.save_model', return_value=None)
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)

    cfg = {
        "method": "cpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "cpo_out"),
        "epochs": 1
    }

    cfg['push_to_hub'] = 'dummy/alignment'
    run_alignment_pipeline(cfg, dummy_data=True)

def test_run_alignment_dpo_with_data(mocker, tmp_path):
    mocker.patch('trl.DPOTrainer.train', return_value=None)
    mocker.patch('trl.DPOTrainer.save_model', return_value=None)
    mocker.patch('src.alignment.pipeline.format_dpo_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'chosen': ['b'], 'rejected': ['c']}))
    mocker.patch('src.alignment.pipeline.format_kto_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'completion': ['b'], 'label': [True]}))
    mocker.patch('src.alignment.pipeline.format_grpo_dataset', return_value=Dataset.from_dict({'prompt': ['a']}))

    cfg = {
        "method": "dpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "dpo_out"),
        "epochs": 1,
        "dataset_path": "dummy_path"
    }

    run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_grpo_with_data(mocker, tmp_path):
    mocker.patch('trl.GRPOTrainer.train', return_value=None)
    mocker.patch('trl.GRPOTrainer.save_model', return_value=None)
    mocker.patch('src.alignment.pipeline.format_dpo_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'chosen': ['b'], 'rejected': ['c']}))
    mocker.patch('src.alignment.pipeline.format_kto_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'completion': ['b'], 'label': [True]}))
    mocker.patch('src.alignment.pipeline.format_grpo_dataset', return_value=Dataset.from_dict({'prompt': ['a']}))

    cfg = {
        "method": "grpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "grpo_out"),
        "reward_funcs": ["accuracy", "variance"],
        "epochs": 1,
        "dataset_path": "dummy_path"
    }

    run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_kto_with_data(mocker, tmp_path):
    mocker.patch('trl.KTOTrainer.train', return_value=None)
    mocker.patch('trl.KTOTrainer.save_model', return_value=None)
    mocker.patch('src.alignment.pipeline.format_dpo_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'chosen': ['b'], 'rejected': ['c']}))
    mocker.patch('src.alignment.pipeline.format_kto_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'completion': ['b'], 'label': [True]}))
    mocker.patch('src.alignment.pipeline.format_grpo_dataset', return_value=Dataset.from_dict({'prompt': ['a']}))

    cfg = {
        "method": "kto",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "kto_out"),
        "epochs": 1,
        "dataset_path": "dummy_path"
    }

    run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_orpo_with_data(mocker, tmp_path):
    mocker.patch('trl.experimental.orpo.ORPOTrainer.train', return_value=None)
    mocker.patch('trl.experimental.orpo.ORPOTrainer.save_model', return_value=None)
    mocker.patch('src.alignment.pipeline.format_dpo_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'chosen': ['b'], 'rejected': ['c']}))
    mocker.patch('src.alignment.pipeline.format_kto_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'completion': ['b'], 'label': [True]}))
    mocker.patch('src.alignment.pipeline.format_grpo_dataset', return_value=Dataset.from_dict({'prompt': ['a']}))

    cfg = {
        "method": "orpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "orpo_out"),
        "epochs": 1,
        "dataset_path": "dummy_path"
    }

    run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_cpo_with_data(mocker, tmp_path):
    mocker.patch('trl.experimental.cpo.CPOTrainer.train', return_value=None)
    mocker.patch('trl.experimental.cpo.CPOTrainer.save_model', return_value=None)
    mocker.patch('src.alignment.pipeline.format_dpo_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'chosen': ['b'], 'rejected': ['c']}))
    mocker.patch('src.alignment.pipeline.format_kto_dataset', return_value=Dataset.from_dict({'prompt': ['a'], 'completion': ['b'], 'label': [True]}))
    mocker.patch('src.alignment.pipeline.format_grpo_dataset', return_value=Dataset.from_dict({'prompt': ['a']}))

    cfg = {
        "method": "cpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "cpo_out"),
        "epochs": 1,
        "dataset_path": "dummy_path"
    }

    run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_no_dataset_path(mocker, tmp_path):
    mocker.patch('trl.DPOTrainer.train', return_value=None)
    mocker.patch('trl.DPOTrainer.save_model', return_value=None)

    cfg = {
        "method": "dpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "dpo_out"),
        "epochs": 1
    }

    with pytest.raises(ValueError, match="dataset_path must be provided in config for DPO"):
        run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_grpo_no_dataset_path(mocker, tmp_path):
    mocker.patch('trl.GRPOTrainer.train', return_value=None)
    mocker.patch('trl.GRPOTrainer.save_model', return_value=None)

    cfg = {
        "method": "grpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "grpo_out"),
        "epochs": 1
    }

    with pytest.raises(ValueError, match="dataset_path must be provided in config for GRPO"):
        run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_kto_no_dataset_path(mocker, tmp_path):
    mocker.patch('trl.KTOTrainer.train', return_value=None)
    mocker.patch('trl.KTOTrainer.save_model', return_value=None)

    cfg = {
        "method": "kto",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "kto_out"),
        "epochs": 1
    }

    with pytest.raises(ValueError, match="dataset_path must be provided in config for KTO"):
        run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_orpo_no_dataset_path(mocker, tmp_path):
    mocker.patch('trl.experimental.orpo.ORPOTrainer.train', return_value=None)
    mocker.patch('trl.experimental.orpo.ORPOTrainer.save_model', return_value=None)

    cfg = {
        "method": "orpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "orpo_out"),
        "epochs": 1
    }

    with pytest.raises(ValueError, match="dataset_path must be provided in config for ORPO"):
        run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_cpo_no_dataset_path(mocker, tmp_path):
    mocker.patch('trl.experimental.cpo.CPOTrainer.train', return_value=None)
    mocker.patch('trl.experimental.cpo.CPOTrainer.save_model', return_value=None)

    cfg = {
        "method": "cpo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "cpo_out"),
        "epochs": 1
    }

    with pytest.raises(ValueError, match="dataset_path must be provided in config for CPO"):
        run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_rejection_sampling_dummy(mocker, tmp_path):
    mocker.patch('transformers.AutoModelForCausalLM.from_pretrained')
    mocker.patch('transformers.AutoTokenizer.from_pretrained')
    mocker.patch('src.alignment.rejection_sampling.pipeline', return_value=lambda *args, **kwargs: [{"generated_text": "text"} for _ in range(kwargs.get('num_return_sequences', 1))])
    mocker.patch('src.alignment.rejection_sampling.run_sft', return_value=None)

    cfg = {
        "method": "rejection_sampling",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "rs_out"),
        "epochs": 1,
        "num_generations": 2
    }

    run_alignment_pipeline(cfg, dummy_data=True)

def test_run_alignment_rejection_sampling_with_data(mocker, tmp_path):
    mocker.patch('transformers.AutoModelForCausalLM.from_pretrained')
    mocker.patch('transformers.AutoTokenizer.from_pretrained')
    mocker.patch('src.alignment.rejection_sampling.pipeline', return_value=lambda *args, **kwargs: [{"generated_text": "text"} for _ in range(kwargs.get('num_return_sequences', 1))])
    mocker.patch('src.alignment.rejection_sampling.run_sft', return_value=None)
    mocker.patch('src.alignment.rejection_sampling.load_dataset', return_value=Dataset.from_dict({'prompt': ['a', 'b']}))

    cfg = {
        "method": "rft",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "rs_out"),
        "epochs": 1,
        "num_generations": 2,
        "dataset_path": "dummy_path"
    }

    run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_rejection_sampling_no_dataset_path(mocker, tmp_path):
    cfg = {
        "method": "rejection_sampling",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "rs_out"),
        "epochs": 1
    }

    with pytest.raises(ValueError, match="dataset_path must be provided in config for Rejection Sampling"):
        run_alignment_pipeline(cfg, dummy_data=False)


def test_run_alignment_spin_dummy(mocker):
    cfg = {
        "method": "spin",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": "./test_spin",
        "learning_rate": 1e-5,
        "batch_size": 2,
        "epochs": 1
    }
    mocker.patch("src.alignment.pipeline.DPOTrainer.train")
    mocker.patch("src.alignment.pipeline.DPOTrainer.save_model")
    mock_push = mocker.patch("transformers.PreTrainedModel.push_to_hub")

    run_alignment_pipeline(cfg, dummy_data=True)

    from src.alignment.pipeline import DPOTrainer
    DPOTrainer.train.assert_called_once()
    DPOTrainer.save_model.assert_called_once()
    mock_push.assert_not_called()
