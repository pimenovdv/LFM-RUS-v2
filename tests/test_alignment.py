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

def test_run_alignment_ppo_reward_dummy(mocker, tmp_path):
    mocker.patch('trl.trainer.reward_trainer.RewardTrainer.train', return_value=None)
    mocker.patch('trl.trainer.reward_trainer.RewardTrainer.save_model', return_value=None)

    cfg = {
        "method": "ppo_reward",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "ppo_reward_out"),
        "epochs": 1
    }

    run_alignment_pipeline(cfg, dummy_data=True)


def test_run_alignment_ppo_dummy(mocker, tmp_path):
    mocker.patch('trl.experimental.ppo.PPOTrainer.train', return_value=None)
    # Let PPOTrainer initialize normally to actually test it, just mock train and save
    # mock save_model for base trainer if PPOTrainer inherits it
    try:
        mocker.patch('trl.experimental.ppo.PPOTrainer.save_model', return_value=None, create=True)
    except Exception:
        pass

    cfg = {
        "method": "ppo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "ppo_out"),
        "epochs": 1
    }

    run_alignment_pipeline(cfg, dummy_data=True)

def test_run_rlaif_pipeline_dummy(mocker, tmp_path):



    # Mock text generation to return 2 responses
    def mock_generate_responses(model, tokenizer, prompts, n, max_length):
        return [["Response A for " + p, "Response B for " + p] for p in prompts]

    mocker.patch('src.alignment.rlaif.generate_responses', side_effect=mock_generate_responses)

    # Mock the LLM judge evaluation to return "A" or "B"
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
            return MockResponse(json.dumps({"preference": "B"}))

    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()

    class MockClient:
        def __init__(self, **kwargs):
            self.chat = MockChat()

    mocker.patch('src.alignment.rlaif.OpenAI', MockClient)

    # Mock the DPO training that happens after generation
    mocker.patch('trl.DPOTrainer.train', return_value=None)

    from src.alignment.pipeline import run_alignment_pipeline
    import os
    import json

    output_dir = str(tmp_path / "rlaif_out")
    cfg = {
        "method": "rlaif",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": output_dir,
        "epochs": 1,
        "openai_api_key": "dummy_key"
    }

    run_alignment_pipeline(cfg, dummy_data=True)

    generated_data_path = os.path.join(output_dir, "rlaif_dataset.jsonl")
    assert os.path.exists(generated_data_path)

    with open(generated_data_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 3 # 3 dummy prompts

        first_item = json.loads(lines[0])
        assert "prompt" in first_item
        assert "chosen" in first_item
        assert "rejected" in first_item

        # Since judge mocked to return "B", chosen should be Response B
        assert "Response B" in first_item["chosen"]
        assert "Response A" in first_item["rejected"]


def test_run_rlaif_pipeline_no_dataset(mocker, tmp_path):
    mocker.patch('src.alignment.rlaif.OpenAI')
    from src.alignment.pipeline import run_alignment_pipeline

    cfg = {
        "method": "rlaif",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "rlaif_out"),
        "epochs": 1,
        "openai_api_key": "dummy"
    }

    with pytest.raises(ValueError, match="dataset_path must be provided in config for RLAIF"):
        run_alignment_pipeline(cfg, dummy_data=False)

def test_run_alignment_ppo_no_reward_model_path(mocker, tmp_path):
    mocker.patch('trl.experimental.ppo.PPOTrainer.train', return_value=None)
    mocker.patch('src.alignment.pipeline.format_grpo_dataset', return_value=Dataset.from_dict({'prompt': ['a']}))

    cfg = {
        "method": "ppo",
        "model_name": "sshleifer/tiny-gpt2",
        "output_dir": str(tmp_path / "ppo_out"),
        "epochs": 1,
        "dataset_path": "dummy_path"
    }

    with pytest.raises(ValueError, match="reward_model_path must be provided in config for PPO"):
        run_alignment_pipeline(cfg, dummy_data=False)
