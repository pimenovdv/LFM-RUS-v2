import pytest
import torch
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMRequest, MDLMContinuousBatchingManager
from src.models.diffusion.configuration_diffusion import DiffusionConfig

def test_prob_threshold_generate(mocker):
    config_dict = {"vocab_size": 100, "hidden_size": 32, "mask_token_id": 99, "base_config_dict": {"model_type": "gpt2", "vocab_size": 100}}
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(*args, **kwargs):
        class Output:
            # Predict some logits
            logits = torch.randn(1, 4, 100)
            # Make one logit very high and others low so probabilities are distinct
            logits[0, 0, 0] = 10.0
            logits[0, 0, 1] = 5.0
            logits[0, 0, 2] = -10.0
            last_hidden_state = torch.randn(1, 4, 32)
        return Output()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    # Run without threshold
    torch.manual_seed(42)
    out_no_threshold = model.generate(
        input_ids,
        prob_threshold_val=0.0,
        max_new_tokens=1,
    )

    # Run with threshold
    torch.manual_seed(42)
    out_with_threshold = model.generate(
        input_ids,
        prob_threshold_val=0.9,
        max_new_tokens=1,
    )

    assert out_with_threshold is not None
    assert out_no_threshold is not None

def test_prob_threshold_schedules(mocker):
    config_dict = {"vocab_size": 100, "hidden_size": 32, "mask_token_id": 99, "base_config_dict": {"model_type": "gpt2", "vocab_size": 100}}
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(*args, **kwargs):
        class Output:
            logits = torch.randn(1, 4, 100)
            last_hidden_state = torch.randn(1, 4, 32)
        return Output()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    schedules = ["constant", "linear", "cosine", "exponential", "cyclic"]
    for sched in schedules:
        out = model.generate(
            input_ids,
            prob_threshold_val=0.5,
            min_prob_threshold_val=0.1,
            prob_threshold_schedule=sched,
            max_new_tokens=1,
        )
        assert out is not None

def test_prob_threshold_continuous_batching(mocker):
    config_dict = {"vocab_size": 100, "hidden_size": 32, "mask_token_id": 99, "base_config_dict": {"model_type": "gpt2", "vocab_size": 100}}
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    manager = MDLMContinuousBatchingManager(model)

    manager.add_request(input_ids=torch.tensor([[1,2,3]]), max_new_tokens=5, total_steps=10, prob_threshold_val=0.5)
    manager.add_request(input_ids=torch.tensor([[4,5]]), max_new_tokens=5, total_steps=10, prob_threshold_val=0.9, min_prob_threshold_val=0.1, prob_threshold_schedule="linear")

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is None:
            input_ids = kwargs.get('inputs_embeds')
        batch_size, seq_len = input_ids.shape[:2]
        class Output:
            logits = torch.randn(batch_size, seq_len, 100)
        return Output()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    for _ in range(5):
        completed = manager.step()
        if len(manager.active_requests) == 0:
            break
