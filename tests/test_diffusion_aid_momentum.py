import pytest
import torch
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager

@pytest.fixture
def dummy_model(mocker):
    config = DiffusionConfig(
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"},
        mask_token_id=0,
        diffusion_steps=2
    )
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        batch, seq_len = input_ids.shape
        vocab_size = config.base_config_dict["vocab_size"]

        class OutputMock:
            def __init__(self):
                # We return a specific logit pattern to ensure predictable probabilities
                # Make token 10 always highly confident
                logits = torch.randn(batch, seq_len, vocab_size) * 0.1
                logits[..., 10] = 10.0
                self.logits = logits
        return OutputMock()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)
    return model


def test_diffusion_denoising_momentum_generate(dummy_model, mocker):
    input_ids = torch.tensor([[1, 2, 3]])

    # Run with denoising_momentum
    output = dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=4,
        steps=2,
        denoising_momentum=0.5
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape[1] == input_ids.shape[1] + 4

def test_diffusion_denoising_momentum_continuous_batching(dummy_model, mocker):
    input_ids = torch.tensor([[1, 2, 3]])

    manager = MDLMContinuousBatchingManager(dummy_model, max_batch_size=2)
    manager.add_request(
        input_ids=input_ids,
        max_new_tokens=4,
        total_steps=2,
        denoising_momentum=0.8
    )

    while manager.active_requests or manager.pending_requests:
        manager.step()

    assert len(manager.completed_requests) == 1
    assert manager.completed_requests[0].x.shape[1] == input_ids.shape[1] + 4
