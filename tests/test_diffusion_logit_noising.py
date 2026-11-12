from src.models.diffusion.modeling_diffusion import MDLMContinuousBatchingManager
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionConfig

@pytest.fixture
def mock_model():
    config = DiffusionConfig(
        vocab_size=100,
        mask_token_id=0,
        pad_token_id=0,
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"}
    )
    model = DiffusionModelForConditionalGeneration(config)
    model.autoencoder = MagicMock()
    model.autoencoder.pre_quant_proj.return_value.reshape.return_value = torch.randn(1, 64)
    model.autoencoder.vq.embedding.weight = torch.randn(100, 64)
    model.autoencoder.decoder_model.return_value.last_hidden_state = torch.randn(1, 10, 64)
    model.autoencoder.post_quant_proj.return_value = torch.randn(1, 10, 64)

    # Mock inner forward



    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get('input_ids', args[0] if args else torch.zeros(1, 133))
        class Output:
            def __init__(self, logits):
                self.logits = logits
        # match input_ids size EXACTLY
        return Output(torch.randn(input_ids.shape[0], input_ids.shape[1], 100))
    model.forward = MagicMock(side_effect=mock_forward)


    return model

def test_gaussian_logit_noising_generate(mock_model, mocker):
    input_ids = torch.zeros((1, 5), dtype=torch.long)
    mock_randn_like = mocker.patch("src.models.diffusion.modeling_diffusion.torch.randn_like", side_effect=lambda x: torch.ones_like(x))

    mock_model.generate(
        input_ids,
        steps=2,
        gaussian_logit_noise_scale=2.0,
        gaussian_logit_noise_schedule="constant"
    )

    assert mock_randn_like.call_count > 0

def test_gumbel_logit_noising_generate(mock_model, mocker):
    input_ids = torch.zeros((1, 5), dtype=torch.long)
    mock_rand_like = mocker.patch("src.models.diffusion.modeling_diffusion.torch.rand_like", side_effect=lambda x: torch.full_like(x, 0.5))

    mock_model.generate(
        input_ids,
        steps=2,
        gumbel_logit_noise_scale=1.5,
        gumbel_logit_noise_schedule="linear"
    )

    assert mock_rand_like.call_count > 0

def test_gaussian_logit_noising_continuous_batching(mock_model, mocker):
    manager = MDLMContinuousBatchingManager(mock_model)
    input_ids = torch.zeros((1, 5), dtype=torch.long)

    manager.add_request(
        input_ids,
        max_new_tokens=5,
        total_steps=2,
        gaussian_logit_noise_scale=2.0,
        gaussian_logit_noise_schedule="cosine"
    )

    mock_randn_like = mocker.patch("src.models.diffusion.modeling_diffusion.torch.randn_like", side_effect=lambda x: torch.ones_like(x))
    manager.step()
    manager.step()

    assert mock_randn_like.call_count > 0

def test_gumbel_logit_noising_continuous_batching(mock_model, mocker):
    manager = MDLMContinuousBatchingManager(mock_model)
    input_ids = torch.zeros((1, 5), dtype=torch.long)

    manager.add_request(
        input_ids,
        max_new_tokens=5,
        total_steps=2,
        gumbel_logit_noise_scale=1.5,
        gumbel_logit_noise_schedule="exponential"
    )

    mock_rand_like = mocker.patch("src.models.diffusion.modeling_diffusion.torch.rand_like", side_effect=lambda x: torch.full_like(x, 0.5))
    manager.step()
    manager.step()

    assert mock_rand_like.call_count > 0
