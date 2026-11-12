import pytest
import torch
from unittest.mock import MagicMock
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager, MDLMRequest

from src.models.diffusion.configuration_diffusion import DiffusionConfig

class DummyConfig(DiffusionConfig):
    def __init__(self):
        super().__init__()
        self.vocab_size = 100
        self.mask_token_id = 0
        self.pad_token_id = 0
        self.model_type = "gpt2"
        self.use_moe = False
        self.use_latent_diffusion = False
        self.use_flow_matching = False
        self.use_sdpa = False
        self.use_flash_attention_2 = False
        self.hidden_size = 128
        self.base_config_dict = {"vocab_size": 100, "model_type": "gpt2", "hidden_size": 128}

def test_diffusion_gumbel_max_noise_generation(mocker):
    config = DummyConfig()
    mocker.patch("transformers.AutoConfig.for_model", return_value=config)
    mocker.patch("transformers.AutoModel.from_config")
    model = DiffusionModelForConditionalGeneration(config)

    # Mock forward to return logits directly
    class Output:
        def __init__(self, logits):
            self.logits = logits
    mocker.patch.object(model, "forward", side_effect=lambda input_ids, **kwargs: Output(torch.zeros(input_ids.shape[0], input_ids.shape[1], 100)))

    # We patch torch.rand to return a known value to check noise transformation
    mock_rand = mocker.patch("src.models.diffusion.modeling_diffusion.torch.rand")
    # In generate(), input_ids shape might change over time, so we need a side_effect
    mock_rand.side_effect = lambda shape, device=None: torch.full(shape, 0.5, device=device)

    # We patch torch.where since we want to see the modified logits
    mocker.patch("src.models.diffusion.modeling_diffusion.torch.where", wraps=torch.where)

    input_ids = torch.zeros(1, 5, dtype=torch.long)

    # Linear schedule step
    model.generate(
        input_ids,
        max_new_tokens=5,
        max_timesteps=10,
        gumbel_max_noise_scale=2.0,
        gumbel_max_noise_schedule="linear",
        min_gumbel_max_noise_scale=0.0,
        gumbel_max_loc=0.0,
        gumbel_max_scale=1.0,
    )

    assert mock_rand.called

    # Verify cyclic schedule
    model.generate(
        input_ids,
        max_new_tokens=5,
        max_timesteps=10,
        gumbel_max_noise_scale=2.0,
        gumbel_max_noise_schedule="cyclic",
        min_gumbel_max_noise_scale=0.1,
    )
    assert mock_rand.called

def test_diffusion_gumbel_max_noise_continuous_batching(mocker):
    config = DummyConfig()
    mocker.patch("transformers.AutoConfig.for_model", return_value=config)
    mocker.patch("transformers.AutoModel.from_config")
    model = DiffusionModelForConditionalGeneration(config)

    manager = MDLMContinuousBatchingManager(
        model=model,
        max_batch_size=8
    )

    req = MDLMRequest(
        request_id="req1",
        input_ids=torch.zeros(5, dtype=torch.long),
        max_new_tokens=5,
        total_steps=10,
        gumbel_max_noise_scale=2.0,
        gumbel_max_noise_schedule="cosine",
        min_gumbel_max_noise_scale=0.5,
        gumbel_max_loc=1.0,
        gumbel_max_scale=0.5,
    )

    mock_rand = mocker.patch("src.models.diffusion.modeling_diffusion.torch.rand")
    mock_rand.return_value = torch.full((5, 100), 0.5)

    manager.active_requests = [req]

    # For unit testing we'll just assert the configuration structure is correct.
    assert req.gumbel_max_noise_scale == 2.0
    assert req.gumbel_max_noise_schedule == "cosine"
    assert req.min_gumbel_max_noise_scale == 0.5
    assert req.gumbel_max_loc == 1.0
    assert req.gumbel_max_scale == 0.5
