import pytest
import torch
from unittest.mock import patch, MagicMock

from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration
from src.models.diffusion.configuration_diffusion import DiffusionConfig

@pytest.fixture
def mock_auto_model(mocker):
    auto_model_cls = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel.from_config")
    mock_model = auto_model_cls.return_value
    mock_model.config.vocab_size = 100
    mock_model.config.hidden_size = 32

    mock_embeddings = MagicMock()
    mock_embeddings.weight = torch.nn.Parameter(torch.randn(100, 32))
    mock_model.get_input_embeddings = MagicMock(return_value=mock_embeddings)
    return mock_model

@patch("torch.distributions.InverseGamma", create=True)
def test_inverse_gamma_noise_injection(mock_inverse_gamma, mock_auto_model, mocker):
    config = DiffusionConfig(
        vocab_size=100,
        mask_token_id=0,
        pad_token_id=0,
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"},
        inverse_gamma_noise_scale=1.0,
        inverse_gamma_noise_schedule="constant",
        min_inverse_gamma_noise_scale=0.0,
    )
    model = DiffusionModelForConditionalGeneration(config)
    model.model = mock_auto_model

    class MockOutput:
        def __init__(self, logits):
            self.logits = logits

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
        else:
            embeds = kwargs.get("inputs_embeds")
            batch_size, seq_len = embeds.shape[0], embeds.shape[1]

        logits = torch.randn(batch_size, seq_len, 100)
        return MockOutput(logits=logits)

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    mock_dist = MagicMock()
    mock_dist.sample.side_effect = lambda shape: torch.ones(shape + (100,))
    mock_inverse_gamma.return_value = mock_dist

    input_ids = torch.zeros((1, 5), dtype=torch.long)

    with patch("src.models.diffusion.modeling_diffusion.torch.multinomial") as mock_multinomial:
        mock_multinomial.return_value = torch.ones((1, 1), dtype=torch.long)

        model.generate(
            input_ids,
            max_new_tokens=2,
            inverse_gamma_noise_scale=1.0,
            inverse_gamma_concentration=2.0
        )
        assert mock_inverse_gamma.called
