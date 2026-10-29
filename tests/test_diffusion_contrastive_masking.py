import pytest
import torch
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

@pytest.fixture
def mock_config():
    return DiffusionConfig(
        base_config_dict={
            "model_type": "gpt2",
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2
        },
        mask_token_id=0,
        max_timesteps=10
    )

def test_diffusion_contrastive_masking(mocker, mock_config):
    main_model = DiffusionModelForConditionalGeneration(mock_config)
    amateur_model = DiffusionModelForConditionalGeneration(mock_config)

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        batch, seq_len = input_ids.shape
        vocab_size = mock_config.base_config_dict["vocab_size"]

        class OutputMock:
            def __init__(self):
                self.logits = torch.randn(batch, seq_len, vocab_size)
        return OutputMock()

    mocker.patch.object(main_model, 'forward', side_effect=mock_forward)
    mocker.patch.object(amateur_model, 'forward', side_effect=mock_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    # Generate with Contrastive Masking
    output_contrastive = main_model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=2,
        amateur_model=amateur_model,
        contrastive_masking=True,
        contrastive_alpha=1.0
    )

    assert isinstance(output_contrastive, torch.Tensor)
    assert output_contrastive.shape[1] == input_ids.shape[1] + 5
