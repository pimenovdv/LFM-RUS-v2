import pytest
from transformers import AutoConfig, AutoModel
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionControlNetModel

def test_diffusion_config_flash_attention():
    config = DiffusionConfig(
        base_config_dict={"model_type": "llama", "hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 8, "vocab_size": 32000, "intermediate_size": 512, "max_position_embeddings": 512},
        use_flash_attention_2=True
    )
    assert config.use_flash_attention_2 is True

def test_diffusion_model_flash_attention(mocker):
    config = DiffusionConfig(
        base_config_dict={"model_type": "llama", "hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 8, "vocab_size": 32000, "intermediate_size": 512, "max_position_embeddings": 512},
        use_flash_attention_2=True
    )
    # Llama won't load FA2 unless torch.cuda.is_available() and flash_attn is installed, so we mock AutoModel.from_config
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel.from_config")

    # Just to bypass some errors in disable causal mask
    mock_inner_model = mocker.MagicMock()
    mock_inner_model.config.is_causal = True
    mock_auto_model.return_value = mock_inner_model

    model = DiffusionModelForConditionalGeneration(config)

    # Assert that from_config was called with a config where _attn_implementation is flash_attention_2
    called_config = mock_auto_model.call_args[0][0]
    assert getattr(called_config, "_attn_implementation", None) == "flash_attention_2"

def test_controlnet_flash_attention(mocker):
    config = DiffusionConfig(
        base_config_dict={"model_type": "llama", "hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 8, "vocab_size": 32000, "intermediate_size": 512, "max_position_embeddings": 512},
        use_flash_attention_2=True
    )
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel.from_config")
    mock_inner_model = mocker.MagicMock()
    mock_inner_model.config.is_causal = True
    mock_auto_model.return_value = mock_inner_model

    model = DiffusionControlNetModel(config)
    called_config = mock_auto_model.call_args[0][0]
    assert getattr(called_config, "_attn_implementation", None) == "flash_attention_2"
