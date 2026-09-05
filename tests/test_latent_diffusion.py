import pytest
import torch
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import LatentDiffusionModelForConditionalGeneration, TextAutoencoder, VectorQuantizer

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
        use_latent_diffusion=True,
        vq_num_embeddings=64,
        vq_embedding_dim=16,
        mask_token_id=0,
        max_timesteps=100
    )

def test_vector_quantizer(mock_config):
    vq = VectorQuantizer(mock_config.vq_num_embeddings, mock_config.vq_embedding_dim)
    inputs = torch.randn(2, 5, mock_config.vq_embedding_dim)
    quantized, loss, indices = vq(inputs)

    assert quantized.shape == inputs.shape
    assert loss.shape == torch.Size([])
    assert indices.shape == (2, 5)

def test_text_autoencoder(mock_config):
    autoencoder = TextAutoencoder(mock_config)
    input_ids = torch.randint(0, mock_config.base_config_dict["vocab_size"], (2, 5))
    outputs = autoencoder(input_ids=input_ids)

    assert outputs.loss.shape == torch.Size([])
    assert outputs.logits.shape == (2, 5, mock_config.base_config_dict["vocab_size"])
    assert outputs.hidden_states.shape == (2, 5, mock_config.vq_embedding_dim)

def test_latent_diffusion_forward(mock_config, mocker):
    model = LatentDiffusionModelForConditionalGeneration(mock_config)
    input_ids = torch.randint(0, mock_config.base_config_dict["vocab_size"], (2, 5))
    timesteps = torch.tensor([10, 20])

    # We mock diffusion forward to bypass deep generation steps logic if necessary
    # Or just let it run
    outputs = model(input_ids=input_ids, timesteps=timesteps, labels=input_ids)

    assert hasattr(outputs, "loss")
    assert outputs.loss is not None
    assert outputs.logits.shape == (2, 5, mock_config.vq_num_embeddings)

def test_latent_diffusion_generate(mock_config, mocker):
    model = LatentDiffusionModelForConditionalGeneration(mock_config)
    input_ids = torch.randint(0, mock_config.base_config_dict["vocab_size"], (1, 5))

    # Mocking internal generate of diffusion model to return dummy latents
    mocker.patch.object(model.diffusion_model, "generate", return_dict=False, return_value=torch.randint(0, mock_config.vq_num_embeddings, (1, 15)))

    generated = model.generate(input_ids=input_ids, max_new_tokens=10)
    assert generated.shape == (1, 15)
