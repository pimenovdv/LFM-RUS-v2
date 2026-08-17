import pytest
import torch
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionConfig

class DummyOutputs:
    def __init__(self, logits):
        self.logits = logits

class DummyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, input_ids, **kwargs):
        vocab_size = getattr(self.config, "vocab_size", 100)
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, vocab_size, device=input_ids.device)
        return DummyOutputs(logits)
    def __call__(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)

@pytest.fixture
def dummy_model(mocker):
    config = DiffusionConfig(
        base_config_dict={"model_type": "gpt2", "vocab_size": 100},
        vocab_size=100,
        hidden_size=32,
        diffusion_steps=4, # Ensure steps matches steps_per_block to avoid index out of bounds
        mask_token_id=0,
        remasking_strategy="random",
        max_timesteps=10,
        block_size=4
    )
    # Patch base config loading so it doesn't fail
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig.for_model", return_value=config)
    class InnerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.embeds = torch.nn.Embedding(100, 32)
        def get_input_embeddings(self):
            return self.embeds
        def forward(self, inputs_embeds, **kwargs):
            batch_size, seq_len, _ = inputs_embeds.shape
            logits = torch.randn(batch_size, seq_len, self.config.vocab_size, device=inputs_embeds.device)
            return DummyOutputs(logits)

    mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel.from_config", return_value=InnerModel())

    model = DiffusionModelForConditionalGeneration(config)
    # We patch model's forward to directly return DummyOutputs containing logits
    mocker.patch.object(model, "forward", side_effect=DummyModel(config))
    mocker.patch.object(model, "__call__", side_effect=DummyModel(config))
    return model

def test_watermark_generation(dummy_model):
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    watermarking_config = {"bias": 2.0, "seeding_scheme": "lefthash", "context_width": 1}

    # Run generation with watermarking
    try:
        output_ids = dummy_model.generate(
            input_ids=input_ids,
            max_new_tokens=4,
            steps=4,
            watermarking_config=watermarking_config
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        pytest.fail(f"Generation with watermarking failed: {e}")

    assert output_ids.shape == (1, 7) # 3 initial + 4 new tokens
