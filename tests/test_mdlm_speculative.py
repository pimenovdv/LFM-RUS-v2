import pytest
import torch
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionConfig
from transformers import AutoConfig

class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.config = AutoConfig.for_model("bert")
        self.vocab_size = vocab_size
        self.embeddings = torch.nn.Embedding(vocab_size, 768)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None:
            logits = torch.randn(input_ids.size(0), input_ids.size(1), self.vocab_size)
            hidden = torch.randn(input_ids.size(0), input_ids.size(1), 768)
        else:
            logits = torch.randn(inputs_embeds.size(0), inputs_embeds.size(1), self.vocab_size)
            hidden = torch.randn(inputs_embeds.size(0), inputs_embeds.size(1), 768)
        class Output:
            pass
        out = Output()
        out.logits = logits
        out.last_hidden_state = hidden
        return out

@pytest.fixture
def dummy_model():
    config = DiffusionConfig(
        base_config_dict={"model_type": "bert", "vocab_size": 100},
        mask_token_id=0,
        diffusion_steps=4,
        block_size=10
    )
    model = DiffusionModelForConditionalGeneration(config)
    model.inner_model = DummyModel(100)
    return model

@pytest.fixture
def dummy_draft():
    config = DiffusionConfig(
        base_config_dict={"model_type": "bert", "vocab_size": 100},
        mask_token_id=0,
        diffusion_steps=4,
        block_size=10
    )
    model = DiffusionModelForConditionalGeneration(config)
    model.inner_model = DummyModel(100)
    return model

def test_speculative_decoding_basic(dummy_model, dummy_draft):
    input_ids = torch.ones(2, 5, dtype=torch.long)

    out = dummy_model.generate(
        input_ids,
        draft_model=dummy_draft,
        speculative_steps=2,
        speculative_threshold=0.5,
        max_new_tokens=10
    )
    assert out is not None
    assert out.shape == (2, 15)

def test_speculative_decoding_high_threshold(dummy_model, dummy_draft):
    input_ids = torch.ones(2, 5, dtype=torch.long)

    out = dummy_model.generate(
        input_ids,
        draft_model=dummy_draft,
        speculative_steps=2,
        speculative_threshold=0.99,
        max_new_tokens=10
    )
    assert out is not None
    assert out.shape == (2, 15)
