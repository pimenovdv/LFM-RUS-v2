import pytest
import torch
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionConfig, MDLMContinuousBatchingManager

class Output:
    def __init__(self, logits):
        self.logits = logits

class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = DiffusionConfig(vocab_size=vocab_size, base_config_dict={"vocab_size": vocab_size, "model_type": "gpt2"}, mask_token_id=100)
    def forward(self, input_ids, **kwargs):
        return Output(torch.randn(input_ids.shape[0], input_ids.shape[1], self.vocab_size))
    def _update_causal_mask(self, *args, **kwargs):
        pass
    def get_input_embeddings(self):
        emb = torch.nn.Embedding(self.vocab_size, 16)
        return emb

@pytest.fixture
def dummy_model():
    base = DummyModel(100)
    model = DiffusionModelForConditionalGeneration(base.config)
    model.model = base
    return model

def test_igrp_generate(dummy_model, mocker):
    mocker.patch.object(dummy_model, "forward", side_effect=lambda input_ids, **kwargs: Output(torch.randn(input_ids.shape[0], input_ids.shape[1], dummy_model.config.vocab_size)))

    input_ids = torch.tensor([[0, 1, 2, 100, 100]])

    out = dummy_model.generate(
        input_ids,
        steps=2,
        max_new_tokens=0,
        repetition_penalty=1.5,
        igrp_scale=0.5,
        igrp_schedule="linear",
        temperature=1.0
    )

    assert out.shape == input_ids.shape


def test_igrp_continuous_batching(dummy_model, mocker):
    mocker.patch.object(dummy_model, "forward", side_effect=lambda input_ids, **kwargs: Output(torch.randn(input_ids.shape[0], input_ids.shape[1], dummy_model.config.vocab_size)))

    manager = MDLMContinuousBatchingManager(dummy_model)
    manager.add_request(
        request_id="req_igrp",
        input_ids=torch.tensor([[0, 1, 100]]),
        max_new_tokens=1,
        total_steps=2,
        repetition_penalty=1.5,
        igrp_scale=0.5,
        igrp_schedule="cosine",
        temperature=1.0
    )

    while len(manager.active_requests) > 0:
        manager.step()
