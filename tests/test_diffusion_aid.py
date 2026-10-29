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

def test_diffusion_aid(mocker, mock_config):
    model = DiffusionModelForConditionalGeneration(mock_config)

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        batch, seq_len = input_ids.shape
        vocab_size = mock_config.base_config_dict["vocab_size"]

        class OutputMock:
            def __init__(self):
                # Ensure high confidence so mean_confidence > aid_threshold
                logits = torch.randn(batch, seq_len, vocab_size) * 0.1
                # Make one token overwhelmingly likely
                logits[..., 10] = 50.0
                self.logits = logits
        return OutputMock()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    # Generate with AID
    output_aid = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=2,
        use_aid=True,
        aid_threshold=0.5
    )

    assert isinstance(output_aid, torch.Tensor)
    assert output_aid.shape[1] == input_ids.shape[1] + 5

    # Generate with Continuous Batching using AID
    from src.models.diffusion.modeling_diffusion import MDLMContinuousBatchingManager
    manager = MDLMContinuousBatchingManager(model)
    manager.add_request(
        input_ids=input_ids,
        max_new_tokens=5,
        total_steps=2,
        use_aid=True,
        aid_threshold=0.5
    )

    while manager.active_requests or manager.pending_requests:
        manager.step()

    assert len(manager.completed_requests) == 1
    assert manager.completed_requests[0].x.shape[1] == input_ids.shape[1] + 5
