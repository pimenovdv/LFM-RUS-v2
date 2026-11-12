import pytest
import torch
import math

from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration


@pytest.fixture
def dummy_model(mocker):
    config = DiffusionConfig(
        base_config_dict={"vocab_size": 100, "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "model_type": "gpt2"},
        mask_token_id=0,
        max_timesteps=10,
    )

    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mock_inner = mocker.MagicMock()
    mock_inner.config.model_type = "gpt2"
    mock_inner.config.is_causal = False

    embed_mock = torch.nn.Embedding(100, 16)
    mock_inner.get_input_embeddings.return_value = embed_mock

    class DummyOutputs:
        def __init__(self, batch_size, seq_len):
            self.last_hidden_state = torch.randn(batch_size, seq_len, 16)
            self.hidden_states = [torch.randn(batch_size, seq_len, 16)] * 3

    def mock_forward(*args, **kwargs):
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is not None:
            bsz, seq_len = inputs_embeds.shape[:2]
        else:
            bsz, seq_len = 1, 10
        return DummyOutputs(bsz, seq_len)

    mock_inner.side_effect = mock_forward
    mock_auto_model.from_config.return_value = mock_inner

    model = DiffusionModelForConditionalGeneration(config)

    def custom_forward(input_ids=None, inputs_embeds=None, *args, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = model.inner_model.get_input_embeddings()(input_ids)
        bsz, seq_len = inputs_embeds.shape[:2]

        class Output:
            logits = torch.randn(bsz, seq_len, 100)
            # Add some spikes to logits to make distributions non-uniform
            logits[0, 0, 10] = 10.0
            logits[0, 0, 20] = 5.0
            logits[0, 0, 30] = 2.0

            def __getitem__(self, idx):
                if idx == 0:
                    return None
                return self.logits

        return Output()

    model.forward = custom_forward
    model.eval()
    return model


def test_quantile_sampling_generate(dummy_model):
    input_ids = torch.tensor([[1, 2, 3]])

    torch.manual_seed(42)
    # Test only top quantile (drops bottom 50%)
    out_top = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        quantile_low=0.0,
        quantile_high=0.5
    )

    torch.manual_seed(42)
    # Test middle quantile (drops top 20% and bottom 20%)
    out_mid = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        quantile_low=0.2,
        quantile_high=0.8
    )

    # Test quantile bottom (drops top 50%)
    torch.manual_seed(42)
    out_bot = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        quantile_low=0.5,
        quantile_high=1.0
    )

    assert out_top.shape == (1, 7)
    assert out_mid.shape == (1, 7)
    assert out_bot.shape == (1, 7)


def test_quantile_sampling_schedules(dummy_model):
    input_ids = torch.tensor([[1, 2, 3]])

    for schedule in ["linear", "cosine", "exponential", "cyclic"]:
        out = dummy_model.generate(
            input_ids,
            max_new_tokens=2,
            steps=2,
            quantile_low=0.2,
            min_quantile_low=0.1,
            quantile_low_schedule=schedule,
            quantile_high=0.8,
            min_quantile_high=0.9,
            quantile_high_schedule=schedule
        )
        assert out.shape == (1, 5)


def test_quantile_sampling_continuous_batching(dummy_model):
    req_config = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "max_new_tokens": 4,
        "total_steps": 2,
        "quantile_low": 0.1,
        "quantile_high": 0.9,
        "quantile_low_schedule": "cosine",
        "quantile_high_schedule": "cosine",
        "min_quantile_low": 0.05,
        "min_quantile_high": 0.95
    }

    outputs = dummy_model.generate_dynamic_batch([req_config])

    assert len(outputs) == 1
    assert outputs[0].shape == (1, 7)


def test_quantile_sampling_speculative(dummy_model):
    input_ids = torch.tensor([[1, 2, 3]])

    out = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        quantile_low=0.1,
        quantile_high=0.9,
        draft_model=dummy_model,
        speculative_steps=2
    )

    assert out.shape == (1, 7)


def test_quantile_sampling_logic_no_nan(mocker):
    # Pure unit test of the quantile masking logic to ensure no NaNs
    import torch
    import torch.nn.functional as F

    logits = torch.randn(2, 5, 100)

    quantile_low = 0.2
    quantile_high = 0.8
    filter_value = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    exclusive_cumulative_probs = cumulative_probs - sorted_probs
    sorted_indices_to_remove = (exclusive_cumulative_probs >= quantile_high) | (cumulative_probs <= quantile_low)

    all_masked = sorted_indices_to_remove.all(dim=-1, keepdim=True)
    sorted_indices_to_remove = sorted_indices_to_remove & (~all_masked | (torch.arange(sorted_indices_to_remove.size(-1), device=logits.device) != 0))

    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    masked_logits = logits.masked_fill(indices_to_remove, filter_value)

    assert not torch.isnan(masked_logits).any()

    # Check that at least one value is not masked in each sequence/position
    assert (masked_logits != filter_value).any(dim=-1).all()
