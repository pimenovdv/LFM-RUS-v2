import pytest
import torch
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager, MDLMRequest
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from transformers import AutoModelForCausalLM, AutoConfig

class DummyAutoModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input_ids, *args, **kwargs):
        # return logits: shape (batch_size, seq_len, vocab_size)
        # We will deterministically return arange to be able to test smooth_k and temp
        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size

        # logit values: [0, 1, 2, ..., vocab_size-1] repeated for each token
        logits = torch.arange(vocab_size, dtype=torch.float, device=input_ids.device)
        logits = logits.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, vocab_size).clone()

        class Output:
            def __init__(self, l):
                self.logits = l
            def __getitem__(self, idx):
                return self.logits
        return Output(logits)

@pytest.fixture
def dummy_model():
    config = DiffusionConfig(
        base_config_dict={"vocab_size": 10, "model_type": "gpt2"},
        mask_token_id=0,
        diffusion_steps=10
    )

    model = DiffusionModelForConditionalGeneration(config)
    dummy_auto = DummyAutoModel(AutoConfig.from_pretrained("gpt2", vocab_size=10))
    model.model = dummy_auto
    return model

def test_smooth_k_sampling_generation(dummy_model, mocker):
    mocker.patch.object(dummy_model, 'forward', side_effect=lambda input_ids, **kw: DummyAutoModel(AutoConfig.from_pretrained("gpt2", vocab_size=10)).forward(input_ids, **kw))

    input_ids = torch.tensor([[1, 2, 3]])

    # We will test the effect by intercepting the logits before they are fed to softmax or topk
    # Since we can't easily intercept the middle, we will mock torch.topk or just observe the generated text.
    # Actually, we can intercept F.softmax to see the logits!

    original_softmax = torch.nn.functional.softmax

    logits_history = []
    def mock_softmax(input, dim=-1, _dtype=None):
        logits_history.append(input.clone())
        return original_softmax(input, dim=dim, dtype=_dtype)

    mocker.patch('torch.nn.functional.softmax', side_effect=mock_softmax)

    dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        steps=1,
        smooth_k=2,
        smooth_k_alpha=100.0,
        temperature=1.0,
        gumbel_temperature=0.0,
        renormalize_logits=False
    )

    # Check the logits right before softmax
    # For steps=1, it will do 1 step generation
    assert len(logits_history) > 0
    logits = logits_history[-1]

    # The original logits were [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # top_k=0 (disabled), smooth_k=2. The top 2 values are 8 and 9.
    # The remaining values should have 100.0 subtracted from them!
    # Let's check the last sequence element
    last_logits = logits[0, -1, :]
    assert abs(last_logits[-1].item() - 9.0) < 1e-4
    assert abs(last_logits[-2].item() - 8.0) < 1e-4
    assert last_logits[-3] < -90.0 # 7.0 - 100.0 = -93.0

def test_positional_temperature_generation(dummy_model, mocker):
    mocker.patch.object(dummy_model, 'forward', side_effect=lambda input_ids, **kw: DummyAutoModel(AutoConfig.from_pretrained("gpt2", vocab_size=10)).forward(input_ids, **kw))

    input_ids = torch.tensor([[1, 2, 3]])

    original_softmax = torch.nn.functional.softmax

    logits_history = []
    def mock_softmax(input, dim=-1, _dtype=None):
        logits_history.append(input.clone())
        return original_softmax(input, dim=dim, dtype=_dtype)

    mocker.patch('torch.nn.functional.softmax', side_effect=mock_softmax)

    dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        steps=1,
        temperature=1.0,
        positional_temperature=1.0,
        gumbel_temperature=0.0,
        renormalize_logits=False
    )

    logits = logits_history[-1]

    # Original logits: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Positional scaling: temp = 1.0 + 1.0 * (pos / (seq_len - 1))
    seq_len = logits.shape[1]

    # Check pos 0: temp = 1.0, logit 9 -> 9.0
    assert abs(logits[0, 0, 9].item() - 9.0) < 1e-4

    # Check last pos (pos seq_len - 1): temp = 2.0, logit 9 -> 4.5
    assert abs(logits[0, -1, 9].item() - 4.5) < 1e-4

def test_continuous_batching_smooth_k_positional_temperature(dummy_model, mocker):
    mocker.patch.object(dummy_model, 'forward', side_effect=lambda input_ids, **kw: DummyAutoModel(AutoConfig.from_pretrained("gpt2", vocab_size=10)).forward(input_ids, **kw))

    manager = MDLMContinuousBatchingManager(dummy_model)

    req_id = manager.add_request(
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=2,
        total_steps=2,
        temperature=1.0,
        positional_temperature=1.0,
        smooth_k=2,
        smooth_k_alpha=50.0
    )

    original_softmax = torch.nn.functional.softmax
    logits_history = []
    def mock_softmax(input, dim=-1, _dtype=None):
        logits_history.append(input.clone())
        return original_softmax(input, dim=dim, dtype=_dtype)

    mocker.patch('torch.nn.functional.softmax', side_effect=mock_softmax)

    manager.step()

    assert len(logits_history) > 0
    logits = logits_history[-1]

    seq_len = logits.shape[1]
    # Check Positional Temperature on top elements
    # SmoothK keeps top 2 intact.
    # top 2 at pos 0: 8 and 9. Temp is 1.0 -> 8.0 and 9.0
    assert abs(logits[0, 0, 9].item() - 9.0) < 1e-4
    # top 2 at last pos: 8 and 9. Temp is 2.0 -> 4.0 and 4.5
    assert abs(logits[0, -1, 9].item() - 4.5) < 1e-4

    # Check SmoothK on bottom elements
    # pos 0, elem 0: original 0. Temp = 1.0 -> 0.0. SmoothK -> -50.0
    assert abs(logits[0, 0, 0].item() - -50.0) < 1e-4

    # pos last, elem 0: original 0. Temp = 2.0 -> 0.0. SmoothK -> -50.0
    assert abs(logits[0, -1, 0].item() - -25.0) < 1e-4
