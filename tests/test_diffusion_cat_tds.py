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
        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size

        # logit values: [0, 1, 2, ..., vocab_size-1] repeated for each token
        logits = torch.arange(vocab_size, dtype=torch.float, device=input_ids.device)
        logits = logits.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, vocab_size).clone()

        # Increase the difference so softmax results in a high probability for the last element
        # Making the last element significantly larger so its prob > cat_max_prob (e.g. 0.9)
        logits[:, :, -1] += 20.0

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

def test_cat_sampling_generation(dummy_model, mocker):
    mocker.patch.object(dummy_model, 'forward', side_effect=lambda input_ids, **kw: DummyAutoModel(AutoConfig.from_pretrained("gpt2", vocab_size=10)).forward(input_ids, **kw))

    input_ids = torch.tensor([[1, 2, 3]])

    original_softmax = torch.nn.functional.softmax
    logits_history = []
    def mock_softmax(input, dim=-1, _dtype=None):
        logits_history.append(input.clone())
        return original_softmax(input, dim=dim, dtype=_dtype)

    # We capture the logits directly using a custom forward wrapper
    logits_history_custom = []
    original_where = torch.where
    def mock_where(*args, **kwargs):
        res = original_where(*args, **kwargs)
        if "float" in str(res.dtype) and len(res.shape) == 3 and res.shape[-1] == 10:
            logits_history_custom.append(res.clone())
        return res

    mocker.patch('torch.where', side_effect=mock_where)

    # With the last logit boosted by 20.0, the max prob will be > 0.99
    # Setting cat_max_prob to 0.9, we expect flatten logic to trigger
    dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        steps=1,
        cat_max_prob=0.9,
        cat_flatten_factor=0.5,
        cat_min_prob=0.1,
        cat_sharpen_factor=2.0,
        renormalize_logits=False,
        temperature=1.0
    )

    assert len(logits_history_custom) > 0
    logits = logits_history_custom[-1]

    # Original logits before cat: [0, 1, 2, 3, 4, 5, 6, 7, 8, 29]
    # Because max_prob > 0.9, they should all be multiplied by cat_flatten_factor (0.5)
    # Resulting logits: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 14.5]
    last_logits = logits[0, -1, :]

    # We only care that it successfully flattened
    assert abs(last_logits[-1].item() - 14.5) < 1e-4
    assert abs(last_logits[-2].item() - 4.0) < 1e-4

def test_tds_sampling_generation(dummy_model, mocker):
    mocker.patch.object(dummy_model, 'forward', side_effect=lambda input_ids, **kw: DummyAutoModel(AutoConfig.from_pretrained("gpt2", vocab_size=10)).forward(input_ids, **kw))

    input_ids = torch.tensor([[1, 2, 3]])

    original_softmax = torch.nn.functional.softmax
    logits_history = []
    def mock_softmax(input, dim=-1, _dtype=None):
        logits_history.append(input.clone())
        return original_softmax(input, dim=dim, dtype=_dtype)

    mocker.patch('torch.nn.functional.softmax', side_effect=mock_softmax)

    # Using a seeded RNG or mocking torch.rand_like to always drop all non-max elements
    original_rand_like = torch.rand_like
    def mock_rand_like(input, *args, **kwargs):
        # return a tensor of zeros so it's always less than token_drop_prob (which is 1.0)
        return torch.zeros_like(input)
    mocker.patch('torch.rand_like', side_effect=mock_rand_like)

    dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        steps=1,
        token_drop_prob=1.0, # Drop everything we can
        renormalize_logits=False
    )

    assert len(logits_history) > 0
    logits = logits_history[-1]

    last_logits = logits[0, -1, :]

    # The max logit is at index -1 (value 29). It should NOT be dropped.
    assert last_logits[-1].item() > 0
    # All other logits should be dropped (-inf)
    assert last_logits[-2].item() == -float("Inf")
    assert last_logits[0].item() == -float("Inf")

def test_continuous_batching_cat_tds(dummy_model, mocker):
    mocker.patch.object(dummy_model, 'forward', side_effect=lambda input_ids, **kw: DummyAutoModel(AutoConfig.from_pretrained("gpt2", vocab_size=10)).forward(input_ids, **kw))

    manager = MDLMContinuousBatchingManager(dummy_model)

    req_id = manager.add_request(
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=2,
        total_steps=2,
        cat_max_prob=0.9,
        cat_flatten_factor=0.5,
        token_drop_prob=1.0,
        temperature=1.0
    )

    original_softmax = torch.nn.functional.softmax
    logits_history = []
    def mock_softmax(input, dim=-1, _dtype=None):
        logits_history.append(input.clone())
        return original_softmax(input, dim=dim, dtype=_dtype)
    mocker.patch('torch.nn.functional.softmax', side_effect=mock_softmax)

    original_rand_like = torch.rand_like
    def mock_rand_like(input, *args, **kwargs):
        return torch.zeros_like(input)
    mocker.patch('torch.rand_like', side_effect=mock_rand_like)

    # We capture the logits directly using a custom forward wrapper
    logits_history_custom = []
    original_where = torch.where
    def mock_where(*args, **kwargs):
        res = original_where(*args, **kwargs)
        if "float" in str(res.dtype) and len(res.shape) == 3 and res.shape[-1] == 10:
            logits_history_custom.append(res.clone())
        return res

    mocker.patch('torch.where', side_effect=mock_where)

    manager.step()

    assert len(logits_history_custom) > 0
    logits = logits_history_custom[-1]

    # Continuous batching step logic evaluates TDS first, then CAT
    # Original: [0, 1, 2, 3, 4, 5, 6, 7, 8, 29]
    # TDS with prob=1.0 drops all except max_prob.
    # So logits become [-inf, -inf, ..., -inf, 29]
    # Then CAT sees max_prob=1.0 > 0.9, so applies flatten factor 0.5.
    # Result should be [-inf, -inf, ..., -inf, 14.5]
    last_logits = logits[0, -1, :]
    assert last_logits[0].item() == -float("Inf")
    assert abs(last_logits[-1].item() - 14.5) < 1e-4

def test_cat_sharpen_logic(dummy_model, mocker):
    # Test sharpening (max_prob < min_prob)
    class DummyAutoModelLowConf(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def forward(self, input_ids, *args, **kwargs):
            batch_size, seq_len = input_ids.shape
            vocab_size = self.config.vocab_size
            # Uniform logits so probabilities are 1/vocab_size (0.1)
            logits = torch.ones(vocab_size, dtype=torch.float, device=input_ids.device)
            logits = logits.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, vocab_size).clone()
            class Output:
                def __init__(self, l):
                    self.logits = l
                def __getitem__(self, idx):
                    return self.logits
            return Output(logits)

    dummy_model.model = DummyAutoModelLowConf(AutoConfig.from_pretrained("gpt2", vocab_size=10))
    mocker.patch.object(dummy_model, 'forward', side_effect=lambda input_ids, **kw: dummy_model.model.forward(input_ids, **kw))

    original_softmax = torch.nn.functional.softmax
    logits_history = []
    def mock_softmax(input, dim=-1, _dtype=None):
        logits_history.append(input.clone())
        return original_softmax(input, dim=dim, dtype=_dtype)

    mocker.patch('torch.nn.functional.softmax', side_effect=mock_softmax)

    dummy_model.generate(
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=2,
        steps=1,
        cat_min_prob=0.2, # Uniform prob is 0.1, so it is < 0.2
        cat_sharpen_factor=2.0,
        renormalize_logits=False
    )

    # We capture the logits directly using a custom forward wrapper
    logits_history_custom = []

    # We monkey patch max_probs and torch.where instead
    original_where = torch.where
    def mock_where(*args, **kwargs):
        res = original_where(*args, **kwargs)
        if "float" in str(res.dtype) and len(res.shape) == 3 and res.shape[-1] == 10:
            logits_history_custom.append(res.clone())
        return res

    mocker.patch('torch.where', side_effect=mock_where)

    dummy_model.generate(
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=2,
        steps=1,
        cat_min_prob=0.2, # Uniform prob is 0.1, so it is < 0.2
        cat_sharpen_factor=2.0,
        renormalize_logits=False,
        temperature=1.0
    )

    assert len(logits_history_custom) > 0
    logits = logits_history_custom[-1]

    # Logits should all be 1.0 * 2.0 = 2.0
    last_logits = logits[0, -1, :]
    assert abs(last_logits[-1].item() - 2.0) < 1e-4
