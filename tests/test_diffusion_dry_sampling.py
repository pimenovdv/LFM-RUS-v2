import torch
import pytest
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

@pytest.fixture
def dummy_config():
    config_dict = {
        "model_type": "gpt2",
        "vocab_size": 100,
        "n_embd": 32,
        "n_layer": 2,
        "n_head": 2,
    }
    return DiffusionConfig(
        base_config_dict=config_dict,
        mask_token_id=99,
        diffusion_steps=1,
        block_size=10,
        remasking_strategy="low_confidence"
    )

@pytest.fixture
def dummy_model(dummy_config, mocker):
    model = DiffusionModelForConditionalGeneration(dummy_config)
    # Mock forward to return controlled logits
    mocker.patch.object(model, 'forward', side_effect=lambda **kwargs: type('obj', (object,), {'logits': torch.zeros(kwargs['input_ids'].shape[0], kwargs['input_ids'].shape[1], dummy_config.base_config_dict['vocab_size'])}))
    return model

def test_dry_sampling_no_penalty(dummy_model):
    """Test that DRY sampling does not penalize when there are no matches."""
    input_ids = torch.tensor([[1, 2, 3, 4]])

    # We want to check the logits of the generated tokens.
    # Let's mock forward to return non-zero logits so we can observe the change.
    def mock_forward(**kwargs):
        batch, seq_len = kwargs['input_ids'].shape
        logits = torch.ones(batch, seq_len, 100) * 10.0 # base logit 10.0
        return type('obj', (object,), {'logits': logits})

    dummy_model.forward = mock_forward

    # Generate 2 tokens
    output = dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        steps=1,
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=2
    )

    assert output.shape == (1, 6)

def test_dry_sampling_penalty_applied(dummy_model, mocker):
    """Test that DRY sampling penalizes correct tokens."""
    input_ids = torch.tensor([[1, 2, 3, 4, 1, 2]]) # Suffix '1, 2' matches prefix '1, 2'. Next token was '3'.

    captured_logits = []

    def mock_forward(**kwargs):
        batch, seq_len = kwargs['input_ids'].shape
        logits = torch.ones(batch, seq_len, 100) * 10.0 # base logit 10.0
        captured_logits.append(logits)
        return type('obj', (object,), {'logits': logits})

    mocker.patch.object(dummy_model, 'forward', side_effect=mock_forward)

    dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=1,
        dry_multiplier=1.0, # Multiplier 1.0
        dry_base=2.0,       # Base 2.0
        dry_allowed_length=2 # Allowed length 2
    )

    # The suffix '1, 2' at the end matches '1, 2' at the beginning.
    # The token following the first '1, 2' is '3'.
    # Match length is 2. Penalty = 1.0 * (2.0 ** (2 - 2)) = 1.0 * (2.0 ** 0) = 1.0.
    # So logit for token '3' at the generated position (index 6, which is block_start) should be 10.0 - 1.0 = 9.0.
    # However, since steps=1, block_size=10, the generation logic processes logits.
    # We can check if token 3 was penalized.
    # Because `forward` returns constant logits, without penalty, argmax would be 0.
    # With penalty on 3, it doesn't change argmax, but we can inspect the captured logits if we could,
    # or we can write a test that verifies the logit modification.

    # A better way is to set logit of 3 higher, and see if penalty reduces it below another token.
    def mock_forward_specific(**kwargs):
        batch, seq_len = kwargs['input_ids'].shape
        logits = torch.zeros(batch, seq_len, 100)
        # Make token 3 the most likely
        logits[:, :, 3] = 10.0
        # Make token 5 slightly less likely
        logits[:, :, 5] = 9.5
        return type('obj', (object,), {'logits': logits})

    mocker.patch.object(dummy_model, 'forward', side_effect=mock_forward_specific)

    output = dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=1,
        dry_multiplier=2.0, # Multiplier 2.0
        dry_base=2.0,       # Base 2.0
        dry_allowed_length=2 # Allowed length 2
    )

    # Penalty on 3: 2.0 * (2.0 ** (2 - 2)) = 2.0
    # New logit for 3: 10.0 - 2.0 = 8.0
    # Logit for 5 is 9.5.
    # Therefore, the model should generate token 5 instead of 3.
    assert output[0, -1] == 5

def test_dry_sampling_longer_match(dummy_model, mocker):
    """Test that longer matches incur exponentially higher penalties."""
    # Prefix '1, 2, 3', Suffix '1, 2, 3'. Next token was '4'.
    input_ids = torch.tensor([[1, 2, 3, 4, 9, 1, 2, 3]])

    def mock_forward_specific(**kwargs):
        batch, seq_len = kwargs['input_ids'].shape
        logits = torch.zeros(batch, seq_len, 100)
        # Make token 4 the most likely
        logits[:, :, 4] = 10.0
        # Make token 5 less likely
        logits[:, :, 5] = 9.0
        return type('obj', (object,), {'logits': logits})

    mocker.patch.object(dummy_model, 'forward', side_effect=mock_forward_specific)

    output = dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=1,
        dry_multiplier=0.6, # Multiplier
        dry_base=2.0,       # Base
        dry_allowed_length=2 # Allowed length 2
    )

    # Match length is 3 ('1, 2, 3').
    # Penalty on 4: 0.6 * (2.0 ** (3 - 2)) = 0.6 * 2.0 = 1.2
    # New logit for 4: 10.0 - 1.2 = 8.8
    # Logit for 5 is 9.0.
    # The model should generate token 5.
    assert output[0, -1] == 5

def test_dry_sampling_sequence_breaker(dummy_model, mocker):
    """Test that sequence breakers stop the penalty and limit match length."""
    # Prefix '9, 1, 2, 3', next is '4'. Suffix '9, 1, 2, 3'.
    # If '9' is a breaker, the match '9, 1, 2, 3' should be broken at '9'.
    # The effective match will only be '1, 2, 3' (length 3).
    input_ids = torch.tensor([[9, 1, 2, 3, 4, 8, 9, 1, 2, 3]])

    def mock_forward_specific(**kwargs):
        batch, seq_len = kwargs['input_ids'].shape
        logits = torch.zeros(batch, seq_len, 100)
        logits[:, :, 4] = 10.0 # base logit for 4
        logits[:, :, 5] = 9.0  # alternative token
        return type('obj', (object,), {'logits': logits})

    mocker.patch.object(dummy_model, 'forward', side_effect=mock_forward_specific)

    # Run with breaker [9]
    output_with_breaker = dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=1,
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=2,
        dry_sequence_breakers=[9] # 9 is a breaker
    )

    # With breaker=9, match is '1, 2, 3' (len=3). Penalty = 1.0 * (2.0 ** (3 - 2)) = 2.0
    # Logit for 4 = 10.0 - 2.0 = 8.0.
    # Logit for 5 = 9.0.
    # Token 5 should be generated.
    assert output_with_breaker[0, -1] == 5

    # Run without breaker
    output_no_breaker = dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=1,
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=2,
        dry_sequence_breakers=None
    )

    # Without breaker, match is '9, 1, 2, 3' (len=4). Penalty = 1.0 * (2.0 ** (4 - 2)) = 4.0
    # Logit for 4 = 10.0 - 4.0 = 6.0.
    # Logit for 5 = 9.0.
    # Token 5 should be generated.
    assert output_no_breaker[0, -1] == 5

    # Test breaker in the base allowed length suffix
    input_ids_short = torch.tensor([[1, 2, 3, 9, 1, 2]])
    def mock_forward_short(**kwargs):
        batch, seq_len = kwargs['input_ids'].shape
        logits = torch.zeros(batch, seq_len, 100)
        logits[:, :, 3] = 10.0
        logits[:, :, 5] = 9.5
        return type('obj', (object,), {'logits': logits})

    mocker.patch.object(dummy_model, 'forward', side_effect=mock_forward_short)

    output_breaker_in_suffix = dummy_model.generate(
        input_ids=input_ids_short,
        max_new_tokens=1,
        steps=1,
        dry_multiplier=2.0,
        dry_base=2.0,
        dry_allowed_length=2,
        dry_sequence_breakers=[2] # 2 is a breaker, inside the suffix '1, 2'
    )

    # Since 2 is in the base suffix '1, 2', the penalty should be completely skipped.
    # Logit for 3 remains 10.0, so 3 is generated.
    assert output_breaker_in_suffix[0, -1] == 3

def test_dry_sampling_multiple_steps(dummy_model, mocker):
    """Test that DRY sampling does not penalize tokens against themselves in multi-step diffusion."""
    input_ids = torch.tensor([[1, 2, 3]])

    # Track the number of forward calls
    forward_calls = [0]

    def mock_forward(**kwargs):
        batch, seq_len = kwargs['input_ids'].shape
        logits = torch.zeros(batch, seq_len, 100)

        # On first step, make token 4 the most likely
        if forward_calls[0] == 0:
            logits[:, :, 4] = 10.0
            logits[:, :, 5] = 9.0
        # On second step, the input will now have token 4 at the end.
        # We still return 4 as the most likely.
        else:
            logits[:, :, 4] = 10.0
            logits[:, :, 5] = 9.0

        forward_calls[0] += 1
        return type('obj', (object,), {'logits': logits})

    mocker.patch.object(dummy_model, 'forward', side_effect=mock_forward)

    # We set max_new_tokens=2, block_size=1 to force step-by-step block generation?
    # Actually, we can just use steps=2. The first step will unmask it partially (or fully).
    # Since dummy_model has unmasking, we'll set unmasking schedule such that it generates tokens over 2 steps.
    # We will generate 2 new tokens to give enough length for a match.
    dummy_model.config.block_size = 2

    output = dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        steps=2,
        dry_multiplier=10.0, # High penalty
        dry_base=2.0,
        dry_allowed_length=1 # Allowed length 1 to trigger easily
    )

    # If the bug is present, on the second step, the target suffix will match itself.
    # And since the token is already unmasked (e.g., token 4), it will be heavily penalized,
    # causing token 5 to be selected.
    # With the bug fixed, it won't match itself, penalty will only apply if token 4 actually appeared earlier.
    # Since it didn't appear earlier (prompt is 1, 2, 3), token 4 should be generated.
    # Wait, the prompt has 1, 2, 3. The first generated token is 4. Next is 4.
    # If it predicts 4 again, then there will be a match (because 4 was generated previously).
    # However, DRY only penalizes tokens based on matches *before* the start_idx.
    # Actually, our bug was that the target suffix matched itself.
    # If the bug was fixed, the token won't penalize itself. But will it penalize the second token?
    # No, because the second token hasn't matched anything length >= 2 (if allowed_length was 2).
    # Here allowed_length is 1. The suffix is length 1.
    # For the second generated token (pos=4), suffix is seq[3:4].
    # seq[3] is 4. It searches for '4' in seq[0:3], which is [1, 2, 3]. '4' is not there.
    # So there is NO match. No penalty applies to the second token.
    # Therefore both generated tokens should be 4.

    # Let's check the first generated token. It should be 4.
    assert output[0, 3] == 4
    # The second generated token sees '4' as its suffix. It searches [1, 2, 3] for '4'. Not found.
    # Thus, token 4 is NOT penalized for the next position.
    # So the second generated token should also be 4.
    assert output[0, 4] == 4
