import torch
import pytest
from transformers import AutoConfig
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

@pytest.fixture
def dummy_model():
    base_config = AutoConfig.for_model("gpt2")
    base_config.n_layer = 2
    base_config.n_head = 2
    base_config.n_embd = 32
    base_config.vocab_size = 100
    config = DiffusionConfig(
        base_config_dict=base_config.to_dict(),
        mask_token_id=0,
        diffusion_steps=4,
        block_size=2,
        timestep_dim=16
    )
    model = DiffusionModelForConditionalGeneration(config)
    return model

def test_generate_dynamic_cfg(dummy_model, mocker):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    uncond_ids = torch.randint(1, 100, (batch_size, seq_len))

    # We mock the math.cos to verify cosine is called or we just test they run without crashing
    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        cfg_scale=2.0,
        cfg_schedule="linear",
        unconditional_input_ids=uncond_ids
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        cfg_scale=2.0,
        cfg_schedule="cosine",
        unconditional_input_ids=uncond_ids
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)

def test_generate_top_k_top_p(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_top_k = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_k=5
    )

    out_top_p = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_p=0.9
    )

    out_both = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_k=5,
        top_p=0.9
    )

    assert out_top_k.shape == (batch_size, seq_len + 2)
    assert out_top_p.shape == (batch_size, seq_len + 2)
    assert out_both.shape == (batch_size, seq_len + 2)

def test_generate_temperature(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_temp = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        temperature=0.8
    )

    assert out_temp.shape == (batch_size, seq_len + 2)

def test_generate_repetition_penalty(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_rep_penalty = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        repetition_penalty=1.2
    )

    assert out_rep_penalty.shape == (batch_size, seq_len + 2)

def test_generate_min_p(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_min_p = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        min_p=0.05
    )

    assert out_min_p.shape == (batch_size, seq_len + 2)

def test_generate_frequency_presence_penalty(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_penalties = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )

    assert out_penalties.shape == (batch_size, seq_len + 2)

def test_generate_logit_bias_and_suppress_tokens(dummy_model):
    batch_size = 1
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Suppress all tokens except one (token_id=42)
    # Give a massive bias to token_id=42
    suppress_tokens = list(range(1, 42)) + list(range(43, 100))
    logit_bias = {42: 1e9}

    out_biased = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        logit_bias=logit_bias,
        suppress_tokens=suppress_tokens
    )

    # With all other tokens suppressed and 42 given massive bias, the generated tokens should be 42
    # The generated tokens are the last `max_new_tokens` items.
    generated_tokens = out_biased[:, seq_len:]
    assert torch.all(generated_tokens == 42).item()

def test_generate_exponential_cfg_and_rescale(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    uncond_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Test the new exponential schedule and guidance rescale functionality
    out = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        cfg_scale=2.0,
        cfg_schedule="exponential",
        guidance_rescale=0.7,
        unconditional_input_ids=uncond_ids
    )

    # Validate output shape
    assert out.shape == (batch_size, seq_len + 2)
    assert out.dtype == input_ids.dtype


def test_dynamic_temperature_schedule(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        temperature=1.0,
        temperature_schedule="linear",
        min_temperature=0.1
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        temperature=1.0,
        temperature_schedule="cosine",
        min_temperature=0.1
    )

    out_exponential = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        temperature=1.0,
        temperature_schedule="exponential",
        min_temperature=0.1
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)
    assert out_exponential.shape == (batch_size, seq_len + 2)


def test_dynamic_top_k_schedule(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_k=50,
        top_k_schedule="linear",
        min_top_k=5
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_k=50,
        top_k_schedule="cosine",
        min_top_k=5
    )

    out_exponential = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_k=50,
        top_k_schedule="exponential",
        min_top_k=5
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)
    assert out_exponential.shape == (batch_size, seq_len + 2)


def test_dynamic_min_p_schedule(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        min_p=0.9,
        min_p_schedule="linear",
        min_min_p=0.1
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        min_p=0.9,
        min_p_schedule="cosine",
        min_min_p=0.1
    )

    out_exponential = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        min_p=0.9,
        min_p_schedule="exponential",
        min_min_p=0.1
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)
    assert out_exponential.shape == (batch_size, seq_len + 2)


def test_dynamic_top_p_schedule(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_p=0.9,
        top_p_schedule="linear",
        min_top_p=0.1
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_p=0.9,
        top_p_schedule="cosine",
        min_top_p=0.1
    )

    out_exponential = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_p=0.9,
        top_p_schedule="exponential",
        min_top_p=0.1
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)
    assert out_exponential.shape == (batch_size, seq_len + 2)


def test_early_stopping_eos(dummy_model, mocker):
    batch_size = 1
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    eos_token_id = 99
    pad_token_id = 100

    # Force the model to generate eos_token_id as the first generated token
    logit_bias = {eos_token_id: 1e9}

    # max_new_tokens is 4, but due to eos_token_id being generated immediately,
    # the sequence should stop early at the first block (block_length is 2),
    # meaning the output length should be seq_len + block_length (4 + 2 = 6).
    # Since it stopped early and padded, the second token in the block should be pad_token_id.

    out = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=4,
        block_length=2,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        logit_bias=logit_bias
    )

    assert out.shape == (batch_size, seq_len + 2)
    # The first token generated (index 4) should be eos_token_id
    assert out[0, seq_len].item() == eos_token_id
    # The second token generated (index 5) should be pad_token_id
    assert out[0, seq_len + 1].item() == pad_token_id

def test_forced_decoder_ids(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Force token 42 at relative index 0, and token 43 at relative index 1
    forced_decoder_ids = [[0, 42], [1, 43]]

    out = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        forced_decoder_ids=forced_decoder_ids
    )

    assert out.shape == (batch_size, seq_len + 4)
    assert torch.all(out[:, seq_len] == 42).item()
    assert torch.all(out[:, seq_len + 1] == 43).item()

def test_forced_eos_token_id(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    forced_eos_token_id = 99
    max_new_tokens = 4

    out = dummy_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        steps=2,
        forced_eos_token_id=forced_eos_token_id
    )

    assert out.shape == (batch_size, seq_len + max_new_tokens)
    assert torch.all(out[:, -1] == forced_eos_token_id).item()

def test_renormalize_logits(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        renormalize_logits=True
    )

    assert out.shape == (batch_size, seq_len + 2)


def test_dynamic_typical_p_schedule(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        typical_p=0.9,
        typical_p_schedule="linear",
        min_typical_p=0.1
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        typical_p=0.9,
        typical_p_schedule="cosine",
        min_typical_p=0.1
    )

    out_exponential = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        typical_p=0.9,
        typical_p_schedule="exponential",
        min_typical_p=0.1
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)
    assert out_exponential.shape == (batch_size, seq_len + 2)

def test_dynamic_top_a_schedule(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_a=0.9,
        top_a_schedule="linear",
        min_top_a=0.1
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_a=0.9,
        top_a_schedule="cosine",
        min_top_a=0.1
    )

    out_exponential = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_a=0.9,
        top_a_schedule="exponential",
        min_top_a=0.1
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)
    assert out_exponential.shape == (batch_size, seq_len + 2)


def test_dynamic_tfs_z_schedule(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        tfs_z=0.9,
        tfs_z_schedule="linear",
        min_tfs_z=1.0
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        tfs_z=0.9,
        tfs_z_schedule="cosine",
        min_tfs_z=1.0
    )

    out_exponential = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        tfs_z=0.9,
        tfs_z_schedule="exponential",
        min_tfs_z=1.0
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)
    assert out_exponential.shape == (batch_size, seq_len + 2)

def test_dynamic_tkg_schedule(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    uncond_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        tkg_scale=2.0,
        tkg_schedule="linear",
        tkg_min_scale=0.1,
        unconditional_input_ids=uncond_ids,
        top_k=5
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        tkg_scale=2.0,
        tkg_schedule="cosine",
        tkg_min_scale=0.1,
        unconditional_input_ids=uncond_ids,
        top_k=5
    )

    out_exponential = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        tkg_scale=2.0,
        tkg_schedule="exponential",
        tkg_min_scale=0.1,
        unconditional_input_ids=uncond_ids,
        top_k=5
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)
    assert out_exponential.shape == (batch_size, seq_len + 2)

def test_generate_tkg_and_cfg_combined(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    uncond_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_combined = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        cfg_scale=1.5,
        tkg_scale=2.0,
        top_k=10,
        unconditional_input_ids=uncond_ids,
        guidance_rescale=0.7
    )

    assert out_combined.shape == (batch_size, seq_len + 2)

def test_dynamic_epsilon_cutoff_schedule(dummy_model):
    input_ids = torch.tensor([[1, 2, 3]])

    out_linear = dummy_model.generate(
        input_ids, max_new_tokens=2, steps=2,
        epsilon_cutoff=0.5, epsilon_cutoff_schedule="linear", min_epsilon_cutoff=0.1
    )

    out_cosine = dummy_model.generate(
        input_ids, max_new_tokens=2, steps=2,
        epsilon_cutoff=0.5, epsilon_cutoff_schedule="cosine", min_epsilon_cutoff=0.1
    )

    out_exp = dummy_model.generate(
        input_ids, max_new_tokens=2, steps=2,
        epsilon_cutoff=0.5, epsilon_cutoff_schedule="exponential", min_epsilon_cutoff=0.1
    )

    assert out_linear.shape == (1, 5)
    assert out_cosine.shape == (1, 5)
    assert out_exp.shape == (1, 5)

def test_dynamic_eta_cutoff_schedule(dummy_model):
    input_ids = torch.tensor([[1, 2, 3]])

    out_linear = dummy_model.generate(
        input_ids, max_new_tokens=2, steps=2,
        eta_cutoff=0.5, eta_cutoff_schedule="linear", min_eta_cutoff=0.1
    )

    out_cosine = dummy_model.generate(
        input_ids, max_new_tokens=2, steps=2,
        eta_cutoff=0.5, eta_cutoff_schedule="cosine", min_eta_cutoff=0.1
    )

    out_exp = dummy_model.generate(
        input_ids, max_new_tokens=2, steps=2,
        eta_cutoff=0.5, eta_cutoff_schedule="exponential", min_eta_cutoff=0.1
    )

    assert out_linear.shape == (1, 5)
    assert out_cosine.shape == (1, 5)
    assert out_exp.shape == (1, 5)

def test_num_return_sequences_support(dummy_model, mocker):
    """
    Test that generate supports returning multiple sequences per prompt
    using num_return_sequences.
    """
    input_ids = torch.tensor([[1, 2], [3, 4]])

    # Each prompt should return 3 sequences, total 6
    output = dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=4,
        steps=2,
        num_return_sequences=3
    )

    assert output.size(0) == 6
    assert output.size(1) == input_ids.size(1) + 4

def test_encoder_repetition_penalty(dummy_model):
    input_ids = torch.tensor([[1, 2, 3, 4]])
    out_normal = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
    )
    # The normal generation might have repeated tokens depending on random init.
    # To test encoder_repetition_penalty, we force it with a very high penalty.
    out_penalized = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        encoder_repetition_penalty=1000.0
    )
    # The generated tokens (indices 4 to 7) in penalized output should have
    # lower chance of being in [1, 2, 3, 4]. We can't guarantee they are absolutely 0,
    # but let's check basic execution.
    assert out_penalized.shape == (1, 8)


def test_encoder_no_repeat_ngram_size(dummy_model):
    # Prompt is [1, 2, 3, 4]
    input_ids = torch.tensor([[1, 2, 3, 4]])

    # We force the model to generate [1, 2, 3, ...] by passing dummy values if possible,
    # or just checking the execution.
    # The true test of logic would use a specific logit bias. Let's use logit bias
    # to force generation of [1, 2]. Then the next token cannot be 3 if encoder_no_repeat_ngram_size=3
    # Wait, logit bias is per-token.
    # Actually just ensuring it runs without crashing is the minimal test.
    out = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        encoder_no_repeat_ngram_size=2
    )
    assert out.shape == (1, 8)

def test_max_length_min_length(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Test max_length truncating default or specified max_new_tokens
    # If max_length = 6, then max_new_tokens should become 2
    out_max = dummy_model.generate(
        input_ids,
        max_length=6,
        steps=2
    )
    assert out_max.shape == (batch_size, 6)

    # Test min_length expanding missing or smaller min_new_tokens
    # If min_length = 8, min_new_tokens is at least 4
    # With max_length=10, max_new_tokens=6.
    out_min = dummy_model.generate(
        input_ids,
        min_length=8,
        max_length=10,
        steps=2
    )
    assert out_min.shape == (batch_size, 10)

def test_cyclic_schedules_coverage(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    uncond_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Test all parameters with cyclic schedule
    out_cyclic = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        cfg_scale=2.0,
        cfg_schedule="cyclic",
        guidance_rescale=0.7,
        guidance_rescale_schedule="cyclic",
        unconditional_input_ids=uncond_ids,
        top_k=50,
        top_k_schedule="cyclic",
        top_p=0.9,
        top_p_schedule="cyclic",
        min_p=0.1,
        min_p_schedule="cyclic",
        top_a=0.9,
        top_a_schedule="cyclic",
        typical_p=0.9,
        typical_p_schedule="cyclic",
        tfs_z=0.9,
        tfs_z_schedule="cyclic",
        epsilon_cutoff=0.1,
        epsilon_cutoff_schedule="cyclic",
        eta_cutoff=0.1,
        eta_cutoff_schedule="cyclic",
        tkg_scale=2.0,
        tkg_schedule="cyclic",
        temperature=1.0,
        temperature_schedule="cyclic",
        repetition_penalty=1.2,
        repetition_penalty_schedule="cyclic",
        frequency_penalty=0.5,
        frequency_penalty_schedule="cyclic",
        presence_penalty=0.5,
        presence_penalty_schedule="cyclic",
        xtc_threshold=0.5,
        xtc_threshold_schedule="cyclic",
        xtc_probability=0.5,
        xtc_probability_schedule="cyclic"
    )

    assert out_cyclic.shape == (batch_size, seq_len + 2)
