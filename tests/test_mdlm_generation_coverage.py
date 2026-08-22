import pytest
import torch
from unittest.mock import MagicMock
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, get_num_transfer_tokens
from src.models.diffusion.configuration_diffusion import DiffusionConfig

@pytest.fixture
def mock_diffusion_model():
    config = DiffusionConfig(
        mask_token_id=0,
        max_timesteps=1000,
        base_config_dict={"vocab_size": 100},
        diffusion_steps=5,
        block_size=5,
        remasking_strategy="confidence"
    )
    model = MagicMock(spec=DiffusionModelForConditionalGeneration)
    model.config = config
    model.device = torch.device("cpu")

    # Mocking the forward pass
    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get('input_ids', args[0] if len(args) > 0 else None)
        batch_size, seq_len = input_ids.shape
        vocab_size = config.base_config_dict["vocab_size"]

        # Return random logits
        logits = torch.randn((batch_size, seq_len, vocab_size), device=input_ids.device)

        class Output:
            pass
        out = Output()
        out.logits = logits
        return out

    model.side_effect = mock_forward
    model.__call__ = mock_forward
    model.generate = DiffusionModelForConditionalGeneration.generate.__get__(model)
    return model

def test_generate_basic(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2, 3]])
    output = mock_diffusion_model.generate(input_ids, max_new_tokens=5, steps=2)
    assert output.shape == (1, 8)

def test_generate_missing_mask_id(mock_diffusion_model):
    mock_diffusion_model.config.mask_token_id = None
    input_ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="mask_token_id must be set"):
        mock_diffusion_model.generate(input_ids, max_new_tokens=5)

def test_generate_batch_multiplier(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    uncond_ids = torch.tensor([[4, 5, 6]])
    output = mock_diffusion_model.generate(
        input_ids,
        max_new_tokens=5,
        num_return_sequences=2,
        attention_mask=attention_mask,
        unconditional_input_ids=uncond_ids,
        cfg_scale=2.0
    )
    assert output.shape == (2, 8)

def test_generate_max_length_logic(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2, 3]])
    # T = 3, max_length = 5 => max_new = 2
    output = mock_diffusion_model.generate(input_ids, max_length=5)
    assert output.shape == (1, 5)

def test_generate_max_length_with_new_tokens(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2, 3]])
    # min of (10, 8 - 3) -> 5
    output = mock_diffusion_model.generate(input_ids, max_length=8, max_new_tokens=10)
    assert output.shape == (1, 8)

def test_get_num_transfer_tokens_schedules():
    mask_index = torch.tensor([[True, True, True, True]])

    # Cosine
    transfer_cosine = get_num_transfer_tokens(mask_index, 4, schedule="cosine")
    assert transfer_cosine.shape == (1, 4)

    # Square
    transfer_square = get_num_transfer_tokens(mask_index, 4, schedule="square")
    assert transfer_square.shape == (1, 4)

    # Exponential
    transfer_exp = get_num_transfer_tokens(mask_index, 4, schedule="exponential")
    assert transfer_exp.shape == (1, 4)

    # Sigmoid
    transfer_sig = get_num_transfer_tokens(mask_index, 4, schedule="sigmoid")
    assert transfer_sig.shape == (1, 4)

def test_generate_error_handling(mock_diffusion_model):
    mock_diffusion_model.config.mask_token_id = None
    input_ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="mask_token_id must be set"):
        mock_diffusion_model.generate(input_ids)

    mock_diffusion_model.config.mask_token_id = 0

def test_generate_negative_prompt_fallback(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    neg_prompt = torch.tensor([[5, 6]])
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, num_return_sequences=2)
    assert out.shape == (2, 4)

def test_generate_bad_words(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    # T=2, max_new=2 => steps=5 => loops
    # Add dummy bad_words
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, bad_words_ids=[[10]])
    assert out.shape == (1, 4)

def test_generate_remasking_entropy(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, remasking="entropy")
    assert out.shape == (1, 4)

def test_generate_classifier_guidance(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    def mock_classifier(logits, ids, timesteps):
        return logits.sum()

    out = mock_diffusion_model.generate(
        input_ids,
        max_new_tokens=2,
        classifier_fn=mock_classifier,
        classifier_scale=1.0,
        classifier_schedule="linear"
    )
    assert out.shape == (1, 4)

def test_generate_length_penalty(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(
        input_ids,
        max_new_tokens=2,
        length_penalty=1.5,
        length_penalty_schedule="linear"
    )
    assert out.shape == (1, 4)

def test_generate_tfs_z(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(
        input_ids,
        max_new_tokens=2,
        tfs_z=0.5,
        tfs_z_schedule="linear"
    )
    assert out.shape == (1, 4)

def test_generate_xtc(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(
        input_ids,
        max_new_tokens=2,
        xtc_threshold=0.5,
        xtc_probability=0.5
    )
    assert out.shape == (1, 4)

def test_generate_logit_bias(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(
        input_ids,
        max_new_tokens=2,
        logit_bias={10: 5.0}
    )
    assert out.shape == (1, 4)

def test_generate_beam_search(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(
        input_ids,
        max_new_tokens=2,
        num_beams=2
    )
    assert out.shape == (1, 4)

def test_generate_cfg_schedules(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    neg_prompt = torch.tensor([[5, 6]])
    # Test linear
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, cfg_scale=2.0, cfg_schedule="linear")
    assert out.shape == (1, 4)
    # Test cosine
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, cfg_scale=2.0, cfg_schedule="cosine")
    assert out.shape == (1, 4)
    # Test exponential
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, cfg_scale=2.0, cfg_schedule="exponential")
    assert out.shape == (1, 4)
    # Test cyclic
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, cfg_scale=2.0, cfg_schedule="cyclic")
    assert out.shape == (1, 4)

def test_generate_tkg_schedules(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    neg_prompt = torch.tensor([[5, 6]])
    # Test linear
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, tkg_scale=2.0, top_k=10, tkg_schedule="linear")
    assert out.shape == (1, 4)
    # Test cosine
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, tkg_scale=2.0, top_k=10, tkg_schedule="cosine")
    assert out.shape == (1, 4)
    # Test exponential
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, tkg_scale=2.0, top_k=10, tkg_schedule="exponential")
    assert out.shape == (1, 4)
    # Test cyclic
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, tkg_scale=2.0, top_k=10, tkg_schedule="cyclic")
    assert out.shape == (1, 4)

def test_generate_guidance_rescale_schedules(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    neg_prompt = torch.tensor([[5, 6]])
    # Test linear
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, cfg_scale=2.0, guidance_rescale=0.5, guidance_rescale_schedule="linear")
    assert out.shape == (1, 4)
    # Test cosine
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, cfg_scale=2.0, guidance_rescale=0.5, guidance_rescale_schedule="cosine")
    assert out.shape == (1, 4)
    # Test exponential
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, cfg_scale=2.0, guidance_rescale=0.5, guidance_rescale_schedule="exponential")
    assert out.shape == (1, 4)
    # Test cyclic
    out = mock_diffusion_model.generate(input_ids, negative_prompt_ids=neg_prompt, max_new_tokens=2, cfg_scale=2.0, guidance_rescale=0.5, guidance_rescale_schedule="cyclic")
    assert out.shape == (1, 4)

def test_generate_repetition_penalty(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, repetition_penalty=2.0, repetition_penalty_schedule="linear")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, repetition_penalty=2.0, repetition_penalty_schedule="cosine")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, repetition_penalty=2.0, repetition_penalty_schedule="exponential")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, repetition_penalty=2.0, repetition_penalty_schedule="cyclic")
    assert out.shape == (1, 4)

def test_generate_frequency_presence_penalty(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, frequency_penalty=2.0, presence_penalty=2.0, frequency_penalty_schedule="linear", presence_penalty_schedule="linear")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, frequency_penalty=2.0, presence_penalty=2.0, frequency_penalty_schedule="cosine", presence_penalty_schedule="cosine")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, frequency_penalty=2.0, presence_penalty=2.0, frequency_penalty_schedule="exponential", presence_penalty_schedule="exponential")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, frequency_penalty=2.0, presence_penalty=2.0, frequency_penalty_schedule="cyclic", presence_penalty_schedule="cyclic")
    assert out.shape == (1, 4)

def test_generate_dynamic_temperature_entropy(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, temperature=1.0, dynamic_temperature_entropy=0.5, dynamic_temperature_entropy_schedule="linear")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, temperature=1.0, dynamic_temperature_entropy=0.5, dynamic_temperature_entropy_schedule="cosine")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, temperature=1.0, dynamic_temperature_entropy=0.5, dynamic_temperature_entropy_schedule="exponential")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, temperature=1.0, dynamic_temperature_entropy=0.5, dynamic_temperature_entropy_schedule="cyclic")
    assert out.shape == (1, 4)

def test_generate_top_k_schedules(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_k=10, top_k_schedule="linear")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_k=10, top_k_schedule="cosine")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_k=10, top_k_schedule="exponential")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_k=10, top_k_schedule="cyclic")
    assert out.shape == (1, 4)

def test_generate_top_p_schedules(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_p=0.8, top_p_schedule="linear")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_p=0.8, top_p_schedule="cosine")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_p=0.8, top_p_schedule="exponential")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_p=0.8, top_p_schedule="cyclic")
    assert out.shape == (1, 4)

def test_generate_typical_p_schedules(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, typical_p=0.8, typical_p_schedule="linear")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, typical_p=0.8, typical_p_schedule="cosine")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, typical_p=0.8, typical_p_schedule="exponential")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, typical_p=0.8, typical_p_schedule="cyclic")
    assert out.shape == (1, 4)

def test_generate_top_a_schedules(mock_diffusion_model):
    input_ids = torch.tensor([[1, 2]])
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_a=0.5, top_a_schedule="linear")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_a=0.5, top_a_schedule="cosine")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_a=0.5, top_a_schedule="exponential")
    assert out.shape == (1, 4)
    out = mock_diffusion_model.generate(input_ids, max_new_tokens=2, top_a=0.5, top_a_schedule="cyclic")
    assert out.shape == (1, 4)
