
import pytest
import torch
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, get_num_transfer_tokens, filter_special_tokens

def test_diffusion_config():
    config = DiffusionConfig(mask_token_id=0, diffusion_steps=1000)
    assert config.mask_token_id == 0
    assert config.diffusion_steps == 1000
    assert config.block_size == 64

def test_get_num_transfer_tokens():
    mask_index = torch.ones((2, 10), dtype=torch.bool)
    mask_index[0, 5:] = False
    # batch 0 has 5 True, batch 1 has 10 True
    # steps = 3
    res = get_num_transfer_tokens(mask_index, 3)
    assert res.shape == (2, 3)
    assert res[0].sum() == 5
    assert res[1].sum() == 10

def test_filter_special_tokens(mocker):
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.bos_token_id = 1
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.pad_token_id = 3

    tokens = torch.tensor([[1, 4, 2, 5, 3]])
    res = filter_special_tokens(tokens, mock_tokenizer, mask_id=0)

    # 1, 2, 3 should be replaced by 0
    assert res[0, 0] == 0
    assert res[0, 1] == 4
    assert res[0, 2] == 0
    assert res[0, 3] == 5
    assert res[0, 4] == 0

def test_diffusion_model_init(mocker):
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")


    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)
    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 100}, timestep_dim=8)


    model = DiffusionModelForConditionalGeneration(config)
    # Give lm_head a real weight so F.linear doesn't fail
    model.lm_head = torch.nn.Linear(12, 100, bias=False)


    assert model.timestep_embedder is not None
    assert model.lm_head is not None

def test_diffusion_model_forward(mocker):
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")

    # Mocking behavior
    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner


    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)
    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 100}, timestep_dim=8)


    model = DiffusionModelForConditionalGeneration(config)
    # Give lm_head a real weight so F.linear doesn't fail
    model.lm_head = torch.nn.Linear(12, 100, bias=False)


    # Fake inner model return
    mock_out = mocker.MagicMock()
    mock_out.last_hidden_state = torch.rand((2, 5, 12))

    def side_effect(*args, **kwargs):
        if 'return_dict' in kwargs:
            del kwargs['return_dict']
        return mock_out
    mock_inner.side_effect = side_effect


    inputs_embeds = torch.rand((2, 5, 12))
    timesteps = torch.tensor([10, 20])

    out = model(inputs_embeds=inputs_embeds, timesteps=timesteps)
    assert getattr(out, 'logits', None) is not None
    assert out.logits.shape == (2, 5, 100)

def test_diffusion_model_generate(mocker):
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")

    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner


    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)
    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 100}, timestep_dim=8, mask_token_id=0, max_timesteps=10)


    model = DiffusionModelForConditionalGeneration(config)
    # Give lm_head a real weight so F.linear doesn't fail
    model.lm_head = torch.nn.Linear(12, 100, bias=False)


    mock_out = mocker.MagicMock()
    mock_out.logits = torch.rand((2, 5, 100))
    model.forward = mocker.MagicMock(return_value=mock_out)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    res = model.generate(input_ids, max_new_tokens=2, steps=2, block_length=2)
    assert res.shape == (2, 5)

def test_typical_p_sampling(mocker):
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner

    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)
    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 10}, timestep_dim=8, mask_token_id=0, max_timesteps=10)
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)

    # Setup controlled logits output
    mock_out = mocker.MagicMock()
    # Logits where some probabilities are much higher
    logits = torch.zeros((1, 4, 10))
    logits[0, :, 0] = 10.0 # High prob token
    logits[0, :, 1] = 5.0
    logits[0, :, 2] = -5.0
    mock_out.logits = logits
    model.forward = mocker.MagicMock(return_value=mock_out)

    input_ids = torch.tensor([[1, 2]])

    # Run typical_p sampling
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, typical_p=0.5)

    # Just checking it doesn't crash and returns the correct shape
    assert res.shape == (1, 4)

def test_top_a_sampling(mocker):
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner

    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)
    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 10}, timestep_dim=8, mask_token_id=0, max_timesteps=10)
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)

    mock_out = mocker.MagicMock()
    logits = torch.zeros((1, 4, 10))
    logits[0, :, 0] = 5.0
    logits[0, :, 1] = 4.0
    logits[0, :, 2] = -5.0
    mock_out.logits = logits
    model.forward = mocker.MagicMock(return_value=mock_out)

    input_ids = torch.tensor([[1, 2]])

    # Run top_a sampling
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, top_a=0.5)

    # Checking it runs correctly
    assert res.shape == (1, 4)
