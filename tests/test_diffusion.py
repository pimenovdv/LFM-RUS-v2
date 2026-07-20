
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

def test_epsilon_cutoff_sampling(mocker):
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

    # Run epsilon_cutoff sampling
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, epsilon_cutoff=0.1)

    assert res.shape == (1, 4)

def test_eta_cutoff_sampling(mocker):
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

    # Run eta_cutoff sampling
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, eta_cutoff=0.1)

    assert res.shape == (1, 4)

def test_tfs_sampling(mocker):
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
    logits[0, :, 0] = 10.0
    logits[0, :, 1] = 9.0
    logits[0, :, 2] = 5.0
    logits[0, :, 3] = 4.0
    logits[0, :, 4] = 4.0
    mock_out.logits = logits
    model.forward = mocker.MagicMock(return_value=mock_out)

    input_ids = torch.tensor([[1, 2]])

    # Run tfs sampling
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, tfs_z=0.5)

    assert res.shape == (1, 4)

def test_dynamic_entropy_temperature(mocker):
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
    # Mix of high and low entropy logits
    logits[0, 2, :] = 1.0 # High entropy
    logits[0, 3, 0] = 10.0 # Low entropy
    mock_out.logits = logits
    model.forward = mocker.MagicMock(return_value=mock_out)

    input_ids = torch.tensor([[1, 2]])

    # Run dynamic entropy temperature sampling
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, temperature=0.0, dynamic_temperature_entropy=1.0)

    assert res.shape == (1, 4)

def test_generation_parameters(mocker):
    # Test top_k, top_p, min_p, temperature schedules, cfg_scale, penalties
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

    # Test top_k, top_p, min_p
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, top_k=2, top_p=0.9, min_p=0.1)
    assert res.shape == (1, 4)

    # Test repetition_penalty, frequency_penalty, presence_penalty
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, repetition_penalty=1.2, frequency_penalty=0.5, presence_penalty=0.5)
    assert res.shape == (1, 4)

    # Test temperature schedules
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, temperature=1.0, temperature_schedule="linear")
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, temperature=1.0, temperature_schedule="cosine")
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, temperature=1.0, temperature_schedule="exponential")
    assert res.shape == (1, 4)

    # Test CFG
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, cfg_scale=1.5, guidance_rescale=0.5, unconditional_input_ids=torch.tensor([[0, 0]]))
    assert res.shape == (1, 4)

    # Test suppress_tokens and logit_bias
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, suppress_tokens=[0], logit_bias={1: 2.0})
    assert res.shape == (1, 4)

    # Test early stopping
    res = model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, eos_token_id=0, pad_token_id=0)


def test_diffusion_model_misc_features(mocker):
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()

    class FakeConfig:
        is_causal = True
        model_type = "gpt2"

    mock_inner.config = FakeConfig()

    mock_emb = mocker.MagicMock()
    mock_emb.weight = torch.nn.Parameter(torch.rand(10, 12))
    mock_inner.get_input_embeddings.return_value = mock_emb

    mock_auto_model.from_config.return_value = mock_inner

    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=True)
    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 10}, timestep_dim=8, mask_token_id=0)
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)

    # Test disable causal mask
    model._disable_causal_mask()

    # Test get_input_embeddings / set_input_embeddings
    model.get_input_embeddings()
    model.set_input_embeddings(mock_emb)

    # Test forward with input_ids and labels
    mock_out = mocker.MagicMock()
    mock_out.last_hidden_state = torch.rand((2, 5, 12))
    def side_effect(*args, **kwargs):
        if 'return_dict' in kwargs:
            del kwargs['return_dict']
        return mock_out
    mock_inner.side_effect = side_effect

    labels = torch.randint(0, 10, (2, 5))
    model(input_ids=torch.tensor([[1,2,3,4,5], [6,7,8,9,0]]), labels=labels, return_dict=False)

    # Test generate with cfg_schedules
    mock_out.logits = torch.rand((1, 4, 10))
    model.forward = mocker.MagicMock(return_value=mock_out)
    input_ids = torch.tensor([[1, 2]])
    model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, cfg_scale=1.5, cfg_schedule="linear", unconditional_input_ids=torch.tensor([[0, 0]]))
    model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, cfg_scale=1.5, cfg_schedule="cosine", unconditional_input_ids=torch.tensor([[0, 0]]))
    model.generate(input_ids, max_new_tokens=2, steps=1, block_length=2, cfg_scale=1.5, cfg_schedule="exponential", unconditional_input_ids=torch.tensor([[0, 0]]))

def test_steering_vector_capture(mocker):
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner

    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)
    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 10}, timestep_dim=8)
    model = DiffusionModelForConditionalGeneration(config)

    # Use real layer for steering
    mock_layer = torch.nn.Linear(12, 12)
    model.inner_model.named_modules = mocker.MagicMock(return_value=[("layer1", mock_layer)])

    # Actually trigger the hook correctly
    def mock_forward(*args, **kwargs):
        # Forward pass through the layer to trigger the hook
        mock_layer(torch.rand((1, 5, 12)))

    model.forward = mocker.MagicMock(side_effect=mock_forward)

    vec = model.get_steering_vector(torch.tensor([[1, 2]]), torch.tensor([[3, 4]]), "layer1")
    assert vec.shape == (1, 12)

def test_min_new_tokens(mocker):
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner

    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)
    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 10}, timestep_dim=8, mask_token_id=0, max_timesteps=10, block_size=2, diffusion_steps=2)
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)

    input_ids = torch.ones((1, 2), dtype=torch.long)
    # block_size is 2, min_new_tokens=4, eos_token_id=5
    # the generator logic will set logit for eos_token_id to -inf

    # Mock forward on the class level to bypass torch.nn.Module.__call__ logic
    mock_forward = mocker.patch.object(DiffusionModelForConditionalGeneration, 'forward')
    # Because x grows, the model receives inputs of length up to input_len + block_size * iter.
    # max_new_tokens=4, block_size=2. input_ids is 1x2. So x will be 1x6.
    logits = torch.randn(1, 6, 10)
    logits[:, :, 5] = 100.0
    mock_output = mocker.MagicMock()
    mock_output.logits = logits
    mock_forward.return_value = mock_output

    # Test with min_new_tokens
    res_min_new = model.generate(input_ids, max_new_tokens=4, min_new_tokens=4, eos_token_id=5)

    assert not (res_min_new == 5).any()

def test_xtc_sampling(mocker):
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner

    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)
    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 10}, timestep_dim=8, mask_token_id=0, max_timesteps=10, block_size=2, diffusion_steps=2)
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)

    input_ids = torch.ones((1, 2), dtype=torch.long)

    mock_forward = mocker.patch.object(DiffusionModelForConditionalGeneration, 'forward')
    mocker.patch('torch.rand_like', return_value=torch.zeros((1, 4, 10)))
    logits = torch.zeros(1, 4, 10)
    logits[:, :, 2] = 50.0  # Top token is 2
    logits[:, :, 3] = 10.0  # Second is 3
    mock_output = mocker.MagicMock()
    mock_output.logits = logits
    mock_forward.return_value = mock_output

    # xtc_threshold=0.5, xtc_probability=1.0 -> should always drop the top token
    # input_ids is 1x2, max_new_tokens=2 -> total length 4
    res = model.generate(input_ids, max_new_tokens=2, xtc_threshold=0.5, xtc_probability=1.0)

    assert not (res == 2).any()
    assert (res[:, 2:] == 3).all()

def test_no_repeat_ngram_size(mocker):
    # Setup mock to simulate model unmasking and n-gram penalty
    from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration
    from src.models.diffusion.configuration_diffusion import DiffusionConfig

    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner
    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)

    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 10}, timestep_dim=8, mask_token_id=0, max_timesteps=10, block_size=1, diffusion_steps=2)
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)

    input_ids = torch.tensor([[1, 2, 3, 1, 2]])
    mask_id = config.mask_token_id

    with torch.no_grad():
        mock_forward = mocker.patch.object(DiffusionModelForConditionalGeneration, 'forward')
        vocab_size = 10

        def forward_side_effect(*args, **kwargs):
            input_seq = kwargs.get('input_ids', args[0] if args else None)
            seq_len = input_seq.shape[1]
            out_logits = torch.zeros(1, seq_len, vocab_size)
            out_logits[:, -1, 3] = 10.0
            out_logits[:, -1, 4] = 5.0

            mock_out = mocker.MagicMock()
            mock_out.logits = out_logits
            return mock_out

        mock_forward.side_effect = forward_side_effect

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            block_length=1,
            steps=1,
            no_repeat_ngram_size=2
        )

        assert outputs[0, -1].item() != 3
        assert outputs[0, -1].item() == 4

def test_bad_words_ids(mocker):
    # Setup mock to simulate model unmasking and bad_words_ids penalty
    from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration
    from src.models.diffusion.configuration_diffusion import DiffusionConfig

    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner
    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)

    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 10}, timestep_dim=8, mask_token_id=0, max_timesteps=10, block_size=1, diffusion_steps=2)
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)

    input_ids = torch.tensor([[1, 2]])

    with torch.no_grad():
        mock_forward = mocker.patch.object(DiffusionModelForConditionalGeneration, 'forward')
        vocab_size = 10

        def forward_side_effect(*args, **kwargs):
            input_seq = kwargs.get('input_ids', args[0] if args else None)
            seq_len = input_seq.shape[1]
            out_logits = torch.zeros(1, seq_len, vocab_size)
            # Give high probability to token 3
            out_logits[:, -1, 3] = 10.0
            # Next best is token 4
            out_logits[:, -1, 4] = 5.0

            mock_out = mocker.MagicMock()
            mock_out.logits = out_logits
            return mock_out

        mock_forward.side_effect = forward_side_effect

        bad_words_ids = [[1, 2, 3]]

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            block_length=1,
            steps=1,
            bad_words_ids=bad_words_ids
        )

        # The sequence is [1, 2, <new token>].
        # If bad_words_ids is [[1, 2, 3]], then token 3 should be blocked since prefix [1, 2] matches.
        # So we expect it to fallback to 4.
        assert outputs[0, -1].item() != 3
        assert outputs[0, -1].item() == 4

def test_bad_words_ids_single_token(mocker):
    # Setup mock to simulate model unmasking and bad_words_ids penalty (single token)
    from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration
    from src.models.diffusion.configuration_diffusion import DiffusionConfig

    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()
    mock_auto_model.from_config.return_value = mock_inner
    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)

    config = DiffusionConfig(base_config_dict={"hidden_size": 12, "vocab_size": 10}, timestep_dim=8, mask_token_id=0, max_timesteps=10, block_size=1, diffusion_steps=2)
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)

    input_ids = torch.tensor([[1, 2]])

    with torch.no_grad():
        mock_forward = mocker.patch.object(DiffusionModelForConditionalGeneration, 'forward')
        vocab_size = 10

        def forward_side_effect(*args, **kwargs):
            input_seq = kwargs.get('input_ids', args[0] if args else None)
            seq_len = input_seq.shape[1]
            out_logits = torch.zeros(1, seq_len, vocab_size)
            # Give high probability to token 3
            out_logits[:, -1, 3] = 10.0
            # Next best is token 4
            out_logits[:, -1, 4] = 5.0

            mock_out = mocker.MagicMock()
            mock_out.logits = out_logits
            return mock_out

        mock_forward.side_effect = forward_side_effect

        # 3 is blocked
        bad_words_ids = [[3], []]

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            block_length=1,
            steps=1,
            bad_words_ids=bad_words_ids
        )

        assert outputs[0, -1].item() != 3
        assert outputs[0, -1].item() == 4

def test_max_time(mocker):
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)

    config = DiffusionConfig(mask_token_id=0, diffusion_steps=4, block_size=2, vocab_size=10, base_config_dict={"hidden_size": 12, "vocab_size": 10})
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)
    model.eval()

    input_ids = torch.tensor([[1, 2]])

    mock_forward = mocker.patch.object(model, "forward")

    # Mock to slow down generation loop
    call_count = 0
    def forward_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Each step we advance "time" using mocker
        return mocker.MagicMock(logits=torch.zeros(1, kwargs["input_ids"].shape[1], 10))

    mock_forward.side_effect = forward_side_effect

    import time
    mock_time = mocker.patch("time.time")

    # First call is start time, subsequent calls increase time
    time_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    def time_side_effect():
        if time_values:
            return time_values.pop(0)
        return 100.0
    mock_time.side_effect = time_side_effect

    # We ask for 4 tokens (2 blocks), 2 steps per block
    # Max time is 1.5. So it should break early in the first block
    # or between steps.
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=4,
        steps=4,
        max_time=1.5
    )

    # Because max_time is reached very quickly in mock time,
    # it should call forward fewer times than (blocks * steps) = 2 * 2 = 4
    assert call_count < 4

def test_remove_invalid_values(mocker):
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)

    config = DiffusionConfig(mask_token_id=0, diffusion_steps=1, block_size=1, vocab_size=10, base_config_dict={"hidden_size": 12, "vocab_size": 10})
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)
    model.eval()

    input_ids = torch.tensor([[1, 2]])

    mock_forward = mocker.patch.object(model, "forward")

    def forward_side_effect(*args, **kwargs):
        seq_len = kwargs["input_ids"].shape[1]
        out_logits = torch.zeros(1, seq_len, 10)
        # Add NaNs and Infs
        out_logits[0, -1, 0] = float("nan")
        out_logits[0, -1, 1] = float("inf")
        out_logits[0, -1, 2] = -float("inf")
        out_logits[0, -1, 3] = 10.0 # Valid high logit

        mock_out = mocker.MagicMock()
        mock_out.logits = out_logits
        return mock_out

    mock_forward.side_effect = forward_side_effect

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=1,
        remove_invalid_values=True
    )

    # Because remove_invalid_values handles NaNs and Infs, generate shouldn't crash.
    # Token 1 was set to inf, so its logit gets set to torch.finfo(float32).max.
    # This becomes the highest value and gets selected by argmax sampling (since temperature is 0).
    assert outputs[0, -1].item() == 1
