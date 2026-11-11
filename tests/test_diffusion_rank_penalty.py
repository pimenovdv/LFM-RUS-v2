import torch
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager
from src.models.diffusion.configuration_diffusion import DiffusionConfig

def test_rank_penalty_generate(mocker):
    config_dict = {"vocab_size": 100, "hidden_size": 32, "mask_token_id": 99, "base_config_dict": {"model_type": "gpt2", "vocab_size": 100}}
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(*args, **kwargs):
        class Output:
            # Predict monotonically decreasing logits so that indices are their ranks
            logits = torch.linspace(10, 1, steps=100).unsqueeze(0).unsqueeze(0)
            last_hidden_state = torch.randn(1, 4, 32)
        return Output()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    # Run without rank penalty
    torch.manual_seed(42)
    out_no_penalty = model.generate(
        input_ids,
        rank_penalty=0.0,
        max_new_tokens=1,
    )

    # Run with rank penalty
    torch.manual_seed(42)
    out_with_penalty = model.generate(
        input_ids,
        rank_penalty=2.0,
        max_new_tokens=1,
    )

    assert out_with_penalty is not None
    assert out_no_penalty is not None

def test_rank_penalty_schedules(mocker):
    config_dict = {"vocab_size": 100, "hidden_size": 32, "mask_token_id": 99, "base_config_dict": {"model_type": "gpt2", "vocab_size": 100}}
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(*args, **kwargs):
        class Output:
            logits = torch.randn(1, 4, 100)
            last_hidden_state = torch.randn(1, 4, 32)
        return Output()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    schedules = ["constant", "linear", "cosine", "exponential", "cyclic"]
    for sched in schedules:
        out = model.generate(
            input_ids,
            rank_penalty=2.0,
            min_rank_penalty=0.5,
            rank_penalty_schedule=sched,
            max_new_tokens=1,
        )
        assert out is not None

def test_rank_penalty_continuous_batching(mocker):
    config_dict = {"vocab_size": 100, "hidden_size": 32, "mask_token_id": 99, "base_config_dict": {"model_type": "gpt2", "vocab_size": 100}}
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    manager = MDLMContinuousBatchingManager(model)

    manager.add_request(input_ids=torch.tensor([[1,2,3]]), max_new_tokens=5, total_steps=10, rank_penalty=1.0)
    manager.add_request(input_ids=torch.tensor([[4,5]]), max_new_tokens=5, total_steps=10, rank_penalty=2.0, min_rank_penalty=0.1, rank_penalty_schedule="linear")

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is None:
            input_ids = kwargs.get('inputs_embeds')
        batch_size, seq_len = input_ids.shape[:2]
        class Output:
            logits = torch.randn(batch_size, seq_len, 100)
        return Output()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    for _ in range(5):
        manager.step()
        if len(manager.active_requests) == 0:
            break
