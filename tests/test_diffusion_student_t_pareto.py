import pytest
import torch
import torch.nn.functional as F
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager, MDLMRequest
from src.models.diffusion.configuration_diffusion import DiffusionConfig
import math


@pytest.fixture
def model_and_config():
    config = DiffusionConfig(
        vocab_size=100,
        base_config_dict={"model_type": "gpt2", "vocab_size": 100},
        mask_token_id=0,
        pad_token_id=1,
    )

    model = DiffusionModelForConditionalGeneration(config)

    # Mocking self.forward logic inside generate to just return a dummy tensor
    # where all logits are random to test the sampling methods.
    def mock_forward(input_ids, **kwargs):
        class Output:
            logits = torch.randn((input_ids.shape[0], input_ids.shape[1], config.vocab_size))
        return Output()

    model.forward = mock_forward
    return model, config


def test_student_t_and_pareto_noise_in_generate(mocker, model_and_config):
    model, config = model_and_config
    input_ids = torch.full((1, 5), config.mask_token_id)

    # We must patch so that the mock sample shape matches the logits shape exactly during generation
    # The generation loop adds tokens so length is changing.
    mock_student_t_sample = mocker.patch("torch.distributions.studentT.StudentT.sample")
    def mock_student_t_sampler(shape):
        return torch.ones(shape)
    mock_student_t_sample.side_effect = mock_student_t_sampler

    mock_pareto_sample = mocker.patch("torch.distributions.pareto.Pareto.sample")
    def mock_pareto_sampler(shape):
        return torch.ones(shape)
    mock_pareto_sample.side_effect = mock_pareto_sampler

    # Mocking self.forward logic inside generate to handle correct input_ids shape during generation
    def mock_forward(input_ids, **kwargs):
        class Output:
            logits = torch.randn((input_ids.shape[0], input_ids.shape[1], config.vocab_size))
        return Output()

    model.forward = mock_forward

    # Test Student-T noise
    model.generate(
        input_ids,
        student_t_noise_scale=0.5,
        student_t_df=2.0,
        student_t_noise_schedule="constant",
        steps_per_block=1,
        max_new_tokens=5
    )
    assert mock_student_t_sample.called
    assert not mock_pareto_sample.called

    mock_student_t_sample.reset_mock()
    mock_pareto_sample.reset_mock()

    # Test Pareto noise
    model.generate(
        input_ids,
        pareto_noise_scale=0.5,
        pareto_alpha=2.0,
        pareto_noise_schedule="constant",
        steps_per_block=1,
        max_new_tokens=5
    )
    assert not mock_student_t_sample.called
    assert mock_pareto_sample.called


def test_student_t_and_pareto_noise_in_continuous_batching(mocker, model_and_config):
    model, config = model_and_config

    def mock_forward(input_ids, **kwargs):
        class Output:
            logits = torch.randn((input_ids.shape[0], input_ids.shape[1], config.vocab_size))
        return Output()

    model.forward = mock_forward

    manager = MDLMContinuousBatchingManager(model)

    # Add requests with student_t and pareto noise
    manager.add_request(
        input_ids=torch.full((1, 5), config.mask_token_id),
        max_new_tokens=5,
        total_steps=1,
        student_t_noise_scale=0.5,
        student_t_df=2.0,
    )
    manager.add_request(
        input_ids=torch.full((1, 5), config.mask_token_id),
        max_new_tokens=5,
        total_steps=1,
        pareto_noise_scale=0.5,
        pareto_alpha=2.0,
    )

    mock_student_t_sample = mocker.patch("torch.distributions.studentT.StudentT.sample")
    def mock_student_t_sampler(shape):
        return torch.ones(shape)
    mock_student_t_sample.side_effect = mock_student_t_sampler

    mock_pareto_sample = mocker.patch("torch.distributions.pareto.Pareto.sample")
    def mock_pareto_sampler(shape):
        return torch.ones(shape)
    mock_pareto_sample.side_effect = mock_pareto_sampler

    manager.step()

    # Because req1 has student_t_noise_scale > 0 and req2 has pareto_noise_scale > 0,
    # both distribution samplers should be called
    assert mock_student_t_sample.called
    assert mock_pareto_sample.called

def test_student_t_and_pareto_schedules(mocker, model_and_config):
    model, config = model_and_config
    input_ids = torch.full((1, 5), config.mask_token_id)

    mock_student_t_sample = mocker.patch("torch.distributions.studentT.StudentT.sample", side_effect=lambda shape: torch.ones(shape))
    mock_pareto_sample = mocker.patch("torch.distributions.pareto.Pareto.sample", side_effect=lambda shape: torch.ones(shape))

    schedules = ["linear", "cosine", "exponential", "cyclic"]

    for schedule in schedules:
        model.generate(
            input_ids,
            student_t_noise_scale=0.5,
            student_t_noise_schedule=schedule,
            pareto_noise_scale=0.5,
            pareto_noise_schedule=schedule,
            steps_per_block=1,
            max_new_tokens=2
        )
        assert mock_student_t_sample.called
        assert mock_pareto_sample.called
        mock_student_t_sample.reset_mock()
        mock_pareto_sample.reset_mock()
