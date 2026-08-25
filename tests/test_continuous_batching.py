import pytest
import torch
from unittest.mock import MagicMock

from src.models.diffusion.modeling_diffusion import MDLMContinuousBatchingManager, MDLMRequest, DiffusionModelForConditionalGeneration
from src.models.diffusion.configuration_diffusion import DiffusionConfig

@pytest.fixture
def mock_diffusion_model():
    config = DiffusionConfig(
        mask_token_id=0,
        max_timesteps=1000,
        base_config_dict={"vocab_size": 100}
    )
    model = MagicMock(spec=DiffusionModelForConditionalGeneration)
    model.config = config

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
    return model

def test_batching_manager_init(mock_diffusion_model):
    manager = MDLMContinuousBatchingManager(mock_diffusion_model, max_batch_size=4)
    assert manager.max_batch_size == 4
    assert len(manager.pending_requests) == 0
    assert len(manager.active_requests) == 0
    assert len(manager.completed_requests) == 0

def test_add_request(mock_diffusion_model):
    manager = MDLMContinuousBatchingManager(mock_diffusion_model)
    input_ids = torch.tensor([[1, 2, 3]])
    req_id = manager.add_request(input_ids, max_new_tokens=10, total_steps=5)

    assert len(manager.pending_requests) == 1
    assert manager.pending_requests[0].request_id == req_id
    assert manager.pending_requests[0].max_new_tokens == 10
    assert manager.pending_requests[0].total_steps == 5

def test_step_processing(mock_diffusion_model):
    manager = MDLMContinuousBatchingManager(mock_diffusion_model, max_batch_size=2)

    # Request 1: 2 tokens prompt, 3 new tokens, 2 steps
    req1_id = manager.add_request(torch.tensor([[1, 2]]), max_new_tokens=3, total_steps=2)

    # Request 2: 3 tokens prompt, 2 new tokens, 3 steps
    req2_id = manager.add_request(torch.tensor([[1, 2, 3]]), max_new_tokens=2, total_steps=3)

    # Request 3: Should stay pending initially
    req3_id = manager.add_request(torch.tensor([[1]]), max_new_tokens=1, total_steps=1)

    assert len(manager.pending_requests) == 3

    # Step 1
    has_active = manager.step()
    assert has_active is True
    assert len(manager.active_requests) == 2
    assert len(manager.pending_requests) == 1
    assert len(manager.completed_requests) == 0

    assert manager.active_requests[0].current_step == 1
    assert manager.active_requests[1].current_step == 1

    # Step 2
    has_active = manager.step()
    assert has_active is True
    assert len(manager.active_requests) == 1 # req2 continues, req1 completes. (req3 is NOT promoted until next step because it happens at the start of step())
    assert len(manager.pending_requests) == 1 # req3 still pending
    assert len(manager.completed_requests) == 1

    assert manager.completed_requests[0].request_id == req1_id
    assert manager.completed_requests[0].status == "completed"

    # Step 3
    has_active = manager.step()
    assert has_active is True
    assert len(manager.active_requests) == 0 # req2 completes, req3 promoted and completes immediately in the same step
    assert len(manager.pending_requests) == 0
    assert len(manager.completed_requests) == 3

    # Both req2 and req3 should be in completed_requests now
    completed_ids = [req.request_id for req in manager.completed_requests]
    assert req2_id in completed_ids
    assert req3_id in completed_ids

    # Step 4
    has_active = manager.step()
    assert has_active is False # Nothing left

def test_generate_dynamic_batch(mock_diffusion_model):
    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 3, "total_steps": 2},
        {"input_ids": torch.tensor([[1, 2, 3]]), "max_new_tokens": 2, "total_steps": 3},
        {"input_ids": torch.tensor([[1]]), "max_new_tokens": 1, "total_steps": 1},
    ]

    # Use the real generate_dynamic_batch method bound to the mock object
    results = DiffusionModelForConditionalGeneration.generate_dynamic_batch(mock_diffusion_model, requests_configs=requests, max_batch_size=2)

    assert len(results) == 3
    # Check shape of results: prompt_len + max_new_tokens
    assert results[0].shape == (1, 5) # 2 + 3
    assert results[1].shape == (1, 5) # 3 + 2
    assert results[2].shape == (1, 2) # 1 + 1
