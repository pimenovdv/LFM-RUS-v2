import pytest
import torch
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

def test_mdlm_rag_retrieval(mocker):
    config = DiffusionConfig(
        base_config_dict={"hidden_size": 16, "vocab_size": 100, "num_hidden_layers": 1, "num_attention_heads": 1, "model_type": "bert"},
        mask_token_id=0,
        diffusion_steps=2,
        use_rag=True
    )
    model = DiffusionModelForConditionalGeneration(config)

    input_ids = torch.tensor([[1, 2, 3]])

    # Mock forward to return some dummy logits matching the input size
    def mock_forward(*args, **kwargs):
        x = kwargs.get("input_ids")
        class Output:
            logits = torch.randn(x.size(0), x.size(1), 100)
        return Output()

    mocker.patch.object(model, "forward", side_effect=mock_forward)

    # Mock rag retriever
    def mock_rag_retriever(x, timesteps=None):
        return torch.tensor([[99, 98]])

    # The generated output shouldn't contain the RAG context because it is removed before returning
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=4,
        steps=2,
        rag_retriever=mock_rag_retriever,
        rag_query_steps=[0]
    )

    assert out.shape == (1, 7)

    # Check NotImplementedError in generate_dynamic_batch
    with pytest.raises(NotImplementedError):
        model.generate_dynamic_batch([{"input_ids": input_ids, "max_new_tokens": 4, "rag_retriever": mock_rag_retriever}])


def test_mdlm_rag_unconditional_and_attention(mocker):
    config = DiffusionConfig(
        base_config_dict={"hidden_size": 16, "vocab_size": 100, "num_hidden_layers": 1, "num_attention_heads": 1, "model_type": "bert"},
        mask_token_id=0,
        diffusion_steps=1,
        use_rag=True
    )
    model = DiffusionModelForConditionalGeneration(config)

    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    unconditional_input_ids = torch.tensor([[4, 5, 6]])

    def mock_forward(*args, **kwargs):
        x = kwargs.get("input_ids")
        class Output:
            logits = torch.randn(x.size(0), x.size(1), 100)
        return Output()

    mocker.patch.object(model, "forward", side_effect=mock_forward)

    # test that rag works over multiple queries and unconditionals update
    call_count = [0]
    def mock_rag_retriever(x, timesteps=None):
        call_count[0] += 1
        return torch.tensor([[99, 98]]) if call_count[0] == 1 else torch.tensor([[97, 96, 95]])

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        unconditional_input_ids=unconditional_input_ids,
        cfg_scale=1.5,  # to force unconditional path logic (if it exists)
        max_new_tokens=4,
        steps=2,
        rag_retriever=mock_rag_retriever,
        rag_query_steps=[0, 1]
    )

    assert out.shape == (1, 7)
    assert call_count[0] == 2
