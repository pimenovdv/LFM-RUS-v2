import pytest
import torch
from transformers import AutoConfig, AutoModel
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, SparseMoEBlock

@pytest.fixture
def moe_config():
    base_config = AutoConfig.for_model("gpt2")
    base_config.vocab_size = 100
    base_config.n_layer = 2
    return DiffusionConfig(
        base_config_dict=base_config.to_dict(),
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2
    )

def test_diffusion_moe_injection(moe_config, mocker):
    # We mock AutoModel.from_config to avoid downloading or instantiating a huge model if not needed
    # However, since we're using gpt2 config, we can just instantiate it normally for a robust test
    # We'll just instantiate the actual model since it's a small 2-layer GPT2

    model = DiffusionModelForConditionalGeneration(moe_config)

    # Assert that at least one SparseMoEBlock was injected
    found_moe = False
    for name, module in model.named_modules():
        if isinstance(module, SparseMoEBlock):
            found_moe = True
            assert module.num_experts == 4
            assert module.num_experts_per_tok == 2
            assert module.hidden_size == model.inner_model.config.hidden_size
            break

    assert found_moe, "SparseMoEBlock was not injected into the model"

def test_diffusion_moe_forward(moe_config):
    model = DiffusionModelForConditionalGeneration(moe_config)
    model.eval()

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, moe_config.base_config_dict["vocab_size"], (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert "logits" in outputs
    assert outputs.logits.shape == (batch_size, seq_len, moe_config.base_config_dict["vocab_size"])
