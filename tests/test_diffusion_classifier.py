import pytest
import torch
import torch.nn.functional as F
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration
from src.models.diffusion.configuration_diffusion import DiffusionConfig

@pytest.fixture
def dummy_diffusion_model(mocker):
    config = DiffusionConfig(
        base_config_dict={"model_type": "gpt2", "vocab_size": 100},
        mask_token_id=0,
        vocab_size=100
    )
    model = DiffusionModelForConditionalGeneration(config)

    # Mock the forward pass to return some dummy logits
    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            input_ids = args[0]
        batch_size, seq_len = input_ids.shape
        # Return random logits that will pass F.log_softmax without NaNs
        logits = torch.randn(batch_size, seq_len, 100)

        class Output:
            def __init__(self, logits):
                self.logits = logits
                self.past_key_values = None
                self.hidden_states = None
                self.attentions = None

        return Output(logits)

    mocker.patch.object(model, "__call__", side_effect=mock_forward)
    return model

def test_classifier_guided_sampling(dummy_diffusion_model):
    input_ids = torch.randint(1, 100, (1, 10))
    # introduce mask tokens
    input_ids[0, 5:] = 0

    # Dummy classifier that encourages the generation of token ID 42
    def dummy_classifier(logits, input_ids, timesteps):
        probs = F.softmax(logits, dim=-1)
        return probs[:, :, 42].sum()

    # Test generation with the classifier
    output_with_classifier = dummy_diffusion_model.generate(
        input_ids.clone(),
        steps=2,
        max_new_tokens=5, # To cap the output length
        classifier_fn=dummy_classifier,
        classifier_scale=10.0,
        classifier_schedule="linear",
        min_classifier_scale=1.0,
        return_dict_in_generate=False
    )

    assert output_with_classifier is not None
    assert output_with_classifier.shape[1] >= 10  # original sequence length

    # Test generation with different schedules
    for schedule in ["cosine", "exponential", "cyclic", "constant"]:
        output_sched = dummy_diffusion_model.generate(
            input_ids.clone(),
            steps=2,
            max_new_tokens=5,
            classifier_fn=dummy_classifier,
            classifier_scale=10.0,
            classifier_schedule=schedule,
            min_classifier_scale=1.0,
            return_dict_in_generate=False
        )
        assert output_sched is not None
        assert output_sched.shape[1] >= 10

    # Run without classifier to ensure it works
    output_without_classifier = dummy_diffusion_model.generate(
        input_ids.clone(),
        steps=2,
        max_new_tokens=5,
        classifier_scale=0.0,
        return_dict_in_generate=False
    )

    assert output_without_classifier is not None
    assert output_without_classifier.shape[1] >= 10
