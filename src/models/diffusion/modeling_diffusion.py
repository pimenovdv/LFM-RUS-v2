import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, AutoModel

from .configuration_diffusion import DiffusionConfig

class UniversalDiffusionLM(PreTrainedModel):
    config_class = DiffusionConfig

    def __init__(self, config: DiffusionConfig):
        super().__init__(config)

        base_config = AutoConfig.for_model(**config.base_config_dict)
        self.inner_model = AutoModel.from_config(base_config)
        self._disable_causal_mask()

        self.timestep_embedder = nn.Sequential(
            nn.Linear(1, config.timestep_dim),
            nn.SiLU(),
            nn.Linear(config.timestep_dim, base_config.hidden_size)
        )

        self.lm_head = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)

        if getattr(base_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.inner_model.get_input_embeddings().weight

    def _disable_causal_mask(self):
        """
        Dynamic patching to disable causality.
        In transformers >= 5.9.0, masking is delegated to masking_utils which
        checks config.is_causal. Setting it to False allows fallback to
        bidirectional mask.
        For older models or specific implementations, further patching can be done.
        """
        if hasattr(self.inner_model, "config"):
            self.inner_model.config.is_causal = False

        # Additionally patch _update_causal_mask if present (from the example)
        if hasattr(self.inner_model, "_update_causal_mask"):
            self.inner_model._update_causal_mask = lambda *args, **kwargs: args[1] if len(args) > 1 else kwargs.get("attention_mask")

        if self.inner_model.config.model_type == "gpt2":
            for module in self.inner_model.modules():
                if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor) and module.bias.dim() == 4:
                    module.bias.data.fill_(1)

    def get_input_embeddings(self):
        return self.inner_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.inner_model.set_input_embeddings(value)

    def forward(self, input_ids=None, timesteps=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            embeddings_layer = self.inner_model.get_input_embeddings()
            inputs_embeds = embeddings_layer(input_ids)

        if timesteps is not None:
            t_embed = self.timestep_embedder(timesteps.unsqueeze(-1).float())
            hidden_states = inputs_embeds + t_embed.unsqueeze(1)
        else:
            hidden_states = inputs_embeds

        outputs = self.inner_model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )

        logits = self.lm_head(outputs.last_hidden_state)

        return logits

UniversalDiffusionLM.register_for_auto_class("AutoModelForCausalLM")
UniversalDiffusionLM.register_for_auto_class("AutoModel")
