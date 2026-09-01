from transformers import PretrainedConfig

class DiffusionConfig(PretrainedConfig):
    model_type = "lfm_masked_diffusion"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        base_config_dict=None,
        mask_token_id=None,
        diffusion_steps=1000,
        remasking_strategy="low_confidence",
        block_size=64,
        timestep_dim=256,
        max_timesteps=1000,
        use_sdpa=True,
        use_moe=False,
        num_experts=8,
        num_experts_per_tok=2,
        use_rag=False,
        rag_query_steps=None,
        rag_max_retrieved=128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_config_dict = base_config_dict or {}
        self.use_rag = use_rag
        self.rag_query_steps = rag_query_steps
        self.rag_max_retrieved = rag_max_retrieved
        self.mask_token_id = mask_token_id
        self.diffusion_steps = diffusion_steps
        self.remasking_strategy = remasking_strategy
        self.block_size = block_size
        self.timestep_dim = timestep_dim
        self.max_timesteps = max_timesteps
        self.use_sdpa = use_sdpa
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

DiffusionConfig.register_for_auto_class("AutoConfig")
