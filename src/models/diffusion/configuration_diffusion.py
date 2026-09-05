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
        use_flash_attention_2=False,
        use_moe=False,
        num_experts=8,
        num_experts_per_tok=2,
        use_rag=False,
        rag_query_steps=None,
        rag_max_retrieved=128,
        continuous_time=False,
        is_consistency_model=False,
        use_latent_diffusion=False,
        vq_num_embeddings=512,
        vq_embedding_dim=64,
        use_flow_matching=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_flow_matching = use_flow_matching
        self.use_latent_diffusion = use_latent_diffusion
        self.vq_num_embeddings = vq_num_embeddings
        self.vq_embedding_dim = vq_embedding_dim
        self.base_config_dict = base_config_dict or {}
        self.use_rag = use_rag
        self.is_consistency_model = is_consistency_model
        self.continuous_time = continuous_time
        self.rag_query_steps = rag_query_steps
        self.rag_max_retrieved = rag_max_retrieved
        self.mask_token_id = mask_token_id
        self.diffusion_steps = diffusion_steps
        self.remasking_strategy = remasking_strategy
        self.block_size = block_size
        self.timestep_dim = timestep_dim
        self.max_timesteps = max_timesteps
        self.use_sdpa = use_sdpa
        self.use_flash_attention_2 = use_flash_attention_2
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

DiffusionConfig.register_for_auto_class("AutoConfig")
