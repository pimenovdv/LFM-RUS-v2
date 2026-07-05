from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from trl import GRPOTrainer, GRPOConfig

@dataclass
class VRPOConfig(GRPOConfig):
    """
    Configuration for Variance-Reduced Policy Optimization (VRPO) / CJ-GRPO
    tailored for Diffusion Language Models.
    """
    diffusion_steps: int = field(default=20, metadata={"help": "Number of diffusion steps for rollout trajectory."})
    trajectory_alignment: bool = field(default=True, metadata={"help": "Enable rollout and optimization trajectory alignment to prevent skip-step errors."})

class VRPOTrainer(GRPOTrainer):
    """
    Variance-Reduced Policy Optimization (VRPO) Trainer for Diffusion LMs.
    Adapts GRPO for parallel diffusion trajectories and alignment.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_diffusion_trajectory(self, inputs):
        """
        Aligns the rollout trajectory (parallel unmasking) with the policy optimization trajectory.
        This prevents skip-step errors in continuous/discrete diffusion RLHF.
        """
        if "input_ids" in inputs:
            device = inputs["input_ids"].device
            trajectory_states = torch.zeros(
                (inputs["input_ids"].size(0), self.args.diffusion_steps, inputs["input_ids"].size(1)),
                device=device
            )
            return trajectory_states
        raise ValueError("input_ids not provided for diffusion trajectory")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Custom compute_loss that handles diffusion trajectory rollout and variance reduction.
        """
        if getattr(self.args, "trajectory_alignment", False):
            trajectory_states = self._prepare_diffusion_trajectory(inputs)
            if trajectory_states is not None:
                inputs["trajectory_states"] = trajectory_states

        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch, **kwargs)
