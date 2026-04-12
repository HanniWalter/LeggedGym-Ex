"""Reward blending utilities for AMP.

Provides functions to blend task rewards with AMP discriminator rewards.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from rsl_rl.utils.utils import Normalizer


def compute_amp_reward(
    discriminator: AMPDiscriminator,
    amp_obs: torch.Tensor,
    next_amp_obs: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer: Optional[Normalizer] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute blended reward using AMP discriminator.

    Args:
        discriminator: AMP discriminator network.
        amp_obs: Current AMP observations. Shape: [num_envs, amp_obs_dim]
        next_amp_obs: Next AMP observations (with terminal states substituted).
            Shape: [num_envs, amp_obs_dim]
        task_reward: Task reward from environment. Shape: [num_envs]
        normalizer: Optional normalizer for AMP observations.

    Returns:
        total_reward: Blended reward. Shape: [num_envs]
        amp_reward: Pure AMP reward component. Shape: [num_envs, 1] or similar
    """
    return discriminator.predict_amp_reward(
        amp_obs, next_amp_obs, task_reward, normalizer=normalizer
    )
