# AMP (Adversarial Motion Priors) injectable module
# This module provides a backend-agnostic AMP implementation that can be used
# with any RL algorithm (PPO, SAC, etc.)

from .manager import AMPManager

__all__ = ["AMPManager"]
