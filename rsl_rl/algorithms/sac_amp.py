"""SAC + AMP algorithm.

Extends SAC to work with AMP discriminator-based reward blending.
The algorithm itself stays pure SAC; AMP orchestration lives in the runner
via AMPManager.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch

from rsl_rl.algorithms.sac import SAC, SACActorCritic


class SAC_AMP(SAC):
    """SAC algorithm with AMP support.

    This is a thin wrapper that adds nothing beyond standard SAC.
    AMP discriminator training and reward blending are handled by the
    SACAMPRunner via AMPManager. This class exists for symmetry with
    PPO_AMP and to allow future AMP-specific algorithm modifications.
    """

    def __init__(
        self,
        actor_critic: SACActorCritic,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor_critic, **kwargs)
