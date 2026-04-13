from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from rsl_rl.modules.flash_sac_layers import (
    EnsembleUnitBatchNorm,
    EnsembleUnitLinear,
    EnsembleUnitRMSNorm,
    UnitBatchNorm,
    UnitLinear,
    UnitRMSNorm,
)
from rsl_rl.modules.flash_sac_networks import FlashSACActor, FlashSACDoubleCritic


class FlashSACActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_num_blocks: int = 2,
        actor_hidden_dim: int = 128,
        critic_num_blocks: int = 2,
        critic_hidden_dim: int = 256,
        critic_num_bins: int = 101,
        critic_min_v: float = -5.0,
        critic_max_v: float = 5.0,
        clip_actions: float = 1.0,
        init_noise_std: float = 1.0,
        activation: str = "relu",
        **kwargs: Any,
    ) -> None:
        del init_noise_std, activation
        if kwargs:
            print(
                "FlashSACActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.actor = FlashSACActor(
            num_blocks=actor_num_blocks,
            input_dim=num_actor_obs,
            hidden_dim=actor_hidden_dim,
            action_dim=num_actions,
        )
        self.critic = FlashSACDoubleCritic(
            num_blocks=critic_num_blocks,
            input_dim=num_critic_obs + num_actions,
            hidden_dim=critic_hidden_dim,
            num_bins=critic_num_bins,
            min_v=critic_min_v,
            max_v=critic_max_v,
        )
        self.num_actions = num_actions
        self.clip_actions = clip_actions
        self.num_bins = critic_num_bins

        self.std = nn.Parameter(torch.ones(num_actions), requires_grad=False)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            mean, _ = self.actor.get_mean_and_std(obs, training=False)
            action = torch.tanh(mean)
            return action, torch.zeros((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)
        action, info = self.actor(obs, training=True)
        return action, info["log_prob"]

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean, _ = self.actor.get_mean_and_std(obs, training=False)
            action = torch.tanh(mean)
        return action

    def evaluate_q(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qs, _ = self.critic(obs, action, training=True)
        return qs[0].unsqueeze(-1), qs[1].unsqueeze(-1)

    def evaluate_q_log_probs(self, obs: torch.Tensor, action: torch.Tensor, training: bool = True) -> torch.Tensor:
        _, infos = self.critic(obs, action, training=training)
        return infos["log_prob"]

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        del dones

    def normalize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (UnitLinear, UnitBatchNorm, UnitRMSNorm,
                                   EnsembleUnitLinear, EnsembleUnitBatchNorm, EnsembleUnitRMSNorm)):
                module.normalize_parameters()

    def forward(self) -> None:
        raise NotImplementedError
