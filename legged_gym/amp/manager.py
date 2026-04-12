"""AMPManager: Backend-agnostic AMP orchestration module.

Owns the discriminator, normalizer, motion data, and AMP replay buffer.
Any runner (PPO or SAC) can use this module to add AMP capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.utils.utils import Normalizer
from legged_gym.amp.motion_dataset import MotionDataset


class AMPManager:
    """Backend-agnostic AMP module that can be used by any runner.

    Owns:
        - AMPDiscriminator network
        - AMP normalizer (running mean/std of AMP observations)
        - Motion dataset (expert data)
        - AMP replay buffer (policy AMP transitions)

    Provides:
        - compute_reward(): Blend task + AMP rewards
        - store_transition(): Store AMP transitions in replay buffer
        - update_discriminator(): Train the discriminator
        - save() / load(): Checkpoint the AMP components

    Args:
        device: Torch device.
        amp_obs_dim: Dimension of AMP observations.
        amp_reward_coef: Reward coefficient for AMP.
        amp_discr_hidden_dims: Hidden layer sizes for discriminator.
        amp_task_reward_lerp: Interpolation factor for task vs AMP reward.
        amp_replay_buffer_size: Size of the AMP replay buffer.
        disc_lr: Learning rate for discriminator optimizer.
        max_grad_norm: Maximum gradient norm for discriminator updates.
        motion_files: List of motion file paths.
        num_dof: Number of DOFs.
        num_key_bodies: Number of key bodies.
        time_between_frames: Simulation timestep.
        num_preload_transitions: Number of expert transitions to preload.
    """

    def __init__(
        self,
        device: Union[str, torch.device],
        amp_obs_dim: int,
        amp_reward_coef: float,
        amp_discr_hidden_dims: List[int],
        amp_task_reward_lerp: float,
        amp_replay_buffer_size: int,
        disc_lr: float,
        max_grad_norm: float,
        motion_files: List[str],
        num_dof: int,
        num_key_bodies: int,
        time_between_frames: float,
        num_preload_transitions: int = 2_000_000,
    ) -> None:
        self.device = torch.device(device)
        self.amp_obs_dim = amp_obs_dim
        self.max_grad_norm = max_grad_norm

        # Motion dataset (expert data)
        self.motion_dataset = MotionDataset(
            device=device,
            num_dof=num_dof,
            num_key_bodies=num_key_bodies,
            time_between_frames=time_between_frames,
            motion_files=motion_files,
            preload_transitions=True,
            num_preload_transitions=num_preload_transitions,
        )

        # Normalizer
        self.normalizer = Normalizer(amp_obs_dim)

        # Discriminator
        self.discriminator = AMPDiscriminator(
            input_dim=amp_obs_dim * 2,
            amp_reward_coef=amp_reward_coef,
            hidden_layer_sizes=amp_discr_hidden_dims,
            device=self.device,
            task_reward_lerp=amp_task_reward_lerp,
        ).to(self.device)

        # AMP replay buffer (policy transitions)
        self.replay_buffer = ReplayBuffer(
            obs_dim=amp_obs_dim,
            buffer_size=amp_replay_buffer_size,
            device=self.device,
        )

        # Discriminator optimizer
        disc_params = [
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.disc_optimizer = optim.Adam(disc_params, lr=disc_lr)

    def compute_reward(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
        task_reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute blended AMP + task reward.

        Args:
            amp_obs: Current AMP observations. Shape: [num_envs, amp_obs_dim]
            next_amp_obs: Next AMP observations (terminal states substituted).
                Shape: [num_envs, amp_obs_dim]
            task_reward: Raw task reward. Shape: [num_envs]

        Returns:
            total_reward: Blended reward. Shape: [num_envs]
            amp_reward: AMP-only reward component.
        """
        return self.discriminator.predict_amp_reward(
            amp_obs, next_amp_obs, task_reward, normalizer=self.normalizer
        )

    def store_transition(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
    ) -> None:
        """Store an AMP transition in the replay buffer.

        Args:
            amp_obs: Current AMP observations.
            next_amp_obs: Next AMP observations (terminal states substituted).
        """
        self.replay_buffer.insert(amp_obs, next_amp_obs)

    def update_discriminator(
        self,
        num_mini_batches: int,
        mini_batch_size: int,
        normalizer_update: bool = True,
    ) -> Tuple[float, float, float, float, float]:
        """Train the discriminator on policy and expert AMP transitions.

        Args:
            num_mini_batches: Number of mini-batches for training.
            mini_batch_size: Size of each mini-batch.
            normalizer_update: Whether to update the normalizer statistics.

        Returns:
            mean_amp_loss: Average AMP loss.
            mean_grad_pen_loss: Average gradient penalty loss.
            mean_policy_pred: Average discriminator prediction on policy samples.
            mean_expert_pred: Average discriminator prediction on expert samples.
            disc_loss_total: Total discriminator loss.
        """
        self.discriminator.train()

        amp_policy_gen = self.replay_buffer.feed_forward_generator(
            num_mini_batches, mini_batch_size
        )
        amp_expert_gen = self.motion_dataset.loader.feed_forward_generator(
            num_mini_batches, mini_batch_size
        )

        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        for (policy_state, policy_next_state), (expert_state, expert_next_state) in zip(
            amp_policy_gen, amp_expert_gen
        ):
            # Normalize if needed
            if self.normalizer is not None:
                with torch.no_grad():
                    policy_state_n = self.normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state_n = self.normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state_n = self.normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state_n = self.normalizer.normalize_torch(expert_next_state, self.device)
            else:
                policy_state_n = policy_state
                policy_next_state_n = policy_next_state
                expert_state_n = expert_state
                expert_next_state_n = expert_next_state

            # Discriminator predictions
            policy_d = self.discriminator(torch.cat([policy_state_n, policy_next_state_n], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state_n, expert_next_state_n], dim=-1))

            # Losses
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state_n, expert_next_state_n, lambda_=10
            )
            disc_loss = amp_loss + grad_pen_loss

            # Optimizer step
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.disc_optimizer.step()

            # Update normalizer
            if normalizer_update and self.normalizer is not None:
                self.normalizer.update(policy_state.cpu().numpy())
                self.normalizer.update(expert_state.cpu().numpy())

            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        mean_amp_loss /= num_mini_batches
        mean_grad_pen_loss /= num_mini_batches
        mean_policy_pred /= num_mini_batches
        mean_expert_pred /= num_mini_batches

        return mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, mean_amp_loss + mean_grad_pen_loss

    def save(self) -> Dict[str, Any]:
        """Return state dict for checkpointing.

        Returns:
            Dictionary containing discriminator state dict and normalizer.
        """
        return {
            "discriminator_state_dict": self.discriminator.state_dict(),
            "amp_normalizer": self.normalizer,
        }

    def load(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state: Dictionary containing discriminator state dict and normalizer.
        """
        self.discriminator.load_state_dict(state["discriminator_state_dict"])
        self.normalizer = state["amp_normalizer"]
