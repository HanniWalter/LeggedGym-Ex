"""SAC Replay Buffer for off-policy training.

Stores (obs, action, reward, next_obs, done) transitions with random batch sampling.
Supports optional n-step returns for improved sample efficiency.
"""

from __future__ import annotations

from typing import Generator, Tuple, Union

import numpy as np
import torch


class SACReplayBuffer:
    """Fixed-size circular replay buffer for SAC with optional n-step returns.

    When n_step > 1, rewards are accumulated as discounted n-step returns:
        R_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1}
    and next_obs points to the state n steps ahead.

    N-step accumulation uses vectorized tensor ring buffers (no Python loops
    over envs), so performance is the same order as 1-step for massively
    parallel environments.

    Args:
        obs_dim: Dimension of observations.
        action_dim: Dimension of actions.
        buffer_size: Maximum number of transitions to store.
        device: Torch device for tensors.
        n_step: Number of steps for n-step return (default 1 = standard TD).
        gamma: Discount factor for n-step accumulation.
        num_envs: Number of parallel environments (needed for n-step tracking).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        buffer_size: int,
        device: Union[str, torch.device],
        n_step: int = 1,
        gamma: float = 0.99,
        num_envs: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_step = max(1, n_step)
        self.gamma = gamma
        self.num_envs = num_envs

        self.observations = torch.zeros(buffer_size, obs_dim, device=self.device)
        self.actions = torch.zeros(buffer_size, action_dim, device=self.device)
        self.rewards = torch.zeros(buffer_size, 1, device=self.device)
        self.next_observations = torch.zeros(buffer_size, obs_dim, device=self.device)
        self.dones = torch.zeros(buffer_size, 1, device=self.device)

        self.step = 0
        self.num_samples = 0

        # Vectorized n-step ring buffer (tensor-based, no Python loops over envs)
        if self.n_step > 1:
            ns = self.n_step
            ne = num_envs
            self._ns_obs = torch.zeros(ns, ne, obs_dim, device=self.device)
            self._ns_act = torch.zeros(ns, ne, action_dim, device=self.device)
            self._ns_rew = torch.zeros(ns, ne, 1, device=self.device)
            self._ns_next = torch.zeros(ns, ne, obs_dim, device=self.device)
            self._ns_done = torch.zeros(ns, ne, 1, device=self.device)
            # Per-env write pointer and valid count
            self._ns_ptr = torch.zeros(ne, dtype=torch.long, device=self.device)
            self._ns_count = torch.zeros(ne, dtype=torch.long, device=self.device)
            self._env_arange = torch.arange(ne, device=self.device)

    def insert(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Add a batch of transitions to the buffer.

        If n_step > 1, transitions are accumulated in a vectorized ring buffer.
        When n steps are collected (or an episode ends), the n-step return is
        computed and stored. All operations are batched over envs — no Python
        loops over num_envs.
        """
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        if self.n_step <= 1:
            self._insert_batch(obs, actions, rewards, next_obs, dones)
            return

        ea = self._env_arange  # (num_envs,)
        ptr = self._ns_ptr     # (num_envs,)

        # 1. Write new transitions into ring buffer (vectorized)
        self._ns_obs[ptr, ea] = obs
        self._ns_act[ptr, ea] = actions
        self._ns_rew[ptr, ea] = rewards
        self._ns_next[ptr, ea] = next_obs
        self._ns_done[ptr, ea] = dones

        self._ns_ptr = (ptr + 1) % self.n_step
        self._ns_count.clamp_(max=self.n_step - 1).add_(1)

        # 2. Determine which envs should emit a transition
        full_mask = self._ns_count >= self.n_step   # ring buffer full
        done_mask = dones.squeeze(-1) > 0.5          # episode ended
        emit_mask = full_mask | done_mask

        if emit_mask.any():
            ei = emit_mask.nonzero(as_tuple=True)[0]
            n_emit = ei.shape[0]
            cnt = self._ns_count[ei]  # (n_emit,)

            # Oldest valid slot per env
            oldest = (self._ns_ptr[ei] - cnt) % self.n_step

            # 3. Compute discounted n-step reward (loop over n_step slots, NOT envs)
            ret = torch.zeros(n_emit, 1, device=self.device)
            alive = torch.ones(n_emit, 1, device=self.device)

            # Track the final next_obs/done (updated when intermediate done is hit)
            newest = (self._ns_ptr[ei] - 1) % self.n_step
            fin_next = self._ns_next[newest, ei].clone()
            fin_done = self._ns_done[newest, ei].clone()

            for j in range(self.n_step):
                slot = (oldest + j) % self.n_step  # (n_emit,)
                in_range = (j < cnt).unsqueeze(-1).float()  # (n_emit, 1)

                r_j = self._ns_rew[slot, ei]
                d_j = self._ns_done[slot, ei]

                ret += in_range * alive * (self.gamma ** j) * r_j

                # If done at step j, record that as the final state and stop
                hit = (in_range * alive * d_j) > 0.5  # (n_emit, 1)
                if hit.any():
                    hd = hit.squeeze(-1)
                    fin_next[hd] = self._ns_next[slot[hd], ei[hd]]
                    fin_done[hd] = 1.0
                    alive = alive * (1.0 - in_range * d_j)

            # 4. Emit to main buffer
            self._insert_batch(
                self._ns_obs[oldest, ei],
                self._ns_act[oldest, ei],
                ret,
                fin_next,
                fin_done,
            )

        # 5. Housekeeping: decrement count for full envs, reset done envs
        full_not_done = full_mask & ~done_mask
        if full_not_done.any():
            self._ns_count[full_not_done] = self.n_step - 1

        if done_mask.any():
            self._ns_count[done_mask] = 0
            self._ns_ptr[done_mask] = 0

    def _insert_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Insert a batch directly into the main circular buffer."""
        batch_size = obs.shape[0]
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        if self.step + batch_size > self.buffer_size:
            # Wrap around
            first_part = self.buffer_size - self.step
            self.observations[self.step:self.buffer_size] = obs[:first_part]
            self.actions[self.step:self.buffer_size] = actions[:first_part]
            self.rewards[self.step:self.buffer_size] = rewards[:first_part]
            self.next_observations[self.step:self.buffer_size] = next_obs[:first_part]
            self.dones[self.step:self.buffer_size] = dones[:first_part]

            remaining = batch_size - first_part
            self.observations[:remaining] = obs[first_part:]
            self.actions[:remaining] = actions[first_part:]
            self.rewards[:remaining] = rewards[first_part:]
            self.next_observations[:remaining] = next_obs[first_part:]
            self.dones[:remaining] = dones[first_part:]
        else:
            end = self.step + batch_size
            self.observations[self.step:end] = obs
            self.actions[self.step:end] = actions
            self.rewards[self.step:end] = rewards
            self.next_observations[self.step:end] = next_obs
            self.dones[self.step:end] = dones

        self.num_samples = min(self.buffer_size, max(self.step + batch_size, self.num_samples))
        self.step = (self.step + batch_size) % self.buffer_size

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (obs, actions, rewards, next_obs, dones), each as tensors.
        """
        idxs = np.random.randint(0, self.num_samples, size=batch_size)
        return (
            self.observations[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_observations[idxs],
            self.dones[idxs],
        )

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self.num_samples

    def clear(self) -> None:
        """Reset the buffer."""
        self.step = 0
        self.num_samples = 0
        if self.n_step > 1:
            self._ns_count.zero_()
            self._ns_ptr.zero_()
