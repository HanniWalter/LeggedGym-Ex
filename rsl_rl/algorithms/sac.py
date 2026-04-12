"""SAC (Soft Actor-Critic) algorithm for off-policy RL.

Implements squashed Gaussian actor, twin Q-networks, target networks,
and automatic entropy (alpha) tuning.

Reference: https://arxiv.org/abs/1812.05905
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from rsl_rl.storage.sac_replay_buffer import SACReplayBuffer


# ──────────────────── Network Modules ──────────────────────


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU(),
    output_activation: Optional[nn.Module] = None,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    """Build a simple MLP with optional LayerNorm.

    When use_layer_norm=True, a LayerNorm is inserted after each hidden layer
    (before activation). This stabilizes critic training with large Q-value
    magnitudes common in locomotion tasks.
    """
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hd in hidden_dims:
        layers.append(nn.Linear(last_dim, hd))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hd))
        layers.append(activation)
        last_dim = hd
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPSILON = 1e-6


class SquashedGaussianActor(nn.Module):
    """Squashed Gaussian policy network for SAC.

    Outputs a mean and log_std, samples via reparameterisation, then applies
    tanh squashing so actions lie in (-1, 1).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256, 256],
        activation: str = "relu",
        clip_actions: float = 100.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        act = _get_activation(activation)
        self.net = _build_mlp(obs_dim, hidden_dims, hidden_dims[-1], act,
                              use_layer_norm=use_layer_norm)
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        self.action_dim = action_dim
        self.clip_actions = clip_actions

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: compute action and log_prob.

        Args:
            obs: Observations. Shape: [batch, obs_dim]
            deterministic: If True, use mean action (no sampling).

        Returns:
            action: Squashed action in (-1, 1). Shape: [batch, action_dim]
            log_prob: Log probability of the action. Shape: [batch, 1]
        """
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()

        # Squash through tanh
        action = torch.tanh(raw_action)

        # Log probability with correction for tanh squashing
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob -= (2.0 * (np.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))).sum(
            dim=-1, keepdim=True
        )

        return action, log_prob

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for inference (no grad)."""
        with torch.no_grad():
            action, _ = self.forward(obs, deterministic=True)
        return action


class TwinQNetwork(nn.Module):
    """Twin Q-network for SAC (clipped double Q)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256, 256],
        activation: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        act = _get_activation(activation)
        self.q1 = _build_mlp(obs_dim + action_dim, hidden_dims, 1, act,
                             use_layer_norm=use_layer_norm)
        self.q2 = _build_mlp(obs_dim + action_dim, hidden_dims, 1, act,
                             use_layer_norm=use_layer_norm)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ──────────────────── SAC Actor-Critic Container ──────────────────────


class SACActorCritic(nn.Module):
    """Container for SAC actor and critic networks.

    Holds a SquashedGaussianActor and a TwinQNetwork.
    This class is analogous to ActorCritic in PPO but for the SAC architecture.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "relu",
        init_noise_std: float = 1.0,  # unused, kept for config compatibility
        clip_actions: float = 100.0,
        use_layer_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "SACActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        self.actor = SquashedGaussianActor(
            num_actor_obs, num_actions, actor_hidden_dims, activation, clip_actions,
            use_layer_norm=use_layer_norm,
        )
        self.critic = TwinQNetwork(
            num_critic_obs, num_actions, critic_hidden_dims, activation,
            use_layer_norm=use_layer_norm,
        )
        self.num_actions = num_actions

        # Expose a dummy std parameter for logging compatibility with OnPolicyRunner.log()
        self.std = nn.Parameter(torch.ones(num_actions), requires_grad=False)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy."""
        return self.actor(obs, deterministic=deterministic)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for inference."""
        return self.actor.act_inference(obs)

    def evaluate_q(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Q-values for given obs-action pairs."""
        return self.critic(obs, action)

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        pass

    def forward(self) -> None:
        raise NotImplementedError


# ──────────────────── SAC Algorithm ──────────────────────


class SAC:
    """Soft Actor-Critic algorithm for massively parallel locomotion.

    Key design decisions for Isaac Gym:
    - UTD ratio is kept low (1-4) because fresh samples are cheap with many envs.
      High UTD wastes wall-clock time on gradient steps.
    - Policy delay (update actor every N critic updates) stabilizes training
      when Q-values are noisy from many parallel environments.
    - Reward normalization prevents Q-value explosion common in locomotion.

    Implements:
        - Squashed Gaussian policy with automatic entropy tuning
        - Clipped double Q-learning
        - Polyak-averaged target networks
        - Policy delay (optional, default every 2 critic updates)
        - N-step return support (via replay buffer)
        - Temporally correlated exploration (optional)
        - Comprehensive diagnostics
    """

    def __init__(
        self,
        actor_critic: SACActorCritic,
        learning_rate: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        batch_size: int = 256,
        utd_ratio: int = 1,
        max_grad_norm: float = 1.0,
        policy_delay: int = 2,
        use_adamw: bool = False,
        n_step: int = 1,
        normalize_rewards: bool = False,
        normalize_observations: bool = True,
        temporal_noise_steps: int = 0,
        device: Union[str, torch.device] = "cpu",
        # Accept and ignore PPO-specific params for config compatibility
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: Optional[float] = None,
        use_spo: bool = False,
        normalize_rewards_compat: bool = False,  # old key alias
        **kwargs: Any,  # absorb extra config keys (e.g. AMP-specific)
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.max_grad_norm = max_grad_norm
        self.policy_delay = max(1, policy_delay)
        self.n_step = max(1, n_step)

        # Observation normalization (Welford running mean/std)
        self.normalize_observations = normalize_observations
        self._obs_normalizer: Optional[EmpiricalNormalization] = None  # created in init_storage

        # Reward normalization (running mean/std)
        self.normalize_rewards = normalize_rewards
        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

        # Temporal noise: hold noise fixed for several steps for smoother exploration.
        # Per-step noise creates jerky motions in locomotion. Holding noise for
        # several steps produces smoother exploratory gaits.
        self.temporal_noise_steps = max(0, temporal_noise_steps)
        self._cached_noise: Optional[torch.Tensor] = None
        self._noise_step_counter = 0

        # Track update steps for policy delay and diagnostics
        self._critic_update_count = 0
        self._total_updates = 0

        # Networks
        self.actor_critic = actor_critic.to(self.device)

        # Determine critic input dimensions from the actor_critic's critic
        # The first Linear layer of q1 has input_dim = obs_dim + action_dim
        critic_input_dim = None
        for layer in actor_critic.critic.q1:
            if isinstance(layer, nn.Linear):
                critic_input_dim = layer.in_features
                break
        if critic_input_dim is None:
            critic_input_dim = actor_critic.critic.q1[0].in_features
        critic_obs_dim = critic_input_dim - actor_critic.num_actions
        critic_hidden_dims = []
        for layer in actor_critic.critic.q1:
            if isinstance(layer, nn.Linear):
                critic_hidden_dims.append(layer.out_features)
        # Last element is the output (1), remove it
        if critic_hidden_dims:
            critic_hidden_dims = critic_hidden_dims[:-1]

        # Detect if source critic uses LayerNorm
        _has_ln = any(isinstance(m, nn.LayerNorm) for m in actor_critic.critic.q1)

        # Target critic (no gradient)
        self.target_critic = TwinQNetwork(
            critic_obs_dim,
            actor_critic.num_actions,
            critic_hidden_dims if critic_hidden_dims else [256, 256, 256],
            use_layer_norm=_has_ln,
        ).to(self.device)
        # Copy weights
        self.target_critic.load_state_dict(actor_critic.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # Optimizers — AdamW with betas=(0.9, 0.95) for faster adaptation
        opt_class = optim.AdamW if use_adamw else optim.Adam
        opt_kwargs: dict = {"lr": learning_rate}
        if use_adamw:
            opt_kwargs["betas"] = (0.9, 0.95)
        self.actor_optimizer = opt_class(
            actor_critic.actor.parameters(), **opt_kwargs
        )
        self.critic_optimizer = opt_class(
            actor_critic.critic.parameters(), **opt_kwargs
        )

        # Expose learning_rate for logging compatibility
        self.learning_rate = learning_rate

        # Entropy coefficient (alpha)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = (
                target_entropy
                if target_entropy is not None
                else -float(actor_critic.num_actions)
            )
            # Initialize log_alpha from init_alpha (not zero!)
            self.log_alpha = torch.tensor(
                [math.log(max(init_alpha, 1e-8))],
                requires_grad=True, device=self.device,
            )
            alpha_opt_kwargs: dict = {"lr": alpha_lr}
            if use_adamw:
                alpha_opt_kwargs["betas"] = (0.9, 0.95)
            self.alpha_optimizer = opt_class([self.log_alpha], **alpha_opt_kwargs)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = init_alpha
            self.log_alpha = None
            self.alpha_optimizer = None

        # Storage
        self.replay_buffer: Optional[SACReplayBuffer] = None

        # Alias for save/load compatibility
        self.optimizer = self.actor_optimizer  # used by runner.save()

    def init_storage(
        self,
        num_envs: int,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        """Initialize the replay buffer and observation normalizer."""
        self.replay_buffer = SACReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            device=self.device,
            n_step=self.n_step,
            gamma=self.gamma,
            num_envs=num_envs,
        )
        if self.normalize_observations:
            self._obs_normalizer = EmpiricalNormalization(
                shape=obs_dim, device=self.device,
            )

    def _normalize_obs(self, obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        """Normalize observations if enabled."""
        if self._obs_normalizer is not None:
            return self._obs_normalizer(obs, update=update)
        return obs

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample actions from the policy.

        If temporal_noise_steps > 0, the noise component is held fixed for
        that many calls to produce temporally correlated exploration.
        """
        with torch.no_grad():
            norm_obs = self._normalize_obs(obs, update=False)
            if self.temporal_noise_steps > 0 and not deterministic:
                action_det, _ = self.actor_critic.act(norm_obs, deterministic=True)
                if self._cached_noise is None or self._noise_step_counter >= self.temporal_noise_steps:
                    action_stoch, _ = self.actor_critic.act(norm_obs, deterministic=False)
                    self._cached_noise = action_stoch - action_det
                    self._noise_step_counter = 0
                action = action_det + self._cached_noise
                action = torch.clamp(action, -1.0, 1.0)
                self._noise_step_counter += 1
                return action
            else:
                action, _ = self.actor_critic.act(norm_obs, deterministic=deterministic)
                return action

    def store_transition(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Store a batch of transitions in the replay buffer."""
        assert self.replay_buffer is not None
        self.replay_buffer.insert(obs, actions, rewards, next_obs, dones)

    def update(self) -> Dict[str, float]:
        """Perform SAC update (critic + actor with policy delay + alpha).

        Performs ``utd_ratio`` gradient steps. Actor is updated every
        ``policy_delay`` critic updates.

        Returns:
            Dictionary of loss metrics including diagnostics.
        """
        assert self.replay_buffer is not None
        assert self.replay_buffer.size >= self.batch_size, "Not enough samples in buffer"

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        total_alpha = 0.0
        actor_updates = 0

        # Diagnostics accumulators
        total_q1_mean = 0.0
        total_q2_mean = 0.0
        total_target_q_mean = 0.0
        total_td_target_mean = 0.0
        total_log_prob_mean = 0.0
        total_action_abs_mean = 0.0
        has_nan = False
        has_inf = False

        # N-step gamma
        gamma_n = self.gamma ** self.n_step

        for update_i in range(self.utd_ratio):
            obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

            # ─── Observation normalization ───
            obs = self._normalize_obs(obs, update=True)
            next_obs = self._normalize_obs(next_obs, update=False)

            # ─── Reward normalization ───
            if self.normalize_rewards:
                rewards = self._normalize_reward(rewards)

            # ─── NaN/Inf check on inputs ───
            if torch.isnan(obs).any() or torch.isnan(rewards).any():
                has_nan = True
            if torch.isinf(rewards).any():
                has_inf = True

            # ─── Critic update ───
            with torch.no_grad():
                next_actions, next_log_probs = self.actor_critic.act(next_obs)
                target_q1, target_q2 = self.target_critic(next_obs, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target_value = target_q - self.alpha * next_log_probs
                td_target = rewards + (1.0 - dones) * gamma_n * target_value

            q1, q2 = self.actor_critic.evaluate_q(obs, actions)
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            self._critic_update_count += 1
            self._total_updates += 1

            # ─── Actor update (with policy delay) ───
            if self._critic_update_count % self.policy_delay == 0:
                new_actions, log_probs = self.actor_critic.act(obs)
                q1_new, q2_new = self.actor_critic.evaluate_q(obs, new_actions)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (self.alpha * log_probs - q_new).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                actor_updates += 1

                # Action diagnostics
                total_action_abs_mean += new_actions.abs().mean().item()
                total_log_prob_mean += log_probs.mean().item()

                # ─── Alpha update ───
                alpha_loss_val = 0.0
                if self.auto_alpha and self.log_alpha is not None and self.alpha_optimizer is not None:
                    alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
                    alpha_loss_val = alpha_loss.item()
                total_alpha_loss += alpha_loss_val
                total_alpha += self.alpha

            # ─── Target network update (Polyak) ───
            with torch.no_grad():
                for p, p_targ in zip(
                    self.actor_critic.critic.parameters(),
                    self.target_critic.parameters(),
                ):
                    p_targ.data.mul_(1.0 - self.tau)
                    p_targ.data.add_(self.tau * p.data)

            total_critic_loss += critic_loss.item()

            # Diagnostics
            total_q1_mean += q1.mean().item()
            total_q2_mean += q2.mean().item()
            total_target_q_mean += target_q.mean().item()
            total_td_target_mean += td_target.mean().item()

        n = self.utd_ratio
        n_actor = max(1, actor_updates)
        metrics = {
            "critic_loss": total_critic_loss / n,
            "actor_loss": total_actor_loss / n_actor,
            "alpha_loss": total_alpha_loss / n_actor,
            "alpha": total_alpha / n_actor,
            # Q-value diagnostics
            "q1_mean": total_q1_mean / n,
            "q2_mean": total_q2_mean / n,
            "target_q_mean": total_target_q_mean / n,
            "td_target_mean": total_td_target_mean / n,
            # Action diagnostics (values near 1.0 = tanh saturated)
            "action_abs_mean": total_action_abs_mean / n_actor if actor_updates > 0 else 0.0,
            "log_prob_mean": total_log_prob_mean / n_actor if actor_updates > 0 else 0.0,
            # Update counts
            "critic_updates": float(n),
            "actor_updates": float(actor_updates),
            "total_updates": float(self._total_updates),
        }

        # Warnings for failure modes
        if has_nan:
            metrics["warning_nan"] = 1.0
            print("[SAC WARNING] NaN detected in observations or rewards!")
        if has_inf:
            metrics["warning_inf"] = 1.0
            print("[SAC WARNING] Inf detected in rewards!")
        if abs(metrics["q1_mean"]) > 1e4:
            print(f"[SAC WARNING] Q-values exploding: q1_mean={metrics['q1_mean']:.1f}")
        if actor_updates > 0 and metrics["action_abs_mean"] > 0.95:
            print(f"[SAC WARNING] Actions saturating at tanh bounds: |a|_mean={metrics['action_abs_mean']:.3f}")

        return metrics

    def update_reward_stats(self, rewards: torch.Tensor) -> None:
        """Update running reward statistics for normalization."""
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item()
        batch_count = rewards.numel()

        delta = batch_mean - self._reward_running_mean
        total_count = self._reward_count + batch_count
        if total_count == 0:
            return
        new_mean = self._reward_running_mean + delta * batch_count / total_count
        m_a = self._reward_running_var * self._reward_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self._reward_count * batch_count / total_count
        self._reward_running_var = m2 / total_count
        self._reward_running_mean = new_mean
        self._reward_count = total_count

    def _normalize_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics."""
        std = max(self._reward_running_var ** 0.5, 1e-6)
        return rewards / std  # only scale, don't center (preserves reward sign)

    def compute_action_stats(self, obs: torch.Tensor) -> Dict[str, float]:
        """Compute action statistics for diagnostics.

        Useful for detecting whether the effective action range is appropriate.
        Actions near +/-1.0 mean tanh saturation.
        """
        with torch.no_grad():
            norm_obs = self._normalize_obs(obs, update=False)
            h = self.actor_critic.actor.net(norm_obs)
            mean = self.actor_critic.actor.mean_head(h)
            log_std = self.actor_critic.actor.log_std_head(h)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = log_std.exp()
            raw_action = mean + std * torch.randn_like(mean)
            action = torch.tanh(raw_action)

        stats = {
            "pre_tanh_abs_max": raw_action.abs().max().item(),
            "post_tanh_abs_mean": action.abs().mean().item(),
            "policy_std_mean": std.mean().item(),
        }
        return stats

    def print_replay_ratio(self, num_envs: int, num_steps_per_env: int) -> None:
        """Print the effective replay ratio / update-to-data ratio."""
        data_per_iter = num_envs * num_steps_per_env
        gradient_steps_per_iter = self.utd_ratio * num_steps_per_env
        replay_ratio = gradient_steps_per_iter / data_per_iter if data_per_iter > 0 else 0

        print(f"\n{'='*60}")
        print(f" SAC Training Configuration Summary")
        print(f"{'='*60}")
        print(f"  num_envs:              {num_envs:,}")
        print(f"  num_steps_per_env:     {num_steps_per_env}")
        print(f"  data_per_iter:         {data_per_iter:,}")
        print(f"  utd_ratio:             {self.utd_ratio}")
        print(f"  gradient_steps/iter:   {gradient_steps_per_iter}")
        print(f"  batch_size:            {self.batch_size}")
        print(f"  replay_ratio:          {replay_ratio:.4f}")
        print(f"  policy_delay:          {self.policy_delay}")
        print(f"  n_step:                {self.n_step}")
        print(f"  gamma:                 {self.gamma}")
        print(f"  tau:                   {self.tau}")
        print(f"  normalize_rewards:     {self.normalize_rewards}")
        print(f"  temporal_noise_steps:  {self.temporal_noise_steps}")
        if self.replay_buffer:
            print(f"  buffer_size:           {self.replay_buffer.buffer_size:,}")
        print(f"{'='*60}\n")

    def test_mode(self) -> None:
        self.actor_critic.eval()

    def train_mode(self) -> None:
        self.actor_critic.train()


# ──────────────────── Helpers ──────────────────────


def _get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "lrelu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
    }
    act = activations.get(name.lower())
    if act is None:
        raise ValueError(f"Unknown activation: {name}")
    return act


class EmpiricalNormalization(nn.Module):
    """Normalize observations using running mean/variance (Welford's algorithm).

    Stores running statistics as buffers so they are saved/loaded with
    the model checkpoint. During training, call with update=True to update
    statistics from sampled batches.
    """

    def __init__(self, shape: int, device: torch.device, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(1, shape, device=device))
        self.register_buffer("_var", torch.ones(1, shape, device=device))
        self.register_buffer("_std", torch.ones(1, shape, device=device))
        self.register_buffer("_count", torch.tensor(0, dtype=torch.long, device=device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.training and update:
            self._update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        batch_size = x.shape[0]
        batch_mean = x.mean(dim=0, keepdim=True)
        batch_var = x.var(dim=0, unbiased=False, keepdim=True)

        new_count = self._count + batch_size
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        delta2 = batch_mean - self._mean
        m_a = self._var * self._count
        m_b = batch_var * batch_size
        m2 = m_a + m_b + delta2.pow(2) * (self._count * batch_size / new_count)
        self._var.copy_(m2 / new_count)
        self._std.copy_(self._var.sqrt())
        self._count.copy_(new_count)
