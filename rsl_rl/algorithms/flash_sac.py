from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.modules.flash_sac_actor_critic import FlashSACActorCritic
from rsl_rl.storage.sac_replay_buffer import SACReplayBuffer


def _build_truncated_zeta_cdf(mu: float, max_n: int, device: torch.device) -> torch.Tensor:
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    pmf = ns.pow(-mu)
    pmf = pmf / pmf.sum()
    return torch.cumsum(pmf, dim=0)


class FlashSAC:
    def __init__(
        self,
        actor_critic: FlashSACActorCritic,
        learning_rate: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        init_alpha: float = 0.01,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        batch_size: int = 2048,
        utd_ratio: int = 2,
        max_grad_norm: float = 1.0,
        policy_delay: int = 2,
        use_adamw: bool = False,
        n_step: int = 1,
        normalize_rewards: bool = True,
        normalize_observations: bool = False,
        temporal_noise_steps: int = 0,
        noise_zeta_mu: float = 2.0,
        noise_zeta_max: int = 16,
        temp_target_sigma: float = 0.15,
        critic_num_bins: int = 101,
        critic_min_v: float = -5.0,
        critic_max_v: float = 5.0,
        learning_rate_end: float = 1.5e-4,
        learning_rate_decay_steps: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        del normalize_observations, temporal_noise_steps, target_entropy, kwargs
        self.device = torch.device(device)
        self.actor_critic = actor_critic
        self.target_critic = copy.deepcopy(actor_critic.critic).to(self.device)
        self.target_critic.eval()

        # FP16 mixed-precision (paper feature)
        self._use_amp = self.device.type == "cuda"
        self._grad_scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        if self._use_amp:
            torch.set_float32_matmul_precision("high")  # enable TF32 tensor cores

        # torch.compile actor & critic for throughput (paper feature)
        try:
            self.actor_critic.actor = torch.compile(self.actor_critic.actor, mode="reduce-overhead")
            self.actor_critic.critic = torch.compile(self.actor_critic.critic, mode="reduce-overhead")
            self.target_critic = torch.compile(self.target_critic, mode="reduce-overhead")
        except Exception:
            pass  # graceful fallback if compile not supported

        opt_class = optim.AdamW if use_adamw else optim.Adam
        self.actor_optimizer = opt_class(self.actor_critic.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = opt_class(self.actor_critic.critic.parameters(), lr=learning_rate)

        self.auto_alpha = auto_alpha
        self.log_alpha = torch.tensor([math.log(max(init_alpha, 1e-8))], device=self.device, requires_grad=True)
        self.alpha_optimizer = opt_class([self.log_alpha], lr=alpha_lr)
        self.alpha = float(self.log_alpha.exp().item())

        self.target_entropy = 0.5 * self.actor_critic.num_actions * math.log(2.0 * math.pi * math.e * temp_target_sigma**2)

        # Cosine decay LR schedulers (paper §4.1)
        self._lr_peak = learning_rate
        self._lr_end = learning_rate_end
        self._lr_decay_steps = learning_rate_decay_steps
        if learning_rate_decay_steps > 0:
            def _cosine_lr(step: int) -> float:
                if step >= learning_rate_decay_steps:
                    return learning_rate_end / learning_rate
                progress = step / learning_rate_decay_steps
                lr = learning_rate_end + (learning_rate - learning_rate_end) * 0.5 * (1.0 + math.cos(math.pi * progress))
                return lr / learning_rate
            self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=_cosine_lr)
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=_cosine_lr)
            self.alpha_scheduler = optim.lr_scheduler.LambdaLR(self.alpha_optimizer, lr_lambda=_cosine_lr)
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None
            self.alpha_scheduler = None

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.max_grad_norm = max_grad_norm
        self.policy_delay = max(1, policy_delay)
        self.n_step = n_step
        self.normalize_rewards = normalize_rewards

        self.replay_buffer: Optional[SACReplayBuffer] = None
        self.optimizer = self.actor_optimizer

        # Optional callback for fresh AMP reward re-computation.
        # Set by the runner: fn(amp_obs, next_amp_obs, task_rewards) -> blended_rewards
        self.amp_reward_fn: Optional[callable] = None

        self._critic_update_count = 0
        self._total_updates = 0

        self.noise_zeta_cdf = _build_truncated_zeta_cdf(noise_zeta_mu, max(1, noise_zeta_max), self.device)
        # Per-env noise repetition (initialized properly in init_storage)
        self._cur_noise_repeat_n: Optional[torch.Tensor] = None   # (num_envs,)
        self._cur_noise_repeat_count: Optional[torch.Tensor] = None  # (num_envs,)
        self._cached_noise: Optional[torch.Tensor] = None  # (num_envs, action_dim)

        self.critic_num_bins = critic_num_bins
        self.critic_min_v = critic_min_v
        self.critic_max_v = critic_max_v

        # Return-based reward normalizer state (initialized in init_storage)
        # Implements paper Eq 6: r_bar = r / max(sqrt(sigma_G^2 + eps), G_t_max / G_max)
        self._G_r: Optional[torch.Tensor] = None          # per-env discounted return tracker
        self._G_r_max = torch.tensor(1.0, device=self.device)  # max |G_r| seen
        self._G_count = torch.tensor(0.0, device=self.device)  # Welford count
        self._G_mean = torch.zeros(1, device=self.device)      # Welford mean
        self._G_M2 = torch.zeros(1, device=self.device)        # Welford sum of squares
        self._G_var = torch.ones(1, device=self.device)         # Welford variance

    def init_storage(self, num_envs: int, buffer_size: int, obs_dim: int, action_dim: int, amp_obs_dim: int = 0) -> None:
        self.replay_buffer = SACReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            device=self.device,
            n_step=self.n_step,
            gamma=self.gamma,
            num_envs=num_envs,
            amp_obs_dim=amp_obs_dim,
        )
        # Init per-env return tracker
        self._G_r = torch.zeros(num_envs, device=self.device)
        # Init per-env noise repetition counters
        self._cur_noise_repeat_n = torch.ones(num_envs, dtype=torch.int32, device=self.device)
        self._cur_noise_repeat_count = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._cached_noise = None

    def _sample_n(self, count: int = 1) -> torch.Tensor:
        """Sample noise repetition lengths from truncated Zeta distribution.

        Args:
            count: Number of samples to draw (one per environment needing reinit).
        Returns:
            Tensor of shape (count,) with int32 repetition lengths >= 1.
        """
        u = torch.rand(count, device=self.device)
        # For each u, find first CDF bin that exceeds it
        idx = torch.argmax((u.unsqueeze(-1) < self.noise_zeta_cdf.unsqueeze(0)).to(torch.int32), dim=-1)
        return (idx + 1).to(torch.int32)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            if deterministic:
                action, _ = self.actor_critic.act(obs, deterministic=True)
                return action

            num_envs = obs.shape[0]
            mean, std = self.actor_critic.actor.get_mean_and_std(obs, training=False)

            # Lazy init of per-env noise state (handles first call before init_storage)
            if self._cached_noise is None or self._cached_noise.shape != mean.shape:
                self._cached_noise = torch.randn_like(mean)
                self._cur_noise_repeat_count = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
                self._cur_noise_repeat_n = torch.ones(num_envs, dtype=torch.int32, device=self.device)

            # Per-env reinit mask: reinit noise where count exhausted
            reinit = (self._cur_noise_repeat_count >= self._cur_noise_repeat_n)
            n_reinit = reinit.sum().item()
            if n_reinit > 0:
                self._cached_noise[reinit] = torch.randn(n_reinit, mean.shape[-1], device=self.device)
                self._cur_noise_repeat_n[reinit] = self._sample_n(n_reinit)
                self._cur_noise_repeat_count[reinit] = 0

            action = torch.tanh(mean + std * self._cached_noise)
            self._cur_noise_repeat_count += 1
            return action

    def store_transition(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        amp_obs: Optional[torch.Tensor] = None,
        next_amp_obs: Optional[torch.Tensor] = None,
    ) -> None:
        assert self.replay_buffer is not None
        self.replay_buffer.insert(obs, actions, rewards, next_obs, dones, amp_obs, next_amp_obs)

    def _select_min_q_log_probs(self, next_qs: torch.Tensor, next_q_log_probs: torch.Tensor) -> torch.Tensor:
        num_bins = next_q_log_probs.shape[-1]
        min_indices = next_qs.argmin(dim=0)
        selected = torch.gather(
            next_q_log_probs,
            dim=0,
            index=min_indices[None, :, None].expand(1, -1, num_bins),
        )[0]
        return selected

    def _compute_categorical_td_target(
        self,
        target_log_probs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        actor_entropy: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = reward.shape[0]
        reward = reward.reshape(-1, 1)
        done = done.reshape(-1, 1)
        actor_entropy = actor_entropy.reshape(-1, 1)

        num_bins = self.critic_num_bins
        min_v = self.critic_min_v
        max_v = self.critic_max_v
        bin_width = (max_v - min_v) / (num_bins - 1)
        bin_values = torch.linspace(min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype).view(1, -1)

        target_bin_values = reward + (self.gamma ** self.n_step) * (bin_values - actor_entropy) * (1.0 - done)
        target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

        b = (target_bin_values - min_v) / bin_width
        lower = torch.floor(b).long()
        upper = torch.clamp(lower + 1, 0, num_bins - 1)
        frac = b - lower.float()

        target_probs_exp = target_log_probs.exp()
        m_l = target_probs_exp * (1.0 - frac)
        m_u = target_probs_exp * frac

        target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)
        target_probs.scatter_add_(1, lower, m_l)
        target_probs.scatter_add_(1, upper, m_u)
        return target_probs

    @staticmethod
    @torch.no_grad()
    def _normalize_net(net: torch.nn.Module) -> None:
        """Normalize weight-normalized layers in a network sub-tree."""
        for m in net.modules():
            fn = getattr(m, "normalize_parameters", None)
            if fn is not None:
                fn()

    def update(self) -> Dict[str, float]:
        assert self.replay_buffer is not None
        assert self.replay_buffer.size >= self.batch_size

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        total_alpha = 0.0
        actor_updates = 0

        total_q1_mean = 0.0
        total_q2_mean = 0.0
        total_target_q_mean = 0.0
        total_log_prob_mean = 0.0
        total_action_abs_mean = 0.0

        for _ in range(self.utd_ratio):
            sample = self.replay_buffer.sample(self.batch_size)
            obs, actions, rewards, next_obs, dones = sample[:5]

            # Re-compute fresh AMP rewards if callback is set
            if self.amp_reward_fn is not None and len(sample) > 5:
                amp_obs_batch, next_amp_obs_batch = sample[5], sample[6]
                rewards = self.amp_reward_fn(amp_obs_batch, next_amp_obs_batch, rewards)

            if self.normalize_rewards:
                rewards = self._normalize_reward(rewards)

            # ── Step 1: Actor update (matches reference: actor before critic) ──
            do_actor_update = (self._critic_update_count % self.policy_delay == 0)
            if do_actor_update:
                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    # Forward actor on [obs, next_obs] for better BatchNorm statistics
                    actor_obs_all = torch.cat([obs, next_obs], dim=0)
                    all_actions, all_info = self.actor_critic.actor(actor_obs_all, training=True)
                    new_actions = torch.chunk(all_actions, 2, dim=0)[0]
                    log_probs = torch.chunk(all_info["log_prob"], 2, dim=0)[0]

                    # Evaluate Q without flowing gradients into critic
                    self.actor_critic.critic.requires_grad_(False)
                    qs, _ = self.actor_critic.critic(obs, new_actions, training=False)
                    q = torch.minimum(qs[0], qs[1])
                    self.actor_critic.critic.requires_grad_(True)

                    alpha = self.log_alpha.exp().detach()
                    actor_loss = (log_probs * alpha - q).mean()

                self.actor_optimizer.zero_grad()
                self._grad_scaler.scale(actor_loss).backward()
                self._grad_scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm)
                self._grad_scaler.step(self.actor_optimizer)
                self._normalize_net(self.actor_critic.actor)

                # ── Step 2: Temperature update (reference formulation) ──
                entropy = -log_probs.detach().mean()
                alpha_val = self.log_alpha.exp()
                alpha_loss = alpha_val * (entropy - self.target_entropy)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().item())

                total_actor_loss += actor_loss.item()
                total_alpha_loss += alpha_loss.item()
                total_alpha += self.alpha
                actor_updates += 1
                total_action_abs_mean += new_actions.detach().abs().mean().item()
                total_log_prob_mean += log_probs.detach().mean().item()

            # ── Step 3: Critic update (uses fresh actor after step 1) ──
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    next_actions, info = self.actor_critic.actor(next_obs, training=False)
                    next_log_probs = info["log_prob"]
                    temp_value = self.log_alpha.exp()
                    next_actor_entropy = temp_value * next_log_probs

                    obs_all = torch.cat([obs, next_obs], dim=0)
                    act_all = torch.cat([actions, next_actions], dim=0)
                    qs_all, q_infos_all = self.target_critic(obs_all, act_all, training=True)
                    next_qs = qs_all.chunk(2, dim=1)[1]
                    next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
                    next_q_log_probs = self._select_min_q_log_probs(next_qs, next_q_log_probs)
                    target_probs = self._compute_categorical_td_target(
                        target_log_probs=next_q_log_probs,
                        reward=rewards,
                        done=dones,
                        actor_entropy=next_actor_entropy,
                    )

            with torch.amp.autocast("cuda", enabled=self._use_amp):
                pred_qs_all, pred_q_infos = self.actor_critic.critic(obs_all, act_all, training=True)
                pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]
                ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)
                critic_loss = ce_loss.mean()

            self.critic_optimizer.zero_grad()
            self._grad_scaler.scale(critic_loss).backward()
            self._grad_scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)
            self._grad_scaler.step(self.critic_optimizer)
            self._normalize_net(self.actor_critic.critic)

            # Update scaler once per UTD step
            self._grad_scaler.update()

            # ── Step 4: Target network EMA ──
            with torch.no_grad():
                for p, p_targ in zip(self.actor_critic.critic.parameters(), self.target_critic.parameters()):
                    p_targ.data.mul_(1.0 - self.tau)
                    p_targ.data.add_(self.tau * p.data)

            self._critic_update_count += 1
            self._total_updates += 1

            total_critic_loss += critic_loss.item()
            with torch.no_grad():
                q1, q2 = self.actor_critic.evaluate_q(obs, actions)
                total_q1_mean += q1.mean().item()
                total_q2_mean += q2.mean().item()
                total_target_q_mean += next_qs.mean().item()

        n = self.utd_ratio
        n_actor = max(1, actor_updates)
        cur_lr = self.actor_optimizer.param_groups[0]["lr"]

        # ── Step 5: LR scheduler (once per iteration) ──
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        if self.alpha_scheduler is not None:
            self.alpha_scheduler.step()
        return {
            "critic_loss": total_critic_loss / n,
            "actor_loss": total_actor_loss / n_actor,
            "alpha_loss": total_alpha_loss / n_actor,
            "alpha": total_alpha / n_actor if actor_updates > 0 else self.alpha,
            "q1_mean": total_q1_mean / n,
            "q2_mean": total_q2_mean / n,
            "target_q_mean": total_target_q_mean / n,
            "action_abs_mean": total_action_abs_mean / n_actor if actor_updates > 0 else 0.0,
            "log_prob_mean": total_log_prob_mean / n_actor if actor_updates > 0 else 0.0,
            "critic_updates": float(n),
            "actor_updates": float(actor_updates),
            "total_updates": float(self._total_updates),
            "learning_rate": cur_lr,
        }

    def update_reward_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Update return-based reward normalizer (parallel Welford + G_r_max)."""
        assert self._G_r is not None
        # Reset return tracker on episode boundaries, then accumulate
        self._G_r = self.gamma * (1.0 - dones.float()) * self._G_r + rewards
        # Track max absolute return seen (for Eq 6 denominator)
        self._G_r_max = torch.maximum(self._G_r_max, torch.max(torch.abs(self._G_r)))
        # Parallel Welford batch update for variance
        batch = self._G_r
        batch_count = batch.numel()
        batch_mean = batch.mean()
        batch_var = batch.var(correction=0)
        delta = batch_mean - self._G_mean
        new_count = self._G_count + batch_count
        self._G_mean = self._G_mean + delta * batch_count / new_count
        self._G_M2 = self._G_M2 + batch_var * batch_count + delta ** 2 * self._G_count * batch_count / new_count
        self._G_count = new_count
        if self._G_count > 1:
            self._G_var = self._G_M2 / self._G_count

    def _normalize_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards per paper Eq 6:
        r_bar = r / max(sqrt(sigma_G^2 + eps), G_t_max / G_max)
        """
        eps = 1e-8
        var_denom = torch.sqrt(self._G_var + eps)
        min_required_denom = self._G_r_max / self.critic_max_v
        denom = torch.maximum(var_denom, min_required_denom)
        return rewards / denom.clamp(min=eps)

    def test_mode(self) -> None:
        self.actor_critic.eval()

    def train_mode(self) -> None:
        self.actor_critic.train()
