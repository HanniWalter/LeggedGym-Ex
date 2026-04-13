"""SAC + AMP training runner.

Extends SACRunner with AMP reward blending and discriminator training
via AMPManager.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from rsl_rl.algorithms.sac import SAC, SACActorCritic
from rsl_rl.algorithms.sac_amp import SAC_AMP
from rsl_rl.env import VecEnv
from rsl_rl.runners.sac_runner import SACRunner, TrainConfig
from legged_gym.amp.manager import AMPManager


class SACAMPRunner(SACRunner):
    """SAC runner with Adversarial Motion Priors (AMP).

    Extends SACRunner to:
        1. Compute blended rewards using AMPManager
        2. Store AMP transitions for discriminator training
        3. Train the discriminator alongside SAC updates

    Args:
        env: Vectorized AMP environment.
        train_cfg: Training configuration dict.
        log_dir: Directory for logging and checkpoints.
        device: Torch device.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: TrainConfig,
        log_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)

    def _init_agent_and_algo(self) -> None:
        """Initialize SAC actor-critic, SAC_AMP algorithm, and AMPManager."""
        super()._init_agent_and_algo()

        # Build AMPManager from runner config
        amp_obs_dim = self.env.get_amp_observations().shape[-1]  # type: ignore[attr-defined]
        self._amp_obs_dim = amp_obs_dim

        self.amp_manager = AMPManager(
            device=self.device,
            amp_obs_dim=amp_obs_dim,
            amp_reward_coef=self.cfg.get("amp_reward_coef", 2.0),
            amp_discr_hidden_dims=self.cfg.get("amp_discr_hidden_dims", [1024, 512]),
            amp_task_reward_lerp=self.cfg.get("amp_task_reward_lerp", 0.3),
            amp_replay_buffer_size=self.alg_cfg.get("amp_replay_buffer_size", 100000),
            disc_lr=self.alg_cfg.get("disc_lr", 1e-4),
            max_grad_norm=self.alg_cfg.get("max_grad_norm", 1.0),
            motion_files=self.cfg.get("amp_motion_files", []),
            num_dof=self.env.num_actions,
            num_key_bodies=len(self.env.simulator.key_body_indices),  # type: ignore[attr-defined]
            time_between_frames=self.env.dt,  # type: ignore[attr-defined]
            num_preload_transitions=self.cfg.get("amp_num_preload_transitions", 2_000_000),
        )

        # Set fresh AMP reward callback on the algorithm so it re-computes
        # AMP rewards at training time instead of using stale stored values.
        if hasattr(self.alg, 'amp_reward_fn'):
            def _fresh_amp_reward(amp_obs: torch.Tensor, next_amp_obs: torch.Tensor, task_rewards: torch.Tensor) -> torch.Tensor:
                blended, _ = self.amp_manager.compute_reward(amp_obs, next_amp_obs, task_rewards.squeeze(-1))
                return blended.unsqueeze(-1)
            self.alg.amp_reward_fn = _fresh_amp_reward

    def _init_storage(self) -> None:
        """Initialize SAC replay buffer with AMP observation storage."""
        obs_dim = self.env.num_obs
        amp_obs_dim = getattr(self, '_amp_obs_dim', 0)
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            buffer_size=self.replay_buffer_size,
            obs_dim=obs_dim,
            action_dim=self.env.num_actions,
            amp_obs_dim=amp_obs_dim,
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run SAC + AMP training loop.

        Same structure as SACRunner.learn() but with AMP reward blending
        and discriminator updates.
        """
        self._pre_learn(init_at_random_ep_len)

        obs = self.env.get_observations().to(self.device)
        amp_obs = self.env.get_amp_observations().to(self.device)  # type: ignore[attr-defined]
        self.alg.actor_critic.train()
        self.amp_manager.discriminator.train()

        ep_infos: List[Dict[str, Any]] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        total_env_steps = 0

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ─── Collection phase ───
            mean_amp_reward_this_iter = 0.0
            steps_this_iter = 0

            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    if total_env_steps < self.warmup_steps:
                        actions = (
                            torch.rand(self.env.num_envs, self.env.num_actions, device=self.device) * 2.0 - 1.0
                        )
                    else:
                        actions = self.alg.act(obs)

                    next_obs, _, rewards, dones, infos = self.env.step(actions)
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # AMP reward blending
                    next_amp_obs = self.env.get_amp_observations().to(self.device)  # type: ignore[attr-defined]

                    # Handle terminal AMP states (from extras)
                    reset_env_ids = infos.get("reset_env_ids", None)
                    terminal_amp_states = infos.get("terminal_amp_states", None)
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    if reset_env_ids is not None and terminal_amp_states is not None and len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    # Blend rewards (for logging only — fresh rewards recomputed during training)
                    blended_rewards, amp_reward = self.amp_manager.compute_reward(
                        amp_obs, next_amp_obs_with_term, rewards
                    )

                    # Store AMP transition for discriminator training
                    self.amp_manager.store_transition(amp_obs, next_amp_obs_with_term)

                    # Save current amp_obs for replay buffer BEFORE advancing
                    amp_obs_for_store = amp_obs.clone()
                    amp_obs = torch.clone(next_amp_obs)

                    # Handle timeouts
                    dones_f = dones.float()
                    if "time_outs" in infos:
                        timeouts = infos["time_outs"].float().to(self.device)
                        effective_dones = dones_f * (1.0 - timeouts)
                    else:
                        effective_dones = dones_f

                    # Store RAW task rewards + AMP obs in replay buffer.
                    # Fresh AMP rewards will be recomputed at training time using
                    # the current discriminator (avoids stale reward problem).
                    self.alg.store_transition(
                        obs, actions, rewards, next_obs, effective_dones,
                        amp_obs=amp_obs_for_store, next_amp_obs=next_amp_obs_with_term,
                    )
                    # Update reward normalization stats with blended rewards
                    # (these track the actual return scale for the normalizer)
                    if self.alg.normalize_rewards:
                        self.alg.update_reward_stats(blended_rewards, dones_f)
                    obs = next_obs.clone()
                    total_env_steps += self.env.num_envs
                    steps_this_iter += 1
                    mean_amp_reward_this_iter += amp_reward.mean().item()

                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            infos["episode"]["rew_amp"] = amp_reward.mean() / self.env.dt  # type: ignore[attr-defined]
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += blended_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # ─── Update phase ───
            start = stop
            update_info: Dict[str, float] = {}
            amp_update_info: Dict[str, float] = {}

            if total_env_steps >= self.warmup_steps and self.alg.replay_buffer.size >= self.alg.batch_size:
                # SAC policy/critic update
                n_updates = self.updates_per_step * self.num_steps_per_env
                self.alg.utd_ratio = n_updates
                update_info = self.alg.update()
                self.alg.utd_ratio = self.updates_per_step

                # AMP discriminator update
                disc_batch_size = min(
                    self.alg.batch_size,
                    self.amp_manager.replay_buffer.num_samples,
                )
                if disc_batch_size > 0:
                    num_disc_batches = max(1, n_updates // 2)  # disc gets half as many updates as policy
                    amp_loss, grad_pen, policy_pred, expert_pred, _ = (
                        self.amp_manager.update_discriminator(
                            num_mini_batches=num_disc_batches,
                            mini_batch_size=disc_batch_size,
                        )
                    )
                    amp_update_info = {
                        "amp_loss": amp_loss,
                        "amp_grad_pen": grad_pen,
                        "amp_policy_pred": policy_pred,
                        "amp_expert_pred": expert_pred,
                    }

            stop = time.time()
            learn_time = stop - start

            # ─── Logging ───
            merged_info = {**update_info, **amp_update_info}
            if steps_this_iter > 0:
                merged_info["mean_amp_reward"] = mean_amp_reward_this_iter / steps_this_iter

            if self.log_dir is not None:
                self._log_amp(
                    it=it,
                    num_learning_iterations=num_learning_iterations,
                    ep_infos=ep_infos,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                    collection_time=collection_time,
                    learn_time=learn_time,
                    update_info=merged_info,
                    total_env_steps=total_env_steps,
                )
            if it % self.save_interval == 0:
                assert self.log_dir is not None
                ckpt_dir = os.path.join(self.log_dir, "checkpoints", f"model_{it}")
                self.save(os.path.join(ckpt_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        assert self.log_dir is not None
        final_iter = self.current_learning_iteration
        ckpt_dir = os.path.join(self.log_dir, "checkpoints", f"model_{final_iter}")
        self.save(os.path.join(ckpt_dir, f"model_{final_iter}.pt"))

    def _log_amp(
        self,
        it: int,
        num_learning_iterations: int,
        ep_infos: list,
        rewbuffer: deque,
        lenbuffer: deque,
        collection_time: float,
        learn_time: float,
        update_info: Dict[str, float],
        total_env_steps: int,
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """Log SAC + AMP training metrics."""
        assert self.writer is not None
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time

        ep_string = ""
        wandb_scalars: Dict[str, Any] = {}

        if ep_infos:
            for key in ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, it)
                wandb_scalars["Episode/" + key] = value.item()
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        fps = int(self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-6))

        critic_loss = update_info.get("critic_loss", 0.0)
        actor_loss = update_info.get("actor_loss", 0.0)
        alpha_val = update_info.get("alpha", self.alg.alpha)
        amp_loss = update_info.get("amp_loss", 0.0)
        amp_grad_pen = update_info.get("amp_grad_pen", 0.0)
        policy_pred = update_info.get("amp_policy_pred", 0.0)
        expert_pred = update_info.get("amp_expert_pred", 0.0)

        self.writer.add_scalar("Loss/critic", critic_loss, it)
        self.writer.add_scalar("Loss/actor", actor_loss, it)
        self.writer.add_scalar("SAC/alpha", alpha_val, it)
        self.writer.add_scalar("Loss/AMP", amp_loss, it)
        self.writer.add_scalar("Loss/AMP_grad", amp_grad_pen, it)
        self.writer.add_scalar("AMP/policy_pred", policy_pred, it)
        self.writer.add_scalar("AMP/expert_pred", expert_pred, it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection time", collection_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        # Q-value and action diagnostics
        for diag_key in ("q1_mean", "q2_mean", "target_q_mean", "td_target_mean",
                         "action_abs_mean", "log_prob_mean",
                         "critic_updates", "actor_updates", "learning_rate"):
            if diag_key in update_info:
                self.writer.add_scalar(f"SAC/{diag_key}", update_info[diag_key], it)
                wandb_scalars[f"SAC/{diag_key}"] = update_info[diag_key]

        wandb_scalars.update({
            "Loss/critic": critic_loss,
            "Loss/actor": actor_loss,
            "SAC/alpha": alpha_val,
            "Loss/AMP": amp_loss,
            "Loss/AMP_grad": amp_grad_pen,
            "AMP/policy_pred": policy_pred,
            "AMP/expert_pred": expert_pred,
            "Perf/total_fps": float(fps),
            "Perf/collection time": collection_time,
            "Perf/learning_time": learn_time,
        })

        if len(rewbuffer) > 0:
            mean_rew = statistics.mean(rewbuffer)
            mean_len = statistics.mean(lenbuffer)
            self.writer.add_scalar("Train/mean_reward", mean_rew, it)
            self.writer.add_scalar("Train/mean_episode_length", mean_len, it)
            wandb_scalars["Train/mean_reward"] = mean_rew
            wandb_scalars["Train/mean_episode_length"] = mean_len

        str_iter = f" \033[1m Learning iteration {it}/{self.current_learning_iteration + num_learning_iterations} \033[0m "

        if len(rewbuffer) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_iter.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                f"""{'Critic loss:':>{pad}} {critic_loss:.4f}\n"""
                f"""{'Actor loss:':>{pad}} {actor_loss:.4f}\n"""
                f"""{'Alpha:':>{pad}} {alpha_val:.4f}\n"""
                f"""{'Q1 mean:':>{pad}} {update_info.get('q1_mean', 0.0):.2f}\n"""
                f"""{'|action| mean:':>{pad}} {update_info.get('action_abs_mean', 0.0):.3f}\n"""
                f"""{'AMP loss:':>{pad}} {amp_loss:.4f}\n"""
                f"""{'AMP grad pen:':>{pad}} {amp_grad_pen:.4f}\n"""
                f"""{'AMP policy pred:':>{pad}} {policy_pred:.4f}\n"""
                f"""{'AMP expert pred:':>{pad}} {expert_pred:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(rewbuffer):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(lenbuffer):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_iter.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                f"""{'Critic loss:':>{pad}} {critic_loss:.4f}\n"""
                f"""{'Actor loss:':>{pad}} {actor_loss:.4f}\n"""
                f"""{'Alpha:':>{pad}} {alpha_val:.4f}\n"""
                f"""{'Q1 mean:':>{pad}} {update_info.get('q1_mean', 0.0):.2f}\n"""
                f"""{'AMP loss:':>{pad}} {amp_loss:.4f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (it + 1) * (num_learning_iterations - it):.1f}s\n"""
        )
        log_string += f"""{'Videos recorded:':>{pad}} {len(self._uploaded_video_iters)}\n"""

        wandb_scalars["train/total_timesteps"] = float(self.tot_timesteps)
        wandb_scalars["train/iteration_time"] = float(iteration_time)
        wandb_scalars["train/total_time"] = float(self.tot_time)
        print(log_string)
        self._wandb_log_scalars(it, wandb_scalars)
        self._check_and_upload_videos(it)

    # ──────────────── Save / Load ────────────────

    def save(self, path: str, infos: Optional[Dict[str, Any]] = None) -> None:
        """Save SAC + AMP checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        amp_state = self.amp_manager.save()
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "target_critic_state_dict": self.alg.target_critic.state_dict(),
                "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
                "log_alpha": self.alg.log_alpha.detach().cpu() if self.alg.log_alpha is not None else None,
                "discriminator_state_dict": amp_state["discriminator_state_dict"],
                "amp_normalizer": amp_state["amp_normalizer"],
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True) -> Optional[Dict[str, Any]]:
        """Load SAC + AMP checkpoint."""
        loaded = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
        if "target_critic_state_dict" in loaded:
            self.alg.target_critic.load_state_dict(loaded["target_critic_state_dict"])
        if load_optimizer:
            if "actor_optimizer_state_dict" in loaded:
                self.alg.actor_optimizer.load_state_dict(loaded["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in loaded:
                self.alg.critic_optimizer.load_state_dict(loaded["critic_optimizer_state_dict"])
        if "log_alpha" in loaded and loaded["log_alpha"] is not None and self.alg.log_alpha is not None:
            self.alg.log_alpha.data.copy_(loaded["log_alpha"].to(self.device))
            self.alg.alpha = self.alg.log_alpha.exp().item()
        if "discriminator_state_dict" in loaded:
            self.amp_manager.load({
                "discriminator_state_dict": loaded["discriminator_state_dict"],
                "amp_normalizer": loaded["amp_normalizer"],
            })
        self.current_learning_iteration = loaded["iter"]
        return loaded.get("infos")
