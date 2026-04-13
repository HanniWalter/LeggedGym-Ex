"""Off-policy SAC training runner.

Collects transitions one step at a time, stores them in a replay buffer,
and performs multiple gradient updates per environment step.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import statistics
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms.sac import SAC, SACActorCritic
from rsl_rl.env import VecEnv
import rsl_rl.algorithms as _algorithms
import rsl_rl.modules as _modules

# Reuse config type aliases from on_policy_runner
RunnerConfig = Dict[str, Any]
AlgorithmConfig = Dict[str, Any]
PolicyConfig = Dict[str, Any]
TrainConfig = Dict[str, Any]


class SACRunner:
    """Off-policy SAC training runner.

    Differences from OnPolicyRunner:
        - Collects single env steps (not full rollouts)
        - Stores transitions in a replay buffer
        - Updates after a warmup period, with configurable UTD ratio
        - No GAE / return computation

    Args:
        env: Vectorized environment.
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
        self.cfg: RunnerConfig = train_cfg["runner"]
        self.alg_cfg: AlgorithmConfig = train_cfg["algorithm"]
        self.policy_cfg: PolicyConfig = train_cfg["policy"]
        self.all_cfg: TrainConfig = train_cfg
        self.device = torch.device(device)
        self.env = env

        self._init_agent_and_algo()

        # SAC-specific runner config
        self.save_interval: int = self.cfg.get("save_interval", 200)
        self.num_steps_per_env: int = self.cfg.get("num_steps_per_env", 24)
        self.replay_buffer_size: int = self.cfg.get("replay_buffer_size", 1_000_000)
        
        # Warmup steps: either absolute value or computed from multiplier
        warmup_steps_cfg = self.cfg.get("warmup_steps", 1000)
        warmup_multiplier = self.cfg.get("warmup_steps_per_env", None)
        if warmup_multiplier is not None:
            self.warmup_steps: int = int(self.env.num_envs * warmup_multiplier)
        else:
            self.warmup_steps: int = warmup_steps_cfg
        
        self.updates_per_step: int = self.alg_cfg.get("utd_ratio", 1)

        self._init_storage()

        # Logging
        self.log_dir = log_dir
        if log_dir is not None:
            self.wandb_run_name = os.path.basename(log_dir)
        else:
            self.wandb_run_name = (
                self.cfg.get("run_name", "sac")
                + "_"
                + datetime.now().strftime("%b%d_%H-%M-%S")
            )
        self.sync_wandb: bool = self.cfg.get("sync_wandb", False)
        self._wandb_sync_tensorboard: bool = False
        self.writer: Optional[SummaryWriter] = None
        self._uploaded_video_iters: set = set()
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.current_learning_iteration: int = 0

        self.env.reset()

    def _init_agent_and_algo(self) -> None:
        """Initialize SAC actor-critic and algorithm.

        NOTE: Unlike PPO, SAC stores transitions in a replay buffer with a single
        observation type.  Both actor and critic therefore receive ``num_obs``
        (not ``num_privileged_obs``) so that dimensions are consistent between
        network construction and the observations sampled from the buffer.
        """
        num_critic_obs = self.env.num_obs  # SAC critic uses same obs as actor

        policy_class_name = self.cfg.get("policy_class_name", "SACActorCritic")
        policy_class = getattr(_algorithms, policy_class_name, None) \
            or getattr(_modules, policy_class_name, None)
        if policy_class is None:
            raise ValueError(f"Unknown policy class: {policy_class_name}")
        actor_critic = policy_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        alg_class_name = self.cfg.get("algorithm_class_name", "SAC")
        alg_class = getattr(_algorithms, alg_class_name, None)
        if alg_class is None:
            raise ValueError(f"Unknown algorithm class: {alg_class_name}")
        self.alg: SAC = alg_class(actor_critic, device=self.device, **self.alg_cfg)

    def _init_storage(self) -> None:
        """Initialize SAC replay buffer."""
        if self.env.num_privileged_obs is not None:
            obs_dim = self.env.num_privileged_obs
        else:
            obs_dim = self.env.num_obs
        # Use actor obs_dim for the buffer since we store actor observations
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            buffer_size=self.replay_buffer_size,
            obs_dim=self.env.num_obs,
            action_dim=self.env.num_actions,
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run the SAC training loop.

        Each "iteration" corresponds to one full sweep of env steps
        (``num_steps_per_env`` steps), mirroring the PPO iteration structure
        so that save_interval and max_iterations remain comparable.

        Args:
            num_learning_iterations: Number of iterations (each = num_steps_per_env env steps).
            init_at_random_ep_len: Whether to randomise initial episode lengths.
        """
        self._pre_learn(init_at_random_ep_len)

        obs = self.env.get_observations().to(self.device)
        self.alg.actor_critic.train()

        ep_infos: List[Dict[str, Any]] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        total_env_steps = 0  # total env steps collected so far (across all iterations)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ─── Collect num_steps_per_env transitions ───
            iter_update_info: Dict[str, float] = {}
            num_updates_this_iter = 0

            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    # Warm-up: random actions; after warm-up: policy actions
                    if total_env_steps < self.warmup_steps:
                        actions = torch.rand(
                            self.env.num_envs, self.env.num_actions, device=self.device
                        ) * 2.0 - 1.0
                    else:
                        actions = self.alg.act(obs)

                    next_obs, _, rewards, dones, infos = self.env.step(actions)
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Handle timeouts (bootstrap)
                    dones_f = dones.float()
                    if "time_outs" in infos:
                        # For SAC we do NOT bootstrap – we just mark them as not done
                        # so the target uses next_obs value. The buffer stores raw done=0.
                        timeouts = infos["time_outs"].float().to(self.device)
                        effective_dones = dones_f * (1.0 - timeouts)
                    else:
                        effective_dones = dones_f

                    self.alg.store_transition(obs, actions, rewards, next_obs, effective_dones)
                    # Update reward normalization stats
                    if self.alg.normalize_rewards:
                        self.alg.update_reward_stats(rewards)
                    obs = next_obs.clone()
                    total_env_steps += self.env.num_envs

                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
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
            if total_env_steps >= self.warmup_steps and self.alg.replay_buffer.size >= self.alg.batch_size:
                # Perform updates_per_step * num_steps_per_env gradient steps
                n_updates = self.updates_per_step * self.num_steps_per_env
                self.alg.utd_ratio = n_updates  # batch all updates in one call
                iter_update_info = self.alg.update()
                self.alg.utd_ratio = self.updates_per_step  # restore
                num_updates_this_iter = n_updates

            stop = time.time()
            learn_time = stop - start

            # ─── Logging ───
            if self.log_dir is not None:
                self._log(
                    it=it,
                    num_learning_iterations=num_learning_iterations,
                    ep_infos=ep_infos,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                    collection_time=collection_time,
                    learn_time=learn_time,
                    update_info=iter_update_info,
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

    # ──────────────── Logging ────────────────

    def _pre_learn(self, init_at_random_ep_len: bool) -> None:
        if self.log_dir is not None and self.writer is None:
            if self.sync_wandb:
                run = wandb.init(
                    project="LeggedGym-Ex",
                    name=self.wandb_run_name,
                    sync_tensorboard=self._wandb_sync_tensorboard,
                    config=self.all_cfg,
                )
                self._write_wandb_run_info(run)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

    def _write_wandb_run_info(self, run) -> None:
        if self.log_dir is None or run is None:
            return
        run_info_path = os.path.join(self.log_dir, "wandb_run.json")
        run_info = {
            "id": run.id,
            "name": run.name,
            "project": run.project,
            "entity": run.entity,
            "path": "/".join(run.path),
            "url": run.url,
        }
        tmp_path = run_info_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(run_info, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, run_info_path)

    def _wandb_log_scalars(self, iteration: int, scalars: Dict[str, Any]) -> None:
        if not self.sync_wandb:
            return
        if self._wandb_sync_tensorboard:
            return
        wandb.log(scalars, step=iteration)

    _VIDEO_RE = re.compile(r"video_(\d+)\.mp4$")

    def _check_and_upload_videos(self, current_it: int) -> None:
        """Check for and upload checkpoint videos (same logic as OnPolicyRunner)."""
        if self.log_dir is None:
            return
        checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.isdir(checkpoints_dir):
            return
        for model_dir in os.scandir(checkpoints_dir):
            if not model_dir.is_dir():
                continue
            for entry in os.scandir(model_dir.path):
                m = self._VIDEO_RE.search(entry.name)
                if m is None:
                    continue
                iteration = int(m.group(1))
                if iteration in self._uploaded_video_iters:
                    continue
                self._uploaded_video_iters.add(iteration)
                if not self.sync_wandb:
                    continue
                try:
                    video_payload = {
                        "Video/checkpoint_video": wandb.Video(
                            entry.path,
                            format="mp4",
                            caption=f"checkpoint {iteration} at iter {current_it}",
                        )
                    }
                    if self._wandb_sync_tensorboard:
                        wandb.log(video_payload)
                    else:
                        wandb.log(video_payload, step=current_it)
                except Exception as exc:
                    print(f"[video-upload] ERROR: {exc}")

    def _log(
        self,
        it: int,
        num_learning_iterations: int,
        ep_infos: List[Dict[str, Any]],
        rewbuffer: deque,
        lenbuffer: deque,
        collection_time: float,
        learn_time: float,
        update_info: Dict[str, float],
        total_env_steps: int,
        width: int = 80,
        pad: int = 35,
    ) -> None:
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

        fps = int(
            self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-6)
        )

        # SAC-specific losses
        critic_loss = update_info.get("critic_loss", 0.0)
        actor_loss = update_info.get("actor_loss", 0.0)
        alpha_loss = update_info.get("alpha_loss", 0.0)
        alpha_val = update_info.get("alpha", self.alg.alpha)

        self.writer.add_scalar("Loss/critic", critic_loss, it)
        self.writer.add_scalar("Loss/actor", actor_loss, it)
        self.writer.add_scalar("Loss/alpha", alpha_loss, it)
        self.writer.add_scalar("SAC/alpha", alpha_val, it)
        self.writer.add_scalar("SAC/replay_buffer_size", self.alg.replay_buffer.size, it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection time", collection_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        wandb_scalars["Loss/critic"] = critic_loss
        wandb_scalars["Loss/actor"] = actor_loss
        wandb_scalars["Loss/alpha"] = alpha_loss
        wandb_scalars["SAC/alpha"] = alpha_val
        wandb_scalars["SAC/replay_buffer_size"] = float(self.alg.replay_buffer.size)
        wandb_scalars["Perf/total_fps"] = float(fps)
        wandb_scalars["Perf/collection time"] = collection_time
        wandb_scalars["Perf/learning_time"] = learn_time

        # Q-value and action diagnostics
        for diag_key in ("q1_mean", "q2_mean", "target_q_mean", "td_target_mean",
                         "action_abs_mean", "log_prob_mean",
                         "critic_updates", "actor_updates"):
            if diag_key in update_info:
                self.writer.add_scalar(f"SAC/{diag_key}", update_info[diag_key], it)
                wandb_scalars[f"SAC/{diag_key}"] = update_info[diag_key]

        if len(rewbuffer) > 0:
            mean_rew = statistics.mean(rewbuffer)
            mean_len = statistics.mean(lenbuffer)
            self.writer.add_scalar("Train/mean_reward", mean_rew, it)
            self.writer.add_scalar("Train/mean_episode_length", mean_len, it)
            self.writer.add_scalar("Train/mean_reward/time", mean_rew, self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", mean_len, self.tot_time)
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
                f"""{'Target Q mean:':>{pad}} {update_info.get('target_q_mean', 0.0):.2f}\n"""
                f"""{'|action| mean:':>{pad}} {update_info.get('action_abs_mean', 0.0):.3f}\n"""
                f"""{'Replay buffer size:':>{pad}} {self.alg.replay_buffer.size}\n"""
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
                f"""{'Replay buffer size:':>{pad}} {self.alg.replay_buffer.size}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Total env steps:':>{pad}} {total_env_steps}\n"""
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

    # ──────────────── Save / Load / Inference ────────────────

    def save(self, path: str, infos: Optional[Dict[str, Any]] = None) -> None:
        """Save SAC checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "target_critic_state_dict": self.alg.target_critic.state_dict(),
                "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
                "log_alpha": self.alg.log_alpha.detach().cpu() if self.alg.log_alpha is not None else None,
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True) -> Optional[Dict[str, Any]]:
        """Load SAC checkpoint."""
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
        self.current_learning_iteration = loaded["iter"]
        return loaded.get("infos")

    def get_inference_policy(
        self, device: Optional[Union[str, torch.device]] = None
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get deterministic inference policy."""
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
