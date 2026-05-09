# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from __future__ import annotations

import time
import os
import json
import shutil
import subprocess
from collections import deque
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import re
import wandb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


# Type aliases for configuration dictionaries
RunnerConfig = Dict[str, Any]
AlgorithmConfig = Dict[str, Any]
PolicyConfig = Dict[str, Any]
TrainConfig = Dict[str, Any]


class OnPolicyRunner:
    """On-policy RL training runner for PPO-style algorithms."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: TrainConfig,
        log_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialize the on-policy runner.

        Args:
            env: Vectorized environment for training.
            train_cfg: Training configuration containing runner, algorithm, and policy configs.
            log_dir: Directory for logging and saving models.
            device: Device to run training on (e.g., 'cpu', 'cuda').
        """
        self.cfg: RunnerConfig = train_cfg["runner"]
        self.alg_cfg: AlgorithmConfig = train_cfg["algorithm"]
        self.policy_cfg: PolicyConfig = train_cfg["policy"]
        self.all_cfg: TrainConfig = train_cfg
        self.device: torch.device = torch.device(device)
        self.env: VecEnv = env
        self._init_agent_and_algo()
        self.num_steps_per_env: int = self.cfg["num_steps_per_env"]
        self.save_interval: int = self.cfg["save_interval"]
        self._init_storage()

        # Log
        self.log_dir: Optional[str] = log_dir
        if log_dir is not None:
            self.wandb_run_name: str = os.path.basename(log_dir)
        else:
            self.wandb_run_name: str = (
                self.cfg["run_name"]
                + "_"
                + datetime.now().strftime("%b%d_%H-%M-%S")
            )
        self.sync_wandb: bool = self.cfg.get("sync_wandb", False)
        self._wandb_sync_tensorboard: bool = True
        self.writer: Optional[SummaryWriter] = None
        self._uploaded_video_iters: set = set()
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.current_learning_iteration: int = 0

        self.env.reset()
    
    def _init_agent_and_algo(self) -> None:
        """Initialize the actor-critic network and PPO algorithm."""
        if self.env.num_privileged_obs is not None:
            num_critic_obs: int = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
    
    def _init_storage(self) -> None:
        """Initialize the rollout storage for the algorithm."""
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env, 
            (self.env.num_obs,),
            (self.env.num_privileged_obs,), 
            (self.env.num_actions,),
        )
    
    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run the training loop for a specified number of iterations.

        Args:
            num_learning_iterations: Number of learning iterations to run.
            init_at_random_ep_len: Whether to initialize episode lengths randomly.
        """
        self._pre_learn(init_at_random_ep_len)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos: List[Dict[str, Any]] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                assert self.log_dir is not None
                ckpt_dir = os.path.join(self.log_dir, 'checkpoints', f'model_{it}')
                self.save(os.path.join(ckpt_dir, f'model_{it}.pt'))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        assert self.log_dir is not None
        final_iter = self.current_learning_iteration
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints', f'model_{final_iter}')
        self.save(os.path.join(ckpt_dir, f'model_{final_iter}.pt'))
        self._wait_and_upload_final_video(final_iter)

    def _wait_and_upload_final_video(self, final_iter: int, timeout_s: float = 120.0) -> None:
        """Wait up to *timeout_s* for the recorder to produce the final checkpoint
        video, then upload it to wandb.  Called once at the very end of training
        so the last checkpoint video is not missed.
        """
        if self.log_dir is None:
            return
        if final_iter in self._uploaded_video_iters:
            return  # already handled inside the loop (edge case)

        video_path = os.path.join(
            self.log_dir, 'checkpoints', f'model_{final_iter}', f'video_{final_iter}.mp4'
        )
        print(
            f"[video-upload] waiting up to {timeout_s:.0f}s for final checkpoint "
            f"video (iteration {final_iter}) …"
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if os.path.isfile(video_path) and self._probe_video_file(video_path):
                break
            time.sleep(2.0)
        else:
            print(
                f"[video-upload] timed out waiting for final checkpoint video "
                f"at {video_path} — skipping upload"
            )
            return

        print(f"[video-upload] final checkpoint video is ready: {video_path}")
        if not self.sync_wandb:
            self._uploaded_video_iters.add(final_iter)
            print("[video-upload] skipping upload (sync_wandb=False)")
            return
        try:
            staged_path = self._prepare_video_for_wandb(video_path, final_iter, final_iter)
            video_payload = {
                "Video/checkpoint_video": wandb.Video(
                    staged_path,
                    format="mp4",
                    caption=f"checkpoint {final_iter} recorded at iteration {final_iter}",
                )
            }
            if self._wandb_sync_tensorboard:
                wandb.log(video_payload)
            else:
                wandb.log(video_payload, step=final_iter)
            self._uploaded_video_iters.add(final_iter)
            print(f"[video-upload] uploaded final checkpoint video (iteration {final_iter})")
        except Exception as exc:
            self._uploaded_video_iters.add(final_iter)
            print(f"[video-upload] ERROR uploading final checkpoint video: {exc}")

    def _pre_learn(self, init_at_random_ep_len: bool) -> None:
        """Prepare for training by initializing logging and episode buffers.

        Args:
            init_at_random_ep_len: Whether to randomize initial episode lengths.
        """
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

    def _video_upload_stage_dir(self) -> Optional[str]:
        if self.log_dir is None:
            return None
        stage_dir = os.path.join(self.log_dir, "wandb_video_staging")
        os.makedirs(stage_dir, exist_ok=True)
        return stage_dir

    def _probe_video_file(self, path: str) -> bool:
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            return os.path.isfile(path) and os.path.getsize(path) > 1024
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,nb_frames,width,height",
                "-of",
                "default=noprint_wrappers=1",
                path,
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _wait_for_stable_video_file(self, path: str, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        last_size = -1
        stable_reads = 0
        while time.monotonic() < deadline:
            if not os.path.isfile(path):
                time.sleep(0.2)
                continue
            current_size = os.path.getsize(path)
            if current_size > 0 and current_size == last_size:
                stable_reads += 1
            else:
                stable_reads = 0
            last_size = current_size
            if stable_reads >= 2 and self._probe_video_file(path):
                return True
            time.sleep(0.2)
        return False

    def _prepare_video_for_wandb(self, src_path: str, iteration: int, current_it: int) -> str:
        if not self._wait_for_stable_video_file(src_path):
            raise RuntimeError(f"source video is not stable/valid yet: {src_path}")

        stage_dir = self._video_upload_stage_dir()
        if stage_dir is None:
            raise RuntimeError("log_dir is not set; cannot stage video for wandb upload")

        staged_path = os.path.join(stage_dir, f"checkpoint_{iteration}_iter_{current_it}.mp4")
        tmp_staged_path = staged_path + ".tmp"

        # Copy the source file to a stable staging location.
        # This avoids race conditions where wandb reads a file still being written.
        shutil.copyfile(src_path, tmp_staged_path)
        os.replace(tmp_staged_path, staged_path)

        if not self._probe_video_file(staged_path):
            raise RuntimeError(f"staged wandb video is not valid: {staged_path}")

        return staged_path

    def _check_and_upload_videos(self, current_it: int) -> None:
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
                print(
                    f"[video-upload] found video for checkpoint {iteration} at training iteration {current_it}: {entry.path}"
                )
                if not self.sync_wandb:
                    self._uploaded_video_iters.add(iteration)
                    print(f"[video-upload] skipping upload (sync_wandb=False)")
                    continue
                try:
                    staged_path = self._prepare_video_for_wandb(entry.path, iteration, current_it)
                    video_payload = {
                        "Video/checkpoint_video": wandb.Video(
                            staged_path,
                            format="mp4",
                            caption=f"checkpoint {iteration} recorded at iteration {current_it}",
                        )
                    }
                    if self._wandb_sync_tensorboard:
                        wandb.log(video_payload)
                    else:
                        wandb.log(video_payload, step=current_it)
                    self._uploaded_video_iters.add(iteration)
                    print(
                        f"[video-upload] uploaded video for checkpoint {iteration} at training iteration {current_it}"
                    )
                except Exception as exc:
                    self._uploaded_video_iters.add(iteration)
                    print(f"[video-upload] ERROR uploading checkpoint {iteration} (will not retry): {exc}")

    def log(
        self,
        locs: Dict[str, Any],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """Log training metrics to tensorboard and console.

        Args:
            locs: Dictionary containing iteration metrics and buffers.
            width: Width of the log output.
            pad: Padding for log formatting.
        """
        assert self.writer is not None
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_scalars: Dict[str, Any] = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                wandb_scalars['Episode/' + key] = value.item()
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        wandb_scalars['Loss/value_function'] = float(locs['mean_value_loss'])
        wandb_scalars['Loss/surrogate'] = float(locs['mean_surrogate_loss'])
        wandb_scalars['Loss/learning_rate'] = float(self.alg.learning_rate)
        wandb_scalars['Policy/mean_noise_std'] = mean_std.item()
        wandb_scalars['Perf/total_fps'] = float(fps)
        wandb_scalars['Perf/collection time'] = float(locs['collection_time'])
        wandb_scalars['Perf/learning_time'] = float(locs['learn_time'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            wandb_scalars['Train/mean_reward'] = float(statistics.mean(locs['rewbuffer']))
            wandb_scalars['Train/mean_episode_length'] = float(statistics.mean(locs['lenbuffer']))
            wandb_scalars['Train/mean_reward/time'] = float(statistics.mean(locs['rewbuffer']))
            wandb_scalars['Train/mean_episode_length/time'] = float(statistics.mean(locs['lenbuffer']))

        str_iter = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        log_string += f"""{'Videos recorded:':>{pad}} {len(self._uploaded_video_iters)}\n"""
        wandb_scalars['train/total_timesteps'] = float(self.tot_timesteps)
        wandb_scalars['train/iteration_time'] = float(iteration_time)
        wandb_scalars['train/total_time'] = float(self.tot_time)
        print(log_string)
        self._wandb_log_scalars(locs['it'], wandb_scalars)
        self._check_and_upload_videos(locs['it'])

    def save(
        self,
        path: str,
        infos: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the model checkpoint to disk.

        Args:
            path: File path to save the checkpoint.
            infos: Optional additional information to save with the checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Load a model checkpoint from disk.

        Args:
            path: File path to load the checkpoint from.
            load_optimizer: Whether to load the optimizer state.

        Returns:
            Optional infos dict stored in the checkpoint.
        """
        try:
            loaded_dict = torch.load(path, weights_only=False)
        except TypeError:
            loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the inference policy function.

        Args:
            device: Device to run inference on. If None, uses current device.

        Returns:
            Callable that takes observations and returns actions.
        """
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
