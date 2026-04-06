import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from legged_gym import PROJECT_ROOT_DIR, SIMULATOR
from legged_gym.envs import *
from legged_gym.simulator.isaacgym_simulator import IsaacGymSimulator
from legged_gym.utils import task_registry


CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(description="Record videos for newly created checkpoints.")
    parser.add_argument("--task", type=str, required=True, help="Task name.")
    parser.add_argument("--log_dir", type=str, default=None, help="Exact run directory to watch. If omitted, all runs for the task experiment are watched.")
    parser.add_argument("--load_run", type=str, default=None, help="Single run name under logs/<experiment> to watch.")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of policy steps per recording.")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Run recorder on CPU.")
    parser.add_argument("--poll_interval", type=float, default=5.0, help="Checkpoint poll interval in seconds.")
    parser.add_argument("--summary_interval", type=float, default=30.0, help="Progress summary interval in seconds.")
    parser.add_argument("--fail_fast", action="store_true", default=False, help="Stop on first checkpoint failure.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO"], help="Console log verbosity.")
    parser.add_argument("--camera_width", type=int, default=1280, help="Recording width.")
    parser.add_argument("--camera_height", type=int, default=720, help="Recording height.")
    parser.add_argument("--camera_pos", nargs=3, type=float, default=[2.0, 2.0, 1.5], help="World-space camera position.")
    parser.add_argument("--camera_lookat", nargs=3, type=float, default=[0.0, 0.0, 0.5], help="World-space camera look-at point.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries per checkpoint.")
    parser.add_argument("--retry_backoff", type=float, default=1.0, help="Initial retry backoff in seconds.")
    return parser.parse_args()


def get_task_type(task: str) -> str:
    parts = task.split("_")
    return "_".join(parts[1:])


def make_runtime_args(task: str, cpu: bool) -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        headless=True,
        cpu=cpu,
        num_envs=1,
        max_iterations=None,
        resume=False,
        sync_wandb=False,
        export_onnx=False,
        debug=False,
        load_run=None,
        ckpt=-1,
        use_joystick=False,
        joystick_type="xbox",
        follow_robot=False,
        motion_file=None,
        motion_out_dir=None,
        distill=False,
        teacher_model_path=None,
        experiment_name=None,
        run_name=None,
        run_timestamp=None,
    )


def override_configs_for_recording(env_cfg, task_type: str):
    env_cfg.env.num_envs = 1
    env_cfg.viewer.rendered_envs_idx = [0]
    env_cfg.viewer.offscreen_render = True
    env_cfg.env.debug = False
    env_cfg.env.debug_draw_height_points_around_base = False
    env_cfg.env.debug_draw_height_points_around_feet = False
    env_cfg.env.debug_draw_terrain_height_points = False
    env_cfg.env.debug_draw_key_body_points = False
    env_cfg.env.debug_draw_depth_images = False
    if task_type in {"cts", "cts_amp"}:
        env_cfg.env.num_teacher = 1
    elif "depth" in task_type:
        env_cfg.env.num_camera_envs = 1

    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.measure_heights = False

    env_cfg.commands.zero_cmd_prob = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_links = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_ctrl_delay = False
    env_cfg.domain_rand.randomize_pd_gain = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_camera_pos = False
    env_cfg.domain_rand.randomize_camera_euler = False


def initialize_observation_state(env, task_type: str):
    if task_type == "depth_ts":
        obs_buf, privileged_obs_buf, depth_image, critic_obs = env.get_observations()
        return {
            "obs_buf": obs_buf,
            "privileged_obs_buf": privileged_obs_buf,
            "depth_image": depth_image,
            "critic_obs": critic_obs,
        }
    if task_type in {"ts", "cat", "cts", "cts_amp"}:
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
        return {
            "obs_buf": obs_buf,
            "privileged_obs_buf": privileged_obs_buf,
            "obs_history": obs_history,
            "critic_obs": critic_obs,
        }
    if task_type == "ee":
        estimator_features, _, _ = env.get_observations()
        return {"estimator_features": estimator_features}
    if task_type == "dreamwaq":
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
        return {
            "obs_buf": obs_buf,
            "privileged_obs_buf": privileged_obs_buf,
            "obs_history": obs_history,
            "explicit_labels": explicit_labels,
            "next_states": next_states,
        }
    return {"obs_buf": env.get_observations()}


def step_task(env, policy, task_type: str, state: dict):
    if task_type == "depth_ts":
        actions = policy(state["obs_buf"], state["depth_image"])
        obs_buf, privileged_obs_buf, depth_image, critic_obs, rews, dones, infos = env.step(actions.detach())
        state.update(
            obs_buf=obs_buf,
            privileged_obs_buf=privileged_obs_buf,
            depth_image=depth_image,
            critic_obs=critic_obs,
        )
        return state, rews, dones, infos
    if task_type in {"ts", "cat", "cts"}:
        actions = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        state.update(
            obs_buf=obs_buf,
            privileged_obs_buf=privileged_obs_buf,
            obs_history=obs_history,
            critic_obs=critic_obs,
        )
        return state, rews, dones, infos
    if task_type == "ee":
        actions = policy(state["estimator_features"].detach())
        estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        state.update(estimator_features=estimator_features, estimator_labels=estimator_labels)
        return state, rews, dones, infos
    if task_type == "dreamwaq":
        actions = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())
        state.update(
            obs_buf=obs_buf,
            privileged_obs_buf=privileged_obs_buf,
            obs_history=obs_history,
            explicit_labels=explicit_labels,
            next_states=next_states,
        )
        return state, rews, dones, infos
    if task_type == "amp":
        actions = policy(state["obs_buf"].detach())
        obs_buf, _, rews, dones, infos, _, _ = env.step(actions.detach())
        state.update(obs_buf=obs_buf)
        return state, rews, dones, infos
    if task_type == "cts_amp":
        actions = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos, _, _ = env.step(actions.detach())
        state.update(
            obs_buf=obs_buf,
            privileged_obs_buf=privileged_obs_buf,
            obs_history=obs_history,
            critic_obs=critic_obs,
        )
        return state, rews, dones, infos

    actions = policy(state["obs_buf"].detach())
    obs_buf, _, rews, dones, infos = env.step(actions.detach())
    state.update(obs_buf=obs_buf)
    return state, rews, dones, infos


class RecorderLogger:
    def __init__(self, log_level: str):
        self._logger = logging.getLogger("checkpoint-recorder")
        self._logger.handlers.clear()
        self._logger.setLevel(getattr(logging, log_level))
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self._logger.addHandler(handler)
        self._logger.propagate = False
        self._jsonl_path = None

    def set_log_path(self, log_path: Optional[Path]):
        self._jsonl_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event: str, level: str = "INFO", message: str = "", **payload):
        getattr(self._logger, level.lower())(message or event)
        record = {
            "ts": time.time(),
            "level": level,
            "event": event,
            **payload,
        }
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


class StatusStore:
    def __init__(self):
        self._path = None
        self.data = {"checkpoints": {}}

    def set_path(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                self.data = json.load(handle)
        else:
            self._write()

    def get(self, key: str):
        return self.data["checkpoints"].get(key)

    def update(self, key: str, **fields):
        current = self.data["checkpoints"].setdefault(key, {})
        current.update(fields)
        self._write()

    def _write(self):
        if self._path is None:
            return
        tmp_path = self._path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self._path)


class RecorderRuntime:
    def __init__(self, args, logger: RecorderLogger):
        self.args = args
        self.logger = logger
        self.task_type = get_task_type(args.task)
        self.env = None
        self.runner = None
        self.policy = None
        self.run_dir = None
        self.last_processed_checkpoint = None

    def set_run_dir(self, run_dir: Path):
        self.run_dir = run_dir

    def ensure_runtime(self):
        if self.env is not None:
            return
        env_cfg, train_cfg = task_registry.get_cfgs(name=self.args.task)
        override_configs_for_recording(env_cfg, self.task_type)
        runtime_args = make_runtime_args(self.args.task, self.args.cpu)
        env, _ = task_registry.make_env(name=self.args.task, args=runtime_args, env_cfg=env_cfg)
        train_cfg.runner.resume = False
        runner, _ = task_registry.make_alg_runner(
            env=env,
            name=self.args.task,
            args=runtime_args,
            train_cfg=train_cfg,
            log_root=None,
        )
        if not isinstance(env.simulator, IsaacGymSimulator):
            raise RuntimeError("record_checkpoints.py currently supports IsaacGym only.")
        env.simulator.create_recording_camera(
            width=self.args.camera_width,
            height=self.args.camera_height,
            position=np.asarray(self.args.camera_pos, dtype=np.float32),
            target=np.asarray(self.args.camera_lookat, dtype=np.float32),
            env_index=0,
        )
        self.env = env
        self.runner = runner


def get_log_root(args) -> Path:
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    return Path(PROJECT_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name


def resolve_run_dirs(args):
    log_root = get_log_root(args)
    if args.log_dir is not None:
        return [Path(args.log_dir).expanduser().resolve()]
    if args.load_run is not None:
        return [(log_root / args.load_run).resolve()]
    if not log_root.exists():
        return []
    runs = [path for path in log_root.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime)
    return runs


def scan_checkpoints(run_dir: Path):
    checkpoints = {}
    nested_root = run_dir / "checkpoints"
    if nested_root.exists():
        for checkpoint_path in nested_root.glob("model_*/model_*.pt"):
            match = CHECKPOINT_RE.search(checkpoint_path.name)
            if match is None:
                continue
            iteration = int(match.group(1))
            checkpoints[iteration] = {
                "run_dir": run_dir,
                "iteration": iteration,
                "checkpoint_path": checkpoint_path,
                "video_path": checkpoint_path.parent / f"video_{iteration}.mp4",
            }
    for checkpoint_path in run_dir.glob("model_*.pt"):
        match = CHECKPOINT_RE.search(checkpoint_path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        checkpoints.setdefault(
            iteration,
            {
                "run_dir": run_dir,
                "iteration": iteration,
                "checkpoint_path": checkpoint_path,
                "video_path": checkpoint_path.parent / f"video_{iteration}.mp4",
            },
        )
    return [checkpoints[key] for key in sorted(checkpoints)]


def record_video(runtime: RecorderRuntime, checkpoint_info: dict):
    runtime.runner.load(str(checkpoint_info["checkpoint_path"]), load_optimizer=False)
    runtime.policy = runtime.runner.get_inference_policy(device=runtime.env.device)
    runtime.env.reset()
    state = initialize_observation_state(runtime.env, runtime.task_type)

    final_video_path = checkpoint_info["video_path"]
    temp_video_path = final_video_path.with_suffix(".tmp.mp4")
    if temp_video_path.exists():
        temp_video_path.unlink()

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        ffmpeg_cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{runtime.args.camera_width}x{runtime.args.camera_height}",
            "-r",
            str(runtime.args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_video_path),
        ]
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for _ in range(runtime.args.num_steps):
                state, _, _, _ = step_task(runtime.env, runtime.policy, runtime.task_type, state)
                frame = runtime.env.simulator.capture_recording_frame()
                frame = np.ascontiguousarray(frame)
                ffmpeg_proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            ffmpeg_proc.kill()
            ffmpeg_proc.wait()
            stderr_text = ffmpeg_proc.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg pipe broke while encoding video: {stderr_text[-2000:]}")

        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        ffmpeg_stderr = ffmpeg_proc.stderr.read()
        if ffmpeg_proc.returncode != 0:
            stderr_text = ffmpeg_stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg failed to encode video: {stderr_text[-2000:]}")
    else:
        writer = cv2.VideoWriter(
            str(temp_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            runtime.args.fps,
            (runtime.args.camera_width, runtime.args.camera_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {temp_video_path}")

        try:
            for _ in range(runtime.args.num_steps):
                state, _, _, _ = step_task(runtime.env, runtime.policy, runtime.task_type, state)
                frame = runtime.env.simulator.capture_recording_frame()
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    os.replace(temp_video_path, final_video_path)
    return final_video_path


def checkpoint_key(checkpoint_info: dict) -> str:
    return str(checkpoint_info["iteration"])


def summarize(status_stores):
    recorded = 0
    failed = 0
    recording = 0
    pending = 0
    total = 0
    for status_store in status_stores.values():
        for entry in status_store.data["checkpoints"].values():
            total += 1
            status = entry.get("status")
            if status == "recorded":
                recorded += 1
            elif status == "failed":
                failed += 1
            elif status == "recording":
                recording += 1
            else:
                pending += 1
    return {
        "pending": pending,
        "recorded": recorded,
        "failed": failed,
        "recording": recording,
        "total": total,
    }


def process_checkpoint(runtime: RecorderRuntime, status_store: StatusStore, logger: RecorderLogger, checkpoint_info: dict):
    key = checkpoint_key(checkpoint_info)
    final_video_path = checkpoint_info["video_path"]
    last_error_message = None

    for attempt in range(1, runtime.args.max_retries + 1):
        status_store.update(
            key,
            status="recording",
            checkpoint_path=str(checkpoint_info["checkpoint_path"]),
            video_path=str(final_video_path),
            attempts=attempt,
            last_attempt_ts=time.time(),
            error_message=None,
        )
        logger.event(
            "recording_started",
            message=f"Recording checkpoint {checkpoint_info['iteration']} (attempt {attempt}/{runtime.args.max_retries})",
            checkpoint_path=str(checkpoint_info["checkpoint_path"]),
            iteration=checkpoint_info["iteration"],
            attempt=attempt,
        )
        started_at = time.perf_counter()
        try:
            runtime.ensure_runtime()
            if not final_video_path.exists():
                record_video(runtime, checkpoint_info)
                logger.event(
                    "recording_done",
                    message=f"Recorded checkpoint {checkpoint_info['iteration']} to {final_video_path}",
                    checkpoint_path=str(checkpoint_info["checkpoint_path"]),
                    video_path=str(final_video_path),
                    iteration=checkpoint_info["iteration"],
                    elapsed_s=time.perf_counter() - started_at,
                )
            status_store.update(
                key,
                status="recorded",
                checkpoint_path=str(checkpoint_info["checkpoint_path"]),
                video_path=str(final_video_path),
                attempts=attempt,
                last_attempt_ts=time.time(),
                error_message=None,
            )
            logger.event(
                "recording_ready",
                message=f"Recorded video for checkpoint {checkpoint_info['iteration']}",
                checkpoint_path=str(checkpoint_info["checkpoint_path"]),
                video_path=str(final_video_path),
                iteration=checkpoint_info["iteration"],
                elapsed_s=time.perf_counter() - started_at,
            )
            runtime.last_processed_checkpoint = str(checkpoint_info["checkpoint_path"])
            return
        except Exception as exc:
            last_error_message = str(exc)
            logger.event(
                "checkpoint_failed",
                level="ERROR",
                message=f"Checkpoint {checkpoint_info['iteration']} failed on attempt {attempt}: {exc}",
                checkpoint_path=str(checkpoint_info["checkpoint_path"]),
                iteration=checkpoint_info["iteration"],
                attempt=attempt,
                elapsed_s=time.perf_counter() - started_at,
                traceback=traceback.format_exc(),
            )
            if attempt >= runtime.args.max_retries:
                status_store.update(
                    key,
                    status="failed",
                    checkpoint_path=str(checkpoint_info["checkpoint_path"]),
                    video_path=str(final_video_path),
                    attempts=attempt,
                    last_attempt_ts=time.time(),
                    error_message=last_error_message,
                )
                if runtime.args.fail_fast:
                    raise
                return
            time.sleep(runtime.args.retry_backoff * (2 ** (attempt - 1)))


def main():
    args = parse_args()
    if SIMULATOR != "isaacgym":
        raise RuntimeError("record_checkpoints.py requires SIMULATOR=isaacgym.")

    logger = RecorderLogger(args.log_level)
    runtime = RecorderRuntime(args, logger)

    logger.event(
        "watcher_started",
        message=(
            f"Recorder started for task={args.task}, run_scope={args.log_dir or args.load_run or 'all-runs'}, "
            f"fps={args.fps}, num_steps={args.num_steps}, camera_pos={args.camera_pos}, camera_lookat={args.camera_lookat}"
        ),
        task=args.task,
        run_scope=args.log_dir or args.load_run or "all-runs",
        fps=args.fps,
        num_steps=args.num_steps,
        camera_pos=args.camera_pos,
        camera_lookat=args.camera_lookat,
    )

    status_stores = {}
    known_runs = set()
    next_summary_time = time.monotonic() + args.summary_interval

    try:
        while True:
            run_dirs = resolve_run_dirs(args)
            if not run_dirs:
                logger.set_log_path(None)
                logger.event("waiting_for_run", message="Waiting for a run directory to appear.")
                time.sleep(args.poll_interval)
                continue

            for run_dir in run_dirs:
                recorder_dir = run_dir / "recorder"
                if run_dir not in status_stores:
                    run_status_store = StatusStore()
                    run_status_store.set_path(recorder_dir / "status.json")
                    status_stores[run_dir] = run_status_store
                else:
                    run_status_store = status_stores[run_dir]

                logger.set_log_path(recorder_dir / "recorder.log.jsonl")
                if run_dir not in known_runs:
                    logger.event(
                        "run_selected",
                        message=f"Watching run directory {run_dir}",
                        run_dir=str(run_dir),
                    )
                    known_runs.add(run_dir)

                runtime.set_run_dir(run_dir)
                checkpoint_infos = scan_checkpoints(run_dir)
                for checkpoint_info in checkpoint_infos:
                    key = checkpoint_key(checkpoint_info)
                    if run_status_store.get(key) is None:
                        run_status_store.update(
                            key,
                            status="pending",
                            checkpoint_path=str(checkpoint_info["checkpoint_path"]),
                            video_path=str(checkpoint_info["video_path"]),
                            attempts=0,
                            last_attempt_ts=None,
                            error_message=None,
                        )
                        logger.event(
                            "checkpoint_detected",
                            message=f"Detected checkpoint {checkpoint_info['iteration']} in {run_dir.name}",
                            checkpoint_path=str(checkpoint_info["checkpoint_path"]),
                            iteration=checkpoint_info["iteration"],
                            run_dir=str(run_dir),
                        )

                    status = (run_status_store.get(key) or {}).get("status")
                    if status == "recorded":
                        continue
                    if status == "recording":
                        continue
                    if status == "failed":
                        continue
                    process_checkpoint(runtime, run_status_store, logger, checkpoint_info)

            if time.monotonic() >= next_summary_time:
                logger.set_log_path(None)
                summary = summarize(status_stores)
                logger.event(
                    "progress_summary",
                    message=(
                        f"Progress summary: pending={summary['pending']}, recording={summary['recording']}, "
                        f"recorded={summary['recorded']}, failed={summary['failed']}, total={summary['total']}"
                    ),
                    **summary,
                )
                next_summary_time = time.monotonic() + args.summary_interval

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.set_log_path(None)
        logger.event("watcher_interrupt", message="Recorder interrupted by user.")
    finally:
        logger.set_log_path(None)
        summary = summarize(status_stores)
        logger.event(
            "watcher_stopped",
            message=(
                f"Recorder stopped: pending={summary['pending']}, recorded={summary['recorded']}, "
                f"failed={summary['failed']}, total={summary['total']}"
            ),
            last_processed_checkpoint=runtime.last_processed_checkpoint,
            **summary,
        )


if __name__ == "__main__":
    main()