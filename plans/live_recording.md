# Checkpoint Video Recording

## Answers to Key Questions

### Can we record videos while headless?
**Not in the same process.** When `headless=True`, IsaacGym sets `graphics_device_id = -1` at sim creation. This disables the entire graphics pipeline — no viewer, no camera images, no rendering. This cannot be toggled mid-run.

However, IsaacGym supports **offscreen rendering** if `graphics_device_id >= 0` and no viewer is created. This works with Xvfb or EGL. A separate process can use this.

### Do we need to stop training?
**No.** A separate watcher process can load checkpoints independently while training continues. PyTorch checkpoints are atomic (write-then-rename), so reading a `.pt` file while training writes the next one is safe.

### Do we need a secondary runtime?
**Yes — a separate process is the clean solution.** Reasons:
- Training process runs `headless=True` with `graphics_device_id=-1` (no graphics pipeline)
- Cannot enable rendering mid-simulation in IsaacGym
- Separate process avoids any training slowdown
- Can run with fewer envs (e.g. 1-4) for lightweight recording

## Architecture

```
Training Process (headless)          Watcher Process (offscreen render)
─────────────────────────            ──────────────────────────────────
train.py --headless                  record_checkpoints.py --task k1_amp
  │                                    │
  ├─ model_1000.pt  ───────────────►   ├─ detect new checkpoint
  ├─ model_1200.pt  ───────────────►   ├─ load model, run N steps
  ├─ model_1400.pt  ───────────────►   ├─ capture frames via camera sensor
  │  ...                               ├─ encode to .mp4 (cv2/ffmpeg)
  │                                    └─ save next to checkpoint
```

## What the video should show

Each recorded video is a short clip (default ~200 steps ≈ 4s at 50 Hz) showing **the robot walking forward** with a fixed command:
- `lin_vel_x = 0.5 m/s`, `lin_vel_y = 0.0`, `ang_vel_yaw = 0.0`
- No joystick, no curriculum terrain — flat plane only
- 1 environment, 1 robot in frame

This is deliberately simple so every checkpoint produces a comparable clip. Over a full training run the progression from "falling over immediately" to "stable forward walking" becomes clearly visible.

## Reusable code from play.py

Most of the recording pipeline reuses existing, tested logic from `play.py`:

| Component | Source in play.py | Reuse |
|-----------|-------------------|-------|
| Task type detection | `task_type = "_".join(task.split("_")[1:])` | 1:1 |
| Config override (small env, flat plane, fixed command) | `override_configs()` | Simplified copy (no joystick, no terrain variants) |
| Env creation | `task_registry.make_env(...)` | 1:1, but with `headless=True` + offscreen render |
| Runner creation + checkpoint load | `task_registry.make_alg_runner(...)` + `runner.load(path)` | 1:1, pointed at specific checkpoint |
| Get inference policy | `ppo_runner.get_inference_policy(device=env.device)` | 1:1 |
| Observation init per task type | `if task_type == "amp": ...` branching | 1:1 |
| Step loop per task type | `if task_type == "amp": obs, _, rews, dones, infos, _, _ = env.step(...)` | 1:1, without sleep/logger/joystick |

**Not reused** (new for recorder):
- Camera sensor setup (`gymapi.CameraProperties` + `IMAGE_COLOR`)
- Frame capture per step (`gym.get_camera_image()`)
- MP4 encoding (`cv2.VideoWriter`)
- Checkpoint watcher loop
- WandB video upload
- Structured logging / status tracking

## Camera setup

We need only a **single, simple camera sensor** — no depth, no segmentation, no multi-view.

```
Camera type:    COLOR only (gymapi.IMAGE_COLOR)
Resolution:     1280 x 720 (720p)
Position:       Fixed world-space position, e.g. (2.0, 2.0, 1.5)
Look-at:        Origin (0, 0, 0.5) — roughly robot center at start
Attached to:    Environment 0 (not the robot body — stays fixed in world)
FOV:            Default (~60°)
```

This avoids complexity:
- No follow-cam (position stays constant — robot walks through frame)
- No depth buffer
- No multi-camera stitching
- One `gym.create_camera_sensor()` call, one `gym.get_camera_image()` per step

## Implementation Plan

### Step 1: Enable offscreen rendering mode
In `isaacgym_simulator.py`, allow `graphics_device_id >= 0` even when `headless=True`  
(new flag: `offscreen_render=True` → keeps `graphics_device_id=0` but no viewer)

### Step 2: Add camera-based frame capture
- Create one `gymapi.CameraProperties` camera sensor with `IMAGE_COLOR`
- Fixed world-space camera pose (position `(2, 2, 1.5)`, look-at `(0, 0, 0.5)`)
- Resolution: **1280x720 (720p)**
- Method: `capture_frame() → np.ndarray (H, W, 3)`
- Per step: `gym.step_graphics()` → `gym.render_all_camera_sensors()` → `gym.get_camera_image()`

### Step 3: Create `record_checkpoints.py` watcher script
```
scripts/record_checkpoints.py --task k1_amp [--log_dir <dir>] [--num_steps 200] [--fps 30]
```
- Watches log directory for new checkpoint folders under `checkpoints/model_*/model_*.pt`
- On new checkpoint:
  1. Create lightweight env (1 env, flat plane, offscreen render, fixed forward command)
  2. Load checkpoint into runner
  3. Get inference policy (same as play.py)
  4. Init observations per task type (same branching as play.py)
  5. Run policy for N steps, step loop per task type (same branching as play.py)
  6. Each step: capture 720p COLOR frame
  7. Encode frames to `video_<iter>.mp4` using cv2.VideoWriter
  8. Save MP4 into checkpoint folder alongside model file
- Upload generated MP4 to WandB automatically
- Skips already-recorded checkpoints (checks for existing .mp4)
- Records **all checkpoints** (no interval skipping)

### Step 3.1: Add watcher observability and debugging
- Write structured logs (JSONL) to `<run_dir>/recorder/recorder.log.jsonl`
- Mirror concise human-readable logs to stdout
- Log lifecycle events: `watcher_started`, `checkpoint_detected`, `recording_started`, `recording_done`, `wandb_upload_done`
- Log failures with traceback and context: checkpoint path, iteration, elapsed time, retry count
- Persist status per checkpoint in `<run_dir>/recorder/status.json`:
  - `pending`, `recording`, `uploaded`, `failed`
  - `error_message`, `last_attempt_ts`, `attempts`
- Retry policy for transient errors (e.g., WandB/network): 3 retries with exponential backoff
- Keep running on per-checkpoint errors (never crash the watcher for one failed checkpoint)
- Optional strict mode: `--fail_fast` to stop immediately for debugging sessions
- Optional verbosity levels: `--log_level INFO|DEBUG`

### Step 4: Convenience wrapper
A single launch script that starts both training and recording:
```bash
./scripts/train_with_video.sh --task k1_amp --headless --sync_wandb
# internally runs:
#   python train.py --task k1_amp --headless --sync_wandb &
#   DISPLAY=:1 python record_checkpoints.py --task k1_amp &
```

## Example: what a recorded step loop looks like (pseudocode)

```python
# Setup (reused from play.py)
env_cfg, train_cfg = task_registry.get_cfgs(name=task)
task_type = get_task_type(task)            # e.g. "amp"
override_configs_for_recording(env_cfg)    # 1 env, flat plane, lin_vel_x=0.5
env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)

# Load checkpoint (reused from play.py)
train_cfg.runner.resume = True
runner, _ = task_registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg)
runner.load(checkpoint_path)
policy = runner.get_inference_policy(device=env.device)

# Init observations (same branching as play.py lines 113-124)
if task_type == "amp":
    obs_buf = env.get_observations()

# Setup camera (NEW — not in play.py)
cam = create_recording_camera(env, width=1280, height=720)
writer = cv2.VideoWriter("video_1000.mp4", fourcc, fps, (1280, 720))

# Step loop (same branching as play.py lines 150-174, minus joystick/logger)
for step in range(num_steps):
    if task_type == "amp":
        actions = policy(obs_buf.detach())
        obs_buf, _, rews, dones, infos, _, _ = env.step(actions.detach())
    # ... other task_type branches identical to play.py ...

    frame = capture_frame(env, cam)        # NEW
    writer.write(frame)                    # NEW

writer.release()
upload_to_wandb("video_1000.mp4", iter=1000)  # NEW
```

## Dependencies
- `opencv-python` (cv2.VideoWriter for MP4 encoding)
- Xvfb (already installed) for offscreen display
- `wandb` (upload recorded videos automatically)

## Debugging Requirements
- Every checkpoint must end in a terminal status: `uploaded` or `failed`
- Failed checkpoints must include actionable reason and traceback in logs
- Recorder startup must print effective config (task, run_dir, fps, num_steps, camera pose)
- Recorder should expose progress summary every N seconds: pending/processed/failed counts
- On shutdown, recorder writes a final summary block with totals and last processed checkpoint

## File Changes

| File | Change |
|------|--------|
| `legged_gym/simulator/isaacgym_simulator.py` | Add offscreen render mode + frame capture API |
| `legged_gym/scripts/record_checkpoints.py` | New: checkpoint watcher + 720p fixed-camera recorder + WandB upload |
| `scripts/train_with_video.sh` | New: convenience launcher |
| `pyproject.toml` | Add opencv-python dependency |

## Final Decisions
- Video resolution: **720p (1280x720)**
- Camera: **fixed world-space pose** (simple, no follow-cam)
- Scene: **flat plane, 1 env, robot walks forward at 0.5 m/s**
- WandB: **upload every generated video automatically**
- Coverage: **record every checkpoint**