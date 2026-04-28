#!/usr/bin/env bash

set -euo pipefail

# Default simulator – override with --sim <isaacgym|genesis|isaaclab>
default_sim="isaacgym"

task=""
recorder_num_steps="200"
recorder_fps="30"
recorder_log_level="INFO"
recorder_fail_fast=""
display_value="${DISPLAY:-:1}"
sim_override=""
num_envs_override=""
experiment_name_override=""
run_name_override=""
train_args=()

if command -v python >/dev/null 2>&1; then
    python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
else
    echo "Neither python nor python3 is available." >&2
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sim)
            sim_override="$2"
            shift 2
            ;;
        --num_envs)
            num_envs_override="$2"
            shift 2
            ;;
        --task)
            task="$2"
            shift 2
            ;;
        --num_steps)
            recorder_num_steps="$2"
            shift 2
            ;;
        --fps)
            recorder_fps="$2"
            shift 2
            ;;
        --log_level)
            recorder_log_level="$2"
            shift 2
            ;;
        --fail_fast)
            recorder_fail_fast="--fail_fast"
            shift
            ;;
        --display)
            display_value="$2"
            shift 2
            ;;
        --experiment_name)
            experiment_name_override="$2"
            shift 2
            ;;
        --run_name)
            run_name_override="$2"
            shift 2
            ;;
        *)
            train_args+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$task" ]]; then
    echo "Missing required --task argument" >&2
    exit 1
fi

# Resolve simulator: CLI flag > env var > default
SIMULATOR="${sim_override:-${SIMULATOR:-$default_sim}}"
export SIMULATOR
echo "[launcher] using simulator: $SIMULATOR"

# ---- activate matching venv -------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$SIMULATOR" in
    isaacgym) venv_dir="$REPO_ROOT/isaacgym_venv" ;;
    genesis)  venv_dir="$REPO_ROOT/genesis_venv"  ;;
    isaaclab) venv_dir="$REPO_ROOT/isaaclab_venv" ;;
    *)
        echo "[launcher] unknown simulator '$SIMULATOR'" >&2
        exit 1
        ;;
esac

if [[ -f "$venv_dir/bin/activate" ]]; then
    echo "[launcher] activating venv: $venv_dir"
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    python_bin="python"
    # IsaacGym C extensions need libpython3.8.so.1.0 at runtime.
    if [[ "$SIMULATOR" == "isaacgym" ]]; then
        py_libdir="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
        if [[ -n "$py_libdir" ]]; then
            export LD_LIBRARY_PATH="${py_libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            echo "[launcher] LD_LIBRARY_PATH prepended with $py_libdir"
        fi
        # Prepend venv nvidia lib dirs so they shadow older system CUDA libs
        # (e.g. libnvJitLink.so.12 from CUDA 12.0 in /usr/lib would otherwise
        # shadow the newer pip-installed version needed by PyTorch 2.4.1+cu121).
        nvidia_libs_path="$(python -c "
import os, glob, site
dirs = []
for sp in site.getsitepackages():
    for d in glob.glob(os.path.join(sp, 'nvidia', '*', 'lib')):
        if os.path.isdir(d):
            dirs.append(d)
print(':'.join(dirs))
")"
        if [[ -n "$nvidia_libs_path" ]]; then
            export LD_LIBRARY_PATH="${nvidia_libs_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            echo "[launcher] LD_LIBRARY_PATH prepended with nvidia venv libs"
        fi
    fi
else
    echo "[launcher] WARNING: venv not found at $venv_dir — run ./setup_simulator.sh $SIMULATOR first" >&2
fi

config_line="$({
    "$python_bin" - "$task" <<'PY'
import json
import sys
from legged_gym.envs import *
from legged_gym.utils import task_registry

env_cfg, train_cfg = task_registry.get_cfgs(name=sys.argv[1])
print(json.dumps({
    "experiment_name": train_cfg.runner.experiment_name,
    "run_name": train_cfg.runner.run_name,
    "num_envs": env_cfg.env.num_envs,
}))
PY
} | tail -n 1)"

default_experiment_name="$({
    CONFIG_LINE="$config_line" "$python_bin" - <<'PY'
import json
import os
config = json.loads(os.environ["CONFIG_LINE"])
print(config["experiment_name"])
PY
} | tail -n 1)"

default_run_name="$({
    CONFIG_LINE="$config_line" "$python_bin" - <<'PY'
import json
import os
config = json.loads(os.environ["CONFIG_LINE"])
print(config["run_name"])
PY
} | tail -n 1)"

default_num_envs="$({
    CONFIG_LINE="$config_line" "$python_bin" - <<'PY'
import json
import os
config = json.loads(os.environ["CONFIG_LINE"])
print(config.get("num_envs", 4096))
PY
} | tail -n 1)"
default_num_envs="${default_num_envs:-4096}"
train_num_envs="${num_envs_override:-$default_num_envs}"

experiment_name="$default_experiment_name"
if [[ -n "$experiment_name_override" ]]; then
    experiment_name="$experiment_name_override"
fi

run_name="$default_run_name"
if [[ -n "$run_name_override" ]]; then
    run_name="$run_name_override"
fi
if [[ -z "$run_name" ]]; then
    run_name="${task}_${SIMULATOR:-run}"
fi

run_timestamp="$(LC_ALL=C date +%b%d_%H-%M-%S)"
run_dir="logs/$experiment_name/${run_timestamp}_${run_name}"
mkdir -p "$run_dir"

launcher_log_file="$run_dir/launcher.log"
training_log_file="$run_dir/training.log"
watcher_log_file="$run_dir/watcher.log"
: > "$launcher_log_file"
: > "$training_log_file"
: > "$watcher_log_file"

log_launcher() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$launcher_log_file"
}

terminate_pid() {
    local pid="$1"
    local label="$2"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" >/dev/null 2>&1; then
        return
    fi

    log_launcher "stopping $label (pid=$pid)"
    kill "$pid" >/dev/null 2>&1 || true
    for _ in {1..50}; do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            return
        fi
        sleep 0.1
    done
    kill -9 "$pid" >/dev/null 2>&1 || true
}

cleanup() {
    terminate_pid "${recorder_pid:-}" "watcher"
    terminate_pid "${train_pid:-}" "training"
}

handle_interrupt() {
    log_launcher "received interrupt, stopping training and watcher"
    cleanup
    exit 130
}

trap handle_interrupt INT TERM
trap cleanup EXIT

log_launcher "using run directory $run_dir"
log_launcher "training log: $training_log_file"
log_launcher "watcher log: $watcher_log_file"
log_launcher "training output is mirrored to the terminal"

log_launcher "training with num_envs=$train_num_envs (config default: $default_num_envs)"

PYTHONUNBUFFERED=1 "$python_bin" -m legged_gym.scripts.train \
    --task "$task" \
    --run_name "$run_name" \
    --experiment_name "$experiment_name" \
    --run_timestamp "$run_timestamp" \
    --num_envs "$train_num_envs" \
    "${train_args[@]}" \
    > >(tee -a "$training_log_file") 2>&1 &
train_pid=$!
log_launcher "started training pid=$train_pid"

DISPLAY="$display_value" PYTHONUNBUFFERED=1 "$python_bin" -m legged_gym.scripts.record_checkpoints \
    --task "$task" \
    --log_dir "$run_dir" \
    --num_steps "$recorder_num_steps" \
    --fps "$recorder_fps" \
    --log_level "$recorder_log_level" \
    $recorder_fail_fast >> "$watcher_log_file" 2>&1 &
recorder_pid=$!
log_launcher "started watcher pid=$recorder_pid"

if wait "$train_pid"; then
    train_exit_code=0
else
    train_exit_code=$?
fi
log_launcher "training exited with code $train_exit_code"
terminate_pid "$recorder_pid" "watcher"
wait "$recorder_pid" || true
exit "$train_exit_code"