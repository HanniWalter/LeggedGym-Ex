#!/usr/bin/env bash
# ==============================================================================
# setup_simulator.sh
#
# Create a dedicated Python venv (in the LeggedGym-Ex root) for one of the
# supported simulators and install LeggedGym-Ex with the matching extras.
#
#   IsaacGym  -> python3.8  -> ./isaacgym_venv
#   Genesis   -> python3.10 -> ./genesis_venv
#   IsaacLab  -> python3.11 -> ./isaaclab_venv
#
# Usage:
#   ./setup_simulator.sh <isaacgym|genesis|isaaclab>
#
# Optional environment variables / flags:
#   ISAACGYM_PATH=<path>     Path to extracted IsaacGym Preview 4 directory
#                            (default: ../isaacgym, relative to repo root)
#   ISAACLAB_PATH=<path>     Path where IsaacLab repo is/should be cloned
#                            (default: ../IsaacLab, relative to repo root)
#   ISAACLAB_VERSION=<tag>   IsaacLab git tag/branch (default: v2.3.2)
#   FORCE=1                  Recreate the venv even if it already exists
# ==============================================================================

set -euo pipefail

# ---- locate repo root --------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---- defaults ----------------------------------------------------------------
ISAACGYM_PATH="${ISAACGYM_PATH:-$REPO_ROOT/../isaacgym}"
ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/../IsaacLab}"
ISAACLAB_VERSION="${ISAACLAB_VERSION:-v2.3.2}"
FORCE="${FORCE:-0}"

# ---- helpers -----------------------------------------------------------------
log()  { echo -e "\033[1;34m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn ]\033[0m $*"; }
err()  { echo -e "\033[1;31m[error]\033[0m $*" >&2; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Try to install a missing python<X.Y> via apt + the deadsnakes PPA.
# Only supports Debian/Ubuntu. Set AUTO_INSTALL_PYTHON=0 to disable.
auto_install_python() {
    local py="$1"               # e.g. python3.8
    local ver="${py#python}"    # e.g. 3.8

    if [[ "${AUTO_INSTALL_PYTHON:-1}" != "1" ]]; then
        return 1
    fi
    if ! have_cmd apt-get; then
        warn "apt-get not available; cannot auto-install $py"
        return 1
    fi

    local SUDO=""
    if [[ $EUID -ne 0 ]]; then
        if ! have_cmd sudo; then
            warn "sudo not available; cannot auto-install $py"
            return 1
        fi
        SUDO="sudo"
    fi

    log "Attempting to auto-install $py via apt + deadsnakes PPA"

    # Make sure we can add a PPA.
    if ! have_cmd add-apt-repository; then
        $SUDO apt-get update -y
        $SUDO apt-get install -y software-properties-common ca-certificates gnupg
    fi

    # Always add deadsnakes — the exact python3.X and python3.X-venv packages
    # only exist there on Ubuntu 22.04+ (apt-cache may match unrelated libs).
    log "Adding ppa:deadsnakes/ppa"
    $SUDO add-apt-repository -y ppa:deadsnakes/ppa
    $SUDO apt-get update -y

    # libpython3.X is needed at runtime by C extensions (e.g. IsaacGym)
    log "Installing ${py} ${py}-venv ${py}-distutils lib${py}"
    $SUDO apt-get install -y "${py}" "${py}-venv" "${py}-distutils" "lib${py}" || \
        $SUDO apt-get install -y "${py}" "${py}-venv" "lib${py}" || \
        $SUDO apt-get install -y "${py}" "${py}-venv"

    have_cmd "$py"
}

require_python() {
    local py="$1"
    if have_cmd "$py"; then
        return 0
    fi
    warn "$py not found, attempting installation..."
    if auto_install_python "$py" && have_cmd "$py"; then
        log "$py installed successfully"
        return 0
    fi
    err "$py not found and auto-install failed."
    err "Install it manually, e.g.:"
    err "  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt-get update"
    err "  sudo apt-get install -y $py ${py}-venv ${py}-distutils"
    err "Or set AUTO_INSTALL_PYTHON=0 to skip auto-install attempts."
    exit 1
}

create_venv() {
    local py="$1"
    local venv_dir="$2"

    if [[ -d "$venv_dir" ]]; then
        if [[ "$FORCE" == "1" ]]; then
            log "Removing existing venv $venv_dir (FORCE=1)"
            rm -rf "$venv_dir"
        else
            log "Reusing existing venv $venv_dir (set FORCE=1 to recreate)"
            return 0
        fi
    fi

    log "Creating venv at $venv_dir using $py"
    "$py" -m venv "$venv_dir"
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    deactivate
}

activate_venv() {
    # shellcheck disable=SC1091
    source "$1/bin/activate"
}

usage() {
    cat <<EOF
Usage: $0 <isaacgym|genesis|isaaclab>

Creates ./isaacgym_venv, ./genesis_venv, or ./isaaclab_venv and installs
LeggedGym-Ex with the corresponding extras.

Environment overrides:
  ISAACGYM_PATH        (default: $ISAACGYM_PATH)
  ISAACLAB_PATH        (default: $ISAACLAB_PATH)
  ISAACLAB_VERSION     (default: $ISAACLAB_VERSION)
  FORCE=1              Recreate the venv even if it already exists
  AUTO_INSTALL_PYTHON  Set to 0 to disable apt+deadsnakes auto-install (default: 1)
EOF
}

# ==============================================================================
# Per-simulator install routines
# ==============================================================================

install_isaacgym() {
    require_python python3.8
    local venv_dir="$REPO_ROOT/isaacgym_venv"

    # ninja is required by PyTorch's JIT extension builder (gymtorch compile)
    if ! have_cmd ninja; then
        log "Installing ninja-build (required by PyTorch cpp_extension / gymtorch)"
        local SUDO=""
        [[ $EUID -ne 0 ]] && SUDO="sudo"
        $SUDO apt-get install -y ninja-build
    fi
    create_venv python3.8 "$venv_dir"
    activate_venv "$venv_dir"

    log "Installing PyTorch 2.4.1 (cu121)"
    pip install torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121

    if [[ ! -d "$ISAACGYM_PATH" ]]; then
        err "IsaacGym directory not found at: $ISAACGYM_PATH"
        err "Download IsaacGym Preview 4 from NVIDIA and extract it, then re-run with"
        err "  ISAACGYM_PATH=/path/to/isaacgym $0 isaacgym"
        deactivate
        exit 1
    fi

    log "Installing IsaacGym from $ISAACGYM_PATH/python"
    pip install -e "$ISAACGYM_PATH/python"

    log "Installing LeggedGym-Ex with [isaacgym] extras"
    pip install -e "$REPO_ROOT[isaacgym]"

    deactivate
    log "IsaacGym venv ready: $venv_dir"
    cat <<EOF

To use it:
  source $venv_dir/bin/activate
  # IsaacGym needs its libs on LD_LIBRARY_PATH:
  export LD_LIBRARY_PATH=$venv_dir/lib:\$LD_LIBRARY_PATH

EOF
}

install_genesis() {
    require_python python3.10
    local venv_dir="$REPO_ROOT/genesis_venv"
    create_venv python3.10 "$venv_dir"
    activate_venv "$venv_dir"

    log "Installing PyTorch 2.8.0 (cu126)"
    pip install torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu126

    log "Installing LeggedGym-Ex with [genesis] extras"
    pip install -e "$REPO_ROOT[genesis]"

    deactivate
    log "Genesis venv ready: $venv_dir"
    cat <<EOF

To use it:
  source $venv_dir/bin/activate
  export SIMULATOR=genesis

EOF
}

install_isaaclab() {
    require_python python3.11
    local venv_dir="$REPO_ROOT/isaaclab_venv"
    create_venv python3.11 "$venv_dir"
    activate_venv "$venv_dir"

    log "Installing IsaacSim 5.1.0"
    pip install "isaacsim[all,extscache]==5.1.0" \
        --extra-index-url https://pypi.nvidia.com

    if [[ ! -d "$ISAACLAB_PATH" ]]; then
        log "Cloning IsaacLab ($ISAACLAB_VERSION) into $ISAACLAB_PATH"
        git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
        git -C "$ISAACLAB_PATH" checkout "$ISAACLAB_VERSION"
    else
        log "Using existing IsaacLab at $ISAACLAB_PATH"
    fi

    log "Running IsaacLab installer (./isaaclab.sh -i)"
    ( cd "$ISAACLAB_PATH" && ./isaaclab.sh -i )

    log "Installing LeggedGym-Ex with [isaaclab] extras"
    pip install -e "$REPO_ROOT[isaaclab]"

    deactivate
    log "IsaacLab venv ready: $venv_dir"
    cat <<EOF

To use it:
  source $venv_dir/bin/activate
  export SIMULATOR=isaaclab

EOF
}

# ==============================================================================
# Dispatch
# ==============================================================================

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

case "${1,,}" in
    isaacgym) install_isaacgym ;;
    genesis)  install_genesis  ;;
    isaaclab) install_isaaclab ;;
    -h|--help|help) usage ;;
    *)
        err "Unknown simulator: $1"
        usage
        exit 1
        ;;
esac
