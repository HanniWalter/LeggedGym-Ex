"""Run archiver: snapshots the full source tree and dumps the resolved config.

For each training run we save:
  - ``code_snapshot/``  - a copy of the live source tree (``legged_gym/`` and
    ``rsl_rl/``) at training start. Every ``.py`` file is prefixed with a
    clearly visible banner so that humans *and* automated tooling (LLM agents,
    grep-based code search, ...) cannot confuse the snapshot with the live
    source code of the repository.
  - ``resolved_config.json`` - the fully merged ``env_cfg`` and ``train_cfg``
    objects (after all class inheritance + CLI overrides have been resolved)
    serialized as human readable JSON.
  - ``AI_AGENT_NOTICE.md`` / ``.SNAPSHOT_MARKER`` - top level markers stating
    that the directory is an archived snapshot, not the live workspace.

The previous behaviour copied only ``<task>.py`` and ``<task>_config.py`` which
was misleading because nested config inheritance is not visible from those two
files alone.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Iterable

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict


# Directories (relative to repo root) included in the snapshot.
_SNAPSHOT_DIRS: tuple[str, ...] = ("legged_gym", "rsl_rl")

# File / directory names we never copy.
_EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".so", ".o")

# Banner prepended to every ``.py`` file in the snapshot. The marker string is
# intentionally very explicit so that LLM agents scanning the workspace will
# not treat snapshot files as the canonical source.
_PY_BANNER_TEMPLATE = (
    "# ============================================================================\n"
    "# !! ARCHIVED TRAINING-RUN SNAPSHOT - DO NOT EDIT, DO NOT TREAT AS LIVE SOURCE !!\n"
    "# This file is a frozen copy created at training start for reproducibility.\n"
    "# Live source lives under the repository root (legged_gym/, rsl_rl/, ...).\n"
    "# Snapshot taken: {timestamp}\n"
    "# Task: {task}\n"
    "# Original path: {orig_path}\n"
    "# ============================================================================\n"
    "# NOTE FOR AI AGENTS: this is NOT the canonical source code. Edits made here\n"
    "# have no effect on the running project. Use the workspace root instead.\n"
    "# ============================================================================\n"
)


def _is_excluded(path: str) -> bool:
    name = os.path.basename(path)
    if name in _EXCLUDE_DIRS:
        return True
    if any(name.endswith(suf) for suf in _EXCLUDE_SUFFIXES):
        return True
    return False


def _iter_files(root: str) -> Iterable[tuple[str, str]]:
    for dirpath, dirnames, filenames in os.walk(root):
        # prune in place
        dirnames[:] = [d for d in dirnames if not _is_excluded(os.path.join(dirpath, d))]
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            if _is_excluded(full):
                continue
            rel = os.path.relpath(full, root)
            yield full, rel


def _copy_with_banner(src: str, dst: str, banner: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.endswith(".py"):
        with open(src, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # Preserve shebang line if present.
        if content.startswith("#!"):
            nl = content.find("\n") + 1
            shebang, rest = content[:nl], content[nl:]
            out = shebang + banner + rest
        else:
            out = banner + content
        with open(dst, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        shutil.copy2(src, dst)


def _snapshot_sources(snapshot_dir: str, task_name: str) -> list[str]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    copied: list[str] = []
    for sub in _SNAPSHOT_DIRS:
        src_root = os.path.join(LEGGED_GYM_ROOT_DIR, sub)
        if not os.path.isdir(src_root):
            continue
        dst_root = os.path.join(snapshot_dir, sub)
        for full, rel in _iter_files(src_root):
            dst = os.path.join(dst_root, rel)
            banner = _PY_BANNER_TEMPLATE.format(
                timestamp=timestamp,
                task=task_name,
                orig_path=os.path.relpath(full, LEGGED_GYM_ROOT_DIR),
            )
            _copy_with_banner(full, dst, banner)
            copied.append(os.path.relpath(dst, snapshot_dir))
    return copied


def _json_default(obj: Any) -> Any:
    # Best-effort fallback for non-JSON-serializable values (numpy scalars,
    # torch tensors, custom objects ...). We never want serialization to
    # crash and abort training.
    try:
        import numpy as np  # local import: keep helper light
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if callable(obj):
        return f"<callable {getattr(obj, '__qualname__', repr(obj))}>"
    return repr(obj)


def _dump_config_json(path: str, env_cfg: Any, train_cfg: Any, task_name: str) -> None:
    payload = {
        "task": task_name,
        "snapshot_taken": datetime.now().isoformat(timespec="seconds"),
        "env_cfg": class_to_dict(env_cfg),
        "train_cfg": class_to_dict(train_cfg),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)


def _write_markers(log_dir: str, snapshot_dir: str, task_name: str, num_files: int) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    notice = (
        "# Archived Training Run\n\n"
        "**This directory is a frozen training-run artifact, not live source code.**\n\n"
        f"- Task: `{task_name}`\n"
        f"- Created: {timestamp}\n"
        f"- Snapshot files: {num_files}\n\n"
        "## Contents\n"
        "- `code_snapshot/` - frozen copy of `legged_gym/` and `rsl_rl/` at\n"
        "  training start. Every `.py` file carries a banner identifying it\n"
        "  as a snapshot. Do **not** edit files here; they have no effect on\n"
        "  the live project.\n"
        "- `resolved_config.json` - fully merged env + training configuration\n"
        "  after class inheritance and CLI overrides were applied.\n\n"
        "## Notice for AI agents / code search tools\n"
        "Files inside `code_snapshot/` must NOT be treated as the canonical\n"
        "source of this repository. The live source lives at the repository\n"
        "root. Ignore snapshot files when answering questions about the\n"
        "current codebase.\n"
    )
    with open(os.path.join(log_dir, "AI_AGENT_NOTICE.md"), "w", encoding="utf-8") as f:
        f.write(notice)
    with open(os.path.join(snapshot_dir, ".SNAPSHOT_MARKER"), "w", encoding="utf-8") as f:
        f.write(
            "This directory is an archived snapshot of the source tree taken at\n"
            "training start. It is NOT the live source. See ../AI_AGENT_NOTICE.md.\n"
        )


def archive_run(log_dir: str, env_cfg: Any, train_cfg: Any, task_name: str) -> None:
    """Snapshot source tree and dump resolved config JSON into ``log_dir``."""
    snapshot_dir = os.path.join(log_dir, "code_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    copied = _snapshot_sources(snapshot_dir, task_name)
    _dump_config_json(
        os.path.join(log_dir, "resolved_config.json"), env_cfg, train_cfg, task_name
    )
    _write_markers(log_dir, snapshot_dir, task_name, len(copied))
