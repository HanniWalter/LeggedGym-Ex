"""Wrapper around AMPLoader for expert motion data sampling."""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from legged_gym.utils.motion_loader import AMPLoader


class MotionDataset:
    """Wrapper around AMPLoader providing convenient expert data access.

    Args:
        device: Torch device for tensors.
        num_dof: Number of degrees of freedom.
        num_key_bodies: Number of key bodies tracked.
        time_between_frames: Simulation time between frames.
        motion_files: List of motion file paths.
        preload_transitions: Whether to preload transitions for fast sampling.
        num_preload_transitions: Number of transitions to preload.
    """

    def __init__(
        self,
        device: Union[str, torch.device],
        num_dof: int,
        num_key_bodies: int,
        time_between_frames: float,
        motion_files: List[str],
        preload_transitions: bool = True,
        num_preload_transitions: int = 2_000_000,
    ) -> None:
        self.loader = AMPLoader(
            device=device,
            num_dof=num_dof,
            num_key_bodies=num_key_bodies,
            time_between_frames=time_between_frames,
            preload_transitions=preload_transitions,
            num_preload_transitions=num_preload_transitions,
            motion_files=motion_files,
        )

    @property
    def observation_dim(self) -> int:
        return self.loader.observation_dim

    def get_full_frame_batch(self, batch_size: int):
        """Sample a batch of full frames from the motion data."""
        return self.loader.get_full_frame_batch(batch_size)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Generate mini-batches of expert transitions."""
        return self.loader.feed_forward_generator(num_mini_batch, mini_batch_size)
