# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push task specific event functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def separate_target_from_cube(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    min_separation: float = 0.10,
):
    """Move target away from cube if they are too close.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to check.
        min_separation: Minimum distance between cube and target (in meters).
    """
    if env_ids is None:
        return
    
    cube = env.scene["cube"]
    target = env.scene["target"]
    
    # Get positions in world frame
    cube_pos = cube.data.root_pos_w[env_ids, :2]  # Only x,y
    target_pos = target.data.root_pos_w[env_ids, :2]  # Only x,y
    
    # Calculate distance
    diff = target_pos - cube_pos
    distance = torch.norm(diff, dim=-1)
    
    # Find environments where target is too close to cube
    too_close = distance < min_separation
    
    if too_close.any():
        # For each environment that's too close, move target away
        for idx, env_id in enumerate(env_ids):
            if too_close[idx]:
                # Calculate direction from cube to target
                direction = diff[idx]
                if torch.norm(direction) < 1e-6:
                    # If exactly on top, push in random direction
                    direction = torch.randn(2, device=env.device)
                
                # Normalize and scale to minimum separation
                direction = direction / torch.norm(direction) * min_separation
                
                # Set new target position
                new_target_pos = cube_pos[idx] + direction
                
                # Get current target pose
                current_pose = target.data.root_pose_w[env_id:env_id+1].clone()
                # Update x,y position, keep z and orientation
                current_pose[0, 0:2] = new_target_pos
                
                # Write to simulation
                target.write_root_pose_to_sim(current_pose, env_ids=torch.tensor([env_id.item()], device=env.device))
                target.write_root_velocity_to_sim(
                    torch.zeros(1, 6, device=env.device),
                    env_ids=torch.tensor([env_id.item()], device=env.device)
                )

