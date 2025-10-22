# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command implementations for push task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

# Import the base command from dexsuite
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.commands.pose_commands_cfg import ObjectUniformPoseCommandCfg
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.commands.pose_commands import ObjectUniformPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class PushObjectUniformPoseCommandCfg(ObjectUniformPoseCommandCfg):
    """Configuration for push task pose command generator with cube avoidance."""
    
    success_threshold: float = 0.12
    """Success threshold radius (in meters) - cube must be within this distance of target to succeed."""
    
    min_gap_from_cube: float = 0.01
    """Minimum gap (in meters) between cube spawn and success boundary."""
    
    cube_half_extent: float = 0.0203
    """Half-extent of the cube in meters (for blue_block.usd with scale 1.0, side=0.0406m, so half=0.0203m)."""


class PushObjectUniformPoseCommand(ObjectUniformPoseCommand):
    """Uniform pose command generator for push task with proper cube avoidance.
    
    This command term samples target object poses while ensuring they don't spawn
    too close to the cube's ACTUAL position, accounting for the cube's volume and 
    the success sphere radius to prevent immediate success states or impossible targets.
    
    Physics:
        - Cube is a 0.0406m sided box (half-extent: 0.0203m from center)
        - Success sphere has radius defined by success_threshold (default 0.12m)
        - For no overlap: distance >= cube_half_extent + success_threshold + min_gap
        - Default: 0.0203 + 0.12 + 0.01 = 0.1503m minimum separation
    """

    cfg: PushObjectUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: PushObjectUniformPoseCommandCfg, env: ManagerBasedEnv):
        # Initialize the base class
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample command with proper cube avoidance using actual cube positions.
        
        This method:
        1. Gets the ACTUAL cube position for each environment (not hardcoded)
        2. Samples target positions uniformly
        3. Checks if target success spheres would overlap with cube volumes
        4. Rejects and resamples any targets that are too close
        
        Args:
            env_ids: Environment IDs to resample commands for
        """
        # Get ACTUAL cube positions (randomized at reset)
        # Shape: (num_envs, 3) in world frame
        cube_pos_w = self.object.data.root_pos_w[env_ids]
        
        # Transform cube positions to robot base frame (where commands are defined)
        from isaaclab.utils.math import subtract_frame_transforms
        cube_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids], 
            cube_pos_w,
            torch.zeros((cube_pos_w.shape[0], 4), device=self.device)
        )
        
        # Convert env_ids to tensor for consistent indexing
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        elif isinstance(env_ids, slice):
            # If slice, get all environment indices
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            # If list or other sequence, convert to tensor
            env_ids_tensor = torch.tensor(list(env_ids), device=self.device)
        
        b = len(env_ids_tensor)
        
        # Calculate minimum distance: cube_half_extent + success_threshold + gap
        # This ensures the success sphere doesn't overlap with the cube volume
        min_distance = self.cfg.cube_half_extent + self.cfg.success_threshold + self.cfg.min_gap_from_cube
        
        # Sample random angles (0 to 2π) for each environment
        angles = torch.rand(b, device=self.device) * 2 * 3.14159
        
        # Calculate max possible distance based on workspace bounds
        # This is distance from origin to corner of workspace
        max_x = max(abs(self.cfg.ranges.pos_x[0]), abs(self.cfg.ranges.pos_x[1]))
        max_y = max(abs(self.cfg.ranges.pos_y[0]), abs(self.cfg.ranges.pos_y[1]))
        max_possible_dist = (max_x**2 + max_y**2)**0.5
        
        # Ensure max distance is at least min_distance
        effective_max_dist = max(max_possible_dist, min_distance)
        
        # Sample distances between [min_distance, effective_max_dist]
        distances = torch.rand(b, device=self.device) * (effective_max_dist - min_distance) + min_distance
        
        # Convert polar coordinates to cartesian, relative to cube position
        target_xy = torch.zeros((b, 2), device=self.device)
        target_xy[:, 0] = cube_pos_b[:, 0] + distances * torch.cos(angles)
        target_xy[:, 1] = cube_pos_b[:, 1] + distances * torch.sin(angles)
        
        # Clamp to table bounds to ensure target stays in workspace
        target_xy[:, 0] = torch.clamp(target_xy[:, 0], self.cfg.ranges.pos_x[0], self.cfg.ranges.pos_x[1])
        target_xy[:, 1] = torch.clamp(target_xy[:, 1], self.cfg.ranges.pos_y[0], self.cfg.ranges.pos_y[1])
        
        # Assign to command buffer
        self.pose_command_b[env_ids_tensor, 0] = target_xy[:, 0]
        self.pose_command_b[env_ids_tensor, 1] = target_xy[:, 1]
        self.pose_command_b[env_ids_tensor, 2] = self.cfg.ranges.pos_z[0]  # Fixed Z height
        
        # Sample orientation (same as base class)
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids_tensor, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        
        from isaaclab.utils.math import quat_from_euler_xyz, quat_unique
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids_tensor, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_metrics(self):
        """Override to use parameterized success threshold for visualization."""
        # Call parent method to update pose_command_w and basic metrics
        super()._update_metrics()
        
        # Update success visualization with our parameterized threshold
        if self.cfg.position_only:
            distance = torch.norm(self.pose_command_w[:, :3] - self.object.data.root_pos_w[:, :3], dim=1)
            success_id = (distance < self.cfg.success_threshold).int()
            # Update goal position visualization with correct threshold
            if hasattr(self, 'goal_visualizer'):
                self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=success_id + 1)
            # Update current object position visualization
            if hasattr(self, 'curr_visualizer'):
                self.curr_visualizer.visualize(self.object.data.root_pos_w, marker_indices=success_id + 1)
