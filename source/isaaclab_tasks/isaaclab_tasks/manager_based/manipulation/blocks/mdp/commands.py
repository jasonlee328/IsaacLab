# Copyright (c) 2024-2025, The ISAAC Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator for blocks manipulation tasks with relative pose targets."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import BlocksPoseCommandCfg


class BlocksPoseCommand(CommandTerm):
    """Relative pose command generator with exclusion zones and optional discrete slicing.
    
    This command term generates target poses relative to an object's current position with:
    - Exclusion zones to prevent targets from being too close
    - Optional discrete slicing for curriculum learning
    - Comprehensive metrics tracking for both position and orientation
    - Support for position-only or full pose targets
    """

    cfg: BlocksPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: BlocksPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and object
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in robot base frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        
        # -- metrics (comprehensive tracking like RandomTableGoalCommand)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)
        
        # Success tracking
        self.position_aligned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.orientation_aligned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.euler_xy_distance = torch.zeros(self.num_envs, device=self.device)
        self.xyz_distance = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "BlocksPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tPosition only: {self.cfg.position_only}\n"
        if hasattr(self.cfg, 'slice_counts'):
            msg += f"\tSlice counts: x={self.cfg.slice_counts.pos_x}, y={self.cfg.slice_counts.pos_y}, z={self.cfg.slice_counts.pos_z}\n"
            if not self.cfg.position_only:
                msg += f"\t             roll={self.cfg.slice_counts.roll}, pitch={self.cfg.slice_counts.pitch}, yaw={self.cfg.slice_counts.yaw}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # Transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        
        # Compute the error between command and object
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        
        # Basic metrics
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        
        # Detailed position and orientation tracking
        self.xyz_distance = self.metrics["position_error"]
        
        # Calculate Euler angle errors for orientation (only if not position_only)
        if not self.cfg.position_only:
            # Convert rotation error to euler angles for more intuitive metrics
            e_x, e_y, e_z = euler_xyz_from_quat(self.object.data.root_state_w[:, 3:7])
            cmd_e_x, cmd_e_y, cmd_e_z = euler_xyz_from_quat(self.pose_command_w[:, 3:])
            
            # Wrap angles to [-pi, pi] and compute absolute differences
            diff_x = wrap_to_pi(cmd_e_x - e_x).abs()
            diff_y = wrap_to_pi(cmd_e_y - e_y).abs()
            diff_z = wrap_to_pi(cmd_e_z - e_z).abs()
            
            # Sum of roll and pitch errors (often used in manipulation)
            self.euler_xy_distance = diff_x + diff_y
        else:
            self.euler_xy_distance = torch.zeros_like(self.xyz_distance)
        
        # Check alignment based on thresholds
        self.position_aligned = self.xyz_distance < self.cfg.success_position_threshold
        if not self.cfg.position_only:
            self.orientation_aligned = self.euler_xy_distance < self.cfg.success_orientation_threshold
        else:
            self.orientation_aligned = torch.ones_like(self.position_aligned)  # Always true if position_only
        
        # Update average metrics
        self.metrics["average_pos_align_error"] = self.xyz_distance
        self.metrics["average_rot_align_error"] = self.euler_xy_distance
        
        # Log end of episode data
        reset_env = self._env.episode_length_buf == 0
        if reset_env.any():
            self.metrics["end_of_episode_pos_align_error"][reset_env] = self.xyz_distance[reset_env]
            self.metrics["end_of_episode_rot_align_error"][reset_env] = self.euler_xy_distance[reset_env]
            last_episode_success = (self.position_aligned & self.orientation_aligned)[reset_env]
            self.metrics["end_of_episode_success_rate"][reset_env] = last_episode_success.float()

    def _resample_command(self, env_ids: Sequence[int]):
        # Get current object position in world frame
        object_pos_w = self.object.data.root_pos_w[env_ids]
        object_quat_w = self.object.data.root_quat_w[env_ids]

        # Transform object position from world to robot base frame
        robot_pos_w = self.robot.data.root_pos_w[env_ids]
        robot_quat_w = self.robot.data.root_quat_w[env_ids]

        # Compute object position in robot base frame
        object_pos_b, object_quat_b = self._transform_world_to_base(
            object_pos_w, object_quat_w, robot_pos_w, robot_quat_w
        )

        # Sample relative offsets
        num_envs = len(env_ids)
        max_attempts = 1000  # Increased for complex constraints
        
        # Initialize offsets
        offset_x = torch.zeros(num_envs, device=self.device)
        offset_y = torch.zeros(num_envs, device=self.device)
        offset_z = torch.zeros(num_envs, device=self.device)
        offset_roll = torch.zeros(num_envs, device=self.device)
        offset_pitch = torch.zeros(num_envs, device=self.device)
        offset_yaw = torch.zeros(num_envs, device=self.device)
        
        # Keep track of which environments still need valid samples
        valid_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        for attempt in range(max_attempts):
            # Sample offsets for environments that don't have valid positions yet
            invalid_indices = ~valid_mask
            if not invalid_indices.any():
                break  # All environments have valid positions
            
            num_invalid = invalid_indices.sum().item()
            
            # Sample positions with optional per-axis slicing
            offset_x[invalid_indices] = self._sample_with_exclusion_and_slice(
                self.cfg.ranges.pos_x, 
                self.cfg.exclusion_ranges.pos_x if hasattr(self.cfg, 'exclusion_ranges') else (0, 0),
                num_invalid,
                self.cfg.slice_counts.pos_x if hasattr(self.cfg, 'slice_counts') else None
            )
            
            offset_y[invalid_indices] = self._sample_with_exclusion_and_slice(
                self.cfg.ranges.pos_y,
                self.cfg.exclusion_ranges.pos_y if hasattr(self.cfg, 'exclusion_ranges') else (0, 0),
                num_invalid,
                self.cfg.slice_counts.pos_y if hasattr(self.cfg, 'slice_counts') else None
            )
            
            offset_z[invalid_indices] = self._sample_with_exclusion_and_slice(
                self.cfg.ranges.pos_z,
                self.cfg.exclusion_ranges.pos_z if hasattr(self.cfg, 'exclusion_ranges') else (0, 0),
                num_invalid,
                self.cfg.slice_counts.pos_z if hasattr(self.cfg, 'slice_counts') else None
            )
            
            # Sample orientations (if not position_only)
            if not self.cfg.position_only:
                offset_roll[invalid_indices] = self._sample_with_exclusion_and_slice(
                    self.cfg.ranges.roll,
                    self.cfg.exclusion_ranges.roll if hasattr(self.cfg, 'exclusion_ranges') else (0, 0),
                    num_invalid,
                    self.cfg.slice_counts.roll if hasattr(self.cfg, 'slice_counts') else None
                )
                
                offset_pitch[invalid_indices] = self._sample_with_exclusion_and_slice(
                    self.cfg.ranges.pitch,
                    self.cfg.exclusion_ranges.pitch if hasattr(self.cfg, 'exclusion_ranges') else (0, 0),
                    num_invalid,
                    self.cfg.slice_counts.pitch if hasattr(self.cfg, 'slice_counts') else None
                )
                
                offset_yaw[invalid_indices] = self._sample_with_exclusion_and_slice(
                    self.cfg.ranges.yaw,
                    self.cfg.exclusion_ranges.yaw if hasattr(self.cfg, 'exclusion_ranges') else (0, 0),
                    num_invalid,
                    self.cfg.slice_counts.yaw if hasattr(self.cfg, 'slice_counts') else None
                )
            
            # All samples are valid since exclusion is handled in sampling
            valid_mask[invalid_indices] = True

        # Add offsets to current object position (in base frame)
        self.pose_command_b[env_ids, 0] = object_pos_b[:, 0] + offset_x
        self.pose_command_b[env_ids, 1] = object_pos_b[:, 1] + offset_y
        self.pose_command_b[env_ids, 2] = object_pos_b[:, 2] + offset_z

        # Handle orientation
        if not self.cfg.position_only:
            # Generate quaternion from sampled euler angles
            quat = quat_from_euler_xyz(offset_roll, offset_pitch, offset_yaw)
            # Combine with object's current orientation if desired
            if self.cfg.relative_orientation:
                # Apply rotation relative to current orientation
                self.pose_command_b[env_ids, 3:] = self._quat_multiply(object_quat_b, quat)
            else:
                # Use absolute orientation
                self.pose_command_b[env_ids, 3:] = quat
            
            # Make quaternion unique if requested
            if self.cfg.make_quat_unique:
                self.pose_command_b[env_ids, 3:] = quat_unique(self.pose_command_b[env_ids, 3:])
        else:
            # Keep current orientation if position_only
            self.pose_command_b[env_ids, 3:] = object_quat_b

    def _sample_with_exclusion_and_slice(
        self,
        range_tuple: tuple[float, float],
        exclusion_tuple: tuple[float, float],
        num_samples: int,
        slice_count: int | None = None,
    ) -> torch.Tensor:
        """Sample values with exclusion zones and optional discrete slicing.
        
        Args:
            range_tuple: (min, max) range for sampling
            exclusion_tuple: (min, max) exclusion zone to avoid
            num_samples: Number of samples to generate
            slice_count: If provided, discretize the range into this many slices
            
        Returns:
            Sampled values that respect the constraints
        """
        min_val, max_val = range_tuple
        excl_min, excl_max = exclusion_tuple
        
        # Handle case where range is a single point (min == max)
        if min_val == max_val:
            # Return the single value for all samples
            return torch.full((num_samples,), min_val, device=self.device)
        
        # Handle case where exclusion zone covers entire range
        if excl_min <= min_val and excl_max >= max_val:
            raise ValueError(f"Exclusion zone [{excl_min}, {excl_max}] covers entire range [{min_val}, {max_val}]")
        
        # If no exclusion zone or exclusion zone is outside range
        if excl_max <= excl_min or excl_max <= min_val or excl_min >= max_val:
            # Simple sampling without exclusion
            if slice_count is None:
                return torch.empty(num_samples, device=self.device).uniform_(min_val, max_val)
            else:
                # Discrete slicing
                slice_vals = torch.linspace(min_val, max_val, slice_count + 1, device=self.device)
                slice_centers = (slice_vals[:-1] + slice_vals[1:]) / 2
                indices = torch.randint(0, slice_count, (num_samples,), device=self.device)
                return slice_centers[indices].float()
        
        # Complex case: exclusion zone intersects with range
        # Create valid intervals
        valid_intervals = []
        
        # Check if there's valid space before exclusion zone
        if min_val < excl_min:
            valid_intervals.append((min_val, min(excl_min, max_val)))
        
        # Check if there's valid space after exclusion zone
        if max_val > excl_max:
            valid_intervals.append((max(excl_max, min_val), max_val))
        
        if not valid_intervals:
            raise ValueError(f"No valid sampling space with range [{min_val}, {max_val}] and exclusion [{excl_min}, {excl_max}]")
        
        # Sample from valid intervals
        if slice_count is None:
            # Continuous sampling
            samples = torch.zeros(num_samples, device=self.device)
            for i in range(num_samples):
                # Randomly choose an interval weighted by its size
                interval_sizes = torch.tensor([iv[1] - iv[0] for iv in valid_intervals], device=self.device)
                interval_probs = interval_sizes / interval_sizes.sum()
                interval_idx = torch.multinomial(interval_probs, 1).item()
                
                # Sample within chosen interval
                iv_min, iv_max = valid_intervals[interval_idx]
                samples[i] = torch.empty(1, device=self.device).uniform_(iv_min, iv_max).item()
            return samples
        else:
            # Discrete slicing with exclusion
            all_slice_centers = []
            for iv_min, iv_max in valid_intervals:
                # Create slices within this interval
                num_slices_in_interval = max(1, int(slice_count * (iv_max - iv_min) / (max_val - min_val)))
                slice_vals = torch.linspace(iv_min, iv_max, num_slices_in_interval + 1, device=self.device)
                slice_centers = (slice_vals[:-1] + slice_vals[1:]) / 2
                all_slice_centers.append(slice_centers)
            
            # Combine all valid slice centers
            all_slice_centers = torch.cat(all_slice_centers)
            indices = torch.randint(0, len(all_slice_centers), (num_samples,), device=self.device)
            return all_slice_centers[indices].float()

    def _transform_world_to_base(
        self,
        pos_w: torch.Tensor,
        quat_w: torch.Tensor,
        robot_pos_w: torch.Tensor,
        robot_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform position and quaternion from world frame to robot base frame."""
        # Compute rotation matrix from robot quaternion
        R_w_b = self._quat_to_rot_matrix(robot_quat_w)

        # Transform position: pos_b = R^T * (pos_w - robot_pos_w)
        pos_b = torch.bmm(R_w_b.transpose(-2, -1), (pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1)

        # Transform quaternion: quat_b = quat_robot_inv * quat_w
        quat_robot_inv = self._quat_inverse(robot_quat_w)
        quat_b = self._quat_multiply(quat_robot_inv, quat_w)

        return pos_b, quat_b

    def _quat_to_rot_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (w, x, y, z) to rotation matrix."""
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        # Compute rotation matrix elements
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        # Build rotation matrix
        R = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
        R[..., 0, 0] = 1 - 2 * (yy + zz)
        R[..., 0, 1] = 2 * (xy - wz)
        R[..., 0, 2] = 2 * (xz + wy)
        R[..., 1, 0] = 2 * (xy + wz)
        R[..., 1, 1] = 1 - 2 * (xx + zz)
        R[..., 1, 2] = 2 * (yz - wx)
        R[..., 2, 0] = 2 * (xz - wy)
        R[..., 2, 1] = 2 * (yz + wx)
        R[..., 2, 2] = 1 - 2 * (xx + yy)

        return R

    def _quat_inverse(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute quaternion inverse (conjugate for unit quaternions)."""
        inv_quat = quat.clone()
        inv_quat[..., 1:] = -quat[..., 1:]
        return inv_quat

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    def _update_command(self):
        """Update command at each step (currently unused)."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable or disable debug visualization."""
        if debug_vis:
            # Create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                # Create goal pose visualizer
                from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
                
                goal_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/blocks_goal_pose",
                    markers=GREEN_ARROW_X_MARKER_CFG.markers
                )
                # Scale down the goal pose markers
                goal_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
                self.goal_pose_visualizer = VisualizationMarkers(goal_cfg)
            if not hasattr(self, "object_pose_visualizer"):
                # Create current object pose visualizer
                from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
                
                object_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/blocks_object_pose",
                    markers=BLUE_ARROW_X_MARKER_CFG.markers
                )
                # Scale down the object pose markers
                object_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
                self.object_pose_visualizer = VisualizationMarkers(object_cfg)
            # Set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.object_pose_visualizer.set_visibility(True)
        else:
            # Set visibility to false
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "object_pose_visualizer"):
                self.object_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update visualization when debug is enabled."""
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return
        
        # Update markers if they exist
        if hasattr(self, "goal_pose_visualizer") and hasattr(self, "object_pose_visualizer"):
            # Visualize goal pose (green arrows)
            self.goal_pose_visualizer.visualize(
                translations=self.pose_command_w[:, :3],
                orientations=self.pose_command_w[:, 3:],
            )
            
            # Visualize current object pose (blue arrows)
            self.object_pose_visualizer.visualize(
                translations=self.object.data.root_pos_w,
                orientations=self.object.data.root_quat_w,
            )
