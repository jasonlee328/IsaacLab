# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing relative pose command generators for manipulation tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import pose_commands_cfg as dex_cmd_cfgs


##
# Command Terms
##


class ObjectRelativePoseCommand(CommandTerm):
    """Simple relative pose command generator for push tasks.

    Samples target poses using 2D polar coordinates (radius and angle) relative to the object's
    current position. This ensures targets are always a specified distance away from the object.

    Key features:
      • Uses polar sampling: radius ∈ [min_radius, max_radius], angle ∈ [0, 2π]
      • Fully vectorized (no loops, no CPU syncs)
      • Position-only mode supported (orientation ignored)

    Frames:
        All targets are computed in the robot's base frame for consistency.

    Outputs:
        Command buffer shape (num_envs, 7): `(x, y, z, qw, qx, qy, qz)`.

    Metrics:
        `position_error`: Distance between target and current object position.
        `orientation_error`: Angular error between target and current object orientation.
    """

    cfg: "ObjectRelativePoseCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "ObjectRelativePoseCommandCfg", env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: Configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # Extract robot and object from scene
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]

        # Create command buffers: (x, y, z, qw, qx, qy, qz)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0  # Initialize quaternion to identity
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # Create metric buffers
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "ObjectRelativePoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tRadius range: [{self.cfg.min_radius:.3f}, {self.cfg.max_radius:.3f}]\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command in robot base frame.
        
        Returns:
            Tensor of shape (num_envs, 7): position (x,y,z) + quaternion (w,x,y,z)
        """
        return self.pose_command_b

    def _update_metrics(self):
        """Compute position and orientation errors between command and current object pose."""
        # Transform command from base frame to world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        
        # Compute pose error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        
        # Store metrics
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new target poses using 2D polar coordinates.
        
        Args:
            env_ids: Environment indices to resample commands for.
        """
        # Get current object and robot poses in world frame
        object_pos_w = self.object.data.root_pos_w[env_ids]
        object_quat_w = self.object.data.root_quat_w[env_ids]
        robot_pos_w = self.robot.data.root_pos_w[env_ids]
        robot_quat_w = self.robot.data.root_quat_w[env_ids]

        # Transform object pose to robot base frame
        object_pos_b, object_quat_b = self._transform_world_to_base(
            object_pos_w, object_quat_w, robot_pos_w, robot_quat_w
        )

        # Sample target using 2D polar coordinates (table-top push)
        num_envs = len(env_ids)
        
        # Sample radius: distance from object center to target
        radius = torch.empty(num_envs, device=self.device)
        radius.uniform_(self.cfg.min_radius, self.cfg.max_radius)
        
        # Sample angle: direction to push (full circle)
        theta = torch.empty(num_envs, device=self.device)
        theta.uniform_(0, 2 * torch.pi)
        
        # Convert polar to Cartesian offsets
        offset_x = radius * torch.cos(theta)
        offset_y = radius * torch.sin(theta)
        offset_z = torch.zeros(num_envs, device=self.device)  # Stay on table

        # Compute target position = current position + offset
        self.pose_command_b[env_ids, 0] = object_pos_b[:, 0] + offset_x
        self.pose_command_b[env_ids, 1] = object_pos_b[:, 1] + offset_y
        self.pose_command_b[env_ids, 2] = object_pos_b[:, 2] + offset_z

        # Handle orientation
        if not self.cfg.position_only:
            # Sample random orientation
            euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
            euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
            quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
            self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        else:
            # Position-only mode: keep current orientation
            self.pose_command_b[env_ids, 3:] = object_quat_b

    def _transform_world_to_base(
        self,
        pos_w: torch.Tensor,
        quat_w: torch.Tensor,
        robot_pos_w: torch.Tensor,
        robot_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform pose from world frame to robot base frame.

        Args:
            pos_w: Position in world frame (N, 3)
            quat_w: Quaternion in world frame (N, 4) in (w, x, y, z) format
            robot_pos_w: Robot base position in world frame (N, 3)
            robot_quat_w: Robot base quaternion in world frame (N, 4)

        Returns:
            Tuple of (position in base frame, quaternion in base frame)
        """
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

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

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
        """Update command (not used in this implementation)."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable or disable debug visualization."""
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
                
                goal_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_pose",
                    markers=GREEN_ARROW_X_MARKER_CFG.markers
                )
                goal_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
                self.goal_pose_visualizer = VisualizationMarkers(goal_cfg)
                
            if not hasattr(self, "object_pose_visualizer"):
                from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
                
                object_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/object_pose",
                    markers=BLUE_ARROW_X_MARKER_CFG.markers
                )
                object_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
                self.object_pose_visualizer = VisualizationMarkers(object_cfg)
                
            self.goal_pose_visualizer.set_visibility(True)
            self.object_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "object_pose_visualizer"):
                self.object_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update visualization markers."""
        if not self.robot.is_initialized:
            return
        
        if hasattr(self, "goal_pose_visualizer") and hasattr(self, "object_pose_visualizer"):
            # Visualize goal pose (green)
            self.goal_pose_visualizer.visualize(
                translations=self.pose_command_w[:, :3],
                orientations=self.pose_command_w[:, 3:],
            )
            # Visualize current object pose (blue)
            self.object_pose_visualizer.visualize(
                translations=self.object.data.root_pos_w,
                orientations=self.object.data.root_quat_w,
            )


##
# Configuration
##


@configclass
class ObjectRelativePoseCommandCfg(CommandTermCfg):
    """Configuration for relative pose command generator using polar sampling.
    
    This configuration generates target poses using 2D polar coordinates (radius and angle)
    relative to the object's current position. This ensures targets are always a specified
    distance away from the object, avoiding rejection sampling.
    
    Key parameters:
        - min_radius: Minimum distance from object to target (e.g., cube size)
        - max_radius: Maximum distance from object to target
        - position_only: If True, only position is commanded (orientation ignored)
    """

    class_type: type = ObjectRelativePoseCommand

    asset_name: str = MISSING
    """Name of the robot asset (reference frame for commands)."""

    object_name: str = MISSING
    """Name of the object to generate commands for."""

    min_radius: float = 0.05
    """Minimum distance from object center to target (in meters)."""

    max_radius: float = 0.30
    """Maximum distance from object center to target (in meters)."""

    position_only: bool = True
    """If True, only generate position commands (ignore orientation)."""

    make_quat_unique: bool = False
    """Whether to make quaternion unique by ensuring positive real part."""

    @configclass
    class Ranges:
        """Orientation ranges (only used if position_only=False)."""

        roll: tuple[float, float] = (0.0, 0.0)
        """Range for roll angle (in radians)."""

        pitch: tuple[float, float] = (0.0, 0.0)
        """Range for pitch angle (in radians)."""

        yaw: tuple[float, float] = (0.0, 0.0)
        """Range for yaw angle (in radians)."""

    ranges: Ranges = Ranges()
    """Orientation ranges for sampling (only used if position_only=False)."""


class PushObjectDistractorAwareCommand(CommandTerm):
    """Relative pose command generator for an object.

    This command term samples target object poses **relative to the object's current position**:
      • Drawing (x, y, z) offsets uniformly within configured ranges relative to the cube
      • Drawing roll-pitch-yaw uniformly within configured ranges, then converting
        to a quaternion (w, x, y, z). Optionally makes quaternions unique by enforcing
        a positive real part.

    This is particularly useful for push tasks where you want to ensure the target
    is always offset from the object's spawn position, preventing trivial successes.

    Frames:
        Targets are computed relative to the object's current position in the robot's base frame.
        For metrics/visualization, targets are transformed into the world frame.

    Outputs:
        The command buffer has shape (num_envs, 7): `(x, y, z, qw, qx, qy, qz)`.

    Metrics:
        `position_error` and `orientation_error` are computed between the commanded
        world-frame pose and the object's current world-frame pose.

    Config:
        `cfg` must provide the relative sampling ranges, whether to enforce quaternion uniqueness,
        and optional visualization settings.
    """

    cfg: dex_cmd_cfgs.ObjectRelativePoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: dex_cmd_cfgs.ObjectRelativePoseCommandCfg, env: ManagerBasedEnv):
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
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "RelativePoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
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
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # Get current object position in world frame
        object_pos_w = self.object.data.root_pos_w[env_ids]
        object_quat_w = self.object.data.root_quat_w[env_ids]

        # Get current robot position in world frame
        robot_pos_w = self.robot.data.root_pos_w[env_ids]
        robot_quat_w = self.robot.data.root_quat_w[env_ids]

        # Compute object position in robot base frame
        object_pos_b, object_quat_b = self._transform_world_to_base(
            object_pos_w, object_quat_w, robot_pos_w, robot_quat_w
        )

        # Sample relative offsets with minimum distance constraint
        num_envs = len(env_ids)
        max_attempts = 100  # Maximum attempts to find valid positions
        
        # Initialize offsets
        offset_x = torch.empty(num_envs, device=self.device)
        offset_y = torch.empty(num_envs, device=self.device)
        offset_z = torch.empty(num_envs, device=self.device)
        
        # Keep track of which environments still need valid samples
        valid_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        for attempt in range(max_attempts):
            # Sample offsets for environments that don't have valid positions yet
            invalid_indices = ~valid_mask
            if not invalid_indices.any():
                break  # All environments have valid positions
            
            num_invalid = invalid_indices.sum().item()
            r = torch.empty(num_invalid, device=self.device)
            
            offset_x[invalid_indices] = r.uniform_(*self.cfg.ranges.pos_x)
            offset_y[invalid_indices] = r.uniform_(*self.cfg.ranges.pos_y)
            offset_z[invalid_indices] = r.uniform_(*self.cfg.ranges.pos_z)
            
            # Check if sampled offsets satisfy minimum distance constraint
            if self.cfg.min_distance > 0.0:
                # Calculate distance from object to target
                distance = torch.sqrt(
                    offset_x[invalid_indices] ** 2 + 
                    offset_y[invalid_indices] ** 2 + 
                    offset_z[invalid_indices] ** 2
                )
                # Mark as valid if distance meets minimum threshold
                newly_valid = distance >= self.cfg.min_distance
                valid_mask[invalid_indices] = newly_valid
            else:
                # No minimum distance constraint, all samples are valid
                valid_mask[invalid_indices] = True
        
        # If some environments still don't have valid positions after max attempts,
        # enforce minimum distance by scaling the offsets
        if not valid_mask.all():
            invalid_indices = ~valid_mask
            offset_magnitude = torch.sqrt(
                offset_x[invalid_indices] ** 2 + 
                offset_y[invalid_indices] ** 2 + 
                offset_z[invalid_indices] ** 2
            )
            # Avoid division by zero
            offset_magnitude = torch.clamp(offset_magnitude, min=1e-6)
            scale = self.cfg.min_distance / offset_magnitude
            offset_x[invalid_indices] *= scale
            offset_y[invalid_indices] *= scale
            offset_z[invalid_indices] *= scale

        # Add offsets to current object position (in base frame)
        self.pose_command_b[env_ids, 0] = object_pos_b[:, 0] + offset_x
        self.pose_command_b[env_ids, 1] = object_pos_b[:, 1] + offset_y
        self.pose_command_b[env_ids, 2] = object_pos_b[:, 2] + offset_z

        # Handle orientation
        if not self.cfg.position_only:
            euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
            euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
            quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
            # make sure the quaternion has real part as positive
            self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        else:
            # Keep current orientation if position_only
            self.pose_command_b[env_ids, 3:] = object_quat_b

    def _transform_world_to_base(
        self,
        pos_w: torch.Tensor,
        quat_w: torch.Tensor,
        robot_pos_w: torch.Tensor,
        robot_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform position and quaternion from world frame to robot base frame.

        Args:
            pos_w: Position in world frame (N, 3)
            quat_w: Quaternion in world frame (N, 4) in (w, x, y, z) format
            robot_pos_w: Robot base position in world frame (N, 3)
            robot_quat_w: Robot base quaternion in world frame (N, 4) in (w, x, y, z) format

        Returns:
            Tuple of (position in base frame, quaternion in base frame)
        """
        # Compute rotation matrix from robot quaternion
        R_w_b = self._quat_to_rot_matrix(robot_quat_w)

        # Transform position: pos_b = R^T * (pos_w - robot_pos_w)
        pos_b = torch.bmm(R_w_b.transpose(-2, -1), (pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1)

        # Transform quaternion: quat_b = quat_robot_inv * quat_w
        quat_robot_inv = self._quat_inverse(robot_quat_w)
        quat_b = self._quat_multiply(quat_robot_inv, quat_w)

        return pos_b, quat_b

    def _quat_to_rot_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (w, x, y, z) to rotation matrix.

        Args:
            quat: Quaternions of shape (N, 4) in (w, x, y, z) format

        Returns:
            Rotation matrices of shape (N, 3, 3)
        """
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
        """Compute quaternion inverse (conjugate for unit quaternions).

        Args:
            quat: Quaternions of shape (N, 4) in (w, x, y, z) format

        Returns:
            Inverse quaternions of shape (N, 4)
        """
        # For unit quaternions, inverse is just negating the vector part
        inv_quat = quat.clone()
        inv_quat[..., 1:] = -quat[..., 1:]
        return inv_quat

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions.

        Args:
            q1: First quaternions of shape (N, 4) in (w, x, y, z) format
            q2: Second quaternions of shape (N, 4) in (w, x, y, z) format

        Returns:
            Product quaternions of shape (N, 4)
        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    def _update_command(self):
        pass

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     # create markers if necessary for the first time
    #     if debug_vis:
    #         if not hasattr(self, "goal_visualizer"):
    #             # -- goal pose
    #             self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
    #             # -- current body pose
    #             self.curr_visualizer = VisualizationMarkers(self.cfg.curr_pose_visualizer_cfg)
    #         # set their visibility to true
    #         self.goal_visualizer.set_visibility(True)
    #         self.curr_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "goal_visualizer"):
    #             self.goal_visualizer.set_visibility(False)
    #             self.curr_visualizer.set_visibility(False)

    # def _debug_vis_callback(self, event):
    #     # check if robot is initialized
    #     # note: this is needed in-case the robot is de-initialized. we can't access the data
    #     if not self.robot.is_initialized:
    #         return
    #     # update the markers
    #     if not self.cfg.position_only:
    #         # -- goal pose
    #         self.goal_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
    #         # -- current object pose
    #         self.curr_visualizer.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
    #     else:
    #         distance = torch.norm(self.pose_command_w[:, :3] - self.object.data.root_pos_w[:, :3], dim=1)
    #         success_id = (distance < 0.05).int()
    #         # note: since marker indices for position is 1(far) and 2(near), we can simply shift the success_id by 1.
    #         # -- goal position
    #         self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=success_id + 1)
    #         # -- current object position
    #         self.curr_visualizer.visualize(self.object.data.root_pos_w, marker_indices=success_id + 1)


    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable or disable debug visualization."""
        if debug_vis:
            # Create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                # Create goal pose visualizer with proper prim path
                from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
                
                goal_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_pose_relative",
                    markers=GREEN_ARROW_X_MARKER_CFG.markers
                )
                # Scale down the goal pose markers
                goal_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
                self.goal_pose_visualizer = VisualizationMarkers(goal_cfg)
            if not hasattr(self, "object_pose_visualizer"):
                # Create current object pose visualizer with proper prim path
                from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
                
                object_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/object_pose_relative",
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
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
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




