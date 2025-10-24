# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

# from ..assembly_keypoints import Offset
# from . import utils
# from .collision_analyzer_cfg import CollisionAnalyzerCfg
# from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import TaskCommand


# class ee_asset_distance_tanh(ManagerTermBase):
#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)
#         self.root_asset_cfg = cfg.params.get("root_asset_cfg")
#         self.target_asset_cfg = cfg.params.get("target_asset_cfg")
#         self.std = cfg.params.get("std")

#         root_asset_offset_metadata_key: str = cfg.params.get("root_asset_offset_metadata_key")
#         target_asset_offset_metadata_key: str = cfg.params.get("target_asset_offset_metadata_key")

#         self.root_asset = env.scene[self.root_asset_cfg.name]
#         root_usd_path = self.root_asset.cfg.spawn.usd_path
#         root_metadata = utils.read_metadata_from_usd_directory(root_usd_path)
#         root_offset_data = root_metadata.get(root_asset_offset_metadata_key)
#         self.root_asset_offset = Offset(pos=root_offset_data.get("pos"), quat=root_offset_data.get("quat"))

#         self.target_asset = env.scene[self.target_asset_cfg.name]
#         if target_asset_offset_metadata_key is not None:
#             target_usd_path = self.target_asset.cfg.spawn.usd_path
#             target_metadata = utils.read_metadata_from_usd_directory(target_usd_path)
#             target_offset_data = target_metadata.get(target_asset_offset_metadata_key)
#             self.target_asset_offset = Offset(pos=target_offset_data.get("pos"), quat=target_offset_data.get("quat"))
#         else:
#             self.target_asset_offset = None

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         root_asset_cfg: SceneEntityCfg,
#         target_asset_cfg: SceneEntityCfg,
#         root_asset_offset_metadata_key: str,
#         target_asset_offset_metadata_key: str | None = None,
#         std: float = 0.1,
#     ) -> torch.Tensor:
#         root_asset_alignment_pos_w, root_asset_alignment_quat_w = self.root_asset_offset.combine(
#             self.root_asset.data.body_link_pos_w[:, root_asset_cfg.body_ids].view(-1, 3),
#             self.root_asset.data.body_link_quat_w[:, root_asset_cfg.body_ids].view(-1, 4),
#         )
#         if self.target_asset_offset is None:
#             target_asset_alignment_pos_w = self.target_asset.data.root_pos_w.view(-1, 3)
#             target_asset_alignment_quat_w = self.target_asset.data.root_quat_w.view(-1, 4)
#         else:
#             target_asset_alignment_pos_w, target_asset_alignment_quat_w = self.target_asset_offset.apply(
#                 self.target_asset
#             )
#         target_asset_in_root_asset_frame_pos, target_asset_in_root_asset_frame_angle_axis = (
#             math_utils.compute_pose_error(
#                 root_asset_alignment_pos_w,
#                 root_asset_alignment_quat_w,
#                 target_asset_alignment_pos_w,
#                 target_asset_alignment_quat_w,
#             )
#         )

#         pos_distance = torch.norm(target_asset_in_root_asset_frame_pos, dim=1)

#         return 1 - torch.tanh(pos_distance / std)


# class ProgressContext(ManagerTermBase):
#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)
#         self.insertive_asset: Articulation | RigidObject = env.scene[cfg.params.get("insertive_asset_cfg").name]  # type: ignore
#         self.receptive_asset: Articulation | RigidObject = env.scene[cfg.params.get("receptive_asset_cfg").name]  # type: ignore

#         insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
#         receptive_meta = utils.read_metadata_from_usd_directory(self.receptive_asset.cfg.spawn.usd_path)
#         self.insertive_asset_offset = Offset(
#             pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
#             quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
#         )
#         self.receptive_asset_offset = Offset(
#             pos=tuple(receptive_meta.get("assembled_offset").get("pos")),
#             quat=tuple(receptive_meta.get("assembled_offset").get("quat")),
#         )

#         self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
#         self.position_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
#         self.euler_xy_distance = torch.zeros((env.num_envs), device=env.device)
#         self.xyz_distance = torch.zeros((env.num_envs), device=env.device)
#         self.success = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
#         self.continuous_success_counter = torch.zeros((self._env.num_envs), dtype=torch.int32, device=self._env.device)

#         success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=1, device=env.device)
#         self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

#     def reset(self, env_ids: torch.Tensor | None = None) -> None:
#         super().reset(env_ids)
#         self.continuous_success_counter[:] = 0

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         insertive_asset_cfg: SceneEntityCfg,
#         receptive_asset_cfg: SceneEntityCfg,
#         command_context: str = "task_command",
#     ) -> torch.Tensor:
#         task_command: TaskCommand = env.command_manager.get_term(command_context)
#         success_position_threshold = task_command.success_position_threshold
#         success_orientation_threshold = task_command.success_orientation_threshold
#         insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
#             self.insertive_asset
#         )
#         receptive_asset_alignment_pos_w, receptive_asset_alignment_quat_w = self.receptive_asset_offset.apply(
#             self.receptive_asset
#         )
#         insertive_asset_in_receptive_asset_frame_pos, insertive_asset_in_receptive_asset_frame_quat = (
#             math_utils.subtract_frame_transforms(
#                 receptive_asset_alignment_pos_w,
#                 receptive_asset_alignment_quat_w,
#                 insertive_asset_alignment_pos_w,
#                 insertive_asset_alignment_quat_w,
#             )
#         )
#         # yaw could be different
#         e_x, e_y, _ = math_utils.euler_xyz_from_quat(insertive_asset_in_receptive_asset_frame_quat)
#         self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
#         self.xyz_distance[:] = torch.norm(insertive_asset_in_receptive_asset_frame_pos, dim=1)
#         self.position_aligned[:] = self.xyz_distance < success_position_threshold
#         self.orientation_aligned[:] = self.euler_xy_distance < success_orientation_threshold
#         self.success[:] = self.orientation_aligned & self.position_aligned

#         # Update continuous success counter
#         self.continuous_success_counter[:] = torch.where(
#             self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
#         )

#         # Update success monitor
#         self.success_monitor.success_update(
#             torch.zeros(env.num_envs, dtype=torch.int32, device=env.device), self.success
#         )

#         return torch.zeros(env.num_envs, device=env.device)


# class GoalProgressContext(ManagerTermBase):
#     """Progress context that uses goal poses instead of receptive object poses."""
    
#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)
#         self.insertive_asset: Articulation | RigidObject = env.scene[cfg.params.get("insertive_asset_cfg").name]  # type: ignore

#         # Read insertive object offset from metadata
#         insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
#         self.insertive_asset_offset = Offset(
#             pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
#             quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
#         )

#         # Initialize tracking variables
#         self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
#         self.position_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
#         self.euler_xy_distance = torch.zeros((env.num_envs), device=env.device)
#         self.euler_xyz_distance = torch.zeros((env.num_envs), device=env.device)
#         self.xyz_distance = torch.zeros((env.num_envs), device=env.device)
#         self.success = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
#         self.continuous_success_counter = torch.zeros((self._env.num_envs), dtype=torch.int32, device=self._env.device)

#         # Initialize success monitor
#         success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=1, device=env.device)
#         self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

#     def reset(self, env_ids: torch.Tensor | None = None) -> None:
#         super().reset(env_ids)
#         self.continuous_success_counter[:] = 0

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         insertive_asset_cfg: SceneEntityCfg,
#         command_context: str = "task_command",
#     ) -> torch.Tensor:
#         # Get the command term to access goal poses and thresholds
#         task_command = env.command_manager.get_term(command_context)
#         success_position_threshold = task_command.success_position_threshold
#         success_orientation_threshold = task_command.success_orientation_threshold
        
#         # Get insertive object pose with offset
#         insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
#             self.insertive_asset
#         )
        
#         # Get goal poses from the command
#         goal_pos = task_command.goal_positions
#         goal_quat = task_command.goal_orientations
        
#         # Calculate insertive object pose relative to goal frame
#         insertive_asset_in_goal_frame_pos, insertive_asset_in_goal_frame_quat = (
#             math_utils.subtract_frame_transforms(
#                 goal_pos,
#                 goal_quat,
#                 insertive_asset_alignment_pos_w,
#                 insertive_asset_alignment_quat_w,
#             )
#         )
        
#         # Calculate alignment errors (same logic as original)
#         e_x, e_y, e_z = math_utils.euler_xyz_from_quat(insertive_asset_in_goal_frame_quat)
#         self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
#         self.euler_xyz_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs() + math_utils.wrap_to_pi(e_z).abs()
#         self.xyz_distance[:] = torch.norm(insertive_asset_in_goal_frame_pos, dim=1)
#         self.position_aligned[:] = self.xyz_distance < success_position_threshold
#         self.orientation_aligned[:] = self.euler_xyz_distance < success_orientation_threshold
#         self.success[:] = self.orientation_aligned & self.position_aligned

#         # Update continuous success counter
#         self.continuous_success_counter[:] = torch.where(
#             self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
#         )

#         # Update success monitor
#         self.success_monitor.success_update(
#             torch.zeros(env.num_envs, dtype=torch.int32, device=env.device), self.success
#         )

#         return torch.zeros(env.num_envs, device=env.device)



def ee_asset_distance_tanh(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    std: float = 0.1,
) -> torch.Tensor:
    """Reward for end-effector proximity to target asset using tanh kernel.
    
    Computes the distance between the end-effector frame and the target asset,
    then applies a tanh function to create a smooth reward that peaks at 1.0
    when the distance is 0 and approaches 0 as distance increases.
    
    Args:
        env: The environment.
        ee_frame_cfg: The end-effector frame configuration (e.g., from FrameTransformer).
        target_asset_cfg: The target asset configuration (e.g., cube).
        std: Standard deviation for the tanh function. Controls reward falloff rate.
    
    Returns:
        Reward tensor of shape (num_envs,) with values in [0, 1].
    """
    # Get end-effector position from frame transformer
    ee_frame = env.scene.sensors[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # Shape: (num_envs, 3)
    
    # Get target asset position
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_pos_w = target_asset.data.root_pos_w  # Shape: (num_envs, 3)
    
    # Compute Euclidean distance
    distance = torch.norm(ee_pos_w - target_pos_w, dim=-1)  # Shape: (num_envs,)
    
    # Apply tanh kernel: reward is 1.0 at distance=0, approaches 0 as distance increases
    reward = 1.0 - torch.tanh(distance / std)
    
    return reward


def dense_success_reward(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str = "ee_pose",
    position_only: bool | None = None
) -> torch.Tensor:
    """Dense reward based on exponential decay of position and orientation errors.
    
    This reward provides a smooth gradient toward the goal by exponentially decaying
    the distance metrics from the command's tracking.
    
    Args:
        env: The environment.
        std: Standard deviation for the exponential decay (controls steepness).
        command_name: Name of the command term that tracks the target pose.
        position_only: If True, only consider position. If None, infer from command config.
        
    Returns:
        Dense reward value between 0 and 1.
    """
    command = env.command_manager.get_term(command_name)
    
    # Infer position_only from command config if not provided
    if position_only is None:
        position_only = getattr(command.cfg, 'position_only', True)
    
    # Get distance metrics from command
    xyz_distance: torch.Tensor = getattr(command, "xyz_distance")
    euler_xy_distance: torch.Tensor = getattr(command, "euler_xy_distance")
    
    # Apply exponential decay to distances
    position_reward = torch.exp(-xyz_distance / std)
    
    if position_only:
        # Only care about position
        return position_reward
    else:
        # Care about both position and orientation
        orientation_reward = torch.exp(-euler_xy_distance / std)
        stacked = torch.stack([position_reward, orientation_reward], dim=0)
        return torch.mean(stacked, dim=0)


def success_reward(
    env: ManagerBasedRLEnv, 
    command_name: str = "ee_pose",
    position_only: bool | None = None,
) -> torch.Tensor:
    """Sparse reward for successful task completion based on command alignment.
    
    Args:
        env: The environment.
        command_name: Name of the command term that tracks alignment.
        position_only: If True, only check position alignment. If False, check both.
                      If None, infer from command's position_only setting.
    
    Returns:
        1.0 if aligned (based on position_only flag), 0.0 otherwise.
    """
    # Get the command term
    command = env.command_manager.get_term(command_name)
    
    # Infer position_only from command if not provided
    if position_only is None:
        position_only = getattr(command.cfg, 'position_only', True)
    
    # Get alignment flags from command
    position_aligned = getattr(command, "position_aligned", torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    orientation_aligned = getattr(command, "orientation_aligned", torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    
    # Return success based on position_only flag
    if position_only:
        return position_aligned.float()
    else:
        return (orientation_aligned & position_aligned).float()


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.clamp(torch.sum(torch.square(env.action_manager.action), dim=1), 0, 1e4)


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.clamp(
        torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1), 0, 1e4
    )


def joint_vel_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1), 0, 1e4)


# class collision_free(ManagerTermBase):
#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)

#         self._env = env

#         self.collision_analyzer_cfg = cfg.params.get("collision_analyzer_cfg")
#         self.collision_analyzer = self.collision_analyzer_cfg.class_type(self.collision_analyzer_cfg, self._env)

#     def __call__(self, env: ManagerBasedRLEnv, collision_analyzer_cfg: CollisionAnalyzerCfg) -> torch.Tensor:
#         all_env_ids = torch.arange(env.num_envs, device=env.device)
#         collision_free = self.collision_analyzer(env, all_env_ids)

#         return collision_free


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Safety reward that penalizes abnormal robot states (e.g., joint velocities exceeding limits).
    
    Returns 1.0 if the robot is in an abnormal state (joint velocities exceed 2x the limits), 0.0 otherwise.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1).float()


def abnormal_robot_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Termination condition for abnormal robot states (e.g., joint velocities exceeding limits).
    
    Returns True if the robot is in an abnormal state (joint velocities exceed 2x the limits), False otherwise.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    threshold: float | None = None,
) -> torch.Tensor:
    """Check if object has reached the goal position defined by the command.
    
    Args:
        env: The environment.
        command_name: Name of the command term that defines the goal.
        object_cfg: Configuration for the object (not used, kept for compatibility).
        threshold: Optional override for position threshold. If None, uses command's threshold.
        
    Returns:
        Boolean tensor indicating success for each environment.
    """
    # Get the command term
    command = env.command_manager.get_term(command_name)
    
    # Use command's built-in success tracking
    if hasattr(command, 'position_aligned'):
        return command.position_aligned.float()
    else:
        # Fallback: compute position error manually
        if threshold is None:
            threshold = getattr(command.cfg, 'success_position_threshold', 0.05)
        
        # Get command target and object position
        pose_command_w = command.pose_command_w if hasattr(command, 'pose_command_w') else command.command
        object: RigidObject = env.scene[object_cfg.name]
        
        # Compute distance
        distance = torch.norm(pose_command_w[:, :3] - object.data.root_pos_w, dim=-1)
        return (distance < threshold).float()


def object_position_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    std: float = 0.1,
) -> torch.Tensor:
    """Dense reward based on distance between object and goal position.
    
    Uses exponential kernel: exp(-distance / std)
    
    Args:
        env: The environment.
        command_name: Name of the command term that defines the goal.
        object_cfg: Configuration for the object.
        std: Standard deviation for the exponential kernel.
        
    Returns:
        Reward tensor with values in (0, 1].
    """
    # Get the command term
    command = env.command_manager.get_term(command_name)
    
    # Use command's built-in distance tracking if available
    if hasattr(command, 'xyz_distance'):
        distance = command.xyz_distance
    else:
        # Fallback: compute distance manually
        pose_command_w = command.pose_command_w if hasattr(command, 'pose_command_w') else command.command
        object: RigidObject = env.scene[object_cfg.name]
        distance = torch.norm(pose_command_w[:, :3] - object.data.root_pos_w, dim=-1)
    
    return torch.exp(-distance / std)

