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

from ..assembly_keypoints import Offset
from . import utils
from .collision_analyzer_cfg import CollisionAnalyzerCfg
from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import TaskCommand


class ee_asset_distance_tanh(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.root_asset_cfg = cfg.params.get("root_asset_cfg")
        self.target_asset_cfg = cfg.params.get("target_asset_cfg")
        self.std = cfg.params.get("std")
        self.enable_debug_vis = cfg.params.get("enable_debug_vis", False)

        root_asset_offset_metadata_key: str = cfg.params.get("root_asset_offset_metadata_key")
        target_asset_offset_metadata_key: str = cfg.params.get("target_asset_offset_metadata_key")

        self.root_asset = env.scene[self.root_asset_cfg.name]
        root_usd_path = self.root_asset.cfg.spawn.usd_path
        root_metadata = utils.read_metadata_from_usd_directory(root_usd_path)
        root_offset_data = root_metadata.get(root_asset_offset_metadata_key)
        self.root_asset_offset = Offset(pos=root_offset_data.get("pos"), quat=root_offset_data.get("quat"))

        self.target_asset = env.scene[self.target_asset_cfg.name]
        if target_asset_offset_metadata_key is not None:
            target_usd_path = self.target_asset.cfg.spawn.usd_path
            target_metadata = utils.read_metadata_from_usd_directory(target_usd_path)
            target_offset_data = target_metadata.get(target_asset_offset_metadata_key)
            self.target_asset_offset = Offset(pos=target_offset_data.get("pos"), quat=target_offset_data.get("quat"))
        else:
            self.target_asset_offset = None

        # Initialize debug visualization markers if enabled
        if self.enable_debug_vis:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
            
            # Goal pose visualizer (green) - shows where end effector should be
            # Note: This will show the target asset position (what the reward is trying to minimize distance to)
            goal_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Rewards/ee_asset_distance/goal_pose",
                markers=GREEN_ARROW_X_MARKER_CFG.markers
            )
            goal_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
            self.goal_pose_visualizer = VisualizationMarkers(goal_cfg)
            
            # End effector pose visualizer (red) - shows current end effector pose
            ee_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Rewards/ee_asset_distance/end_effector_pose",
                markers=RED_ARROW_X_MARKER_CFG.markers
            )
            ee_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
            self.ee_pose_visualizer = VisualizationMarkers(ee_cfg)
            
            self.goal_pose_visualizer.set_visibility(True)
            self.ee_pose_visualizer.set_visibility(True)
        else:
            self.goal_pose_visualizer = None
            self.ee_pose_visualizer = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        root_asset_cfg: SceneEntityCfg,
        target_asset_cfg: SceneEntityCfg,
        root_asset_offset_metadata_key: str,
        target_asset_offset_metadata_key: str | None = None,
        std: float = 0.1,
        enable_debug_vis: bool = False,  # Optional parameter for initialization (not used in call)
    ) -> torch.Tensor:
        # Compute end effector pose with offset (exactly as used in reward computation)
        root_asset_alignment_pos_w, root_asset_alignment_quat_w = self.root_asset_offset.combine(
            self.root_asset.data.body_link_pos_w[:, root_asset_cfg.body_ids].view(-1, 3),
            self.root_asset.data.body_link_quat_w[:, root_asset_cfg.body_ids].view(-1, 4),
        )
        
        # Compute target asset pose (with offset if specified)
        if self.target_asset_offset is None:
            target_asset_alignment_pos_w = self.target_asset.data.root_pos_w.view(-1, 3)
            target_asset_alignment_quat_w = self.target_asset.data.root_quat_w.view(-1, 4)
        else:
            target_asset_alignment_pos_w, target_asset_alignment_quat_w = self.target_asset_offset.apply(
                self.target_asset
            )
        
        # Update debug visualization if enabled
        if self.enable_debug_vis and self.goal_pose_visualizer is not None and self.ee_pose_visualizer is not None:
            # Visualize insertive object pose (green) - this is what the reward is trying to minimize distance to
            # This shows the insertive object's root pose (no offset applied since target_asset_offset_metadata_key is None)
            self.goal_pose_visualizer.visualize(
                translations=target_asset_alignment_pos_w,
                orientations=target_asset_alignment_quat_w,
            )
            # Visualize end effector pose with gripper_offset (red) - current end effector TCP pose
            # This is the actual pose used in the reward computation
            self.ee_pose_visualizer.visualize(
                translations=root_asset_alignment_pos_w,
                orientations=root_asset_alignment_quat_w,
            )
        
        target_asset_in_root_asset_frame_pos, target_asset_in_root_asset_frame_angle_axis = (
            math_utils.compute_pose_error(
                root_asset_alignment_pos_w,
                root_asset_alignment_quat_w,
                target_asset_alignment_pos_w,
                target_asset_alignment_quat_w,
            )
        )

        pos_distance = torch.norm(target_asset_in_root_asset_frame_pos, dim=1)

        return 1 - torch.tanh(pos_distance / std)


class ProgressContext(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.params.get("insertive_asset_cfg").name]  # type: ignore
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.params.get("receptive_asset_cfg").name]  # type: ignore

        insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
        receptive_meta = utils.read_metadata_from_usd_directory(self.receptive_asset.cfg.spawn.usd_path)
        self.insertive_asset_offset = Offset(
            pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
            quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
        )
        self.receptive_asset_offset = Offset(
            pos=tuple(receptive_meta.get("assembled_offset").get("pos")),
            quat=tuple(receptive_meta.get("assembled_offset").get("quat")),
        )

        self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.position_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.euler_xy_distance = torch.zeros((env.num_envs), device=env.device)
        self.xyz_distance = torch.zeros((env.num_envs), device=env.device)
        self.success = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.continuous_success_counter = torch.zeros((self._env.num_envs), dtype=torch.int32, device=self._env.device)

        success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=1, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.continuous_success_counter[:] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        insertive_asset_cfg: SceneEntityCfg,
        receptive_asset_cfg: SceneEntityCfg,
        command_context: str = "task_command",
    ) -> torch.Tensor:
        task_command: TaskCommand = env.command_manager.get_term(command_context)
        success_position_threshold = task_command.success_position_threshold
        success_orientation_threshold = task_command.success_orientation_threshold
        insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
            self.insertive_asset
        )
        receptive_asset_alignment_pos_w, receptive_asset_alignment_quat_w = self.receptive_asset_offset.apply(
            self.receptive_asset
        )
        insertive_asset_in_receptive_asset_frame_pos, insertive_asset_in_receptive_asset_frame_quat = (
            math_utils.subtract_frame_transforms(
                receptive_asset_alignment_pos_w,
                receptive_asset_alignment_quat_w,
                insertive_asset_alignment_pos_w,
                insertive_asset_alignment_quat_w,
            )
        )
        # yaw could be different
        e_x, e_y, _ = math_utils.euler_xyz_from_quat(insertive_asset_in_receptive_asset_frame_quat)
        self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xyz_distance[:] = torch.norm(insertive_asset_in_receptive_asset_frame_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < success_orientation_threshold
        self.success[:] = self.orientation_aligned & self.position_aligned

        # Update continuous success counter
        self.continuous_success_counter[:] = torch.where(
            self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
        )

        # Update success monitor
        self.success_monitor.success_update(
            torch.zeros(env.num_envs, dtype=torch.int32, device=env.device), self.success
        )

        return torch.zeros(env.num_envs, device=env.device)


class GoalProgressContext(ManagerTermBase):
    """Progress context that uses goal poses instead of receptive object poses."""
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.params.get("insertive_asset_cfg").name]  # type: ignore

        # Read insertive object offset from metadata
        insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
        self.insertive_asset_offset = Offset(
            pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
            quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
        )

        # Boolean flag for orientation alignment
        self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        # Boolean flag for position alignment
        self.position_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        # Distance between insertive object and goal frame
        self.euler_xyz_distance = torch.zeros((env.num_envs), device=env.device)
        # Distance between insertive object and goal frame
        self.xyz_distance = torch.zeros((env.num_envs), device=env.device)
        # Boolean flag for success
        self.success = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        # Continuous success counter
        self.continuous_success_counter = torch.zeros((self._env.num_envs), dtype=torch.int32, device=self._env.device)

        # Initialize success monitor
        success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=1, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.continuous_success_counter[:] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        insertive_asset_cfg: SceneEntityCfg,
        command_context: str = "task_command",
    ) -> torch.Tensor:
        # Get the command term to access goal poses and thresholds
        task_command = env.command_manager.get_term(command_context)
        success_position_threshold = task_command.success_position_threshold
        success_orientation_threshold = task_command.success_orientation_threshold
        
        # Get insertive object pose without offset
        insertive_asset_pos_w = self.insertive_asset.data.root_pos_w
        insertive_asset_quat_w = self.insertive_asset.data.root_quat_w
        
        # Get goal poses from the command
        goal_pos = task_command.goal_positions
        goal_quat = task_command.goal_orientations
        
        # Calculate insertive object pose relative to goal frame
        insertive_asset_in_goal_frame_pos, insertive_asset_in_goal_frame_quat = (
            math_utils.subtract_frame_transforms(
                goal_pos,
                goal_quat,
                insertive_asset_pos_w,
                insertive_asset_quat_w,
            )
        )
        
        # Calculate alignment errors (same logic as original)
        e_x, e_y, e_z = math_utils.euler_xyz_from_quat(insertive_asset_in_goal_frame_quat)
        self.euler_xyz_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs() + math_utils.wrap_to_pi(e_z).abs()
        self.xyz_distance[:] = torch.norm(insertive_asset_in_goal_frame_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < success_position_threshold
        self.orientation_aligned[:] = self.euler_xyz_distance < success_orientation_threshold
        self.success[:] = self.orientation_aligned & self.position_aligned

        # Update continuous success counter
        self.continuous_success_counter[:] = torch.where(
            self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
        )

        # Update success monitor
        self.success_monitor.success_update(
            torch.zeros(env.num_envs, dtype=torch.int32, device=env.device), self.success
        )

        return torch.zeros(env.num_envs, device=env.device)


def dense_success_reward(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:

    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    angle_diff: torch.Tensor = getattr(context_term, "euler_xyz_distance")
    xyz_distance: torch.Tensor = getattr(context_term, "xyz_distance")

    # Normalize the distances by std
    angle_diff = torch.exp(-angle_diff / std)
    xyz_distance = torch.exp(-xyz_distance / std)
    stacked = torch.stack([angle_diff, xyz_distance], dim=0)
    return torch.mean(stacked, dim=0)


def success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_aligned: torch.Tensor = getattr(context_term, "position_aligned")
    return torch.where(orientation_aligned & position_aligned, 1.0, 0.0)


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


class collision_free(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._env = env

        self.collision_analyzer_cfg = cfg.params.get("collision_analyzer_cfg")
        self.collision_analyzer = self.collision_analyzer_cfg.class_type(self.collision_analyzer_cfg, self._env)

    def __call__(self, env: ManagerBasedRLEnv, collision_analyzer_cfg: CollisionAnalyzerCfg) -> torch.Tensor:
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        collision_free = self.collision_analyzer(env, all_env_ids)

        return collision_free
