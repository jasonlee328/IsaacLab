# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

# Copyright (c) 2022-2024, The Octi Lab and  Isaac Lab Project Developers.
# All rights reserved.

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkersCfg
from ..assembly_keypoints import Offset
from . import utils
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkersCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TaskCommandCfg, TaskDependentCommandCfg, RandomTableGoalCommandCfg


class TaskDependentCommand(CommandTerm):
    cfg: TaskDependentCommandCfg

    def __init__(self, cfg: TaskDependentCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.reset_terms_when_resample = cfg.reset_terms_when_resample
        self.interval_reset_terms = []
        self.reset_terms = []
        self.ALL_INDICES = torch.arange(self.num_envs, device=self.device)
        for name, term_cfg in self.reset_terms_when_resample.items():
            if not (term_cfg.mode == "reset" or term_cfg.mode == "interval"):
                raise ValueError(f"Term '{name}' in 'reset_terms_when_resample' must have mode 'reset' or 'interval'")
            if inspect.isclass(term_cfg.func):
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
            if term_cfg.mode == "reset":
                self.reset_terms.append(term_cfg)
            elif term_cfg.mode == "interval":
                if term_cfg.interval_range_s != (0, 0):
                    raise ValueError(
                        "task dependent events term with interval mode current only supports range of (0, 0)"
                    )
                self.interval_reset_terms.append(term_cfg)

    def _resample_command(self, env_ids: Sequence[int]):
        for term in self.reset_terms:
            func = term.func
            func(self._env, env_ids, **term.params)
        for term in self.interval_reset_terms:
            func = term.func
            func.reset(env_ids)

    def _update_command(self):
        for term in self.interval_reset_terms:
            func = term.func
            func(self._env, self.ALL_INDICES, **term.params)

    def get_event(self, event_term_name: str):
        """Get the event term by name."""
        return self.reset_terms_when_resample.get(event_term_name).func


class TaskCommand(TaskDependentCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TaskCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TaskCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.insertive_asset_cfg.name]
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.receptive_asset_cfg.name]
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
        self.success_position_threshold: float = cfg.success_position_threshold
        self.success_orientation_threshold: float = cfg.success_orientation_threshold

        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.orientation_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.position_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.euler_xy_distance = torch.zeros((self._env.num_envs), device=self._env.device)
        self.xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs end of episode data
        reset_env = self._env.episode_length_buf == 0
        self.metrics["end_of_episode_rot_align_error"][reset_env] = self.euler_xy_distance[reset_env]
        self.metrics["end_of_episode_pos_align_error"][reset_env] = self.xyz_distance[reset_env]
        last_episode_success = (self.orientation_aligned & self.position_aligned)[reset_env]
        self.metrics["end_of_episode_success_rate"][reset_env] = last_episode_success.float()

        # logs current data
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
        e_x, e_y, _ = math_utils.euler_xyz_from_quat(insertive_asset_in_receptive_asset_frame_quat)
        self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xyz_distance[:] = torch.norm(insertive_asset_in_receptive_asset_frame_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < self.success_orientation_threshold
        self.metrics["average_rot_align_error"][:] = self.euler_xy_distance
        self.metrics["average_pos_align_error"][:] = self.xyz_distance

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


class RandomTableGoalCommand(TaskDependentCommand):
    """Command generator that generates random goal poses on the table for the insertive object.

    This command generator samples random goal positions and orientations on the table
    for the insertive object to move to, similar to the reset state sampling in
    ObjectAnywhereEEAnywhereEventCfg.
    """

    cfg: "RandomTableGoalCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "RandomTableGoalCommandCfg", env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the assets
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.insertive_asset_cfg.name]
        self.table_asset: Articulation | RigidObject = env.scene[cfg.table_asset_cfg.name]
        
        # Read metadata for insertive object offset
        insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
        self.insertive_asset_offset = Offset(
            pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
            quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
        )
        
        self.success_position_threshold: float = cfg.success_position_threshold
        self.success_orientation_threshold: float = cfg.success_orientation_threshold

        # Parse goal pose ranges
        goal_pose_range = cfg.goal_pose_range
        range_list = [goal_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.goal_ranges = torch.tensor(range_list, device=env.device)

        # Initialize goal poses (will be set during resample)
        self.goal_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_orientations = torch.zeros((self.num_envs, 4), device=self.device)

        # Metrics for tracking progress
        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.orientation_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.position_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.euler_xy_distance = torch.zeros((self._env.num_envs), device=self._env.device)
        self.xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs end of episode data
        reset_env = self._env.episode_length_buf == 0
        self.metrics["end_of_episode_rot_align_error"][reset_env] = self.euler_xy_distance[reset_env]
        self.metrics["end_of_episode_pos_align_error"][reset_env] = self.xyz_distance[reset_env]
        last_episode_success = (self.orientation_aligned & self.position_aligned)[reset_env]
        self.metrics["end_of_episode_success_rate"][reset_env] = last_episode_success.float()

        # logs current data
        insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
            self.insertive_asset
        )
        
        # Calculate error between insertive object and goal
        insertive_asset_in_goal_frame_pos, insertive_asset_in_goal_frame_quat = (
            math_utils.subtract_frame_transforms(
                self.goal_positions,
                self.goal_orientations,
                insertive_asset_alignment_pos_w,
                insertive_asset_alignment_quat_w,
            )
        )
        
        e_x, e_y, _ = math_utils.euler_xyz_from_quat(insertive_asset_in_goal_frame_quat)
        self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xyz_distance[:] = torch.norm(insertive_asset_in_goal_frame_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < self.success_orientation_threshold
        self.metrics["average_rot_align_error"][:] = self.euler_xy_distance
        self.metrics["average_pos_align_error"][:] = self.xyz_distance

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        
        # Sample new goal poses for the specified environments
        num_envs = len(env_ids)
        samples = math_utils.sample_uniform(
            self.goal_ranges[:, 0], self.goal_ranges[:, 1], (num_envs, 6), device=self.device
        )
        
        # Get table positions for the environments
        table_pos = self.table_asset.data.root_pos_w[env_ids]
        table_quat = self.table_asset.data.root_quat_w[env_ids]
        
        # Calculate goal positions relative to table
        goal_positions_relative = samples[:, 0:3]
        goal_orientations_relative = math_utils.quat_from_euler_xyz(
            samples[:, 3], samples[:, 4], samples[:, 5]
        )
        
        # Transform to world coordinates
        goal_positions_world, goal_orientations_world = math_utils.combine_frame_transforms(
            table_pos, table_quat, goal_positions_relative, goal_orientations_relative
        )
        
        # Store the new goal poses
        self.goal_positions[env_ids] = goal_positions_world
        self.goal_orientations[env_ids] = goal_orientations_world

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable or disable debug visualization."""
        if debug_vis:
            # Create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                # Create goal pose visualizer with proper prim path
                from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
                from isaaclab.markers import VisualizationMarkersCfg
                goal_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_poses",
                    markers=GREEN_ARROW_X_MARKER_CFG.markers
                )
                # Scale down the goal pose markers
                goal_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
                self.goal_pose_visualizer = VisualizationMarkers(goal_cfg)
            if not hasattr(self, "insertive_object_visualizer"):
                # Create insertive object visualizer with proper prim path
                from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
                from isaaclab.markers import VisualizationMarkersCfg
                insertive_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/insertive_object",
                    markers=BLUE_ARROW_X_MARKER_CFG.markers
                )
                # Scale down the insertive object markers
                insertive_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
                self.insertive_object_visualizer = VisualizationMarkers(insertive_cfg)
            # Set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.insertive_object_visualizer.set_visibility(True)
        else:
            # Set visibility to false
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "insertive_object_visualizer"):
                self.insertive_object_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update visualization when debug is enabled."""
        if hasattr(self, "goal_pose_visualizer") and hasattr(self, "insertive_object_visualizer"):
            # Visualize goal poses (green arrows)
            self.goal_pose_visualizer.visualize(
                translations=self.goal_positions,
                orientations=self.goal_orientations,
            )
            
            # Visualize insertive object poses (blue arrows)
            insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
                self.insertive_asset
            )
            self.insertive_object_visualizer.visualize(
                translations=insertive_asset_alignment_pos_w,
                orientations=insertive_asset_alignment_quat_w,
            )


class ReceptiveObjectGoalCommand(TaskDependentCommand):
    """Command generator that generates goal poses above the receptive object for the insertive object.
    
    This command places the goal for the insertive object at a specified z height above the receptive object,
    while keeping x, y, roll, pitch, and yaw the same as the receptive object.
    """

    cfg: "ReceptiveObjectGoalCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "ReceptiveObjectGoalCommandCfg", env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the assets
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.insertive_asset_cfg.name]
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.receptive_asset_cfg.name]

        self.success_position_threshold: float = cfg.success_position_threshold
        self.success_orientation_threshold: float = cfg.success_orientation_threshold
        self.goal_height_offset: float = cfg.goal_height_offset

        # Initialize goal poses (will be set during resample)
        self.goal_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_orientations = torch.zeros((self.num_envs, 4), device=self.device)

        # Metrics for tracking progress
        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.orientation_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.position_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.euler_xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)
        self.xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs end of episode data
        reset_env = self._env.episode_length_buf == 0
        self.metrics["end_of_episode_rot_align_error"][reset_env] = self.euler_xyz_distance[reset_env]
        self.metrics["end_of_episode_pos_align_error"][reset_env] = self.xyz_distance[reset_env]
        last_episode_success = (self.orientation_aligned & self.position_aligned)[reset_env]
        self.metrics["end_of_episode_success_rate"][reset_env] = last_episode_success.float()

        # logs current data - get insertive asset poses directly without offsets
        insertive_asset_pos_w = self.insertive_asset.data.root_pos_w
        insertive_asset_quat_w = self.insertive_asset.data.root_quat_w
        
        # Calculate error between insertive object and goal
        insertive_asset_in_goal_frame_pos, insertive_asset_in_goal_frame_quat = (
            math_utils.subtract_frame_transforms(
                self.goal_positions,
                self.goal_orientations,
                insertive_asset_pos_w,
                insertive_asset_quat_w,
            )
        )
        #Calculate euler angles and wrap to pi
        e_x, e_y, e_z = math_utils.euler_xyz_from_quat(insertive_asset_in_goal_frame_quat)
        self.euler_xyz_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs() + math_utils.wrap_to_pi(e_z).abs()
        
        
        #Calculate distance between insertive object and goal frame
        self.xyz_distance[:] = torch.norm(insertive_asset_in_goal_frame_pos, dim=1)

        #Calculate boolean flags for alignment and success  
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        self.orientation_aligned[:] = self.euler_xyz_distance < self.success_orientation_threshold

        #Update metrics
        self.metrics["average_rot_align_error"][:] = self.euler_xyz_distance
        self.metrics["average_pos_align_error"][:] = self.xyz_distance

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        
        # Get receptive object poses for the specified environments
        receptive_pos = self.receptive_asset.data.root_pos_w[env_ids]
        receptive_quat = self.receptive_asset.data.root_quat_w[env_ids]
        
        # Set goal positions: copy x, y from receptive object, add height offset to z
        goal_positions = receptive_pos.clone()
        goal_positions[:, 2] += self.goal_height_offset
        
        # Set goal orientations: copy exactly from receptive object
        goal_orientations = receptive_quat.clone()
        
        # Store the new goal poses
        self.goal_positions[env_ids] = goal_positions
        self.goal_orientations[env_ids] = goal_orientations

    def _update_command(self):
        super()._update_command()
        
        # Dynamically update goal poses based on current receptive object pose
        # This makes the goal track the receptive object if it moves
        receptive_pos = self.receptive_asset.data.root_pos_w
        receptive_quat = self.receptive_asset.data.root_quat_w
        
        # Set goal positions: copy x, y from receptive object, add height offset to z
        self.goal_positions[:] = receptive_pos.clone()
        self.goal_positions[:, 2] += self.goal_height_offset
        
        # Set goal orientations: copy exactly from receptive object
        self.goal_orientations[:] = receptive_quat.clone()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable or disable debug visualization."""
        if debug_vis:
            # Create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                # Create goal pose visualizer with proper prim path

                goal_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_poses",
                    markers=GREEN_ARROW_X_MARKER_CFG.markers
                )
                # Scale down the goal pose markers
                goal_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
                self.goal_pose_visualizer = VisualizationMarkers(goal_cfg)
            if not hasattr(self, "insertive_object_visualizer"):
                # Create insertive object visualizer with proper prim path

                insertive_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/insertive_object",
                    markers=BLUE_ARROW_X_MARKER_CFG.markers
                )
                # Scale down the insertive object markers
                insertive_cfg.markers["arrow"].scale = (0.05, 0.05, 0.15)
                self.insertive_object_visualizer = VisualizationMarkers(insertive_cfg)
            # Set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.insertive_object_visualizer.set_visibility(True)
        else:
            # Set visibility to false
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "insertive_object_visualizer"):
                self.insertive_object_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update visualization when debug is enabled."""
        if hasattr(self, "goal_pose_visualizer") and hasattr(self, "insertive_object_visualizer"):
            # Visualize goal poses (green arrows) - where insertive object SHOULD be
            # These are computed from receptive object root pose + height offset
            self.goal_pose_visualizer.visualize(
                translations=self.goal_positions,
                orientations=self.goal_orientations,
            )
            
            # Visualize insertive object poses (blue arrows) - where insertive object ACTUALLY is
            # Using root pose directly (no offset) to match what's used in goal computation
            insertive_asset_pos_w = self.insertive_asset.data.root_pos_w
            insertive_asset_quat_w = self.insertive_asset.data.root_quat_w
            self.insertive_object_visualizer.visualize(
                translations=insertive_asset_pos_w,
                orientations=insertive_asset_quat_w,
            )
