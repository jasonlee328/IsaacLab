# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains the functions for push task MDP terms.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Import event functions
from .events import *  # noqa: F401, F403

# Import reward functions
from .rewards import *  # noqa: F401, F403

# Import observation functions from isaaclab
from isaaclab.envs.mdp.observations import generated_commands  # noqa: F401

# Import command configurations from isaaclab
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg  # noqa: F401

# Import object command from dexsuite for cube target positions
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.commands.pose_commands_cfg import (  # noqa: F401
    ObjectUniformPoseCommandCfg,
)


def object_to_object_distance(
    env: ManagerBasedEnv,
    object1_cfg: SceneEntityCfg,
    object2_cfg: SceneEntityCfg,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Distance between two objects with exponential kernel."""
    object1: RigidObject = env.scene[object1_cfg.name]
    object2: RigidObject = env.scene[object2_cfg.name]
    
    # Get positions
    pos1 = object1.data.root_pos_w[:, :3] - env.scene.env_origins
    pos2 = object2.data.root_pos_w[:, :3] - env.scene.env_origins
    
    # Calculate distance
    distance = torch.norm(pos1 - pos2, dim=-1)
    
    # Apply exponential kernel for reward shaping
    return torch.exp(-distance / sigma)


# def object_reached_goal(
#     env: ManagerBasedEnv,
#     object_cfg: SceneEntityCfg,
#     goal_cfg: str,
#     threshold,
# ) -> torch.Tensor:
#     """Binary reward for object reaching goal within threshold."""
#     object_asset: RigidObject = env.scene[object_cfg.name]
     
#     print("=== DEBUG INFO ===")
#     print(f"Object world pos shape: {object_asset.data.root_pos_w[:, :3].shape}")
#     print(f"Object world pos (first env): {object_asset.data.root_pos_w[0, :3]}")
#     print(f"Env origins shape: {env.scene.env_origins.shape}")
#     print(f"Env origins (first env): {env.scene.env_origins[0]}")
#     # Get positions
#     obj_pos = object_asset.data.root_pos_w[:, :3] - env.scene.env_origins
#     goal_pos = env.command_manager.get_command(goal_cfg)[:, :3] - env.scene.env_origins
#     # goal_asset.data.root_pos_w[:, :3] - env.scene.env_origins
    
#     print(f"Object local pos (first env): {obj_pos[0]}")
#     print(f"Goal world pos (first env): {env.command_manager.get_command(goal_cfg)[0, :3]}")
#     print(f"Goal local pos (first env): {goal_pos[0]}")
    
#     # Calculate distance
#     distance = torch.norm(obj_pos - goal_pos, dim=-1)
#     print(f"Distance (first env): {distance[0]}")
    
#     # Binary reward
#     return (distance < threshold).float()
    


def object_reached_goal(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
    goal_cfg: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Binary reward for object reaching goal within threshold."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    
    # Get the command term directly to access world coordinates
    command_term = env.command_manager._terms[goal_cfg]

    # Use the world-transformed command positions
    goal_pos_world = command_term.pose_command_w[:, :3]  # World coordinates
    obj_pos_world = object_asset.data.root_pos_w[:, :3]  # World coordinates

    # Convert both to environment-relative coordinates
    obj_pos = obj_pos_world - env.scene.env_origins
    goal_pos = goal_pos_world - env.scene.env_origins
    
    # Calculate distance
    distance = torch.norm(obj_pos - goal_pos, dim=-1)

    # Binary reward
    success_mask = (distance < threshold).float()

    return success_mask


def distance_orientation_goal(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
    goal_cfg: str,
    distance_threshold: float = 0.05,
    orientation_threshold: float = 0.1,
) -> torch.Tensor:
    """Binary reward for object reaching goal within both distance and orientation thresholds.
    
    Uses the same error calculation method as the command metrics (compute_pose_error).
    
    Args:
        env: The environment.
        object_cfg: Scene entity config for the object to track.
        goal_cfg: Name of the command term that defines the goal pose.
        distance_threshold: Maximum position error (in meters) to consider success.
        orientation_threshold: Maximum orientation error (in radians) to consider success.
    
    Returns:
        Binary reward tensor of shape (num_envs,). Returns 1.0 when both position
        and orientation are within thresholds, 0.0 otherwise.
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    
    # Get the command term directly to access world coordinates
    command_term = env.command_manager._terms[goal_cfg]

    # Use the world-transformed command positions and orientations
    goal_pos_world = command_term.pose_command_w[:, :3]  # (num_envs, 3)
    goal_quat_world = command_term.pose_command_w[:, 3:]  # (num_envs, 4) - (qw, qx, qy, qz)
    
    obj_pos_world = object_asset.data.root_pos_w[:, :3]  # (num_envs, 3)
    obj_quat_world = object_asset.data.root_quat_w  # (num_envs, 4) - (qw, qx, qy, qz)
    
    # Compute pose error using the same method as metrics
    pos_error, rot_error = compute_pose_error(
        goal_pos_world,
        goal_quat_world,
        obj_pos_world,
        obj_quat_world,
    )
    
    # Calculate position and orientation error magnitudes
    position_error = torch.norm(pos_error, dim=-1)
    orientation_error = torch.norm(rot_error, dim=-1)
    
    # Binary reward: both position and orientation must be within thresholds
    position_success = position_error < distance_threshold
    orientation_success = orientation_error < orientation_threshold
    success_mask = (position_success & orientation_success).float()

    return success_mask


def orientation_goal(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
    goal_cfg: str,
    orientation_threshold: float = 0.1,
) -> torch.Tensor:
    """Binary reward for object reaching goal orientation (ignores position).
    
    Only checks if the object's roll, pitch, yaw match the goal orientation within threshold.
    Position is completely ignored.
    
    Uses the same error calculation method as the command metrics (compute_pose_error).
    
    Args:
        env: The environment.
        object_cfg: Scene entity config for the object to track.
        goal_cfg: Name of the command term that defines the goal pose.
        orientation_threshold: Maximum orientation error (in radians) to consider success.
    
    Returns:
        Binary reward tensor of shape (num_envs,). Returns 1.0 when orientation
        is within threshold, 0.0 otherwise. Position is ignored.
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    
    # Get the command term directly to access world coordinates
    command_term = env.command_manager._terms[goal_cfg]

    # Use the world-transformed command orientations
    goal_quat_world = command_term.pose_command_w[:, 3:]  # (num_envs, 4) - (qw, qx, qy, qz)
    obj_quat_world = object_asset.data.root_quat_w  # (num_envs, 4) - (qw, qx, qy, qz)
    
    # We still need to pass positions to compute_pose_error, but we'll ignore the position error
    goal_pos_world = command_term.pose_command_w[:, :3]  # (num_envs, 3)
    obj_pos_world = object_asset.data.root_pos_w[:, :3]  # (num_envs, 3)
    
    # Compute pose error using the same method as metrics
    pos_error, rot_error = compute_pose_error(
        goal_pos_world,
        goal_quat_world,
        obj_pos_world,
        obj_quat_world,
    )
    
    # Calculate only orientation error magnitude (ignore position error)
    orientation_error = torch.norm(rot_error, dim=-1)
    
    # Binary reward: only orientation must be within threshold
    success_mask = (orientation_error < orientation_threshold).float()

    return success_mask



def ee_object_distance(
    env: ManagerBasedEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the robot for reaching the object using tanh-kernel.
    
    Returns a reward in [0, 1] range where 1 means end-effector is very close to object.
    
    Args:
        env: The environment.
        std: Standard deviation for the tanh kernel (controls reward sharpness).
        object_cfg: Scene entity config for the target object.
        ee_frame_cfg: Scene entity config for the end-effector frame.
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)