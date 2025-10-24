# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for push task with debug capabilities."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_frame_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector position in robot base frame (relative to robot base)."""
    # Get the robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector body index (assuming it's the last body)
    ee_body_idx = robot.num_bodies - 1
    
    # Get end-effector position in world frame
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    
    # Convert to robot base frame (relative to robot base)
    ee_pos_rel, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, ee_pos_w, torch.zeros_like(robot_quat_w)
    )
    
    return ee_pos_rel


def ee_frame_quat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector quaternion in robot base frame (relative to robot base)."""
    # Get the robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector body index (assuming it's the last body)
    ee_body_idx = robot.num_bodies - 1
    
    # Get end-effector quaternion in world frame
    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
    robot_quat_w = robot.data.root_quat_w
    
    # Convert to robot base frame (relative to robot base)
    _, ee_quat_rel = math_utils.subtract_frame_transforms(
        torch.zeros_like(robot.data.root_pos_w), robot_quat_w, 
        torch.zeros_like(robot.data.root_pos_w), ee_quat_w
    )
    
    return ee_quat_rel


def cube_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube position relative to robot base frame."""
    from isaaclab.assets import RigidObject
    
    cube: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]
    
    # Get positions in world frame
    cube_pos_w = cube.data.root_pos_w
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    
    # Convert cube position to robot base frame
    cube_pos_rel, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, cube_pos_w, torch.zeros_like(robot_quat_w)
    )
    
    return cube_pos_rel


def target_pos_rel(env: ManagerBasedRLEnv, command_name: str = "ee_pose") -> torch.Tensor:
    """Target position relative to robot base frame."""
    # Get the command
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :3]  # First 3 elements are position
    
    return target_pos


def target_rot_rel(env: ManagerBasedRLEnv, command_name: str = "ee_pose") -> torch.Tensor:
    """Target rotation relative to robot base frame."""
    # Get the command
    command = env.command_manager.get_command(command_name)
    target_rot = command[:, 3:]  # Last 3 elements are rotation
    
    return target_rot