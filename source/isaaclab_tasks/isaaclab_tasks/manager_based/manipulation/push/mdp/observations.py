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

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset=None,
    root_asset_offset=None,
    use_axis_angle: bool = False,
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids
    root_body_idx = 0 if isinstance(root_asset_cfg.body_ids, slice) else root_asset_cfg.body_ids

    target_pos = target_asset.data.body_link_pos_w[:, target_body_idx].view(-1, 3)
    target_quat = target_asset.data.body_link_quat_w[:, target_body_idx].view(-1, 4)
    root_pos = root_asset.data.body_link_pos_w[:, root_body_idx].view(-1, 3)
    root_quat = root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

    if root_asset_offset is not None:
        root_pos, root_quat = root_asset_offset.combine(root_pos, root_quat)
    if target_asset_offset is not None:
        target_pos, target_quat = target_asset_offset.combine(target_pos, target_quat)

    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)

    if use_axis_angle:
        axis_angle = math_utils.axis_angle_from_quat(target_quat_b)
        return torch.cat([target_pos_b, axis_angle], dim=1)
    else:
        return torch.cat([target_pos_b, target_quat_b], dim=1)
    
    

def ee_frame_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector position in robot base frame (relative to robot base)."""
    # Get the robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector body index (assuming it's the last body)
    ee_body_idx = robot.num_bodies - 1
    
    # Get end-effector pose in world frame
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    
    # Convert to robot base frame (relative to robot base)
    ee_pos_rel, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w
    )
    
    return ee_pos_rel



def ee_frame_quat_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector orientation in robot base frame (relative to robot base)."""
    # Get the robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector body index (assuming it's the last body)
    ee_body_idx = robot.num_bodies - 1
    
    # Get end-effector pose in world frame
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    
    # Convert to robot base frame (relative to robot base)
    _, ee_quat_rel = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w
    )
    
    return ee_quat_rel



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


def cube_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube position in world frame."""
    from isaaclab.assets import RigidObject
    cube: RigidObject = env.scene[asset_cfg.name]
    return cube.data.root_pos_w

def cube_rot_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube orientation in world frame."""
    from isaaclab.assets import RigidObject
    cube: RigidObject = env.scene[asset_cfg.name]
    return cube.data.root_quat_w

def target_pos_rel(env: ManagerBasedRLEnv, command_name: str = "ee_pose") -> torch.Tensor:
    """Target position relative to robot base frame."""
    # Get the command
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :3]  # First 3 elements are position
    
    return target_pos

def target_quat_rel(env: ManagerBasedRLEnv, command_name: str = "ee_pose") -> torch.Tensor:
    """Target orientation relative to robot base frame."""
    # Get the command
    command = env.command_manager.get_command(command_name)
    target_quat = command[:, 3:]  # Last 4 elements are orientation
    
    return target_quat

def target_pos_w(env: ManagerBasedRLEnv, command_name: str = "ee_pose") -> torch.Tensor:
    """Target position in world frame."""
    # Get the command term and access its world frame pose
    command_term = env.command_manager._terms[command_name]
    target_pos_w = command_term.pose_command_w[:, :3]  # First 3 elements are position
    
    return target_pos_w

def target_quat_w(env: ManagerBasedRLEnv, command_name: str = "ee_pose") -> torch.Tensor:
    """Target orientation in world frame."""
    # Get the command term and access its world frame pose
    command_term = env.command_manager._terms[command_name]
    target_quat_w = command_term.pose_command_w[:, 3:]  # Last 4 elements are orientation
    
    return target_quat_w


def cube_in_target_frame(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    use_axis_angle: bool = True,
) -> torch.Tensor:
    """Get cube pose relative to target frame.
    
    This provides a frame-invariant observation: the relative pose between the cube
    and the target doesn't change if both move together in world space.
    
    Args:
        env: The environment instance.
        command_name: The name of the command term for the target.
        asset_cfg: The scene entity configuration for the cube.
        use_axis_angle: If True, return axis-angle for orientation (3D).
                       If False, return quaternion (4D).
        
    Returns:
        Tensor of shape (num_envs, 6) if use_axis_angle=True (pos + axis_angle),
        or (num_envs, 7) if use_axis_angle=False (pos + quat).
    """
    from isaaclab.assets import RigidObject
    
    # Get the cube pose in world frame
    cube: RigidObject = env.scene[asset_cfg.name]
    cube_pos_w = cube.data.root_pos_w
    cube_quat_w = cube.data.root_quat_w
    
    # Get the target pose in world frame
    command_term = env.command_manager._terms[command_name]
    target_pos_w = command_term.pose_command_w[:, :3]
    target_quat_w = command_term.pose_command_w[:, 3:]
    
    # Transform cube pose to target frame
    cube_pos_target, cube_quat_target = math_utils.subtract_frame_transforms(
        target_pos_w, target_quat_w, cube_pos_w, cube_quat_w
    )
    
    # Convert orientation representation if requested
    if use_axis_angle:
        cube_orientation = math_utils.axis_angle_from_quat(cube_quat_target)
    else:
        cube_orientation = cube_quat_target
    
    return torch.cat([cube_pos_target, cube_orientation], dim=-1)


def ee_velocity_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector linear and angular velocity in robot base frame.
    
    Returns a 6D vector: [linear_vel (3D), angular_vel (3D)]
    """
    # Get the robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector body index (assuming it's the last body)
    ee_body_idx = robot.num_bodies - 1
    
    # Get end-effector velocity in world frame
    ee_lin_vel_w = robot.data.body_lin_vel_w[:, ee_body_idx]
    ee_ang_vel_w = robot.data.body_ang_vel_w[:, ee_body_idx]
    
    # Get robot base orientation
    robot_quat_w = robot.data.root_quat_w
    
    # Rotate velocities to robot base frame (velocities are vectors, just need rotation)
    ee_lin_vel_rel = math_utils.quat_apply_inverse(robot_quat_w, ee_lin_vel_w)
    ee_ang_vel_rel = math_utils.quat_apply_inverse(robot_quat_w, ee_ang_vel_w)
    
    return torch.cat([ee_lin_vel_rel, ee_ang_vel_rel], dim=-1)


def cube_velocity_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Cube linear and angular velocity in robot base frame.
    
    Returns a 6D vector: [linear_vel (3D), angular_vel (3D)]
    """
    from isaaclab.assets import RigidObject
    
    # Get the cube asset
    cube: RigidObject = env.scene[asset_cfg.name]
    cube_lin_vel_w = cube.data.root_lin_vel_w
    cube_ang_vel_w = cube.data.root_ang_vel_w
    
    # Get robot base orientation
    robot: Articulation = env.scene[root_asset_cfg.name]
    robot_quat_w = robot.data.root_quat_w
    
    # Rotate velocities to robot base frame
    cube_lin_vel_rel = math_utils.quat_apply_inverse(robot_quat_w, cube_lin_vel_w)
    cube_ang_vel_rel = math_utils.quat_apply_inverse(robot_quat_w, cube_ang_vel_w)
    
    return torch.cat([cube_lin_vel_rel, cube_ang_vel_rel], dim=-1)


def cube_velocity_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube linear and angular velocity in world frame.
    
    Returns a 6D vector: [linear_vel (3D), angular_vel (3D)]
    """
    from isaaclab.assets import RigidObject
    
    cube: RigidObject = env.scene[asset_cfg.name]
    return torch.cat([cube.data.root_lin_vel_w, cube.data.root_ang_vel_w], dim=-1)



##
# Reorientation-specific observations
##

def cube_yaw_angle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current yaw angle of the cube (rotation around Z-axis).
    
    Extracts the yaw angle from the cube's quaternion orientation.
    This is useful for reorientation tasks where you care about rotation around vertical axis.
    
    Returns:
        torch.Tensor: Yaw angle in radians, shape (num_envs, 1)
    """
    from isaaclab.assets import RigidObject
    
    # Get the cube asset
    cube: RigidObject = env.scene[asset_cfg.name]
    
    # Get cube quaternion (w, x, y, z)
    quat = cube.data.root_quat_w  # shape: (num_envs, 4)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Convert quaternion to yaw angle using standard formula
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    return yaw.unsqueeze(-1)  # shape: (num_envs, 1)


def target_yaw_angle(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target yaw angle from the command manager.
    
    Returns:
        torch.Tensor: Target yaw angle in radians, shape (num_envs, 1)
    """
    command_term = env.command_manager.get_term(command_name)
    
    # Extract yaw from goal_orientations (assumed to be in euler or contains yaw)
    # The command manager stores orientations - extract the Z-rotation component
    goal_quat = command_term.pose_command_w[:, 3:]
    
    
    # Convert quaternion to yaw
    w, x, y, z = goal_quat[:, 0], goal_quat[:, 1], goal_quat[:, 2], goal_quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    return yaw.unsqueeze(-1)  # shape: (num_envs, 1)


def orientation_delta(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Signed angular difference between current cube yaw and target yaw.
    
    This computes the shortest rotation needed to reach the target orientation.
    The result is wrapped to the range (-π, π] to ensure the agent learns the
    shortest rotation direction.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the cube asset
        command_name: Name of the command term containing the target orientation
    
    Returns:
        torch.Tensor: Signed angular difference in radians, range (-π, π], shape (num_envs, 1)
    """
    # Get current cube yaw
    current_yaw = cube_yaw_angle(env, asset_cfg).squeeze(-1)  # (num_envs,)
    
    # Get target yaw
    target_yaw = target_yaw_angle(env, command_name).squeeze(-1)  # (num_envs,)
    
    # Compute signed delta: goal - current
    delta = target_yaw - current_yaw
    
    # Wrap to (-π, π] to get shortest rotation
    pi = torch.tensor(torch.pi, device=delta.device, dtype=delta.dtype)
    two_pi = 2 * pi
    delta = torch.remainder(delta + pi, two_pi) - pi
    
    return delta.unsqueeze(-1)  # shape: (num_envs, 1)


