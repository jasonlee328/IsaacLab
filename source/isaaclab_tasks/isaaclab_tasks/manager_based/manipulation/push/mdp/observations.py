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
    
    

# def ee_frame_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """End-effector position in robot base frame (relative to robot base)."""
#     # Get the robot asset
#     robot: Articulation = env.scene[asset_cfg.name]
    
#     # Get end-effector body index (assuming it's the last body)
#     ee_body_idx = robot.num_bodies - 1
    
#     # Get end-effector pose in world frame
#     ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
#     ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
#     robot_pos_w = robot.data.root_pos_w
#     robot_quat_w = robot.data.root_quat_w
    
#     # Convert to robot base frame (relative to robot base)
#     ee_pos_rel, _ = math_utils.subtract_frame_transforms(
#         robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w
#     )
    
#     return ee_pos_rel



# def ee_frame_quat_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """End-effector orientation in robot base frame (relative to robot base)."""
#     # Get the robot asset
#     robot: Articulation = env.scene[asset_cfg.name]
    
#     # Get end-effector body index (assuming it's the last body)
#     ee_body_idx = robot.num_bodies - 1
    
#     # Get end-effector pose in world frame
#     ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
#     ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
#     robot_pos_w = robot.data.root_pos_w
#     robot_quat_w = robot.data.root_quat_w
    
#     # Convert to robot base frame (relative to robot base)
#     _, ee_quat_rel = math_utils.subtract_frame_transforms(
#         robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w
#     )
    
#     return ee_quat_rel


def ee_frame_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector position in robot base frame (relative to robot base) - uses FrameTransformer with TCP offset."""
    # Get the FrameTransformer sensor (includes TCP offset!)
    ee_frame = env.scene.sensors["ee_frame"]
    
    # Get TCP position in world frame (already includes the -0.210m offset)
    ee_pos_w = ee_frame.data.target_pos_w[:, 0]  # First target frame is "end_effector"
    ee_quat_w = ee_frame.data.target_quat_w[:, 0]
    
    # Get robot base pose
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    
    # Convert to robot base frame (relative to robot base)
    ee_pos_rel, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w
    )
    
    return ee_pos_rel



def ee_frame_quat_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector orientation in robot base frame (relative to robot base) - uses FrameTransformer with TCP offset."""
    # Get the FrameTransformer sensor (includes TCP offset!)
    ee_frame = env.scene.sensors["ee_frame"]
    
    # Get TCP pose in world frame (already includes the -0.210m offset)
    ee_pos_w = ee_frame.data.target_pos_w[:, 0]  # First target frame is "end_effector"
    ee_quat_w = ee_frame.data.target_quat_w[:, 0]
    
    # Get robot base pose
    robot: Articulation = env.scene[asset_cfg.name]
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


def distractor_positions_rel(env: ManagerBasedRLEnv, 
                            distractor_1_cfg: SceneEntityCfg = SceneEntityCfg("distractor_1")) -> torch.Tensor:
    """Distractor cube positions relative to robot base.
    
    Returns concatenated positions of both distractor cubes in robot base frame.
    Shape: (num_envs, 6) - [dist1_x, dist1_y, dist1_z, dist2_x, dist2_y, dist2_z]
    """
    from isaaclab.assets import RigidObject
    
    # Extract distractors
    distractor_1: RigidObject = env.scene[distractor_1_cfg.name]
    # distractor_2: RigidObject = env.scene[distractor_2_cfg.name]
    
    # Get positions in world frame
    dist1_pos_w = distractor_1.data.root_pos_w[:, :3]
    # dist2_pos_w = distractor_2.data.root_pos_w[:, :3]
    
    # Convert to environment-relative coordinates
    dist1_pos = dist1_pos_w - env.scene.env_origins
    # dist2_pos = dist2_pos_w - env.scene.env_origins
    
    # Concatenate positions
    return dist1_pos


def distractor_orientations_current(env: ManagerBasedRLEnv,
                                   distractor_1_cfg: SceneEntityCfg = SceneEntityCfg("distractor_1")) -> torch.Tensor:
    """Current distractor cube orientation as quaternion.
    
    Returns the current orientation of the distractor cube.
    Shape: (num_envs, 4) - [qw, qx, qy, qz]
    """
    from isaaclab.assets import RigidObject
    
    # Extract distractor
    distractor_1: RigidObject = env.scene[distractor_1_cfg.name]
    
    # Get current orientation
    dist1_quat_w = distractor_1.data.root_quat_w
    
    return dist1_quat_w


def distractor_yaw_current(env: ManagerBasedRLEnv,
                           distractor_1_cfg: SceneEntityCfg = SceneEntityCfg("distractor_1")) -> torch.Tensor:
    """Current distractor cube yaw angle only.
    
    Returns only the yaw (z-axis rotation) of the distractor cube.
    Shape: (num_envs, 1)
    """
    from isaaclab.assets import RigidObject
    
    # Extract distractor
    distractor_1: RigidObject = env.scene[distractor_1_cfg.name]
    
    # Get current orientation quaternion
    quat = distractor_1.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Convert quaternion to yaw angle
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    return yaw.unsqueeze(-1)  # shape: (num_envs, 1)


def distractor_quats_rel(env: ManagerBasedRLEnv,
                        distractor_1_cfg: SceneEntityCfg = SceneEntityCfg("distractor_1"),
                        distractor_2_cfg: SceneEntityCfg = SceneEntityCfg("distractor_2")) -> torch.Tensor:
    """Distractor cube orientations as quaternions.
    
    Returns concatenated quaternions of both distractor cubes.
    Shape: (num_envs, 8) - [dist1_qw, dist1_qx, dist1_qy, dist1_qz, dist2_qw, dist2_qx, dist2_qy, dist2_qz]
    """
    from isaaclab.assets import RigidObject
    
    # Extract distractors
    distractor_1: RigidObject = env.scene[distractor_1_cfg.name]
    distractor_2: RigidObject = env.scene[distractor_2_cfg.name]
    
    # Get orientations
    dist1_quat_w = distractor_1.data.root_quat_w
    dist2_quat_w = distractor_2.data.root_quat_w
    
    # Concatenate quaternions
    return torch.cat([dist1_quat_w, dist2_quat_w], dim=-1)


def distractor_initial_positions_rel(env: ManagerBasedRLEnv,
                                     distractor_1_cfg: SceneEntityCfg = SceneEntityCfg("distractor_1")) -> torch.Tensor:
    """Initial distractor cube positions relative to environment origin.
    
    Returns the initial spawning positions of distractor cubes (stored during reset).
    This allows the policy to know where distractors should be and guide them back if moved.
    
    Shape: (num_envs, 3) - [dist1_x, dist1_y, dist1_z]
    
    Note: Requires the environment to have 'distractor_initial_poses_w' attribute,
          which is set by ReorientWithDistractorsEnv during reset.
    """
    # Check if initial poses are stored
    if not hasattr(env, 'distractor_initial_poses_w'):
        # Return zeros if not initialized (shouldn't happen in practice)
        return torch.zeros(env.num_envs, 3, device=env.device)
    
    distractor_name = distractor_1_cfg.name
    
    # Check if this specific distractor is tracked
    if distractor_name not in env.distractor_initial_poses_w:
        # Return zeros if this distractor isn't tracked
        return torch.zeros(env.num_envs, 3, device=env.device)
    
    # Get initial position in world frame
    initial_pos_w, _ = env.distractor_initial_poses_w[distractor_name]
    
    # Convert to environment-relative coordinates
    initial_pos_rel = initial_pos_w - env.scene.env_origins
    
    return initial_pos_rel


def distractor_initial_orientations(env: ManagerBasedRLEnv,
                                    distractor_1_cfg: SceneEntityCfg = SceneEntityCfg("distractor_1")) -> torch.Tensor:
    """Initial distractor cube orientations as quaternions.
    
    Returns the initial spawning orientations of distractor cubes (stored during reset).
    This allows the policy to know the target orientation for distractors.
    
    Shape: (num_envs, 4) - [dist1_qw, dist1_qx, dist1_qy, dist1_qz]
    
    Note: Requires the environment to have 'distractor_initial_poses_w' attribute,
          which is set by ReorientWithDistractorsEnv during reset.
    """
    # Check if initial poses are stored
    if not hasattr(env, 'distractor_initial_poses_w'):
        # Return identity quaternion if not initialized
        identity_quat = torch.zeros(env.num_envs, 4, device=env.device)
        identity_quat[:, 0] = 1.0  # w component
        return identity_quat
    
    distractor_name = distractor_1_cfg.name
    
    # Check if this specific distractor is tracked
    if distractor_name not in env.distractor_initial_poses_w:
        # Return identity quaternion if this distractor isn't tracked
        identity_quat = torch.zeros(env.num_envs, 4, device=env.device)
        identity_quat[:, 0] = 1.0  # w component
        return identity_quat
    
    # Get initial quaternion
    _, initial_quat_w = env.distractor_initial_poses_w[distractor_name]
    
    return initial_quat_w


def distractor_yaw_initial(env: ManagerBasedRLEnv,
                           distractor_1_cfg: SceneEntityCfg = SceneEntityCfg("distractor_1")) -> torch.Tensor:
    """Initial distractor cube yaw angle only.
    
    Returns only the initial yaw (z-axis rotation) of the distractor cube.
    Shape: (num_envs, 1)
    
    Note: Requires the environment to have 'distractor_initial_poses_w' attribute.
    """
    # Check if initial poses are stored
    if not hasattr(env, 'distractor_initial_poses_w'):
        # Return zero yaw if not initialized
        return torch.zeros(env.num_envs, 1, device=env.device)
    
    distractor_name = distractor_1_cfg.name
    
    # Check if this specific distractor is tracked
    if distractor_name not in env.distractor_initial_poses_w:
        # Return zero yaw if this distractor isn't tracked
        return torch.zeros(env.num_envs, 1, device=env.device)
    
    # Get initial quaternion
    _, initial_quat_w = env.distractor_initial_poses_w[distractor_name]
    
    # Convert quaternion to yaw angle
    w, x, y, z = initial_quat_w[:, 0], initial_quat_w[:, 1], initial_quat_w[:, 2], initial_quat_w[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    return yaw.unsqueeze(-1)  # shape: (num_envs, 1)


##
# Reorientation-specific observations
##

def arm_joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Arm joint positions only (excludes gripper joints).
    
    Returns only the 7 Franka arm joints (panda_joint1-7), excluding gripper joints.
    
    Returns:
        torch.Tensor: Arm joint positions, shape (num_envs, 7)
    """
    from isaaclab.assets import Articulation
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get all joint names
    joint_names = asset.data.joint_names
    
    # Find indices of arm joints (panda_joint1 through panda_joint7)
    arm_joint_indices = []
    for i, name in enumerate(joint_names):
        if name.startswith("panda_joint") and name[-1].isdigit():
            # This is an arm joint (panda_joint1-7)
            arm_joint_indices.append(i)
    
    # Return arm joint positions relative to default
    arm_joint_indices = torch.tensor(arm_joint_indices, device=asset.device)
    return asset.data.joint_pos[:, arm_joint_indices] - asset.data.default_joint_pos[:, arm_joint_indices]


def arm_joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Arm joint velocities only (excludes gripper joints).
    
    Returns only the 7 Franka arm joints (panda_joint1-7), excluding gripper joints.
    
    Returns:
        torch.Tensor: Arm joint velocities, shape (num_envs, 7)
    """
    from isaaclab.assets import Articulation
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get all joint names
    joint_names = asset.data.joint_names
    
    # Find indices of arm joints (panda_joint1 through panda_joint7)
    arm_joint_indices = []
    for i, name in enumerate(joint_names):
        if name.startswith("panda_joint") and name[-1].isdigit():
            # This is an arm joint (panda_joint1-7)
            arm_joint_indices.append(i)
    
    # Return arm joint velocities relative to default
    arm_joint_indices = torch.tensor(arm_joint_indices, device=asset.device)
    return asset.data.joint_vel[:, arm_joint_indices] - asset.data.default_joint_vel[:, arm_joint_indices]


def gripper_state_binary(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
                         threshold: float = 0.4) -> torch.Tensor:
    """Binary gripper state: 0 = open, 1 = closed.
    
    Uses the outer_knuckle_joint positions to determine if gripper is open or closed.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        threshold: Joint position threshold to determine closed state (default: 0.4 radians)
    
    Returns:
        torch.Tensor: Binary gripper state, shape (num_envs, 1)
            0 = open (joint position < threshold)
            1 = closed (joint position >= threshold)
    """
    from isaaclab.assets import Articulation
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get all joint names
    joint_names = asset.data.joint_names
    
    # Find gripper joint indices (outer_knuckle_joints)
    gripper_joint_indices = []
    for i, name in enumerate(joint_names):
        if "outer_knuckle_joint" in name:
            gripper_joint_indices.append(i)
    
    if len(gripper_joint_indices) == 0:
        # Fallback: return zeros if no gripper joints found
        return torch.zeros(asset.num_instances, 1, device=asset.device)
    
    # Get average gripper joint position
    gripper_joint_indices = torch.tensor(gripper_joint_indices, device=asset.device)
    gripper_pos = asset.data.joint_pos[:, gripper_joint_indices].mean(dim=1)
    
    # Threshold to binary: 0 if open, 1 if closed
    gripper_binary = (gripper_pos >= threshold).float()
    
    return gripper_binary.unsqueeze(-1)  # shape: (num_envs, 1)


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


def cube_orientation_euler(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current orientation of the cube as euler angles (roll, pitch, yaw).
    
    Extracts the full orientation from the cube's quaternion as euler angles.
    This is useful for flip/reorientation tasks where all rotation axes matter.
    
    Returns:
        torch.Tensor: Euler angles (roll, pitch, yaw) in radians, shape (num_envs, 3)
    """
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import euler_xyz_from_quat
    
    # Get the cube asset
    cube: RigidObject = env.scene[asset_cfg.name]
    
    # Get cube quaternion (w, x, y, z)
    quat = cube.data.root_quat_w  # shape: (num_envs, 4)
    
    # Convert quaternion to euler angles
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    
    # Stack into single tensor
    return torch.stack([roll, pitch, yaw], dim=-1)  # shape: (num_envs, 3)


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


def target_orientation_euler(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target orientation as euler angles (roll, pitch, yaw) from the command manager.
    
    Extracts the full target orientation from the command as euler angles.
    This is useful for flip/reorientation tasks where all rotation axes matter.
    
    Returns:
        torch.Tensor: Target euler angles (roll, pitch, yaw) in radians, shape (num_envs, 3)
    """
    from isaaclab.utils.math import euler_xyz_from_quat
    
    command_term = env.command_manager.get_term(command_name)
    
    # Get target quaternion in world frame
    goal_quat = command_term.pose_command_w[:, 3:]  # shape: (num_envs, 4)
    
    # Convert quaternion to euler angles
    roll, pitch, yaw = euler_xyz_from_quat(goal_quat)
    
    # Stack into single tensor
    return torch.stack([roll, pitch, yaw], dim=-1)  # shape: (num_envs, 3)


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


