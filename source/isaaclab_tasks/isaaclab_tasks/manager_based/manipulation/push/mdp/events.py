# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push task specific event functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def separate_target_from_cube(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    min_separation: float = 0.10,
):
    """Move target away from cube if they are too close.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to check.
        min_separation: Minimum distance between cube and target (in meters).
    """
    if env_ids is None:
        return
    
    cube = env.scene["cube"]
    target = env.scene["target"]
    
    # Get positions in world frame
    cube_pos = cube.data.root_pos_w[env_ids, :2]  # Only x,y
    target_pos = target.data.root_pos_w[env_ids, :2]  # Only x,y
    
    # Calculate distance
    diff = target_pos - cube_pos
    distance = torch.norm(diff, dim=-1)
    
    # Find environments where target is too close to cube
    too_close = distance < min_separation
    
    if too_close.any():
        # For each environment that's too close, move target away
        for idx, env_id in enumerate(env_ids):
            if too_close[idx]:
                # Calculate direction from cube to target
                direction = diff[idx]
                if torch.norm(direction) < 1e-6:
                    # If exactly on top, push in random direction
                    direction = torch.randn(2, device=env.device)
                
                # Normalize and scale to minimum separation
                direction = direction / torch.norm(direction) * min_separation
                
                # Set new target position
                new_target_pos = cube_pos[idx] + direction
                
                # Get current target pose
                current_pose = target.data.root_pose_w[env_id:env_id+1].clone()
                # Update x,y position, keep z and orientation
                current_pose[0, 0:2] = new_target_pos
                
                # Write to simulation
                target.write_root_pose_to_sim(current_pose, env_ids=torch.tensor([env_id.item()], device=env.device))
                target.write_root_velocity_to_sim(
                    torch.zeros(1, 6, device=env.device),
                    env_ids=torch.tensor([env_id.item()], device=env.device)
                )


def position_ee_near_cube_simple(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    joint_positions: list[float] = None,
    randomize_joints: bool = True,
    joint_noise_std: float = 0.05,
    debug: bool = False,
):
    """Position the robot using a predefined joint configuration (simple, no IK).
    
    This is a simpler alternative to IK-based positioning. It sets the robot to a
    predefined "ready" pose that positions the end-effector in a good starting position
    for pushing tasks.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        robot_cfg: Configuration for the robot asset.
        joint_positions: List of joint positions [j1, j2, ..., j7, gripper_left, gripper_right].
                        If None, uses a default "ready to push" pose.
        randomize_joints: Whether to add noise to joint positions for variety.
        joint_noise_std: Standard deviation of noise to add to joints (in radians).
        debug: Whether to print debug information.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = len(env_ids)
    
    # Default "ready to push" pose - EE extended forward and slightly above table
    # This pose positions the EE at approximately (0.6, 0.0, 0.15) in robot base frame
    if joint_positions is None:
        joint_positions = [
            0.0,      # j1: base rotation
            -0.5,     # j2: shoulder forward
            0.0,      # j3: elbow rotation
            -2.2,     # j4: elbow bend
            0.0,      # j5: wrist rotation
            1.7,      # j6: wrist bend
            0.785,    # j7: flange rotation (45 degrees)
            0.04,     # gripper left
            0.04,     # gripper right (open)
        ]
    
    # Create joint position tensor
    joint_pos = torch.tensor(joint_positions, device=env.device).unsqueeze(0).repeat(num_envs, 1)
    
    # Add randomization if requested
    if randomize_joints and joint_noise_std > 0:
        # Only randomize arm joints (not gripper)
        joint_noise = torch.randn(num_envs, 7, device=env.device) * joint_noise_std
        joint_pos[:, :7] += joint_noise
    
    # Clamp to joint limits
    joint_pos_limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(
        joint_pos_limits[:, :, 0],
        joint_pos_limits[:, :, 1]
    )
    
    # Set joint positions
    joint_vel = torch.zeros_like(joint_pos)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    if debug and num_envs > 0:
        print("\n" + "="*80)
        print("DEBUG: position_ee_near_cube_simple")
        print("="*80)
        print(f"Number of environments being reset: {num_envs}")
        print(f"Joint positions: {joint_pos[0, :7].cpu().numpy()}")
        print(f"Randomization: {randomize_joints} (std={joint_noise_std})")
        print("="*80 + "\n")


def position_ee_near_cube_ik(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    cube_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    offset_range: tuple[float, float, float] = (-0.01, 0.0, 0.10),
    offset_std: tuple[float, float, float] = (0.02, 0.03, 0.02),
    ik_iterations: int = 50,
    ik_dt: float = 0.01,
    debug: bool = False,
):
    """Position the robot's end-effector near the cube using inverse kinematics.
    
    This function uses damped least squares (DLS) IK to solve for joint positions
    that place the end-effector at the desired position near the cube.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        robot_cfg: Configuration for the robot asset.
        cube_cfg: Configuration for the cube asset.
        ee_frame_cfg: Configuration for the end-effector frame transformer.
        offset_range: Mean offset from cube (x, y, z) in meters. Default (-0.01, 0.0, 0.10)
                     means 1cm behind cube, same Y, and 10cm above.
        offset_std: Standard deviation for offset randomization (x, y, z) in meters.
        ik_iterations: Number of IK iterations to perform.
        ik_dt: Time step for each IK iteration.
        debug: Whether to print debug information about EE positioning.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get the body index for the end-effector (needed for Jacobian)
    # The ee_frame tracks a specific body, we need to find its index in the robot
    ee_body_name = ee_frame.cfg.target_frames[0].prim_path.split("/")[-1]
    ee_body_idx, _ = robot.find_bodies(ee_body_name)
    ee_body_idx = ee_body_idx[0]  # Get first match
    
    # Get the arm joint indices (first 7 joints for Franka)
    arm_joint_ids = list(range(7))
    
    # Get cube positions for specified environments
    cube_pos_w = cube.data.root_pos_w[env_ids]  # (num_envs, 3)
    cube_quat_w = cube.data.root_quat_w[env_ids]  # (num_envs, 4)
    
    # Sample random offsets
    num_envs = len(env_ids)
    offset = torch.tensor(offset_range, device=env.device).unsqueeze(0).repeat(num_envs, 1)
    
    # Add randomization
    if any(std > 0 for std in offset_std):
        noise = torch.randn(num_envs, 3, device=env.device) * torch.tensor(offset_std, device=env.device)
        offset = offset + noise
    
    # Calculate target end-effector position (in world frame)
    target_ee_pos_w = cube_pos_w + offset
    # Keep same orientation as cube (pointing down for pushing)
    target_ee_quat_w = cube_quat_w.clone()
    
    if debug and len(env_ids) > 0:
        first_idx = 0
        print("\n" + "="*80)
        print("DEBUG: position_ee_near_cube (IK-based)")
        print("="*80)
        print(f"Number of environments being reset: {num_envs}")
        print(f"Offset range: {offset_range}")
        print(f"Offset std: {offset_std}")
        print(f"IK iterations: {ik_iterations}")
        print(f"\nEnvironment {env_ids[first_idx].item()}:")
        print(f"  Cube position (world): {cube_pos_w[first_idx].cpu().numpy()}")
        print(f"  Actual offset: {offset[first_idx].cpu().numpy()}")
        print(f"  Target EE position (world): {target_ee_pos_w[first_idx].cpu().numpy()}")
        print(f"  Distance from cube to target EE: {torch.norm(offset[first_idx]).item():.4f}m")
    
    # Initialize with current joint positions
    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    
    # Perform IK iterations
    for iteration in range(ik_iterations):
        # Get current end-effector pose
        curr_ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]  # (num_envs, 3)
        curr_ee_quat_w = ee_frame.data.target_quat_w[env_ids, 0, :]  # (num_envs, 4) - w,x,y,z
        
        # Compute pose error
        pos_error = target_ee_pos_w - curr_ee_pos_w
        
        # For orientation, compute axis-angle error
        # quat_error = target_quat * curr_quat^-1
        curr_ee_quat_inv = curr_ee_quat_w.clone()
        curr_ee_quat_inv[:, 1:] *= -1  # Conjugate (inverse for unit quaternions)
        
        # Quaternion multiplication: q1 * q2
        w1, x1, y1, z1 = target_ee_quat_w[:, 0], target_ee_quat_w[:, 1], target_ee_quat_w[:, 2], target_ee_quat_w[:, 3]
        w2, x2, y2, z2 = curr_ee_quat_inv[:, 0], curr_ee_quat_inv[:, 1], curr_ee_quat_inv[:, 2], curr_ee_quat_inv[:, 3]
        
        quat_error_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        quat_error_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        quat_error_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        quat_error_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        # Convert to axis-angle
        quat_error = torch.stack([quat_error_w, quat_error_x, quat_error_y, quat_error_z], dim=-1)
        axis_angle_error = 2.0 * quat_error[:, 1:]  # Simplified for small angles
        
        # Combine position and orientation errors
        pose_error = torch.cat([pos_error, axis_angle_error], dim=-1)  # (num_envs, 6)
        
        # Get Jacobian for the end-effector from the robot
        # The Jacobian relates joint velocities to end-effector velocities
        # PhysX returns Jacobians for all bodies, we need the EE body
        # Note: ee_jacobi_idx = ee_body_idx - 1 (PhysX indexing)
        ee_jacobi_idx = ee_body_idx - 1
        jacobian_w = robot.root_physx_view.get_jacobians()[env_ids, ee_jacobi_idx, :, :][:, :, arm_joint_ids]  # (num_envs, 6, 7)
        
        # Convert Jacobian from world frame to robot base frame
        jacobian = jacobian_w.clone()
        root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w[env_ids]))
        jacobian[:, :3, :] = torch.bmm(root_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian[:, 3:, :])
        
        # Solve using Damped Least Squares (DLS)
        # delta_q = J^T * (J * J^T + lambda^2 * I)^-1 * pose_error
        lambda_damping = 0.05  # Damping factor
        
        # J * J^T + lambda^2 * I
        JJT = torch.bmm(jacobian, jacobian.transpose(-2, -1))  # (num_envs, 6, 6)
        damping_matrix = lambda_damping ** 2 * torch.eye(6, device=env.device).unsqueeze(0).repeat(num_envs, 1, 1)
        JJT_damped = JJT + damping_matrix
        
        # Solve: (J * J^T + lambda^2 * I)^-1 * pose_error
        try:
            JJT_inv_pose_error = torch.linalg.solve(JJT_damped, pose_error.unsqueeze(-1)).squeeze(-1)
        except:
            # If solve fails, use pseudo-inverse
            JJT_inv = torch.linalg.pinv(JJT_damped)
            JJT_inv_pose_error = torch.bmm(JJT_inv, pose_error.unsqueeze(-1)).squeeze(-1)
        
        # delta_q = J^T * JJT_inv_pose_error
        delta_joint_pos = torch.bmm(
            jacobian.transpose(-2, -1),
            JJT_inv_pose_error.unsqueeze(-1)
        ).squeeze(-1)  # (num_envs, 7)
        
        # Update joint positions
        joint_pos[:, :7] += delta_joint_pos * ik_dt
        
        # Clamp to joint limits
        joint_pos_limits = robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[:, :7] = joint_pos[:, :7].clamp(
            joint_pos_limits[:, :7, 0],
            joint_pos_limits[:, :7, 1]
        )
        
        # Write to simulation
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Check convergence (optional - could break early if error is small)
        if iteration % 10 == 0:
            max_pos_error = torch.norm(pos_error, dim=-1).max().item()
            if max_pos_error < 0.001:  # 1mm tolerance
                if debug:
                    print(f"  IK converged at iteration {iteration}, max error: {max_pos_error:.6f}m")
                break
    
    # Set final joint positions as target
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    
    if debug and len(env_ids) > 0:
        # Check final EE position after IK
        final_ee_pos = ee_frame.data.target_pos_w[env_ids, 0, :]
        final_distance = torch.norm(final_ee_pos[first_idx] - target_ee_pos_w[first_idx]).item()
        print(f"\n  After IK ({ik_iterations} iterations):")
        print(f"    Final EE position: {final_ee_pos[first_idx].cpu().numpy()}")
        print(f"    Target EE position: {target_ee_pos_w[first_idx].cpu().numpy()}")
        print(f"    Final error: {final_distance:.6f}m ({final_distance*1000:.2f}mm)")
        print(f"    Joint positions: {joint_pos[first_idx, :7].cpu().numpy()}")
        print("="*80 + "\n")


def store_distractor_initial_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distractor_1_cfg: SceneEntityCfg,
    distractor_2_cfg: SceneEntityCfg,
):
    """Store initial positions of distractor cubes for movement detection.
    
    This should be called during environment reset to track the initial spawn positions
    of distractor cubes. The positions are stored as environment attributes for later
    comparison in termination conditions.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices that are being reset.
        distractor_1_cfg: Configuration for the first distractor cube.
        distractor_2_cfg: Configuration for the second distractor cube.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    
    # Initialize storage if it doesn't exist
    if not hasattr(env, '_distractor_initial_pos'):
        num_envs = env.scene.num_envs
        device = env.device
        env._distractor_initial_pos = {
            'distractor_1': torch.zeros((num_envs, 3), device=device),
            'distractor_2': torch.zeros((num_envs, 3), device=device),
        }
    
    # Get distractor objects
    distractor_1: RigidObject = env.scene[distractor_1_cfg.name]
    distractor_2: RigidObject = env.scene[distractor_2_cfg.name]
    
    # Store current positions as initial positions for the reset environments
    env._distractor_initial_pos['distractor_1'][env_ids] = distractor_1.data.root_pos_w[env_ids, :3].clone()
    env._distractor_initial_pos['distractor_2'][env_ids] = distractor_2.data.root_pos_w[env_ids, :3].clone()


def randomize_all_cubes_with_command_separation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.15,
    min_command_separation: float = 0.15,
    command_name: str = "ee_pose",
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    """Randomize cube positions ensuring separation from each other AND the command position.
    
    Args:
        env: The environment.
        env_ids: Environment IDs to randomize.
        asset_cfgs: List of asset configurations to randomize.
        min_separation: Minimum distance between objects (in meters).
        min_command_separation: Minimum distance from command target position (in meters).
        command_name: Name of the command to get target position from.
        pose_range: Dictionary with position/orientation ranges.
        max_sample_tries: Maximum sampling attempts per object.
    """
    if env_ids is None:
        return
    
    import random
    import math
    from isaaclab.utils import math as math_utils
    
    # Get command positions in world frame
    command_term = env.command_manager._terms[command_name]
    command_pos_w = command_term.pose_command_w[:, :3]  # (num_envs, 3)
    
    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        # Sample positions ensuring separation from command
        valid_poses = []
        
        for attempt in range(max_sample_tries):
            # Sample random pose
            pose = []
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]:
                range_val = pose_range.get(key, (0.0, 0.0))
                pose.append(random.uniform(range_val[0], range_val[1]))
            
            # Check distance from command position
            cmd_pos = command_pos_w[cur_env]
            pose_pos = torch.tensor(pose[:3], device=env.device)
            cmd_dist = torch.norm(pose_pos - cmd_pos[:3]).item()
            
            if cmd_dist < min_command_separation:
                continue  # Too close to command, try again
                
            # Check distance from other objects
            is_valid = True
            for other_pose in valid_poses:
                dist = math.dist(pose[:3], other_pose[:3])
                if dist < min_separation:
                    is_valid = False
                    break
                    
            if is_valid:
                valid_poses.append(pose)
                
            if len(valid_poses) == len(asset_cfgs):
                break
        
        # If we couldn't find valid positions for all, use what we have
        while len(valid_poses) < len(asset_cfgs):
            # Add a pose far from everything
            angle = random.uniform(0, 2 * math.pi)
            dist = max(min_separation, min_command_separation) * 2
            x = pose_range["x"][0] + (pose_range["x"][1] - pose_range["x"][0]) / 2
            y = pose_range["y"][0] + (pose_range["y"][1] - pose_range["y"][0]) / 2
            pose = [
                x + dist * math.cos(angle),
                y + dist * math.sin(angle), 
                pose_range["z"][0],
                0, 0, random.uniform(pose_range.get("yaw", (0,0))[0], pose_range.get("yaw", (0,0))[1])
            ]
            valid_poses.append(pose)
        
        # Apply poses to objects
        for i, asset_cfg in enumerate(asset_cfgs):
            asset = env.scene[asset_cfg.name]
            
            # Create pose tensor
            pose_tensor = torch.tensor([valid_poses[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            
            # Write to simulation
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), 
                env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), 
                env_ids=torch.tensor([cur_env], device=env.device)
            )

