#!/usr/bin/env python3
# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.

"""Script to visualize the blocks environment with checkpoint inference."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize blocks environment with checkpoint inference.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Push-Cube-Franka-Easy-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for inference.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from inference import load_checkpoint
from isaaclab.sim.schemas import activate_contact_sensors
import omni.usd

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Inference agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # DEBUG: Print what events are registered
    print(f"\n[DEBUG] Event manager available modes: {env.unwrapped.event_manager.available_modes}")
    if "reset" in env.unwrapped.event_manager.available_modes:
        print(f"[DEBUG] Events in 'reset' mode: {env.unwrapped.event_manager._mode_term_names['reset']}")
    print()
    
    # Get robot body names to find gripper fingers
    robot = env.unwrapped.scene["robot"]
    print(f"[INFO]: Robot body names: {robot.body_names}")
    
    # Use wrist (panda_link7) for contact sensing - simpler and more reliable
    # This will detect contact for the entire end-effector assembly
    wrist_body = "panda_link7"
    print(f"[INFO]: Using wrist body for contact sensing: {wrist_body}")
    
    # Activate contact sensors on wrist
    activated = []
    try:
        # Get the stage using Omniverse USD API
        stage = omni.usd.get_context().get_stage()
        
        # Build the concrete prim path for env_0
        # The prim path template is like "{ENV_REGEX_NS}/Robot"
        # For env_0, it becomes "/World/envs/env_0/Robot"
        base_prim_path = "/World/envs/env_0/Robot"
        
        # Activate contact sensor on wrist link
        try:
            wrist_path = f"{base_prim_path}/{wrist_body}"
            print(f"[INFO]: Activating contact sensor on path: {wrist_path}")
            activate_contact_sensors(wrist_path, threshold=0.01, stage=stage)
            activated.append(wrist_body)
            print(f"[INFO]: ✓ Activated contact sensor on: {wrist_body}")
        except Exception as e:
            print(f"[WARNING]: Failed to activate sensor on {wrist_body}: {e}")
        
        if activated:
            print(f"\n[SUCCESS]: Activated contact sensors on wrist")
        else:
            print("[WARNING]: Could not activate contact sensors on wrist")
            
    except Exception as e:
        print(f"[ERROR]: Failed to activate contact sensors: {e}")
    
    # Load checkpoint for inference if provided
    policy = None
    if args_cli.checkpoint:
        policy = load_checkpoint(args_cli.checkpoint, env, device=args_cli.device)
        print("[INFO]: Using checkpoint for inference")
    else:
        print("[INFO]: No checkpoint provided, using random actions")
    
    # reset environment
    obs, _ = env.reset()
    print(f"[INFO]: Observation dimension: {obs.shape[-1]}")
    print("[INFO]: Environment reset - observe target position")
    
    # simulate environment
    step_count = 0
    reset_interval = 10
    # Reset every 200 steps to see randomization
    
    # Get wrist body index for contact force reading
    wrist_body_idx = None
    if activated:
        try:
            wrist_body_idx = robot.body_names.index(wrist_body)
            print(f"[INFO]: Wrist body index for contact reading: {wrist_body_idx}")
        except ValueError:
            print(f"[WARNING]: Could not find wrist body '{wrist_body}' in robot body names")
    
    # We'll use the robot's articulation data directly for contact detection
    # IsaacLab articulations have built-in contact force data when sensors are activated
    print(f"[INFO]: Will monitor wrist-cube proximity and attempt to read contact forces")
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if policy is not None:
                # Use checkpoint inference
                actions = policy(obs)
        
            else:
                # Fallback to random actions
                actions = 4 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 2
            
            # apply actions
            obs, _, _, _, _ = env.step(actions)
            
            # Read and display contact information for wrist
            if step_count % 1 == 0 and wrist_body_idx is not None:
                try:
                    # Get wrist position
                    wrist_pos_w = robot.data.body_pos_w[0, wrist_body_idx]
                    
                    # Get cube position 
                    cube = env.unwrapped.scene["cube"]
                    cube_pos = cube.data.root_pos_w[0]
                    
                    # Calculate distance
                    distance = torch.norm(wrist_pos_w - cube_pos).item()
                    
                    # Method 1: Try to get contact forces from PhysX (if available)
                    contact_force = None
                    if activated:
                        try:
                            # Try to access contact forces through the articulation's PhysX view
                            # When contact sensors are activated, PhysX tracks contact forces
                            import omni.physics.tensors.impl.api as physx_api
                            
                            # Get the PhysX view
                            physx_view = robot._root_physx_view
                            if hasattr(physx_view, 'get_link_incoming_joint_force'):
                                # Try to get forces on the link
                                forces = physx_view.get_link_incoming_joint_force()
                                if forces is not None and forces.shape[1] > wrist_body_idx:
                                    link_force = forces[0, wrist_body_idx]
                                    force_mag = torch.norm(link_force).item()
                                    if force_mag > 0.1:
                                        contact_force = link_force
                        except Exception as e:
                            if step_count == 10:  # Print once
                                print(f"[DEBUG] PhysX contact force reading not available: {e}")
                    
                    # Display contact force if available
                    if contact_force is not None:
                        force_mag = torch.norm(contact_force).item()
                        print(f"\n[CONTACT] Step {step_count}:")
                        print(f"  Wrist force magnitude: {force_mag:.3f} N")
                        print(f"  Force vector [X Y Z]: [{contact_force[0]:.3f}, {contact_force[1]:.3f}, {contact_force[2]:.3f}] N")
                        print(f"  Wrist-Cube distance: {distance:.4f} m")
                    
                    # Display proximity information when close (fallback/always works)
                    elif distance < 0.15:  # Within 15cm
                        print(f"\n[PROXIMITY] Step {step_count}:")
                        print(f"  Wrist-Cube distance: {distance:.4f} m")
                        if distance < 0.05:
                            print(f"  ⚠️  Very close contact! (< 5cm)")
                        print(f"  Wrist pos: [{wrist_pos_w[0]:.3f}, {wrist_pos_w[1]:.3f}, {wrist_pos_w[2]:.3f}]")
                        print(f"  Cube pos:  [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
                        
                except Exception as e:
                    if step_count % 100 == 0:
                        print(f"[DEBUG] Monitoring error: {e}")
            
            # Periodic reset to visualize target randomization
            step_count += 1
            if step_count % reset_interval == 0:
                obs, _ = env.reset()
                print(f"[INFO]: Reset at step {step_count} - observe new target position")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
