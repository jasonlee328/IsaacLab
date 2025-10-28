#!/usr/bin/env python3
# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.

"""Script to visualize the blocks environment with random actions."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize blocks environment with random actions.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Push-Cube-Franka-Easy-v0", help="Name of the task.")
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

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Random actions agent with Isaac Lab environment."""
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
    
    # reset environment
    env.reset()
    print("[INFO]: Environment reset - observe target position")
    
    # simulate environment
    step_count = 0
    reset_interval = 20
    # Reset every 200 steps to see randomization
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 4 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 2
            # apply actions
            env.step(actions)
            
            # Periodic reset to visualize target randomization
            step_count += 1
            if step_count % reset_interval == 0:
                env.reset()
                print(f"[INFO]: Reset at step {step_count} - observe new target position")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
