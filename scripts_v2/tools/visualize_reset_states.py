# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Script to visualize saved states from HDF5 dataset."""

from __future__ import annotations

import argparse
import time
import torch
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize saved reset states from a dataset directory.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./reset_state_datasets",
    help="Directory containing reset-state datasets saved as <hash>.pt",
)
parser.add_argument("--reset_interval", type=float, default=0.1, help="Time interval between resets in seconds.")

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import contextlib
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.utils.hydra import hydra_task_config_programmatic

from isaaclab_tasks.manager_based.manipulation.reset_states.mdp import events as task_mdp

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config_programmatic(args_cli.task, "env_cfg_entry_point", remaining_args)
def main(env_cfg, agent_cfg) -> None:
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic for diverse pose discovery
    env_cfg.seed = None

    # Set up the MultiResetManager to load states from the computed dataset
    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [args_cli.dataset_dir],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

    # Add the reset manager to the environment configuration
    env_cfg.events.reset_from_reset_states = reset_from_reset_states

    # create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    env.reset()

    # Initialize variables
    print(f"Starting visualization of saved states from {args_cli.dataset_dir}")
    print("Press Ctrl+C to stop")

    with contextlib.suppress(KeyboardInterrupt):
        while True:
            asset = env.unwrapped.scene["robot"]
            # Determine gripper closure metric (handles UR5 and Franka variants)
            close_cmd_expr = env_cfg.actions.gripper.close_command_expr
            close_target = torch.tensor(list(close_cmd_expr.values())[0], device=env.device, dtype=torch.float32)
            # Try to locate Franka-style outer knuckle joints; fallback to UR5 finger joint
            knuckle_match = asset.find_joints([".*_outer_knuckle_joint"])
            if knuckle_match and len(knuckle_match[0]) > 0:
                joint_indices = knuckle_match[0]
                joint_positions = asset.data.joint_pos[:, joint_indices]
                gripper_metric = joint_positions.abs().mean(dim=1)
            else:
                finger_match = asset.find_joints(["finger_joint"])
                joint_index = finger_match[0][0]
                joint_positions = asset.data.joint_pos[:, joint_index]
                gripper_metric = joint_positions.abs()
            gripper_closed_fraction = gripper_metric / close_target.abs().clamp(min=1e-6)
            gripper_mask = gripper_closed_fraction > 0.1
            # Step the simulation
            for _ in range(5):
                action = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
                action[gripper_mask, -1] = -1.0
                action[~gripper_mask, -1] = 1.0
                env.step(action)
            for _ in range(5):
                env.unwrapped.sim.step()
            success = env.unwrapped.reward_manager.get_term_cfg("progress_context").func.success
            print("Success: ", success)

            # Wait for the specified interval
            time.sleep(args_cli.reset_interval)

            # Reset the environment to load a new state
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
