# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--sync_task", type=str, required=False, help="The task to be executed in sync mode.")
"""Launch Isaac Sim Simulator first."""
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, remaining_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from gymnasium import Env

import isaaclab_tasks  # noqa: F401
from octilab_apps.apps.experiment_workflows import ExperimentManagerCfgFactory

import octilab_tasks  # noqa: F401


def synchronize_wrap(env: Env) -> Env:
    from isaaclab_tasks.utils import parse_env_cfg
    from octilab_apps.utils.wrappers.synchronize import SyncWrapperCfg

    sync_task = args_cli.sync_task
    sync_env_cfg = parse_env_cfg(sync_task)
    sync_env = gym.make(sync_task, cfg=sync_env_cfg)
    wrapper_cfg = SyncWrapperCfg(sync_mode="follow")
    sync_env = wrapper_cfg.class_type(wrapper_cfg, leader=env, follower=sync_env)
    return sync_env


def main(args_cli: argparse.Namespace):
    """Running keyboard teleoperation with Orbit manipulation environment."""
    # parse configuration
    experiment_manager_cfg = ExperimentManagerCfgFactory("solo").create()
    if args_cli.sync_task is not None:
        experiment_manager_cfg.custom_wrapper = synchronize_wrap
    exp_mgr = experiment_manager_cfg.class_type(experiment_manager_cfg)

    exp_mgr.add_experiment_args(parser)
    args_cli, remaining_args = parser.parse_known_args()
    args_cli, hydra_args = parser.parse_known_args()
    exp_mgr.update_experiment_cfg(args_cli, "teleop", "teleop_cfg_entry_point")

    exp_mgr.make_cfg(hydra_args)
    exp_mgr.update_env_cfg()
    exp_mgr.env_cfg.terminations.time_out = None  # type: ignore

    exp_mgr.dump_cfg(exp_mgr.env_cfg, exp_mgr.agent_cfg)

    sim_env = gym.make(args_cli.task, cfg=exp_mgr.env_cfg)

    if not hasattr(exp_mgr.agent_cfg, args_cli.teleop_device):
        raise ValueError(
            f"Teleoperation device {args_cli.teleop_device} not found in agent configuration.\n"
            f"Please choose from {list(exp_mgr.agent_cfg.__dict__.keys())}"
        )
    teleop_cfg = getattr(exp_mgr.agent_cfg, args_cli.teleop_device)
    teleop = teleop_cfg.class_type(teleop_cfg, sim_env)

    env = exp_mgr.wrap(env=sim_env)

    def reset():
        env.reset()
        teleop.reset()

    reset()
    # create controller
    teleop.add_callback("L", reset)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command in WORLD coordinates
            command_b = teleop.advance()
            env.step(command_b)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main(args_cli)
    # close sim app
    simulation_app.close()
