# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL Agent.")
parser.add_argument(
    "--experiment_workflow",
    type=str,
    default="solo",
    choices=["solo", "orchestrated"],
    help="experiment type, solo, orchestrated.",
)

parser.add_argument(
    "--rl_framework",
    type=str,
    default="skrl",
    choices=["skrl", "rslrl"],
    help="RL framework, skrl, rsl_rl, stable_baselines3, etc.",
)

parser.add_argument("--sync_task", type=str, required=True, help="The task to be executed in sync mode.")

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from octilab_apps.apps.application_workflows import RlWorkflow
from octilab_apps.apps.experiment_workflows import ExperimentManagerCfgFactory
from octilab_apps.utils.wrappers.synchronize import SyncWrapperCfg

from octilab_rl.rl_framework_factory import RlFrameworkFactory

import octilab_tasks  # noqa: F401


def main(args_cli: argparse.Namespace):
    experiment_manager_cfg = ExperimentManagerCfgFactory(args_cli.experiment_workflow).create()

    def synchronize_wrap(env):
        sync_task = args_cli.sync_task
        sync_env_cfg = parse_env_cfg(sync_task)
        sync_env = gym.make(sync_task, cfg=sync_env_cfg)
        wrapper_cfg = SyncWrapperCfg(sync_mode="follow")
        sync_env = wrapper_cfg.class_type(wrapper_cfg, leader=env, follower=sync_env)
        return sync_env

    experiment_manager_cfg.custom_wrapper = synchronize_wrap
    experiment_manager = experiment_manager_cfg.class_type(experiment_manager_cfg)
    rl_framework = RlFrameworkFactory(args_cli.rl_framework).create()

    rl_workflow = RlWorkflow(
        simulation_app=simulation_app,
        algo_class=rl_framework,
        experiment_manager=experiment_manager,
    )

    args_cli = rl_workflow.add_args(parser)

    rl_workflow.make_config(args_cli)

    rl_workflow.evaluate()


if __name__ == "__main__":
    main(args_cli)
