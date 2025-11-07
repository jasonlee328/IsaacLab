# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

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
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import random

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from octilab_apps.apps.application_workflows import RlWorkflow
from octilab_apps.apps.experiment_workflows import ExperimentManagerCfgFactory

from octilab.genes import GenomeCfg

from octilab_rl.rl_framework_factory import RlFrameworkFactory

import octilab_tasks  # noqa: F401


def main(args_cli: argparse.Namespace):
    experiment_manager_cfg = ExperimentManagerCfgFactory(args_cli.experiment_workflow).create()
    experiment_manager = experiment_manager_cfg.class_type(experiment_manager_cfg)
    rl_framework = RlFrameworkFactory(args_cli.rl_framework).create()

    def cfg_modifier(env_cfg, agent_cfg):
        genome_cfg: GenomeCfg = load_cfg_from_registry(args_cli.task, "genome_entry_point")  # type: ignore
        # it should be noted that the seed here only randomize the behavior of genome mutation, but does not affect
        # the seed of the simulation and training.
        genome_cfg.seed = random.randint(0, 10000)
        genome = genome_cfg.class_type(genome_cfg)
        genome.activate(env_cfg, agent_cfg)
        genome.gene_initialize()
        genome.mutate()

    rl_workflow = RlWorkflow(
        simulation_app=simulation_app,
        algo_class=rl_framework,
        experiment_manager=experiment_manager,
        config_modifier=cfg_modifier,
    )

    args_cli = rl_workflow.add_args(parser)

    rl_workflow.make_config(args_cli)

    rl_workflow.launch()


if __name__ == "__main__":
    main(args_cli)
