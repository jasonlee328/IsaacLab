from __future__ import annotations

import argparse
import gymnasium as gym

from .experiment_manager import ExperimentManagerBase, ExperimentManagerBaseCfg
from .rl_framework_factory import RlFrameworkFactory

class RlWorkflow:
    def __init__(self, simulation_app, algo_class, experiment_manager: ExperimentManagerBase, config_modifier=None, env_warpper=None):
        self.simulation_app = simulation_app
        self.algo = algo_class(experiment_manager)
        self.exp_mgr = experiment_manager
        self.config_modifier = config_modifier
        self.env_warpper = env_warpper
        self.hydra_args = None

    def add_args(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        self.exp_mgr.add_experiment_args(parser)
        args_cli, remaining = parser.parse_known_args()
        self.algo.add_algo_args(parser)
        args_cli, hydra_args = parser.parse_known_args()
        self.hydra_args = hydra_args
        return args_cli

    def make_config(self, args_cli: argparse.Namespace) -> None:
        self.exp_mgr.make_cfg(self.hydra_args)
        self.exp_mgr.update_env_cfg()
        self.algo.update_algo_cfg(args_cli)

    def launch(self) -> None:
        if self.exp_mgr.cfg.job_type == "train":
            self.train()
        elif self.exp_mgr.cfg.job_type == "eval":
            self.evaluate()
        elif self.exp_mgr.cfg.job_type == "play":
            self.play()
        else:
            raise ValueError(f"Invalid job type: {self.exp_mgr.cfg.job_type}")

    def train(self) -> None:
        if self.config_modifier:
            self.config_modifier(self.exp_mgr.env_cfg, self.exp_mgr.agent_cfg)
        self.algo.learn()
        self.simulation_app.close()

    def evaluate(self) -> None:
        if self.config_modifier:
            self.config_modifier(self.exp_mgr.env_cfg, self.exp_mgr.agent_cfg)
        self.algo.inference()
        self.simulation_app.close()

    def play(self) -> None:
        # Alias to evaluate
        self.evaluate()

def build_experiment_manager(args_cli: argparse.Namespace) -> ExperimentManagerBase:
    cfg = ExperimentManagerBaseCfg()
    exp_mgr = ExperimentManagerBase(cfg)
    return exp_mgr