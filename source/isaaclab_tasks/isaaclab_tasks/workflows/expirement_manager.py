from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

from isaaclab_tasks.utils.hydra import (
    register_task_to_hydra_programmatic,
    replace_strings_with_slices,
)
from hydra import compose, initialize
from omegaconf import OmegaConf
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.spaces import replace_strings_with_env_cfg_spaces
from isaaclab.utils.io import dump_yaml, dump_pickle

@dataclass
class ExperimentManagerBaseCfg:
    job_type: str = "train"                      # train | eval | play
    task: str | None = None
    num_envs: int | None = None
    resume: bool = False
    checkpoint: str | None = None
    checkpoint_dir: str | None = None
    video: bool = False
    video_length: int = 600
    logger: str | None = None
    log_project_name: str | None = None

class ExperimentManagerBase:
    def __init__(self, cfg: ExperimentManagerBaseCfg):
        self.cfg = cfg
        self.env_cfg = None
        self.agent_cfg = None
        self.log_root_path = None
        self.run_dir = None

    # CLI wiring (OctiLab-like)
    @staticmethod
    def add_experiment_args(parser: argparse.ArgumentParser) -> None:
        arg_group = parser.add_argument_group("experiment", description="Experiment workflow args.")
        arg_group.add_argument("--job_type", type=str, default="train", choices={"train", "eval", "play"})
        arg_group.add_argument("--task", type=str, default=None)
        arg_group.add_argument("--num_envs", type=int, default=None)
        arg_group.add_argument("--resume", action="store_true", default=False)
        arg_group.add_argument("--checkpoint", type=str, default=None)
        arg_group.add_argument("--checkpoint_dir", type=str, default=None)
        arg_group.add_argument("--video", action="store_true", default=False)
        arg_group.add_argument("--video_length", type=int, default=600)
        arg_group.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
        arg_group.add_argument("--log_project_name", type=str, default=None)

    def update_experiment_cfg(self, args_cli: argparse.Namespace, rl_framework: str, agent_cfg_entry_point: str) -> None:
        self.cfg.job_type = args_cli.job_type
        self.cfg.task = args_cli.task
        self.cfg.num_envs = args_cli.num_envs
        self.cfg.resume = args_cli.resume
        self.cfg.checkpoint = args_cli.checkpoint
        self.cfg.checkpoint_dir = args_cli.checkpoint_dir
        self.cfg.video = args_cli.video if hasattr(args_cli, "video") else False
        self.cfg.video_length = args_cli.video_length if hasattr(args_cli, "video_length") else 600
        self.cfg.logger = args_cli.logger
        self.cfg.log_project_name = args_cli.log_project_name

        self._rl_framework = rl_framework
        self._agent_cfg_entry_point = agent_cfg_entry_point

    # Hydra programmatic config (variants supported)
    def make_cfg(self, hydra_args: list[str]) -> None:
        env_cfg, agent_cfg = register_task_to_hydra_programmatic(self.cfg.task.split(":")[-1], self._agent_cfg_entry_point)
        with initialize(config_path=None, version_base="1.3"):
            hydra_cfg = compose(config_name=self.cfg.task, overrides=hydra_args)
        hydra_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)
        hydra_cfg = replace_strings_with_slices(hydra_cfg)
        # env
        env_cfg.from_dict(hydra_cfg["env"])
        env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
        # agent
        if agent_cfg is None or isinstance(agent_cfg, dict):
            agent_cfg = hydra_cfg["agent"]
        else:
            agent_cfg.from_dict(hydra_cfg["agent"])
        # attach
        self.env_cfg, self.agent_cfg = env_cfg, agent_cfg

    def update_env_cfg(self) -> None:
        # Set num_envs and logging roots
        if self.cfg.num_envs is not None:
            self.env_cfg.scene.num_envs = self.cfg.num_envs
        exp_name = getattr(self.agent_cfg, "experiment_name", "experiment") if not isinstance(self.agent_cfg, dict) else self.agent_cfg.get("experiment_name", "experiment")
        self.log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", exp_name))
        run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = getattr(self.agent_cfg, "run_name", "") if not isinstance(self.agent_cfg, dict) else self.agent_cfg.get("run_name", "")
        if run_name:
            run_dir += f"_{run_name}"
        self.run_dir = os.path.join(self.log_root_path, run_dir)
        self.env_cfg.log_dir = self.run_dir  # consistent with IsaacLab

    def dump_cfg(self, env_cfg, agent_cfg) -> None:
        dump_yaml(os.path.join(self.run_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(self.run_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(self.run_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(self.run_dir, "params", "agent.pkl"), agent_cfg)

    # Resolve checkpoint: explicit file > latest in dir
    def resolve_checkpoint_path(self) -> str | None:
        if self.cfg.checkpoint:
            return os.path.abspath(self.cfg.checkpoint)
        if self.cfg.checkpoint_dir:
            pattern = os.path.join(os.path.abspath(self.cfg.checkpoint_dir), "model_*.pt")
            cks = sorted(glob.glob(pattern))
            return cks[-1] if cks else None
        return None