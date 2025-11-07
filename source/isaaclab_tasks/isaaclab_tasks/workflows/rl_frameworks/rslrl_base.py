from __future__ import annotations

import os
import gymnasium as gym

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner, DistillationRunner  # both supported
from isaaclab.utils.dict import print_dict
import torch
import time

from ..rl_base import RlBase

class RslRlBase(RlBase):
    def learn(self) -> None:
        env_cfg = self.exp_mgr.env_cfg
        agent_cfg = self.exp_mgr.agent_cfg
        log_dir = self.exp_mgr.run_dir

        # Build env
        render_mode = "rgb_array" if self.exp_mgr.cfg.video else None
        env = gym.make(self.exp_mgr.cfg.task, cfg=env_cfg, render_mode=render_mode)
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None) if not isinstance(agent_cfg, dict) else agent_cfg.get("clip_actions"))

        # Runner
        runner = OnPolicyRunner(env, agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg, log_dir=log_dir, device=getattr(agent_cfg, "device", "cuda:0"))
        runner.add_git_repo_to_log(__file__)

        # Resume/checkpoint (optional)
        ckpt = self.exp_mgr.resolve_checkpoint_path()
        if self.exp_mgr.cfg.resume and ckpt:
            runner.load(ckpt)

        # Dump cfgs
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        self.exp_mgr.dump_cfg(env_cfg, agent_cfg)

        # Learn
        max_iters = getattr(agent_cfg, "max_iterations", None) if not isinstance(agent_cfg, dict) else agent_cfg.get("max_iterations")
        if max_iters is None:
            max_iters = 0 if self.exp_mgr.cfg.job_type == "eval" else 40000
        runner.learn(num_learning_iterations=max_iters, init_at_random_ep_len=True)

        env.close()

    def inference(self) -> None:
        env_cfg = self.exp_mgr.env_cfg
        agent_cfg = self.exp_mgr.agent_cfg
        log_dir = self.exp_mgr.run_dir

        # Build env
        render_mode = "rgb_array" if self.exp_mgr.cfg.video else None
        env = gym.make(self.exp_mgr.cfg.task, cfg=env_cfg, render_mode=render_mode)
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None) if not isinstance(agent_cfg, dict) else agent_cfg.get("clip_actions"))

        # Runner
        runner = OnPolicyRunner(env, agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg, log_dir=log_dir, device=getattr(agent_cfg, "device", "cuda:0"))
        runner.add_git_repo_to_log(__file__)

        # Require checkpoint for eval
        ckpt = self.exp_mgr.resolve_checkpoint_path()
        if not ckpt:
            raise FileNotFoundError("No checkpoint provided for eval. Use --checkpoint or --checkpoint_dir.")
        runner.load(ckpt)

        # Dump cfgs for completeness
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        self.exp_mgr.dump_cfg(env_cfg, agent_cfg)

        # Obtain inference policy (matches IsaacLab play.py)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Eval loop: roll out for a fixed horizon (use video_length if provided)
        dt = getattr(env.unwrapped, "step_dt", 0.0)
        steps = self.exp_mgr.cfg.video_length if self.exp_mgr.cfg.video else 1000

        # Reset and step
        obs = env.get_observations()
        for _ in range(steps):
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                # Some policies may return a tuple; take the first element
                if isinstance(actions, tuple):
                    actions = actions[0]
                # Ensure actions are 2D: (num_envs, action_dim)
                if actions.ndim == 1:
                    actions = actions.unsqueeze(0)
                actions = actions.to(env.unwrapped.device).float()
                obs, _, _, _ = env.step(actions)
            if self.exp_mgr.cfg.video and dt > 0:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        env.close()