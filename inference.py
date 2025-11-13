#!/usr/bin/env python3
# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.

"""Inference module for loading and running RSL-RL checkpoints."""

import torch
from rsl_rl.runners import OnPolicyRunner, DistillationRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path


class CheckpointInference:
    """Wrapper for running inference with a trained RSL-RL checkpoint."""
    
    def __init__(self, checkpoint_path: str, env, agent_cfg=None, device: str = "cuda:0"):
        """Initialize the inference module.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
            env: The Isaac Lab environment (will be wrapped for RSL-RL)
            agent_cfg: Agent configuration. If None, will use default configuration.
            device: Device to run inference on.
        """
        self.checkpoint_path = retrieve_file_path(checkpoint_path)
        self.device = device
        
        # Wrap environment for RSL-RL if not already wrapped
        if not isinstance(env, RslRlVecEnvWrapper):
            self.env = RslRlVecEnvWrapper(env, clip_actions=None)
        else:
            self.env = env
        
        # Create default agent config if not provided
        if agent_cfg is None:
            agent_cfg = self._create_default_config()
        
        # Determine runner type from config or use default
        runner_class = agent_cfg.get("class_name", "OnPolicyRunner")
        
        # Load the model
        print(f"[INFO]: Loading checkpoint from: {self.checkpoint_path}")
        if runner_class == "OnPolicyRunner":
            self.runner = OnPolicyRunner(self.env, agent_cfg, log_dir=None, device=self.device)
        elif runner_class == "DistillationRunner":
            self.runner = DistillationRunner(self.env, agent_cfg, log_dir=None, device=self.device)
        else:
            raise ValueError(f"Unsupported runner class: {runner_class}")
        
        self.runner.load(self.checkpoint_path)
        
        # Get inference policy
        self.policy = self.runner.get_inference_policy(device=self.env.unwrapped.device)
        print("[INFO]: Checkpoint loaded successfully!")
    
    def _create_default_config(self):
        """Create a minimal default configuration for the runner."""
        return {
            "class_name": "OnPolicyRunner",
            "seed": 42,
            "device": self.device,
            "num_steps_per_env": 24,
            "max_iterations": 0,  # We're not training
            "empirical_normalization": False,
            "save_interval": 50,
            "experiment_name": "inference",
            "run_name": "",
            "logger": "tensorboard",
            "resume": False,
            "load_run": -1,
            "load_checkpoint": -1,
            "log_interval": 1,
            "obs_groups": {"policy": ["policy"]},
            "clip_actions": None,
            "policy": {
                "class_name": "ActorCritic",
                "init_noise_std": 0.6,
                "actor_hidden_dims": [256, 256, 256],
                "critic_hidden_dims": [256, 256, 256],
                "activation": "tanh",
                "actor_obs_normalization": True,
                "critic_obs_normalization": True,
            },
            "algorithm": {
                "class_name": "PPO",
                "value_loss_coef": 0.5,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.006,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 3.0e-4,
                "schedule": "constant",
                "gamma": 0.99,
                "lam": 0.9,
                "desired_kl": 0.1,
                "max_grad_norm": 0.5,
                "normalize_advantage_per_mini_batch": True,
            },
        }
    
    def get_action(self, obs):
        """Get action from observation using the loaded checkpoint.
        
        Args:
            obs: Observation tensor from the environment
            
        Returns:
            Action tensor to apply to the environment
        """
        with torch.inference_mode():
            actions = self.policy(obs)
        return actions
    
    def __call__(self, obs):
        """Allow the class to be called directly as a function."""
        return self.get_action(obs)


def load_checkpoint(checkpoint_path: str, env, agent_cfg=None, device: str = "cuda:0"):
    """Convenience function to load a checkpoint for inference.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        env: The Isaac Lab environment
        agent_cfg: Agent configuration. If None, will use default configuration.
        device: Device to run inference on.
        
    Returns:
        CheckpointInference object that can be called to get actions
        
    Example:
        >>> policy = load_checkpoint("/path/to/model.pt", env)
        >>> obs = env.get_observations()
        >>> actions = policy(obs)
        >>> env.step(actions)
    """
    return CheckpointInference(checkpoint_path, env, agent_cfg, device)


