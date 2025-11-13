#!/usr/bin/env python3


import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

# Configuration
checkpoint_path = "/home/jason/IsaacLab/logs/rsl_rl/franka_robotiq_2f85/model_15700.pt"
obs_dim = 48
device = "cuda:0"

print(f"Loading checkpoint: {checkpoint_path}")

# Minimal dummy environment
class DummyEnv:
    num_envs = 1
    num_actions = 8  
    device = device
    
    def get_observations(self):
        return TensorDict({"policy": torch.zeros(self.num_envs, obs_dim, device=device)}, batch_size=[1])

# Minimal config
config = {
    "class_name": "OnPolicyRunner",
    "seed": 42,
    "device": device,
    "num_steps_per_env": 24,
    "max_iterations": 0,
    "empirical_normalization": False,
    "save_interval": 50,
    "experiment_name": "inference",
    "run_name": "",
    "logger": "tensorboard",
    "resume": False,
    "load_run": -1,
    "load_checkpoint": -1,
    "log_interval": 1,
    "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    "clip_actions": None,
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 0.6,
        "actor_hidden_dims": [256, 256, 256],
        "critic_hidden_dims": [256, 256, 256],
        "activation": "elu",
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


runner = OnPolicyRunner(DummyEnv(), config, log_dir=None, device=device)
runner.load(checkpoint_path)
policy = runner.get_inference_policy(device=device)

print("Loaded successfully! Running inference loop (Ctrl+C to stop)...")
print()


step = 0
while True:
    obs = TensorDict({"policy": torch.randn(DummyEnv.num_envs, obs_dim, device=device)}, batch_size=[1])
    actions = policy(obs)
    print(f"Step {step}: {actions[0].detach().cpu().numpy()}")
    step += 1
