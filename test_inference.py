#!/usr/bin/env python3


import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

# Configuration
checkpoint_path = "/home/jason/IsaacLab/logs/rsl_rl/franka_robotiq_2f85/model_1700.pt"
obs_dim = 45
device = "cuda:0"

print(f"Loading checkpoint: {checkpoint_path}")

# Minimal dummy environment
class DummyEnv:
    num_envs = 1
    num_actions = 8  
    device = device
    
    def get_observations(self):
        # return TensorDict({"policy": torch.zeros(self.num_envs, obs_dim, device=device)}, batch_size=[1])
        return TensorDict({'policy': torch.tensor([[ 4.0694e-03,  1.3767e-03,  9.4668e-04, -4.1459e-03,  1.8414e-02,
          2.2905e-03, -4.8596e-03,  6.4868e-02,  2.8641e-02,  2.6248e-02,
         -6.7317e-02,  1.9198e-01,  1.6404e-02, -4.8287e-02,  0.0000e+00,
          6.1471e-01,  4.0290e-03,  6.6459e-02, -5.2168e-03,  9.9943e-01,
         -2.8685e-03,  3.3375e-02,  6.2402e-01,  4.4779e-07,  2.0296e-02,
          1.6469e-01,  5.7556e-01,  7.4428e-03,  2.0300e-02,  1.5595e+00,
          1.3948e+00, -6.8958e-03, -4.8546e-02, -4.2971e-06, -1.1635e-06,
         -3.1681e-07, -1.3948e+00,  5.7506e-01, -3.8721e-02,  2.0296e-02,
          1.5624e+00,  5.7556e-01, -3.9557e-02,  2.0300e-02,  1.5595e+00]],
       device='cuda:0')}, batch_size=[1])

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
    break
