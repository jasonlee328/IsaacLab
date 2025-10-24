# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaPushCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO training of Franka Push Cube task.
    
    Matches CleanRL-style hyperparameters:
    - num_steps: 50 (your config)
    - num_envs: 512 (will be set via --num_envs CLI arg)
    - total_timesteps: 10M
    - gamma: 0.8
    - gae_lambda: 0.9
    - num_minibatches: 32
    - update_epochs: 4
    """
    
    # Training settings
    num_steps_per_env = 100  # num_steps: steps per env before policy update
    max_iterations = 20000  # ~10M timesteps with 512 envs (512*50*20000 = 512M, adjust via CLI)
    save_interval = 20
    
    # Experiment naming
    experiment_name = "franka_push_cube"
    run_name = "oct17-pushcube-er003-obs-debug"
    
    # Logging
    logger = "wandb"  # Options: "tensorboard", "wandb", "neptune", None
    wandb_project = "Isaac-RL"  # Default wandb project name
    log_interval = 10  # Log every 10 iterations
    
    # Policy network configuration
    # Architecture: 3 layers of 256 units with Tanh activation
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,  # Corresponds to actor_logstd = -0.5
        actor_obs_normalization=False,  # Disabled to match your architecture
        critic_obs_normalization=False,  # Disabled to match your architecture
        actor_hidden_dims=[256, 256, 256],  # 3 layers of 256 units
        critic_hidden_dims=[256, 256, 256],  # 3 layers of 256 units
        activation="tanh",  # Tanh activation
    )
    
    # PPO algorithm configuration (matching your CleanRL config)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,  # vf_coef: 0.5
        use_clipped_value_loss=False,  # clip_vloss: False
        clip_param=0.2,  # clip_coef: 0.2
        entropy_coef=0.006, # ent_coef: 0.0 (no entropy bonus)
        num_learning_epochs=5,  # update_epochs: 4
        num_mini_batches=4,  # num_minibatches: 32
        learning_rate=3.0e-4,  # learning_rate: 3e-4
        schedule="constant",  # annealing handled by schedule
        gamma=0.99,  # gamma: 0.8
        lam=0.9,  # gae_lambda: 0.9
        desired_kl=0.1,  # target_kl: 0.1
        max_grad_norm=0.5,  # max_grad_norm: 0.5
        normalize_advantage_per_mini_batch=True,
    )



