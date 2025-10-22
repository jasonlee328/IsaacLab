#!/bin/bash
# Example training runs with different hyperparameter configurations
# These demonstrate how to override parameters from the command line

# ============================================================================
# EXAMPLE 1: Baseline run
# ============================================================================
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 8192 \
  agent.run_name=baseline

# ============================================================================
# EXAMPLE 2: Tune learning rate
# ============================================================================
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 8192 \
  agent.algorithm.learning_rate=1e-4 \
  agent.run_name=lr_1e-4

# ============================================================================
# EXAMPLE 3: Cube closer to robot
# ============================================================================
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 8192 \
  env.events.randomize_cube_position.params.pose_range.x=[0.3,0.3] \
  env.events.randomize_cube_position.params.pose_range.y=[0.0,0.0] \
  agent.run_name=cube_close

# ============================================================================
# EXAMPLE 4: Target position range adjustment
# ============================================================================
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 8192 \
  env.commands.ee_pose.ranges.pos_x=[0.15,0.35] \
  env.commands.ee_pose.ranges.pos_y=[-0.15,0.15] \
  agent.run_name=target_closer

# ============================================================================
# EXAMPLE 5: Different robot starting pose
# ============================================================================
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 8192 \
  'env.events.init_franka_arm_pose.params.default_pose=[0.0,-0.3,0.0,-2.0,0.0,2.7,0.0,0.04,0.04]' \
  agent.run_name=robot_alt_pose

# ============================================================================
# EXAMPLE 6: Larger network with more exploration
# ============================================================================
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 8192 \
  agent.policy.actor_hidden_dims=[512,512,512] \
  agent.policy.critic_hidden_dims=[512,512,512] \
  agent.policy.init_noise_std=0.8 \
  agent.algorithm.entropy_coef=0.01 \
  agent.run_name=large_net_entropy

# ============================================================================
# EXAMPLE 7: Easier goal threshold
# ============================================================================
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 8192 \
  env.rewards.reaching_goal.params.threshold=0.08 \
  agent.run_name=easier_goal

# ============================================================================
# EXAMPLE 8: Full custom configuration
# ============================================================================
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 8192 \
  --max_iterations 15000 \
  --seed 42 \
  agent.algorithm.learning_rate=2e-4 \
  agent.algorithm.gamma=0.98 \
  agent.algorithm.lam=0.95 \
  agent.algorithm.num_learning_epochs=6 \
  agent.algorithm.num_mini_batches=128 \
  agent.policy.actor_hidden_dims=[256,256,256] \
  agent.policy.critic_hidden_dims=[256,256,256] \
  agent.num_steps_per_env=150 \
  env.events.randomize_cube_position.params.pose_range.x=[0.35,0.35] \
  env.events.randomize_cube_position.params.pose_range.y=[0.0,0.0] \
  env.commands.ee_pose.ranges.pos_x=[0.2,0.4] \
  env.commands.ee_pose.ranges.pos_y=[-0.1,0.1] \
  env.rewards.reaching_goal.params.threshold=0.04 \
  agent.run_name=custom_full_config

# ============================================================================
# EXAMPLE 9: Quick test run (fewer envs and iterations)
# ============================================================================
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 512 \
  --max_iterations 100 \
  agent.run_name=quick_test

# ============================================================================
# EXAMPLE 10: Multiple parameter sweep (run sequentially)
# ============================================================================
# Learning rate sweep
for lr in 5e-4 3e-4 1e-4; do
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Push-Cube-Franka-Easy-v0 \
    --headless \
    --logger wandb \
    --log_project_name isaac-lab-push \
    --num_envs 8192 \
    agent.algorithm.learning_rate=$lr \
    agent.run_name=lr_sweep_${lr}
done

