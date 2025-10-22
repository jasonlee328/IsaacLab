#!/usr/bin/env python3
"""
Example showing how to use the push environment with success metrics for wandb logging.

This shows how to run training with the new environment that logs success metrics to wandb.
"""

# Example command to run training with success metrics:
"""
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 2048 \
  agent.run_name=push_with_success_metrics
"""

# The success metrics will automatically appear in wandb under:
# - Metrics/episode_success_rate: Percentage of environments that maintained success at episode end
# - Metrics/overall_success_rate: Percentage of environments that achieved success at least once
# - Metrics/avg_success_time: Average time (in steps) when success was first achieved

print("To use the push environment with success metrics:")
print("1. Use PushEnvWithMetrics instead of the regular push environment")
print("2. Run training with --logger wandb")
print("3. Check wandb dashboard for success metrics under 'Metrics/' section")
print("\nExample command:")
print("./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\")
print("  --task Isaac-Push-Cube-Franka-Easy-v0 \\")
print("  --headless \\")
print("  --logger wandb \\")
print("  --log_project_name isaac-lab-push \\")
print("  --num_envs 2048 \\")
print("  agent.run_name=push_with_success_metrics")
