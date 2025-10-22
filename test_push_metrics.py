#!/usr/bin/env python3
"""
Test script to demonstrate the new push environment with success metrics.

This script shows how the success metrics are logged to wandb during training.
The metrics will appear in wandb under the "Metrics/" prefix.
"""

import torch
from isaaclab_tasks.manager_based.manipulation.push import PushEnvWithMetrics
from isaaclab_tasks.manager_based.manipulation.push.config.franka.push_joint_pos_env_cfg import FrankaPushCubeEnvCfg


def test_push_environment_with_metrics():
    """Test the push environment with success metrics."""
    
    # Create environment configuration
    cfg = FrankaPushCubeEnvCfg()
    cfg.scene.num_envs = 4  # Small number for testing
    cfg.episode_length_s = 5.0  # Short episodes for testing
    
    # Create environment
    env = PushEnvWithMetrics(cfg)
    
    print("Testing Push Environment with Success Metrics")
    print("=" * 50)
    
    # Reset environment
    obs, _ = env.reset()
    print(f"Environment reset. Number of environments: {env.num_envs}")
    
    # Run a few episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        print("-" * 20)
        
        # Reset environment
        obs, _ = env.reset()
        
        # Run episode
        for step in range(50):  # 50 steps should be enough for short episodes
            # Random actions
            actions = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device)
            
            # Step environment
            obs, rewards, terminated, truncated, extras = env.step(actions)
            
            # Print step info
            if step % 10 == 0:
                print(f"  Step {step}: Rewards = {rewards.mean().item():.3f}")
                
                # Check if any environments are at goal
                curr_successes = env._get_curr_successes()
                success_count = curr_successes.sum().item()
                if success_count > 0:
                    print(f"    {success_count} environments at goal (sparse reward: +1)")
            
            # Check if episode ended
            if terminated.any() or truncated.any():
                print(f"  Episode ended at step {step}")
                
                # Print success metrics from extras (these will also be logged to wandb)
                if "log" in extras and "episode_success_rate" in extras["log"]:
                    episode_success = extras["log"]["episode_success_rate"]
                    overall_success = extras["log"]["overall_success_rate"]
                    avg_success_time = extras["log"]["avg_success_time"]
                    
                    print(f"    Episode Success Rate: {episode_success:.2f}")
                    print(f"    Overall Success Rate: {overall_success:.2f}")
                    print(f"    Avg Success Time: {avg_success_time:.1f} steps")
                    print("    (These metrics are also logged to wandb under 'Metrics/' prefix)")
                break
    
    # Close environment
    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_push_environment_with_metrics()
