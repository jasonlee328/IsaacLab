#!/usr/bin/env python3
"""Quick test to verify ee_object_distance reward works."""

import argparse
from isaaclab.app import AppLauncher

# Create launcher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

def test_ee_reward():
    """Test if ee_frame exists and reward function works."""
    
    print("\n" + "="*60)
    print("Testing EE-to-Cube Distance Reward")
    print("="*60)
    
    # Create environment
    print("\n1. Creating environment...")
    from isaaclab_tasks.manager_based.manipulation.push.config.franka import FrankaPushCubeEasyEnvCfg
    env_cfg = FrankaPushCubeEasyEnvCfg()
    env_cfg.scene.num_envs = 4
    env = gym.make("Isaac-Push-Cube-Franka-Easy-v0", cfg=env_cfg)
    
    # Check scene entities
    print("\n2. Checking scene entities...")
    scene_entities = list(env.unwrapped.scene.keys())
    print(f"   Available entities: {scene_entities}")
    
    # Check if ee_frame exists
    if "ee_frame" in scene_entities:
        print("   ✓ ee_frame found in scene")
        ee_frame = env.unwrapped.scene["ee_frame"]
        print(f"   ✓ ee_frame type: {type(ee_frame).__name__}")
    else:
        print("   ✗ ERROR: ee_frame NOT found in scene!")
        env.close()
        return False
    
    # Check if cube exists
    if "cube" in scene_entities:
        print("   ✓ cube found in scene")
    else:
        print("   ✗ ERROR: cube NOT found in scene!")
        env.close()
        return False
    
    # Reset environment
    print("\n3. Resetting environment...")
    obs, _ = env.reset()
    print(f"   ✓ Environment reset successful")
    
    # Try to access ee_frame data
    print("\n4. Accessing ee_frame data...")
    try:
        ee_pos = ee_frame.data.target_pos_w[..., 0, :]
        print(f"   ✓ EE position shape: {ee_pos.shape}")
        print(f"   ✓ EE position (env 0): {ee_pos[0].cpu().numpy()}")
    except Exception as e:
        print(f"   ✗ ERROR accessing ee_frame data: {e}")
        env.close()
        return False
    
    # Try to compute the reward
    print("\n5. Testing ee_object_distance reward...")
    try:
        from isaaclab_tasks.manager_based.manipulation.push import mdp as push_mdp
        from isaaclab.managers import SceneEntityCfg
        
        reward = push_mdp.ee_object_distance(
            env=env.unwrapped,
            std=0.1,
            object_cfg=SceneEntityCfg("cube"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
        )
        
        print(f"   ✓ Reward computed successfully!")
        print(f"   ✓ Reward shape: {reward.shape}")
        print(f"   ✓ Reward values (first 4 envs): {reward[:4].cpu().numpy()}")
        print(f"   ✓ Reward range: [{reward.min().item():.4f}, {reward.max().item():.4f}]")
        
    except Exception as e:
        print(f"   ✗ ERROR computing reward: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False
    
    # Step environment and test reward again
    print("\n6. Testing reward after environment step...")
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    obs, reward_dict, terminated, truncated, info = env.step(action)
    
    # Check if distance_reward exists in reward manager
    if hasattr(env.unwrapped, 'reward_manager'):
        print(f"   ✓ Reward manager exists")
        reward_terms = list(env.unwrapped.reward_manager._term_names)
        print(f"   ✓ Reward terms: {reward_terms}")
        
        if "distance_reward" in reward_terms:
            print(f"   ✓ distance_reward is active")
        else:
            print(f"   ⚠ distance_reward not found in active terms")
    
    env.close()
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    try:
        success = test_ee_reward()
        simulation_app.close()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        exit(1)

