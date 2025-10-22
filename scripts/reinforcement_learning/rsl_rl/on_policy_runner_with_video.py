# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended OnPolicyRunner that records videos at each checkpoint save."""

import os
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner


class OnPolicyRunnerWithVideo(OnPolicyRunner):
    """Extended OnPolicyRunner that automatically records videos when saving checkpoints."""
    
    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        video_length: int = 200,
        record_video: bool = True,
    ):
        """
        Initialize the runner with video recording capability.
        
        Args:
            env: The environment
            train_cfg: Training configuration dictionary
            log_dir: Directory for logging
            device: Device to run on
            video_length: Length of video to record (in steps)
            record_video: Whether to record videos at checkpoints
        """
        super().__init__(env, train_cfg, log_dir, device)
        self.video_length = video_length
        self.record_video = record_video
        self._video_env = None
        
    def save(self, path: str, infos=None):
        """
        Save checkpoint and record a video rollout.
        
        Args:
            path: Path to save the checkpoint
            infos: Additional information to save
        """
        # Call parent save method
        super().save(path, infos)
        
        # Record video if enabled
        if self.record_video and self.log_dir is not None:
            self._record_video(path)
    
    def _record_video(self, checkpoint_path: str):
        """
        Record a video rollout for the checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_filename = os.path.basename(checkpoint_path)
        checkpoint_name = os.path.splitext(checkpoint_filename)[0]
        
        video_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.mp4")
        
        print(f"[INFO] Recording video for {checkpoint_name}...")
        
        # Switch to eval mode
        self.eval_mode()
        
        try:
            # Create temporary video directory
            temp_video_dir = os.path.join(checkpoint_dir, f"temp_video_{checkpoint_name}")
            os.makedirs(temp_video_dir, exist_ok=True)
            
            # Wrap environment with video recorder
            video_kwargs = {
                "video_folder": temp_video_dir,
                "step_trigger": lambda step: step == 0,
                "video_length": self.video_length,
                "disable_logger": True,
            }
            
            # Get the underlying IsaacLab environment
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            if isinstance(self.env, RslRlVecEnvWrapper):
                isaac_env = self.env.env
            else:
                isaac_env = self.env
            
            # Wrap with RecordVideo
            video_env = gym.wrappers.RecordVideo(isaac_env, **video_kwargs)
            
            # Rewrap with RslRlVecEnvWrapper
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            video_env_wrapped = RslRlVecEnvWrapper(video_env)
            
            # Run rollout
            obs = video_env_wrapped.get_observations().to(self.device)
            
            with torch.inference_mode():
                for step in range(self.video_length):
                    # Get action from policy
                    actions = self.alg.act(obs)
                    
                    # Step environment
                    obs, rewards, dones, extras = video_env_wrapped.step(actions.to(video_env_wrapped.device))
                    obs = obs.to(self.device)
                    
                    # Break if first environment is done
                    if dones[0]:
                        break
            
            # Close video environment
            video_env.close()
            
            # Move the video file to final location
            import shutil
            video_files = [f for f in os.listdir(temp_video_dir) if f.endswith('.mp4')]
            if video_files:
                temp_video_path = os.path.join(temp_video_dir, video_files[0])
                shutil.move(temp_video_path, video_path)
                print(f"[INFO] Video saved to: {video_path}")
            else:
                print(f"[WARNING] No video file generated for {checkpoint_name}")
            
            # Clean up
            shutil.rmtree(temp_video_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"[WARNING] Failed to record video for {checkpoint_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Switch back to train mode
            self.train_mode()


