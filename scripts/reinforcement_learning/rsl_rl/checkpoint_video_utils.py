# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for recording videos at each checkpoint during RSL-RL training."""

import gymnasium as gym
import os
import shutil
import subprocess
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.runners import OnPolicyRunner


def record_checkpoint_video_async(
    task_name: str,
    checkpoint_path: str,
    video_length: int = 200,
    headless: bool = True,
) -> None:
    """
    Record a video rollout for a checkpoint by launching a separate play.py process.
    This runs asynchronously and doesn't block training.
    
    Args:
        task_name: Name of the task (e.g., "Isaac-Push-Cube-Franka-Easy-v0")
        checkpoint_path: Path to the checkpoint file (e.g., model_100.pt)
        video_length: Number of steps to record in the video
        headless: Whether to run in headless mode
    """
    # Extract checkpoint iteration and base path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_filename = os.path.basename(checkpoint_path)
    checkpoint_name = os.path.splitext(checkpoint_filename)[0]  # e.g., "model_100"
    
    # Temporary video folder
    temp_video_folder = os.path.join(checkpoint_dir, "videos", f"checkpoint_{checkpoint_name}")
    
    # Build the command to run play.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    play_script = os.path.join(script_dir, "play.py")
    isaaclab_sh = os.path.join(os.path.dirname(script_dir), "..", "..", "isaaclab.sh")
    
    cmd = [
        isaaclab_sh,
        "-p",
        play_script,
        "--task", task_name,
        "--checkpoint", checkpoint_path,
        "--video",
        "--video_length", str(video_length),
        "--num_envs", "1",  # Use single environment for video
    ]
    
    if headless:
        cmd.append("--headless")
    
    # Run the command in the background
    print(f"[INFO] Starting video recording for checkpoint: {checkpoint_name}")
    try:
        # Run asynchronously (fire and forget)
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent process
        )
    except Exception as e:
        print(f"[WARNING] Failed to start video recording: {e}")


def record_checkpoint_video_sync(
    runner: "OnPolicyRunner",
    checkpoint_path: str,
    video_length: int = 200,
) -> None:
    """
    Record a video rollout for a checkpoint synchronously using the existing environment.
    This temporarily wraps the environment with video recording.
    
    Note: This approach may have issues with the Isaac Sim environment.
    Consider using record_checkpoint_video_postprocess instead.
    
    Args:
        runner: The RSL-RL OnPolicyRunner instance
        checkpoint_path: Path to the checkpoint file (e.g., model_100.pt)
        video_length: Number of steps to record in the video
    """
    # Extract checkpoint iteration and base path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_filename = os.path.basename(checkpoint_path)
    checkpoint_name = os.path.splitext(checkpoint_filename)[0]  # e.g., "model_100"
    
    # Set up temporary video directory
    temp_video_dir = os.path.join(checkpoint_dir, "temp_video")
    os.makedirs(temp_video_dir, exist_ok=True)
    
    # Switch to eval mode
    runner.eval_mode()
    
    print(f"[INFO] Recording video for checkpoint: {checkpoint_name}")
    
    try:
        # Get observations
        obs = runner.env.get_observations().to(runner.device)
        
        # Collect frames manually
        frames = []
        
        with torch.inference_mode():
            for step in range(video_length):
                # Render frame if environment supports it
                if hasattr(runner.env, 'render'):
                    frame = runner.env.render()
                    if frame is not None:
                        frames.append(frame)
                
                # Get action from policy
                actions = runner.alg.act(obs)
                
                # Step environment
                obs, _, dones, _ = runner.env.step(actions.to(runner.env.device))
                obs = obs.to(runner.device)
                
                # Check if done (for first environment)
                if dones[0]:
                    break
        
        # Save frames as video if we collected any
        if frames:
            video_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.mp4")
            save_frames_as_video(frames, video_path)
            print(f"[INFO] Video saved to: {video_path}")
        else:
            print(f"[WARNING] No frames rendered for checkpoint: {checkpoint_path}")
            
    except Exception as e:
        print(f"[WARNING] Failed to record video: {e}")
    finally:
        # Switch back to train mode
        runner.train_mode()
        
        # Clean up temporary directory
        shutil.rmtree(temp_video_dir, ignore_errors=True)


def save_frames_as_video(frames, output_path: str, fps: int = 30):
    """Save a list of frames as an MP4 video using OpenCV."""
    try:
        import cv2
        import numpy as np
        
        if not frames:
            return
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
    except ImportError:
        print("[WARNING] OpenCV not available. Cannot save video.")
    except Exception as e:
        print(f"[WARNING] Failed to save video: {e}")

