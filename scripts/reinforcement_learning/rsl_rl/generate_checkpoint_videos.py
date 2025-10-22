#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to generate video rollouts for all checkpoints in a training run.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/generate_checkpoint_videos.py \
        --task Isaac-Push-Cube-Franka-Easy-v0 \
        --run_dir /path/to/logs/rsl_rl/franka_push_cube_easy/2025-10-16_19-12-47 \
        --video_length 200 \
        --num_envs 4
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_checkpoints(run_dir: str, pattern: str = "model_*.pt") -> list[str]:
    """
    Find all checkpoint files in the run directory.
    
    Args:
        run_dir: Path to the training run directory
        pattern: Glob pattern for checkpoint files
        
    Returns:
        List of checkpoint file paths, sorted by iteration number
    """
    checkpoint_paths = glob.glob(os.path.join(run_dir, pattern))
    
    # Sort by iteration number
    def get_iteration(path):
        try:
            filename = os.path.basename(path)
            # Extract number from "model_100.pt" -> 100
            num_str = filename.replace("model_", "").replace(".pt", "")
            return int(num_str)
        except:
            return 0
    
    checkpoint_paths.sort(key=get_iteration)
    return checkpoint_paths


def video_exists(checkpoint_path: str) -> bool:
    """Check if a video already exists for this checkpoint."""
    video_path = checkpoint_path.replace(".pt", ".mp4")
    return os.path.exists(video_path)


def generate_video(
    task_name: str,
    checkpoint_path: str,
    video_length: int,
    headless: bool,
    skip_existing: bool,
    num_envs: int,
) -> bool:
    """
    Generate a video for a single checkpoint.
    
    Args:
        task_name: Name of the task
        checkpoint_path: Path to checkpoint file
        video_length: Length of video in steps
        headless: Whether to run headless
        skip_existing: Whether to skip if video already exists
        num_envs: Number of environments to run for video generation
        
    Returns:
        True if successful, False otherwise
    """
    checkpoint_name = os.path.basename(checkpoint_path).replace(".pt", "")
    video_path = checkpoint_path.replace(".pt", ".mp4")
    
    # Check if video already exists
    if skip_existing and os.path.exists(video_path):
        print(f"[SKIP] Video already exists for {checkpoint_name}")
        return True
    
    print(f"[INFO] Generating video for {checkpoint_name}...")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    play_script = os.path.join(script_dir, "play.py")
    isaaclab_sh = os.path.join(script_dir, "..", "..", "..", "isaaclab.sh")
    
    # Temporary video directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    temp_video_dir = os.path.join(checkpoint_dir, "videos", "play")
    
    # Build command
    cmd = [
        isaaclab_sh,
        "-p",
        play_script,
        "--task", task_name,
        "--checkpoint", checkpoint_path,
        "--video",
        "--video_length", str(video_length),
        "--num_envs", str(num_envs),
    ]
    
    if headless:
        cmd.append("--headless")
        cmd.append("--enable_cameras")
    
    try:
        # Run the play script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Failed to generate video for {checkpoint_name}")
            print(f"STDERR: {result.stderr}")
            return False
        
        # Find the generated video
        if os.path.exists(temp_video_dir):
            video_files = glob.glob(os.path.join(temp_video_dir, "*.mp4"))
            if video_files:
                # Move the first video file
                src_video = video_files[0]
                shutil.move(src_video, video_path)
                print(f"[SUCCESS] Video saved to: {video_path}")
                
                # Clean up temp directory
                shutil.rmtree(temp_video_dir, ignore_errors=True)
                return True
        
        print(f"[WARNING] No video file found for {checkpoint_name}")
        return False
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Timeout generating video for {checkpoint_name}")
        return False
    except Exception as e:
        print(f"[ERROR] Exception generating video for {checkpoint_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate video rollouts for all checkpoints in a training run."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Name of the task (e.g., Isaac-Push-Cube-Franka-Easy-v0)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the training run directory containing checkpoints",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Length of each video in steps (default: 200)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode (default: True)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip checkpoints that already have videos (default: True)",
    )
    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        default="model_*.pt",
        help="Glob pattern for checkpoint files (default: model_*.pt)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments to run for video generation (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Validate run directory
    if not os.path.exists(args.run_dir):
        print(f"[ERROR] Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.run_dir, args.checkpoint_pattern)
    
    if not checkpoints:
        print(f"[ERROR] No checkpoints found in {args.run_dir}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(checkpoints)} checkpoints")
    
    # Generate videos
    success_count = 0
    for checkpoint_path in checkpoints:
        success = generate_video(
            args.task,
            checkpoint_path,
            args.video_length,
            args.headless,
            args.skip_existing,
            args.num_envs,
        )
        if success:
            success_count += 1
    
    print(f"\n[INFO] Successfully generated {success_count}/{len(checkpoints)} videos")


if __name__ == "__main__":
    main()


