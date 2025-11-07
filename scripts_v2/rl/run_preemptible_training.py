#!/usr/bin/env python3
"""
Smart training script that automatically handles experiment resumption based on existing checkpoints.
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse
import glob
import logging
from datetime import datetime


def setup_logging(experiment_id: str, task: str):
    """
    Set up logging configuration for cluster environments.
    Logs to both stdout and a file for better visibility.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs") / "training_scripts"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{experiment_id}_{task}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Console output
            logging.FileHandler(log_file)       # File output
        ]
    )
    
    # Force immediate flushing for real-time visibility
    for handler in logging.getLogger().handlers:
        handler.flush()
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def find_latest_checkpoint(experiment_dir: str) -> str | None:
    """
    Find the latest checkpoint in the experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    if not os.path.exists(experiment_dir):
        return None
    
    # Look for .pt files in all timestamped directories
    checkpoint_pattern = os.path.join(experiment_dir, "*", "model_*.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Sort by modification time and return the latest
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint


def check_experiment_exists(experiment_id: str, task: str) -> tuple[bool, str | None]:
    """
    Check if experiment exists and return the latest checkpoint if available.
    
    Args:
        experiment_id: The experiment ID to check
        task: The task name
        
    Returns:
        Tuple of (experiment_exists, latest_checkpoint_path)
    """
    logs_dir = Path("logs/rsl_rl") / task / experiment_id
    
    if not logs_dir.exists():
        return False, None
    
    latest_checkpoint = find_latest_checkpoint(str(logs_dir))
    return True, latest_checkpoint


def build_command(args: argparse.Namespace, checkpoint_path: str | None = None) -> list[str]:
    """
    Build the training command with appropriate arguments.
    
    Args:
        args: Parsed command line arguments
        checkpoint_path: Optional path to specific checkpoint
        
    Returns:
        List of command arguments
    """
    cmd = [
        "python", "-m", "torch.distributed.run",
        "--nnodes", args.nnodes,
        "--nproc_per_node", args.nproc_per_node,
        "scripts_v2/rl/main.py",
        "--task", args.task,
        "--num_envs", str(args.num_envs),
        "--job_type", args.job_type,
        "--rl_framework", args.rl_framework,
        "--logger", args.logger,
        "--headless",
        "--distributed",
        "--experiment_id", args.experiment_id,
    ]
    
    # Add resume flag and checkpoint if available
    if checkpoint_path:
        cmd.extend(["--resume", ''])
        cmd.extend(["--checkpoint_dir", checkpoint_path])
        
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        logging.info(f"Starting new training run for experiment: {args.experiment_id}")
    
    # Add environment configuration overrides
    if args.env_overrides:
        cmd.extend(args.env_overrides)
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Smart RL training script with automatic resume detection")
    parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelJointPos-State-v0", 
                       help="Task name")
    parser.add_argument("--num_envs", type=int, default=256, 
                       help="Number of environments")
    parser.add_argument("--job_type", default="train", 
                       help="Job type (train/eval/play)")
    parser.add_argument("--rl_framework", default="rslrl", 
                       help="RL framework")
    parser.add_argument("--nnodes", default="1", 
                       help="Number of nodes to distribute the job on")
    parser.add_argument("--nproc_per_node", default="1", 
                       help="Number of GPU processes per node")
    parser.add_argument("--logger", default="wandb", 
                       help="Logger type")
    parser.add_argument("--experiment_id", default="AAAA", 
                       help="Experiment ID")                   
    parser.add_argument("--force_new", action="store_true", 
                       help="Force new training run even if experiment exists")
    parser.add_argument("--env_overrides", nargs="*",
                        default=["env.scene.insertive_object=cube", "env.scene.receptive_object=cube"],
                        help="Environment configuration overrides")
    
    args = parser.parse_args()
    
    # Set up logging first
    log_file = setup_logging(args.experiment_id, args.task)
    
    logging.info(f"Checking experiment: {args.experiment_id}")
    logging.info(f"Task: {args.task}")
    
    # Check if experiment exists and find latest checkpoint
    experiment_exists, latest_checkpoint = check_experiment_exists(args.experiment_id, args.task)
    
    if experiment_exists and not args.force_new:
        if latest_checkpoint:
            logging.info(f"Found existing experiment with checkpoint: {latest_checkpoint}")
            # checkpoint_name = os.path.basename(latest_checkpoint)
            cmd = build_command(args, latest_checkpoint)
        else:
            logging.info("Found existing experiment but no checkpoints. Starting new run.")
            cmd = build_command(args, None)
    else:
        if args.force_new:
            logging.info("Force new run requested. Starting fresh training.")
        else:
            logging.info("No existing experiment found. Starting new training.")
        cmd = build_command(args, None)
    
    logging.info("Executing command:")
    logging.info(" ".join(cmd))
    logging.info("")
    
    # Execute the command
    try:
        # Use Popen for better signal handling
        process = subprocess.Popen(cmd)
        
        # Wait for the process to complete
        returncode = process.wait()
        
        if returncode == 0:
            logging.info(f"Training completed successfully with exit code: {returncode}")
        else:
            logging.error(f"Training failed with exit code: {returncode}")
            sys.exit(returncode)
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        # Try to terminate the subprocess gracefully
        if 'process' in locals():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during training: {e}")
        # Try to terminate the subprocess if it exists
        if 'process' in locals():
            process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
