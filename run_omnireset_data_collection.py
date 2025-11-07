#!/usr/bin/env python3
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_grasps", type=int, default=1000)
    parser.add_argument("--num_reset_states", type=int, default=10000)
    args = parser.parse_args()
    
    # 1. Grasp Sampling
    subprocess.run([
        "python", "scripts_v2/tools/record_grasps.py",
        "--task", "OmniReset-FrankaRobotiq2f85-GraspSampling-v0",
        "--num_envs", "2048",
        "--num_grasps", str(args.num_grasps),
        "--dataset_dir", "./grasp_datasets",
        "--headless",
        "env.scene.object=cube"
    ], check=True)
    
    # 2. Reset States - Object Anywhere, End-Effector Anywhere
    subprocess.run([
        "python", "scripts_v2/tools/record_reset_states.py",
        "--task", "OmniReset-FrankaRobotiq2f85-ObjectAnywhereEEAnywhere-v0",
        "--num_envs", "2048",
        "--num_reset_states", str(args.num_reset_states),
        "--headless",
        "--dataset_dir", "./reset_state_datasets/ObjectAnywhereEEAnywhere",
        "env.scene.insertive_object=cube",
        "env.scene.receptive_object=cube"
    ], check=True)
    
    # 3. Reset States - Object Resting, End-Effector Grasped
    subprocess.run([
        "python", "scripts_v2/tools/record_reset_states.py",
        "--task", "OmniReset-FrankaRobotiq2f85-ObjectRestingEEGrasped-v0",
        "--num_envs", "2048",
        "--num_reset_states", str(args.num_reset_states),
        "--headless",
        "--dataset_dir", "./reset_state_datasets/ObjectRestingEEGrasped",
        "env.scene.insertive_object=cube",
        "env.scene.receptive_object=cube"
    ], check=True)
    
    # 4. Reset States - Object Anywhere, End-Effector Grasped
    subprocess.run([
        "python", "scripts_v2/tools/record_reset_states.py",
        "--task", "OmniReset-FrankaRobotiq2f85-ObjectAnywhereEEGrasped-v0",
        "--num_envs", "2048",
        "--num_reset_states", str(args.num_reset_states),
        "--headless",
        "--dataset_dir", "./reset_state_datasets/ObjectAnywhereEEGrasped",
        "env.scene.insertive_object=cube",
        "env.scene.receptive_object=cube"
    ], check=True)
    
    # 5. Reset States - Object Near Receptive, End-Effector Grasped
    subprocess.run([
        "python", "scripts_v2/tools/record_reset_states.py",
        "--task", "OmniReset-FrankaRobotiq2f85-ObjectNearReceptiveEEGrasped-v0",
        "--num_envs", "2048",
        "--num_reset_states", str(args.num_reset_states),
        "--headless",
        "--dataset_dir", "./reset_state_datasets/ObjectNearReceptiveEEGrasped",
        "env.scene.insertive_object=cube",
        "env.scene.receptive_object=cube"
    ], check=True)


if __name__ == "__main__":
    sys.exit(main())
