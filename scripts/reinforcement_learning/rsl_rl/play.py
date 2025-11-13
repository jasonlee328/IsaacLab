# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.sim.schemas import activate_contact_sensors
import omni.usd

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


class ContactForceVideoWrapper(gym.Wrapper):
    """Wrapper that overlays contact force information on video frames."""
    
    def __init__(self, env, wrist_body_name="panda_link7"):
        super().__init__(env)
        self.wrist_body_name = wrist_body_name
        self.wrist_body_idx = None
        self.contact_activated = False
        self.current_contact_force = None
        self.current_force_mag = 0.0
        
        # Try to get robot and activate contact sensors
        try:
            robot = self.env.unwrapped.scene["robot"]
            self.wrist_body_idx = robot.body_names.index(wrist_body_name)
            
            # Activate contact sensor on wrist
            stage = omni.usd.get_context().get_stage()
            base_prim_path = "/World/envs/env_0/Robot"
            wrist_path = f"{base_prim_path}/{wrist_body_name}"
            activate_contact_sensors(wrist_path, threshold=0.01, stage=stage)
            self.contact_activated = True
            print(f"[ContactForceVideoWrapper] ✓ Activated contact sensor on: {wrist_body_name}")
        except Exception as e:
            print(f"[ContactForceVideoWrapper] WARNING: Could not activate contact sensors: {e}")
    
    def _read_contact_force(self):
        """Read contact force from wrist."""
        if not self.contact_activated or self.wrist_body_idx is None:
            return None, 0.0
        
        try:
            robot = self.env.unwrapped.scene["robot"]
            # Try to access contact forces through the articulation's PhysX view
            import omni.physics.tensors.impl.api as physx_api
            
            physx_view = robot._root_physx_view
            if hasattr(physx_view, 'get_link_incoming_joint_force'):
                forces = physx_view.get_link_incoming_joint_force()
                if forces is not None and forces.shape[1] > self.wrist_body_idx:
                    link_force = forces[0, self.wrist_body_idx]
                    force_mag = torch.norm(link_force).item()
                    if force_mag > 0.1:
                        return link_force, force_mag
        except:
            pass
        
        return None, 0.0
    
    def render(self):
        """Render with contact force overlay."""
        # Get the base frame from environment
        frame = self.env.render()
        
        if frame is None:
            return frame
        
        # Read current contact force
        self.current_contact_force, self.current_force_mag = self._read_contact_force()
        
        # Overlay contact force text on frame
        if self.current_contact_force is not None and self.current_force_mag > 0.1:
            # Convert numpy array to PIL Image
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            
            # Try to use a better font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Prepare text
            force_vec = self.current_contact_force.cpu().numpy()
            text_lines = [
                f"CONTACT FORCE: {self.current_force_mag:.3f} N",
                f"X: {force_vec[0]:+.3f} N",
                f"Y: {force_vec[1]:+.3f} N",
                f"Z: {force_vec[2]:+.3f} N",
            ]
            
            # Draw semi-transparent background
            padding = 10
            line_height = 30
            bg_height = len(text_lines) * line_height + 2 * padding
            bg_width = 350
            
            # Draw background rectangle (semi-transparent red)
            draw.rectangle([10, 10, 10 + bg_width, 10 + bg_height], fill=(220, 50, 50, 180))
            
            # Draw text
            y_offset = 10 + padding
            for i, line in enumerate(text_lines):
                text_font = font if i == 0 else small_font
                draw.text((20, y_offset), line, fill=(255, 255, 255), font=text_font)
                y_offset += line_height
            
            # Convert back to numpy array
            frame = np.array(img)
        
        return frame


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # Add contact force overlay wrapper BEFORE video recording
        env = ContactForceVideoWrapper(env)
        
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos with contact force overlay.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
