# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains the functions for push task MDP terms.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Import event functions
from .events import *  # noqa: F401, F403

# Import observation functions from isaaclab
from isaaclab.envs.mdp.observations import generated_commands  # noqa: F401

# Import command configurations from isaaclab
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg  # noqa: F401

# Import object command from dexsuite for cube target positions
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.commands.pose_commands_cfg import (  # noqa: F401
    ObjectUniformPoseCommandCfg,
)


def object_to_object_distance(
    env: ManagerBasedEnv,
    object1_cfg: SceneEntityCfg,
    object2_cfg: SceneEntityCfg,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Distance between two objects with exponential kernel."""
    object1: RigidObject = env.scene[object1_cfg.name]
    object2: RigidObject = env.scene[object2_cfg.name]
    
    # Get positions
    pos1 = object1.data.root_pos_w[:, :3] - env.scene.env_origins
    pos2 = object2.data.root_pos_w[:, :3] - env.scene.env_origins
    
    # Calculate distance
    distance = torch.norm(pos1 - pos2, dim=-1)
    
    # Apply exponential kernel for reward shaping
    return torch.exp(-distance / sigma)


def object_reached_goal(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
    goal_cfg: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Binary reward for object reaching goal within threshold."""
    object_asset: RigidObject = env.scene[object_cfg.name]
     
    
    # Get positions
    obj_pos = object_asset.data.root_pos_w[:, :3] - env.scene.env_origins
    goal_pos = env.command_manager.get_command(goal_cfg)[:, :3] - env.scene.env_origins
    # goal_asset.data.root_pos_w[:, :3] - env.scene.env_origins
    
    # Calculate distance
    distance = torch.norm(obj_pos - goal_pos, dim=-1)
    
    # Binary reward
    return (distance < threshold).float()
