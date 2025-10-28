from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def distractor_moved(
    env: ManagerBasedEnv,
    distractor_1_cfg: SceneEntityCfg,
    distractor_2_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Termination condition for distractor cubes moving from their initial positions.
    
    This function tracks if any distractor cube has moved beyond a threshold distance
    from its initial spawn position. Used to fail the episode if the robot disturbs
    the distractor cubes while trying to push the target cube.
    
    Note: Requires store_distractor_initial_positions event to be called during reset.
    
    Args:
        env: The environment.
        distractor_1_cfg: Scene entity config for the first distractor cube.
        distractor_2_cfg: Scene entity config for the second distractor cube.
        threshold: Maximum allowed displacement (in meters) before termination.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    # Check if initial positions have been stored
    if not hasattr(env, '_distractor_initial_pos'):
        # Return False for all environments if not initialized yet
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get distractor objects
    distractor_1: RigidObject = env.scene[distractor_1_cfg.name]
    distractor_2: RigidObject = env.scene[distractor_2_cfg.name]
    
    # Get current positions
    dist1_pos_current = distractor_1.data.root_pos_w[:, :3]
    dist2_pos_current = distractor_2.data.root_pos_w[:, :3]
    
    # Get initial positions
    dist1_pos_initial = env._distractor_initial_pos['distractor_1']
    dist2_pos_initial = env._distractor_initial_pos['distractor_2']
    
    # Calculate displacements
    dist1_displacement = torch.norm(dist1_pos_current - dist1_pos_initial, dim=-1)
    dist2_displacement = torch.norm(dist2_pos_current - dist2_pos_initial, dim=-1)
    
    # Check if any distractor has moved beyond threshold
    dist1_moved = dist1_displacement > threshold
    dist2_moved = dist2_displacement > threshold
    
    # Terminate if either distractor has moved
    termination = torch.logical_or(dist1_moved, dist2_moved)
    
    return termination