# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class ClippedBinaryJointPositionAction(BinaryJointPositionAction):
    """Clipped binary joint action term.

    This action term clips the input actions to the specified range before processing them
    as binary joint actions. Unlike the base BinaryJointAction which clips the processed
    joint commands, this class clips the raw input actions first.
    """

    cfg: actions_cfg.ClippedBinaryJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.ClippedBinaryJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the parent class
        super().__init__(cfg, env)

        # set up input clipping range
        self._input_clip: tuple[float, float] = cfg.input_clip
        self._preprocessed_actions = torch.zeros_like(self.raw_actions)

    def process_actions(self, actions: torch.Tensor):
        # clip the input actions before processing
        clipped_actions = torch.clamp(actions, min=self._input_clip[0], max=self._input_clip[1])
        self._preprocessed_actions[:] = clipped_actions

        # call parent's process_actions with clipped inputs
        super().process_actions(clipped_actions)

    @property
    def preprocessed_actions(self) -> torch.Tensor:
        return self._preprocessed_actions
