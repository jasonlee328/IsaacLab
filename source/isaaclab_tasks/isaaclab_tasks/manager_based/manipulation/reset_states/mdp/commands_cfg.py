# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

# Copyright (c) 2022-2024, The Octi Lab and  Isaac Lab Project Developers.
# All rights reserved.

from dataclasses import MISSING
from typing import Any

from isaaclab.managers import CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .commands import TaskCommand, TaskDependentCommand, RandomTableGoalCommand, ReceptiveObjectGoalCommand


@configclass
class TaskDependentCommandCfg(CommandTermCfg):
    class_type: type = TaskDependentCommand

    reset_terms_when_resample: dict[str, EventTerm] = {}


@configclass
class TaskCommandCfg(TaskDependentCommandCfg):
    class_type: type = TaskCommand

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    success_position_threshold: Any = MISSING

    success_orientation_threshold: Any = MISSING

    insertive_asset_cfg: Any = MISSING

    receptive_asset_cfg: Any = MISSING


@configclass
class RandomTableGoalCommandCfg(TaskDependentCommandCfg):
    class_type: type = RandomTableGoalCommand

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    success_position_threshold: Any = MISSING

    success_orientation_threshold: Any = MISSING

    insertive_asset_cfg: Any = MISSING

    table_asset_cfg: Any = MISSING

    goal_pose_range: Any = MISSING


@configclass
class ReceptiveObjectGoalCommandCfg(TaskDependentCommandCfg):
    class_type: type = ReceptiveObjectGoalCommand

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    success_position_threshold: Any = MISSING

    success_orientation_threshold: Any = MISSING

    insertive_asset_cfg: Any = MISSING

    receptive_asset_cfg: Any = MISSING

    goal_height_offset: Any = MISSING
