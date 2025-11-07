# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from .differential_ik_multi import MultiConstraintDifferentialIKController


@configclass
class MultiConstraintDifferentialIKControllerCfg(DifferentialIKControllerCfg):
    """Configuration for multi-constraint differential inverse kinematics controller."""

    class_type: type = MultiConstraintDifferentialIKController
    """The associated controller class."""
