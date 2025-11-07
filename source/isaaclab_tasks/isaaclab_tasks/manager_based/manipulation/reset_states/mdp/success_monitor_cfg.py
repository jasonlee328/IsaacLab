# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from .success_monitor import SuccessMonitor


@configclass
class SuccessMonitorCfg:

    class_type: type[SuccessMonitor] = SuccessMonitor

    monitored_history_len: int = 100
    """The total length of success entry recorded, monitoring table size: (num_monitored_data, monitored_history_len)"""

    num_monitored_data: int = MISSING
    """Number of success monitored. monitoring table size: (num_monitored_data, monitored_history_len)"""

    device: str = "cpu"
    """The device used to maintain success table data structure"""
