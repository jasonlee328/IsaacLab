# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Reset states tasks for IsaacLab - Franka Robotiq 2F85."""

import gymnasium as gym

from . import agents


# Register the grasp sampling environment
gym.register(
    id="OmniReset-FrankaRobotiq2f85-GraspSampling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.grasp_sampling_cfg:Robotiq2f85GraspSamplingCfg"},
    disable_env_checker=True,
)

# Register reset states environments
gym.register(
    id="OmniReset-FrankaRobotiq2f85-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-FrankaRobotiq2f85-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-FrankaRobotiq2f85-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEGraspedResetStatesCfg"},
)


gym.register(
    id="OmniReset-FrankaRobotiq2f85-ObjectRestingEEAroundInsertive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectRestingEERoundInsertiveResetStatesCfg"},
)

gym.register(
    id="OmniReset-FrankaRobotiq2f85-ObjectsStacked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectStackedOnReceptiveResetStatesCfg"},
)

gym.register(
    id="OmniReset-FrankaRobotiq2f85-ObjectNearReceptiveEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectNearReceptiveEEGraspedResetStatesCfg"},
)


# Register RL state environments
gym.register(
    id="OmniReset-FrankaRobotiq2f85-RelJointPos-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaRobotiq2f85RelJointPosTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaRobotiq2f85-RelJointPos-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaRobotiq2f85RelJointPosEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaRobotiq2f85-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaRobotiq2f85RelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaRobotiq2f85-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaRobotiq2f85RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)
