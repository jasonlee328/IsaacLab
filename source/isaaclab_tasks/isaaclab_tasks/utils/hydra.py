# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for the hydra configuration system."""


import functools
from collections.abc import Callable, Mapping

try:
    import hydra
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def setattr_nested(obj: object, attr_path: str, value: object) -> None:
    """Set a nested attribute on an object using a dot-separated path."""
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        if isinstance(obj, Mapping):
            obj = obj[attr]
        else:
            obj = getattr(obj, attr)
    last = attrs[-1]
    if isinstance(obj, Mapping):
        obj[last] = value
    else:
        setattr(obj, last, value)


def getattr_nested(obj: object, attr_path: str) -> object:
    """Get a nested attribute from an object using a dot-separated path."""
    attrs = attr_path.split(".")
    for attr in attrs:
        if isinstance(obj, Mapping):
            obj = obj[attr]
        else:
            obj = getattr(obj, attr)
    return obj


def register_task_to_hydra(
    task_name: str, agent_cfg_entry_point: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, dict]:
    """Register the task configuration to the Hydra configuration store.

    This function resolves the configuration file for the environment and agent based on the task's name.
    It then registers the configurations to the Hydra configuration store.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        A tuple containing the parsed environment and agent configuration objects.
    """
    # load the configurations
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = None
    if agent_cfg_entry_point:
        agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)
    # replace gymnasium spaces with strings because OmegaConf does not support them.
    # this must be done before converting the env configs to dictionary to avoid internal reinterpretations
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    # convert the configs to dictionary
    env_cfg_dict = env_cfg.to_dict()
    if isinstance(agent_cfg, dict) or agent_cfg is None:
        agent_cfg_dict = agent_cfg
    else:
        agent_cfg_dict = agent_cfg.to_dict()
    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
    # replace slices with strings because OmegaConf does not support slices
    cfg_dict = replace_slices_with_strings(cfg_dict)
    config_store = ConfigStore.instance()
    env_default_groups: list[str] = []
    agent_default_groups: list[str] = []
    # --- ENV variants → register groups + record defaults
    if isinstance(env_cfg_dict, dict) and "variants" in env_cfg_dict:
        for root_config_name, root_config_dict in env_cfg_dict["variants"].items():
            group_path = f"env.{root_config_name}"
            env_default_groups.append(group_path)
            config_store.store(group=group_path, name="default", node=getattr_nested(cfg_dict, group_path))
            for group_name, group_val in root_config_dict.items():
                config_store.store(group=group_path, name=group_name, node=group_val)

    # --- AGENT variants → register groups + record defaults
    if isinstance(agent_cfg_dict, dict) and "variants" in agent_cfg_dict:
        for root_config_name, root_config_dict in agent_cfg_dict["variants"].items():
            group_path = f"agent.{root_config_name}"
            agent_default_groups.append(group_path)
            config_store.store(group=group_path, name="default", node=getattr_nested(cfg_dict, group_path))
            for group_name, group_val in root_config_dict.items():
                config_store.store(group=group_path, name=group_name, node=group_val)

    # Set defaults list only if variants exist
    if env_default_groups or agent_default_groups:
        cfg_dict["defaults"] = (
            ["_self_"] + [{g: "default"} for g in env_default_groups] + [{g: "default"} for g in agent_default_groups]
        )
    # store the configuration to Hydra
    config_store.store(name=task_name, node=OmegaConf.create(cfg_dict) if "defaults" in cfg_dict else cfg_dict)
    return env_cfg, agent_cfg


def hydra_task_config(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Decorator to handle the Hydra configuration for a task.

    This decorator registers the task to Hydra and updates the environment and agent configurations from Hydra parsed
    command line arguments.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The decorated function with the envrionment's and agent's configurations updated from command line arguments.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # register the task to Hydra
            env_cfg, agent_cfg = register_task_to_hydra(task_name.split(":")[-1], agent_cfg_entry_point)

            # define the new Hydra main function
            @hydra.main(config_path=None, config_name=task_name.split(":")[-1], version_base="1.3")
            def hydra_main(hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg):
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                # replace strings that represent gymnasium spaces because OmegaConf does not support them.
                # this must be done after converting the env configs from dictionary to avoid internal reinterpretations
                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
                # get agent configs
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator


def register_task_to_hydra_programmatic(
    task_name: str, agent_cfg_entry_point: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, dict]:
    """Register the task configuration to the Hydra configuration store (programmatic version).

    This version supports variants and is used by hydra_task_config_programmatic.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        A tuple containing the parsed environment and agent configuration objects.
    """
    # load the configurations
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = None
    if agent_cfg_entry_point:
        agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)
    # replace gymnasium spaces with strings because OmegaConf does not support them.
    # this must be done before converting the env configs to dictionary to avoid internal reinterpretations
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    # convert the configs to dictionary
    env_cfg_dict = env_cfg.to_dict()
    if isinstance(agent_cfg, dict) or agent_cfg is None:
        agent_cfg_dict = agent_cfg
    else:
        agent_cfg_dict = agent_cfg.to_dict()
    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
    # replace slices with strings because OmegaConf does not support slices
    cfg_dict = replace_slices_with_strings(cfg_dict)
    config_store = ConfigStore.instance()
    env_default_groups: list[str] = []
    agent_default_groups: list[str] = []
    # --- ENV variants → register groups + record defaults
    if isinstance(env_cfg_dict, dict) and "variants" in env_cfg_dict:
        for root_config_name, root_config_dict in env_cfg_dict["variants"].items():
            group_path = f"env.{root_config_name}"
            env_default_groups.append(group_path)
            config_store.store(group=group_path, name="default", node=getattr_nested(cfg_dict, group_path))
            for group_name, group_val in root_config_dict.items():
                config_store.store(group=group_path, name=group_name, node=group_val)

    # --- AGENT variants → register groups + record defaults
    if isinstance(agent_cfg_dict, dict) and "variants" in agent_cfg_dict:
        for root_config_name, root_config_dict in agent_cfg_dict["variants"].items():
            group_path = f"agent.{root_config_name}"
            agent_default_groups.append(group_path)
            config_store.store(group=group_path, name="default", node=getattr_nested(cfg_dict, group_path))
            for group_name, group_val in root_config_dict.items():
                config_store.store(group=group_path, name=group_name, node=group_val)

        agent_root_defaults = ["_self_"] + [{grp: "default"} for grp in agent_default_groups]
        cfg_dict = {"defaults": agent_root_defaults, **cfg_dict}

    # Set a single defaults list (once)
    cfg_dict["defaults"] = (
        ["_self_"] + [{g: "default"} for g in env_default_groups] + [{g: "default"} for g in agent_default_groups]
    )
    config_store.store(name=task_name, node=OmegaConf.create(cfg_dict), group=None)

    return env_cfg, agent_cfg


def hydra_task_config_programmatic(task_name: str, agent_cfg_entry_point: str, hydra_args: list) -> Callable:
    """Decorator to handle the Hydra configuration for a task (programmatic version).

    This version uses initialize/compose instead of hydra.main, allowing it to be used in scripts
    that need to handle command-line arguments differently or want more control over Hydra initialization.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.
        hydra_args: List of Hydra command-line arguments to use for configuration.

    Returns:
        The decorated function with the environment's and agent's configurations updated from command line arguments.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # register the task to Hydra
            env_cfg, agent_cfg = register_task_to_hydra_programmatic(task_name.split(":")[-1], agent_cfg_entry_point)

            # Initialize Hydra programmatically
            with initialize(config_path=None, version_base="1.3"):
                # Compose the configuration
                hydra_cfg = compose(config_name=task_name, overrides=hydra_args)
                # convert to a native dictionary
                hydra_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_cfg = replace_strings_with_slices(hydra_cfg)
                # update the group configs with Hydra command line arguments
                if "variants" in hydra_cfg["env"]:
                    hydra_env_choice_cfg = {
                        hydra_choice.split("=")[0]: hydra_choice.split("=")[1]
                        for hydra_choice in hydra_args
                        if hydra_choice.startswith("env")
                    }
                    for env_group_key, env_group_choice in hydra_env_choice_cfg.items():
                        key = env_group_key[4:]  # remove "env."
                        if env_group_choice != "default":
                            setattr_nested(env_cfg, key, env_cfg.variants[key][env_group_choice])
                            setattr_nested(hydra_cfg["env"], key, env_cfg.variants[key][env_group_choice].to_dict())
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_cfg["env"])
                # replace strings that represent gymnasium spaces because OmegaConf does not support them.
                # this must be done after converting the env configs from dictionary to avoid internal reinterpretations
                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
                # get agent configs
                if agent_cfg is not None and "variants" in hydra_cfg["agent"]:
                    hydra_agent_choice_cfg = {
                        hydra_choice.split("=")[0]: hydra_choice.split("=")[1]
                        for hydra_choice in hydra_args
                        if hydra_choice.startswith("agent")
                    }
                    for agent_group_key, agent_group_choice in hydra_agent_choice_cfg.items():
                        key = agent_group_key[6:]  # remove "agent."
                        if agent_group_choice != "default":
                            setattr_nested(agent_cfg, key, agent_cfg.variants[key][agent_group_choice])
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    agent_cfg = hydra_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_cfg["agent"])
            # call the original function
            return func(env_cfg, agent_cfg, *args, **kwargs)

        return wrapper

    return decorator
