# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

import argparse
import h5py
import numpy as np

from scriptsv2.robomimic.tools.split_train_val import split_train_val_from_hdf5


def remove_actionless_steps(f, demo_key):
    actions = f["data"][demo_key]["actions"][:]
    successes = f["data"][demo_key]["success"][:]
    valid_steps = (np.sum(actions[:, :3], axis=1) > 0.001) | (successes == 1)

    obs_tmp = f["data"][demo_key]["obs"][valid_steps]
    next_obs_tmp = f["data"][demo_key]["next_obs"][valid_steps]
    success_tmp = f["data"][demo_key]["success"][valid_steps]
    dones_tmp = f["data"][demo_key]["dones"][valid_steps]
    rewards_tmp = f["data"][demo_key]["rewards"][valid_steps]
    actions_tmp = f["data"][demo_key]["actions"][valid_steps]

    # Delete existing datasets if they exist
    dataset_paths = ["obs", "next_obs", "success", "dones", "rewards", "actions"]
    for dataset_path in dataset_paths:
        full_path = f"data/{demo_key}/{dataset_path}"
        if dataset_path in f["data"][demo_key]:
            del f[full_path]

    # Ensure groups for nested data structures exist
    for subkey in ["obs", "next_obs"]:
        if subkey not in f["data"][demo_key]:
            f["data"][demo_key].create_group(subkey)

    # Recreate datasets within their specific groups
    f["data"][demo_key]["obs"].create_dataset("state", data=obs_tmp)
    f["data"][demo_key]["next_obs"].create_dataset("state", data=next_obs_tmp)
    f["data"][demo_key].create_dataset("success", data=success_tmp)
    f["data"][demo_key].create_dataset("dones", data=dones_tmp)
    f["data"][demo_key].create_dataset("rewards", data=rewards_tmp)
    f["data"][demo_key].create_dataset("actions", data=actions_tmp)
    # Return the number of valid steps after filtering
    return len(success_tmp)


def filter_success(hdf5_path: str, max_length=-1, filter_key=None, remove_actionless=False):
    f = h5py.File(hdf5_path, "r+")
    success_mask = []
    success_sample_count = 0
    failure_sample_count = 0
    for demo_key in f["data"].keys():
        # Delete existing datasets
        if remove_actionless:
            # Filter out actionless steps
            num_step_samples = remove_actionless_steps(f, demo_key)
            f["data"][demo_key].attrs["num_samples"] = num_step_samples
        # Check if trajectory length is within the specified max length
        num_step_samples = len(f["data"][demo_key]["success"])
        if f["data"][demo_key]["success"][-1] and max_length != -1 and num_step_samples < max_length:
            success_mask.append(demo_key.encode("utf-8"))
            print(f"{demo_key} has {num_step_samples} steps is stored as successful")
            success_sample_count += num_step_samples
        else:
            print(f"{demo_key} has {num_step_samples} steps is not stored as successful")
            failure_sample_count += num_step_samples

    # Check if the "mask" group exists and delete it if it does
    if "mask" in f:
        del f["mask"]

    # Create a new "mask" group and add the "successful" dataset
    mask_group = f.create_group("mask")
    print(f"total success samples: {success_sample_count}")
    print(f"total failure samples: {failure_sample_count}")
    mask_group.create_dataset(filter_key, data=np.array(success_mask))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument(
        "--remove_actionless", action="store_true", help="whether to remove actionless state action pair"
    )
    parser.add_argument(
        "--max_episode_length", type=int, default=9999, help="the maximum length allowed for successful episode"
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help=(
            "If provided, split the subset of trajectories in the file that correspond to this filter key"
            " into a training and validation set of trajectories, instead of splitting the full set of"
            " trajectories."
        ),
    )
    parser.add_argument("--ratio", type=float, default=0.1, help="validation ratio, in (0, 1)")
    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)
    filter_key = None
    if args.success_only:
        filter_key = "successful"
        filter_success(
            args.dataset,
            max_length=args.max_episode_length,
            filter_key=filter_key,
            remove_actionless=args.remove_actionless,
        )
    split_train_val_from_hdf5(args.dataset, val_ratio=args.ratio, filter_key=filter_key)
