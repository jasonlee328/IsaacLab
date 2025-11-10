# Franka Robotiq 2F85 Run Commands

## 1. Grasp Sampling

Collect grasps for the cube object:

```bash
python scripts_v2/tools/record_grasps.py \
  --task OmniReset-FrankaRobotiq2f85-GraspSampling-v0 \
  --num_envs 4096 \
  --num_grasps 1000 \
  --dataset_dir ./grasp_datasets \
  --headless \
  env.scene.object=cube
```

## 2. Reset State Collection

Collect reset states for various scenarios:

### ObjectAnywhereEEAnywhere
```bash
python scripts_v2/tools/record_reset_states.py \
  --task OmniReset-FrankaRobotiq2f85-ObjectAnywhereEEAnywhere-v0 \
  --num_envs 2048 \
  --num_reset_states 1000 \
  --dataset_dir ./reset_state_datasets/ObjectAnywhereEEAnywhere \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

### ObjectRestingEEGrasped
```bash
python scripts_v2/tools/record_reset_states.py \
  --task OmniReset-FrankaRobotiq2f85-ObjectRestingEEGrasped-v0 \
  --num_envs 8192 \
  --num_reset_states 1000 \
  --dataset_dir ./reset_state_datasets/ObjectRestingEEGrasped \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

### ObjectAnywhereEEGrasped
```bash
python scripts_v2/tools/record_reset_states.py \
  --task OmniReset-FrankaRobotiq2f85-ObjectAnywhereEEGrasped-v0 \
  --num_envs 4096 \
  --num_reset_states 1000 \
  --dataset_dir ./reset_state_datasets/ObjectAnywhereEEGrasped \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

### ObjectPartiallyAssembledEEAnywhere
```bash
python scripts_v2/tools/record_reset_states.py \
  --task OmniReset-FrankaRobotiq2f85-ObjectPartiallyAssembledEEAnywhere-v0 \
  --num_envs 8192 \
  --num_reset_states 1000 \
  --dataset_dir ./reset_state_datasets/ObjectPartiallyAssembledEEAnywhere \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

### ObjectPartiallyAssembledEEGrasped
```bash
python scripts_v2/tools/record_reset_states.py \
  --task OmniReset-FrankaRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 \
  --num_envs 8192 \
  --num_reset_states 1000 \
  --dataset_dir ./reset_state_datasets/ObjectPartiallyAssembledEEGrasped \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

### ObjectRestingEEAroundInsertive
```bash
python scripts_v2/tools/record_reset_states.py \
  --task OmniReset-FrankaRobotiq2f85-ObjectRestingEEAroundInsertive-v0 \
  --num_envs 2024 \
  --num_reset_states 1000 \
  --dataset_dir ./reset_state_datasets/ObjectRestingEEAroundInsertive \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

### ObjectsStacked
```bash
python scripts_v2/tools/record_reset_states.py \
  --task OmniReset-FrankaRobotiq2f85-ObjectsStacked-v0 \
  --num_envs 8192 \
  --num_reset_states 1000 \
  --dataset_dir ./reset_state_datasets/ObjectsStacked \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

### ObjectNearReceptiveEEGrasped
```bash
python scripts_v2/tools/record_reset_states.py \
  --task OmniReset-FrankaRobotiq2f85-ObjectNearReceptiveEEGrasped-v0 \
  --num_envs 4096 \
  --num_reset_states 1000 \
  --dataset_dir ./reset_state_datasets/ObjectNearReceptiveEEGrasped \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

## 3. Visualize Reset States

Visualize collected reset states:

```bash
python scripts_v2/tools/visualize_reset_states.py \
  --task OmniReset-FrankaRobotiq2f85-RelCartesianOSC-State-Play-v0 \
  --num_envs 4 \
  --dataset_dir ./reset_state_datasets/ObjectAnywhereEEAnywhere \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

## 4. RL Training

### Relative Cartesian OSC (Recommended)
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-FrankaRobotiq2f85-RelCartesianOSC-State-v0 \
  --num_envs 8192 \
  --logger wandb \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

### Relative Joint Position
```bash
python scripts_v2/rl/main.py \
  --task OmniReset-FrankaRobotiq2f85-RelJointPos-State-v0 \
  --num_envs 8192 \
  --job_type train \
  --rl_framework rslrl \
  --logger wandb \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

## 5. RL Evaluation

Evaluate a trained policy:

```bash
python scripts_v2/rl/main.py \
  --task OmniReset-FrankaRobotiq2f85-RelCartesianOSC-State-v0 \
  --num_envs 1 \
  --job_type eval \
  --rl_framework rslrl \
  --checkpoint /path/to/checkpoint/model_XXXX.pt \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

With video recording:

```bash
python scripts_v2/rl/main.py \
  --task OmniReset-FrankaRobotiq2f85-RelCartesianOSC-State-v0 \
  --num_envs 1 \
  --job_type eval \
  --rl_framework rslrl \
  --checkpoint /path/to/checkpoint/model_XXXX.pt \
  --video \
  --video_length 600 \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

## Notes

- All commands assume you're in the IsaacLab root directory (`/home/tyler2/JasonLab/IsaacLab`)
- Replace `cube` with other object variants if needed (e.g., `peg`, `peg_hole`)
- Checkpoint paths are typically in `logs/rsl_rl/franka_robotiq_2f85_reset_states_agent/YYYY-MM-DD_HH-MM-SS/model_XXXX.pt`
- For distributed training, add `--distributed` flag (requires multiple GPUs)

