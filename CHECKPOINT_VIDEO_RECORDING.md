# Checkpoint Video Recording for RSL-RL Training

This guide explains how to automatically record video rollouts for each checkpoint during RSL-RL training.

## Problem
You want to save a video rollout (e.g., `model_100.mp4`) for each checkpoint (e.g., `model_100.pt`) that gets saved during training.

## Solutions

### Solution 1: Post-Processing Script (RECOMMENDED)

The easiest and most reliable approach is to generate videos after training completes (or during training in a separate terminal).

#### Usage:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/generate_checkpoint_videos.py \
    --task Isaac-Push-Cube-Franka-Easy-v0 \
    --run_dir /weka/oe-training-default/jasonl/IsaacLab/logs/rsl_rl/franka_push_cube_easy/2025-10-16_19-12-47_oct16-pushcube-er010-noise \
    --video_length 200 \
    --headless
```

#### Features:
- Automatically finds all checkpoints in the run directory
- Generates videos with the same name as checkpoints (e.g., `model_100.pt` → `model_100.mp4`)
- Skips checkpoints that already have videos (use `--no-skip_existing` to regenerate)
- Can run while training is still in progress
- Videos are saved in the same directory as checkpoints

#### Example Output:
```
logs/rsl_rl/franka_push_cube_easy/2025-10-16_19-12-47/
├── model_100.pt
├── model_100.mp4  ← Video rollout
├── model_200.pt
├── model_200.mp4  ← Video rollout
├── model_300.pt
├── model_300.mp4  ← Video rollout
...
```

### Solution 2: Modified Training Script (Advanced)

For automatic video recording during training, you can modify the RSL-RL OnPolicyRunner.

#### Step 1: Use the Extended Runner

Replace the standard `OnPolicyRunner` in `scripts/reinforcement_learning/rsl_rl/train.py`:

```python
# Import the extended runner
from on_policy_runner_with_video import OnPolicyRunnerWithVideo

# Replace this line:
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

# With:
runner = OnPolicyRunnerWithVideo(
    env, 
    agent_cfg.to_dict(), 
    log_dir=log_dir, 
    device=agent_cfg.device,
    video_length=200,  # Length of videos
    record_video=True  # Enable video recording
)
```

#### Pros:
- Videos generated automatically during training
- No post-processing needed

#### Cons:
- May slow down training
- More complex to set up
- Potential issues with environment wrapping during training

### Solution 3: Manual Approach

Generate videos manually for specific checkpoints:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Push-Cube-Franka-Easy-v0 \
    --headless \
    --video \
    --video_length 200 \
    --checkpoint /path/to/model_100.pt
```

Then rename the video:
```bash
# The video is saved in: logs/.../videos/play/rl-video-step-0.mp4
# Move and rename it:
mv logs/.../videos/play/rl-video-step-0.mp4 logs/.../model_100.mp4
```

## Recommended Workflow

1. **Run training** normally:
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
       --task Isaac-Push-Cube-Franka-Easy-v0 \
       --headless \
       --num_envs 8192
   ```

2. **Generate videos** after training (or in a separate terminal during training):
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/generate_checkpoint_videos.py \
       --task Isaac-Push-Cube-Franka-Easy-v0 \
       --run_dir logs/rsl_rl/franka_push_cube_easy/YYYY-MM-DD_HH-MM-SS_run_name \
       --video_length 200
   ```

3. **View videos**: Each checkpoint now has a corresponding video in the same directory.

## Tips

- **Video Length**: Set `--video_length` to match your episode length or desired visualization duration
- **Parallel Processing**: You can modify the post-processing script to generate videos in parallel for faster processing
- **Storage**: Videos can be large (10-50MB each). Monitor disk usage.
- **Skip Checkpoints**: To only generate videos for every N checkpoints, modify the checkpoint pattern:
  ```bash
  # Only generate videos for checkpoints 100, 200, 300, etc.
  --checkpoint_pattern "model_*00.pt"
  ```

## Troubleshooting

**Issue**: No video file generated
- **Solution**: Ensure `ffmpeg` is installed and cameras are enabled (`--enable_cameras`)

**Issue**: Video recording slows down training
- **Solution**: Use the post-processing approach (Solution 1) instead

**Issue**: Out of disk space
- **Solution**: Delete older videos or reduce video length/frequency


