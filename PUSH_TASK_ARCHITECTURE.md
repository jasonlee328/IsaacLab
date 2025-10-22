# Push Task Architecture Explained

## Directory Structure

```
push/
├── push_env_cfg.py              # Base environment configuration (abstract)
├── mdp/                         # MDP (Markov Decision Process) components
│   ├── __init__.py              # Exports MDP functions
│   ├── observations.py          # How to observe the world
│   ├── commands.py              # Target generation logic
│   ├── events.py                # Environment resets & randomization
│   └── command_cfg.py           # Command configurations
└── config/                      # Robot-specific configurations
    └── franka/                  # Franka robot configs
        ├── __init__.py          # Easy/Hard task variants
        ├── push_joint_pos_env_cfg.py  # Franka base config
        └── agents/              # RL algorithm configs
            └── rsl_rl_ppo_cfg.py  # PPO hyperparameters
```

## Component Roles

### 1. `push_env_cfg.py` - The Abstract Blueprint

**Role:** Defines the **structure** of a push task, but no concrete implementation.

**What it contains:**
- **Scene structure** (`PushSceneCfg`): What objects exist (robot, cube, table, lights)
- **MDP managers** (`ActionsCfg`, `ObservationsCfg`, `CommandsCfg`, `RewardsCfg`, `TerminationsCfg`):
  - What observations to provide to the policy
  - What actions the robot can take
  - How to generate target positions
  - How to compute rewards
  - When episodes should terminate

**Think of it as:** A template or interface that says "every push task needs these things"

**Example:**
```python
@configclass
class PushEnvCfg(ManagerBasedRLEnvCfg):
    # Central threshold parameter
    threshold: float = 0.10
    
    # Scene structure (abstract - will be filled by derived classes)
    scene: PushSceneCfg = PushSceneCfg(...)
    
    # MDP components (concrete - same for all push tasks)
    observations: ObservationsCfg = ObservationsCfg()  # What to observe
    actions: ActionsCfg = ActionsCfg()                # What actions exist
    commands: CommandsCfg = CommandsCfg()              # How to generate targets
    rewards: RewardsCfg = RewardsCfg()                # How to reward
    terminations: TerminationsCfg = TerminationsCfg() # When to end episode
```

---

### 2. `mdp/` - The Intelligence Layer

**Role:** Implements the **actual functions** that make the task work.

This is where the **"brain" of the task** lives. Each file handles a different aspect of the MDP:

#### `mdp/observations.py` - What the Robot Sees

**Role:** Functions that compute observations from the simulation state.

**Examples:**
```python
def ee_frame_pos_w(env, asset_cfg):
    """Get end-effector position relative to robot base."""
    # Read simulation state → return observation
    
def cube_pos_rel(env, asset_cfg):
    """Get cube position relative to robot base."""
    # Read simulation state → return observation
    
def target_pos_rel(env, command_name):
    """Get target position from command manager."""
    # Read command → return observation
```

**Used by:** `ObservationsCfg` in `push_env_cfg.py` (lines 95-117)
```python
class PolicyCfg(ObsGroup):
    joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos_rel)
    ee_pos = ObsTerm(func=push_observations.ee_frame_pos_w)    # ← Uses mdp/observations.py
    cube_pos = ObsTerm(func=push_observations.cube_pos_rel)     # ← Uses mdp/observations.py
    target_pos = ObsTerm(func=push_observations.target_pos_rel) # ← Uses mdp/observations.py
```

#### `mdp/events.py` - Environment Reset & Randomization

**Role:** Functions that randomize the environment when episodes reset.

**Examples:**
```python
def separate_target_from_cube(env, env_ids, min_separation):
    """Move target away from cube if they spawn too close."""
    # Called during reset to ensure valid initial state
    
def position_ee_near_cube_simple(env, env_ids, robot_cfg, joint_positions):
    """Position robot's end-effector near the cube at episode start."""
    # Called during reset to give robot a good starting pose
```

**Used by:** `PushEventCfg` in `push_joint_pos_env_cfg.py` (lines 30-106)
```python
@configclass
class PushEventCfg:
    randomize_cube_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,  # ← Uses mdp/events.py
        mode="reset",
        params={
            "pose_range": {"x": (0.50, 0.60), "y": (0.03, 0.07), ...}
        }
    )
```

#### `mdp/__init__.py` - Reward Functions

**Role:** Functions that compute rewards from the simulation state.

**Examples:**
```python
def object_reached_goal(env, object_cfg, goal_cfg, threshold):
    """Binary reward: +1 if cube is within threshold of target, 0 otherwise."""
    obj_pos = get_object_position(...)
    goal_pos = get_target_position(...)
    distance = norm(obj_pos - goal_pos)
    return (distance < threshold).float()  # 1.0 if success, 0.0 otherwise
    
def ee_object_distance(env, std, object_cfg, ee_frame_cfg):
    """Smooth reward for end-effector getting close to cube."""
    ee_pos = get_ee_position(...)
    cube_pos = get_cube_position(...)
    distance = norm(ee_pos - cube_pos)
    return 1 - tanh(distance / std)  # Ranges from 0 to 1
```

**Used by:** `RewardsCfg` in `push_env_cfg.py` (lines 177-189)
```python
@configclass
class RewardsCfg:
    reaching_goal = RwdTerm(
        func=push_mdp.object_reached_goal,  # ← Uses mdp/__init__.py
        params={"object_cfg": SceneEntityCfg("cube"), "threshold": 0.01},
        weight=1.0
    )
```

#### `mdp/commands.py` - Target Generation Logic

**Role:** Classes that generate target positions for the robot to push the cube to.

**Example:**
```python
class ObjectRelativePoseCommand:
    """Generates target positions RELATIVE to the cube's current position."""
    
    def __call__(self, env, env_ids):
        # Get cube position
        cube_pos = env.scene["cube"].data.root_pos_w[env_ids]
        
        # Generate random offset based on configured ranges
        offset_x = uniform(self.ranges.pos_x[0], self.ranges.pos_x[1])
        offset_y = uniform(self.ranges.pos_y[0], self.ranges.pos_y[1])
        
        # Target = cube position + offset
        target_pos = cube_pos + offset
        
        return target_pos
```

**Used by:** `CommandsCfg` in `push_env_cfg.py` (lines 131-174)
```python
@configclass
class CommandsCfg:
    ee_pose = ObjectRelativePoseCommandCfg(  # ← Uses mdp/commands.py
        object_name="cube",
        ranges=ObjectRelativePoseCommandCfg.Ranges(
            pos_x=(-0.10, 0.10),  # Target spawns within ±10cm of cube in X
            pos_y=(-0.10, 0.10),  # Target spawns within ±10cm of cube in Y
        )
    )
```

---

### 3. `config/franka/` - Robot-Specific Implementations

**Role:** Makes the abstract task **concrete** for a specific robot (Franka Panda).

#### `config/franka/push_joint_pos_env_cfg.py` - Franka Base Config

**Role:** Defines Franka-specific components (now mostly empty in our refactored version).

**What it used to contain:**
- Event configurations (cube randomization)
- Robot model (FRANKA_PANDA_CFG)
- Initial robot joint positions
- Action configurations (joint position control)
- Cube spawn configuration
- End-effector frame transformer

**Now:** Just a placeholder - all configuration moved to specific variants.

#### `config/franka/__init__.py` - Task Variants

**Role:** Defines **specific task difficulties** with all parameters in one place.

**Contains:**
- `FrankaPushCubeEasyEnvCfg`: Easy difficulty (5cm threshold, close targets)
- `FrankaPushCubeHardEnvCfg`: Hard difficulty (3cm threshold, far targets)

**What each variant configures:**
```python
@configclass
class FrankaPushCubeEasyEnvCfg(...):
    def __post_init__(self):
        # ═══ PARAMETERS ═══
        threshold = 0.05                    # How precise cube placement needs to be
        cube_x_range = (0.50, 0.60)         # Where cube spawns
        target_x_range = (-0.15, 0.15)      # Where target spawns (relative to cube)
        franka_joint_pos = {...}            # Initial robot pose
        
        # ═══ APPLY CONFIG ═══
        # Sets up robot, cube, physics, rewards, observations, etc.
```

#### `config/franka/agents/rsl_rl_ppo_cfg.py` - RL Algorithm Config

**Role:** PPO (Proximal Policy Optimization) hyperparameters for training.

**Contains:**
```python
@configclass
class FrankaPushCubePPORunnerCfg:
    # Network architecture
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    
    # Training hyperparameters
    learning_rate = 1e-3
    num_steps_per_env = 24      # Steps before policy update
    max_iterations = 1500        # Total training iterations
    
    # PPO specific
    clip_param = 0.2
    entropy_coef = 0.005
```

---

## Data Flow: How Everything Connects

### During Training:

```
1. EPISODE RESET
   ├─> events.py: randomize_cube_position()
   │   └─> Cube spawns at random (x,y) within configured range
   ├─> events.py: position_ee_near_cube_simple() (if enabled)
   │   └─> Robot moves to "ready" position
   └─> commands.py: ObjectRelativePoseCommand()
       └─> Target position generated relative to cube

2. EACH TIMESTEP (25 Hz)
   ├─> observations.py functions called
   │   ├─> ee_frame_pos_w() → robot end-effector position
   │   ├─> cube_pos_rel() → cube position  
   │   └─> target_pos_rel() → target position
   │   └─> All observations stacked → given to policy
   │
   ├─> Policy network outputs actions
   │   └─> Actions applied to robot joints
   │
   ├─> Simulation steps forward (4 physics steps @ 100Hz = 0.04s)
   │
   └─> mdp/__init__.py reward functions called
       └─> object_reached_goal() → +1 if cube at target, 0 otherwise
       └─> Reward given to policy for learning

3. EPISODE END
   ├─> Termination conditions checked
   │   ├─> Time limit reached? (10 seconds)
   │   └─> Cube fell off table?
   └─> Go back to step 1 (reset)
```

### Configuration Hierarchy:

```
push_env_cfg.py (abstract blueprint)
    ↓
    Defines: What observations? What rewards? What commands?
    ↓
config/franka/push_joint_pos_env_cfg.py (Franka base - now empty)
    ↓
    Just inheritance - no logic
    ↓
config/franka/__init__.py (concrete variants)
    ↓
    FrankaPushCubeEasyEnvCfg: ALL CONFIGURATION IN ONE PLACE
        ├─> Sets threshold = 0.05
        ├─> Sets cube_x_range = (0.50, 0.60)
        ├─> Sets target_x_range = (-0.15, 0.15)
        ├─> Configures robot (FRANKA_PANDA_CFG)
        ├─> Configures cube (RigidObjectCfg)
        ├─> Configures events (PushEventCfg)
        ├─> Configures actions (JointPositionActionCfg)
        └─> Everything else
```

---

## Key Concepts

### MDP (Markov Decision Process)

The task is structured as an MDP with:
- **State (s)**: Joint positions, cube position, target position, etc.
  - Defined in `observations.py`
- **Action (a)**: Joint position targets for the robot
  - Defined in `ActionsCfg` → uses `mdp.JointPositionActionCfg`
- **Reward (r)**: +1 when cube reaches target
  - Defined in `RewardsCfg` → uses `object_reached_goal()` from `mdp/__init__.py`
- **Transition (s → s')**: Physics simulation steps the world forward
  - Handled by Isaac Sim
- **Episode termination**: Timeout or cube falls off
  - Defined in `TerminationsCfg`

### Manager-Based RL

Isaac Lab uses a "manager-based" architecture:
- **Observation Manager**: Calls observation functions from `mdp/observations.py`
- **Action Manager**: Processes actions and applies to robot
- **Reward Manager**: Calls reward functions from `mdp/__init__.py`
- **Command Manager**: Generates targets using `mdp/commands.py`
- **Event Manager**: Handles resets using `mdp/events.py`
- **Termination Manager**: Checks if episode should end

Each manager uses the **configuration** from `push_env_cfg.py` and the **implementations** from `mdp/`.

---

## Summary: Who Does What?

| Component | Role | Example |
|-----------|------|---------|
| `push_env_cfg.py` | Abstract blueprint | "Every push task has observations, rewards, commands" |
| `mdp/observations.py` | Observation implementations | `cube_pos_rel()` - how to get cube position |
| `mdp/events.py` | Reset implementations | `randomize_cube_position()` - where to spawn cube |
| `mdp/__init__.py` | Reward implementations | `object_reached_goal()` - how to compute reward |
| `mdp/commands.py` | Target generation | `ObjectRelativePoseCommand` - where to spawn target |
| `config/franka/__init__.py` | Concrete task variants | Easy: 5cm threshold, close targets<br>Hard: 3cm threshold, far targets |
| `config/franka/agents/` | RL algorithm config | PPO hyperparameters for training |

---

## Analogy: Building a House

- **`push_env_cfg.py`**: Architectural blueprint ("house needs kitchen, bedrooms, bathrooms")
- **`mdp/`**: Construction workers (electricians, plumbers, carpenters) who know HOW to build each part
- **`config/franka/`**: Specific house designs ("3-bedroom colonial" vs "4-bedroom ranch")
- **`agents/`**: Interior designer (how to arrange furniture / how to train policy)

You can have many different house designs (Easy, Hard, Custom) using the same blueprint and same workers!

