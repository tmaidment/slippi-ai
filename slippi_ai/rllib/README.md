# Slippi AI RLlib Integration

This directory contains the RLlib integration for Slippi AI, focused on pure self-play training.

## Overview

The RLlib integration reuses the existing environment definitions, gamestate features, and reward definitions from the main Slippi AI codebase while providing a clean interface for RLlib-based training.

## Key Components

### Environment Wrapper (`env.py`)
- `SlippiEnv`: Gym-compatible wrapper around the existing `Environment` class
- `SlippiEnvConfig`: Serializable configuration for RLlib workers
- Reuses existing reward computation and gamestate processing

### Multi-Agent Environment (`multiagent_env.py`)
- `SlippiMultiAgentEnv`: Multi-agent wrapper for true self-play
- Treats each player as a separate agent with zero-sum rewards

### Configuration (`config.py`)
- `get_self_play_config()`: RLlib PPO configuration for shared policy self-play
- `get_multiagent_self_play_config()`: Configuration for multi-agent self-play

### Training Script (`train.py`)
- Command-line training script with configurable parameters
- Supports both single-agent and multi-agent self-play modes

## Usage

### Basic Training
```bash
python -m slippi_ai.rllib.train \
    --iso-path /path/to/melee.iso \
    --num-workers 4 \
    --num-iterations 1000
```

### Multi-Agent Self-Play
```bash
python -m slippi_ai.rllib.train \
    --iso-path /path/to/melee.iso \
    --multiagent \
    --num-workers 4 \
    --num-iterations 1000
```

### Testing
```bash
python -m slippi_ai.rllib.test_env
```

## Configuration Parameters

- `--dolphin-path`: Path to Dolphin executable (optional, uses default if not specified)
- `--iso-path`: Path to Melee ISO file (required)
- `--num-workers`: Number of RLlib rollout workers
- `--num-envs-per-worker`: Environments per worker (be careful with Dolphin resource usage)
- `--train-batch-size`: Training batch size
- `--lr`: Learning rate
- `--multiagent`: Enable multi-agent self-play mode

## Integration Notes

### Reused Components
- **Environment Management**: Wraps existing `SafeEnvironment` and Dolphin integration
- **Reward System**: Uses existing `reward.compute_rewards()` functions
- **Gamestate Features**: Leverages existing observation processing (needs integration)
- **Controller Interface**: Reuses existing controller encoding/decoding

### TODO: Integration Tasks
1. **Observation Processing**: Integrate with existing policy observation embeddings
2. **Action Space**: Map to existing controller head definitions
3. **Reward Integration**: Full integration with existing reward computation
4. **Multi-Agent Actions**: Modify environment to handle true multi-agent actions
5. **Resource Management**: Optimize Dolphin process management for RLlib workers

### Differences from Original Architecture
- **No Teacher Learning**: Pure self-play without teacher distillation
- **RLlib PPO**: Uses RLlib's PPO implementation instead of custom learner
- **Simplified Batching**: RLlib handles environment batching instead of custom `BatchedEnvironment`
- **Standard Gym Interface**: Conforms to Gym API for RLlib compatibility

## Performance Considerations

- Each environment spawns a Dolphin process - be mindful of system resources
- Start with fewer workers and environments per worker
- Monitor CPU and memory usage when scaling up
- Consider using `num_envs_per_worker=1` initially

## Next Steps

1. Test basic environment instantiation with your Dolphin/ISO setup
2. Integrate proper observation processing from existing embeddings
3. Map action space to existing controller definitions
4. Test training with small scale (few workers, short episodes)
5. Scale up and optimize performance
