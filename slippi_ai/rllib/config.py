"""RLlib configuration for pure self-play training."""

import ray
from ray import tune
from ray.rllib.algorithms import appo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.policy.policy import PolicySpec
import gymnasium as gym

from slippi_ai.rllib.env import SlippiEnv, SlippiEnvConfig


def env_creator(env_config: EnvContext = None) -> SlippiEnv:
    """Environment creator function for RLlib."""
    # Handle case where env_config is None (e.g., when RLlib calls gym.make)
    if env_config is None:
        env_config = {}
    
    # Provide default values for required parameters when missing
    defaults = {
        'dolphin_path': '/home/theedman/SlippiApp/Slippi_Online-x86_64-ExiAI.AppImage',
        'iso_path': '/home/theedman/Documents/Dolphin/melee.iso',
    }
    
    # Merge defaults with provided config
    merged_config = {**defaults, **env_config}
    
    # Convert RLlib env_config to our SlippiEnvConfig
    slippi_config = SlippiEnvConfig(**merged_config)
    return SlippiEnv(slippi_config)


def get_self_play_config(
    dolphin_path: str = None,
    iso_path: str = None,
    num_workers: int = 4,
    num_envs_per_worker: int = 1,
    train_batch_size: int = 4000,
    sgd_minibatch_size: int = 128,
    num_sgd_iter: int = 10,
    lr: float = 3e-4,
    checkpoint_freq: int = 10,
) -> appo.APPOConfig:
    """Create RLlib configuration for pure self-play training."""
    
    # Register the environment (if not already registered)
    try:
        gym.register(
            id="SlippiEnv-v0",
            entry_point=env_creator,
        )
    except gym.error.Error:
        pass  # Already registered
    
    # Base environment configuration
    env_config = {
        "dolphin_path": dolphin_path,
        "iso_path": iso_path,
        "stage": "FINAL_DESTINATION",
        "online_delay": 0,
        "headless": True,
        "render": False,
        "console_timeout": 30.0,
        "p1_character": "FOX",
        "p2_character": "FOX",
        # Reward parameters
        "damage_ratio": 0.01,
        "ledge_grab_penalty": 0.0,
        "approaching_factor": 0.0,
        "stalling_penalty": 0.0,
        "stalling_threshold": 0.3,
    }
    
    # Create PPO configuration
    config = (
        appo.APPOConfig()
        .environment(
            env="SlippiEnv-v0",
            env_config=env_config,
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=num_workers,
            num_cpus_per_env_runner=1,
            num_envs_per_env_runner=1,
            rollout_fragment_length=200,  # Length of rollout fragments
        )
        .training(
            train_batch_size=train_batch_size,
            # sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=num_sgd_iter,
            lr=lr,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            # PPO-specific settings
            kl_coeff=0.2,
            kl_target=0.01,
        )
        .resources(
            num_gpus=1,
            num_cpus_per_worker=1,
        )
        .debugging(
            log_level="INFO",
        )
        # .checkpointing(
        #     checkpoint_frequency=checkpoint_freq,
        # )
    )
    
    return config


def get_multiagent_self_play_config(
    dolphin_path: str = None,
    iso_path: str = None,
    num_workers: int = 4,
    num_envs_per_worker: int = 1,
    train_batch_size: int = 4000,
    sgd_minibatch_size: int = 128,
    num_sgd_iter: int = 10,
    lr: float = 3e-4,
    checkpoint_freq: int = 10,
) -> appo.APPOConfig:
    """Create RLlib configuration for multi-agent self-play training.
    
    This version treats each player as a separate agent for true multi-agent self-play.
    """
    
    # Register the environment (if not already registered)
    try:
        gym.register(
            id="SlippiEnv-v0",
            entry_point=env_creator,
        )
    except gym.error.Error:
        pass  # Already registered
    
    # Base environment configuration
    env_config = {
        "dolphin_path": dolphin_path,
        "iso_path": iso_path,
        "stage": "FINAL_DESTINATION",
        "online_delay": 0,
        "headless": True,
        "render": False,
        "console_timeout": 30.0,
        "p1_character": "FOX",
        "p2_character": "FOX",
        # Reward parameters
        "damage_ratio": 0.01,
        "ledge_grab_penalty": 0.0,
        "approaching_factor": 0.0,
        "stalling_penalty": 0.0,
        "stalling_threshold": 0.3,
    }
    
    # Create PPO configuration with multi-agent setup
    config = (
        appo.APPOConfig()
        .environment(
            env="SlippiEnv-v0",
            env_config=env_config,
        )
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=num_envs_per_worker,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=train_batch_size,
            lr=lr,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            kl_coeff=0.2,
            kl_target=0.01,
        )
        .resources(
            num_gpus=1 if ray.is_initialized() and len(ray.get_gpu_ids()) > 0 else 0,
            num_cpus_per_worker=1,
        )
        .debugging(
            log_level="INFO",
        )
        .checkpointing(
            checkpoint_frequency=checkpoint_freq,
        )
        # Multi-agent self-play configuration
        .multi_agent(
            policies={
                "player_1": PolicySpec(),
                "player_2": PolicySpec(),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"player_{agent_id}",
            policies_to_train=["player_1", "player_2"],
        )
    )
    
    return config
