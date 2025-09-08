#!/usr/bin/env python3
"""Test script for SlippiMultiAgentEnv."""

import numpy as np
from slippi_ai.rllib.env import SlippiMultiAgentEnv, SlippiEnvConfig


def test_multiagent_env():
    """Test basic multi-agent environment functionality."""
    print("Testing SlippiMultiAgentEnv...")
    
    # Create config
    config = SlippiEnvConfig(
        dolphin_path="/home/theedman/SlippiApp/Slippi_Online-x86_64-ExiAI.AppImage",
        iso_path="/home/theedman/Documents/Dolphin/melee.iso",
        headless=True,
        render=False,
    )
    
    # Create environment
    env = SlippiMultiAgentEnv(config)
    
    print("✓ Environment created successfully")
    
    # Test observation and action spaces
    print(f"Observation space keys: {list(env.observation_space.keys())}")
    print(f"Action space keys: {list(env.action_space.keys())}")
    
    # Test reset
    try:
        observations, infos = env.reset()
        print("✓ Reset successful")
        print(f"Observation keys: {list(observations.keys())}")
        print(f"Observation shapes: {[obs.shape for obs in observations.values()]}")
        print(f"Info keys: {list(infos.keys())}")
        
        # Test step with sample actions
        sample_actions = {}
        for agent_id in ["p1", "p2"]:
            sample_actions[agent_id] = {
                'button_A': 0,
                'button_B': 0,
                'button_X': 0,
                'button_Y': 0,
                'button_Z': 0,
                'button_L': 0,
                'button_R': 0,
                'button_D_UP': 0,
                'main_stick': np.array([0.0, 0.0], dtype=np.float32),
                'c_stick': np.array([0.0, 0.0], dtype=np.float32),
                'l_shoulder': np.array([0.0], dtype=np.float32),
                'r_shoulder': np.array([0.0], dtype=np.float32),
            }
        
        obs, rewards, terminated, truncated, infos = env.step(sample_actions)
        print("✓ Step successful")
        print(f"Rewards: {rewards}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        
    except Exception as e:
        print(f"✗ Error during environment test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("✓ Environment closed")


if __name__ == "__main__":
    test_multiagent_env()
