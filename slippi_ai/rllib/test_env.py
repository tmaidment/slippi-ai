"""Test script for the RLlib Slippi environment."""

import numpy as np
from slippi_ai.rllib.env import SlippiEnv, SlippiEnvConfig
from slippi_ai.rllib.multiagent_env import SlippiMultiAgentEnv


def test_single_agent_env():
    """Test the single-agent SlippiEnv wrapper."""
    print("Testing single-agent SlippiEnv...")
    
    # Create configuration
    config = SlippiEnvConfig(
        dolphin_path="/home/theedman/SlippiApp/Slippi_Online-x86_64-ExiAI.AppImage",  # Will use default
        iso_path="/home/theedman/Documents/Dolphin/melee.iso",  # Update this path
        stage="FINAL_DESTINATION",
        headless=True,
        render=False,
        console_timeout=30.0,
    )
    
    try:
        # Create environment
        env = SlippiEnv(config)
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Test reset
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            # Sample random action
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"Step {i+1}: reward={reward:.3f}, done={done}, obs_shape={obs.shape}")
            
            if done:
                obs = env.reset()
                print("Environment reset")
        
        env.close()
        print("Single-agent test completed successfully!")
        
    except Exception as e:
        print(f"Single-agent test failed: {e}")


def test_multiagent_env():
    """Test the multi-agent SlippiMultiAgentEnv wrapper."""
    print("\nTesting multi-agent SlippiMultiAgentEnv...")
    
    # Create configuration
    config = SlippiEnvConfig(
        dolphin_path="/home/theedman/SlippiApp/Slippi_Online-x86_64-ExiAI.AppImage",  # Will use default
        iso_path="/home/theedman/Documents/Dolphin/melee.iso",  # Update this path
        stage="FINAL_DESTINATION",
        headless=True,
        render=False,
        console_timeout=30.0,
    )
    
    try:
        # Create environment
        env = SlippiMultiAgentEnv(config)
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        print(f"Agent IDs: {env._agent_ids}")
        
        # Test reset
        obs_dict = env.reset()
        print(f"Initial observations: {list(obs_dict.keys())}")
        for agent_id, obs in obs_dict.items():
            print(f"  Agent {agent_id} obs shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            # Sample random actions for both agents
            action_dict = {
                agent_id: env.action_space.sample()
                for agent_id in env._agent_ids
            }
            
            obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
            
            print(f"Step {i+1}:")
            for agent_id in env._agent_ids:
                print(f"  Agent {agent_id}: reward={reward_dict[agent_id]:.3f}, done={done_dict[agent_id]}")
            
            if done_dict.get("__all__", False):
                obs_dict = env.reset()
                print("Environment reset")
        
        env.close()
        print("Multi-agent test completed successfully!")
        
    except Exception as e:
        print(f"Multi-agent test failed: {e}")


def test_action_observation_compatibility():
    """Test that actions and observations are properly formatted."""
    print("\nTesting action/observation compatibility...")
    
    config = SlippiEnvConfig(
        dolphin_path="/home/theedman/SlippiApp/Slippi_Online-x86_64-ExiAI.AppImage",  # Will use default
        iso_path="/home/theedman/Documents/Dolphin/melee.iso",  # Update this path
        stage="FINAL_DESTINATION",
        headless=True,
        render=False,
        console_timeout=30.0,
    )
    
    try:
        env = SlippiEnv(config)
        
        # Test action space sampling
        for i in range(3):
            action = env.action_space.sample()
            print(f"Sample action {i+1}: {action}")
            
            # Verify action format - individual button actions
            assert 'button_A' in action
            assert 'button_B' in action
            assert 'button_X' in action
            assert 'button_Y' in action
            assert 'button_Z' in action
            assert 'button_L' in action
            assert 'button_R' in action
            assert 'button_D_UP' in action
            assert 'main_stick' in action
            assert 'c_stick' in action
            assert 'l_shoulder' in action
            assert 'r_shoulder' in action
            
            # Verify button values are 0 or 1
            for button_key in ['button_A', 'button_B', 'button_X', 'button_Y', 
                             'button_Z', 'button_L', 'button_R', 'button_D_UP']:
                assert action[button_key] in [0, 1], f"{button_key} should be 0 or 1"
            
            # Verify action value ranges
            assert action['main_stick'].shape == (2,)
            assert np.all(action['main_stick'] >= -1.0) and np.all(action['main_stick'] <= 1.0)
            assert action['c_stick'].shape == (2,)
            assert np.all(action['c_stick'] >= -1.0) and np.all(action['c_stick'] <= 1.0)
        
        print("Action format tests passed!")
        
        # Test observation space
        obs = env.reset()
        assert obs.shape == (954,), f"Expected obs shape (954,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        
        print("Observation format tests passed!")
        
        env.close()
        
    except Exception as e:
        print(f"Compatibility test failed: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Slippi RLlib Environment Tests")
    print("=" * 50)
    
    print("\nNOTE: These tests require:")
    print("1. Dolphin emulator installed")
    print("2. Melee ISO file")
    print("3. Update the iso_path in the test functions")
    print()
    
    # Run tests
    test_action_observation_compatibility()
    
    # Uncomment these when you have Dolphin and ISO set up
    # test_single_agent_env()
    # test_multiagent_env()
    
    print("\n" + "=" * 50)
    print("Tests completed!")
