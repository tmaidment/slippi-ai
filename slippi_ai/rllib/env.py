"""Gym-compatible environment wrapper for Slippi AI."""

import dataclasses
import logging
from typing import Dict, Optional, Tuple, Any

import gymnasium as gym
import numpy as np
import portpicker
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from slippi_ai.envs import Environment, SafeEnvironment, AsyncEnvMP
from slippi_ai.rllib.torch_embed import create_torch_game_embedding
from slippi_ai.reward import RewardConfig
from slippi_ai.types import Game, Controller
from slippi_ai.controller_lib import send_controller
from slippi_ai import observations
import torch


@dataclasses.dataclass
class SlippiEnvConfig:
    """Configuration for SlippiEnv that can be serialized for RLlib workers."""
    dolphin_path: Optional[str] = None
    iso_path: Optional[str] = None
    stage: str = "FINAL_DESTINATION"  # Will be converted to melee.Stage
    online_delay: int = 0
    headless: bool = True
    render: bool = False
    console_timeout: Optional[float] = 30.0
    evaluation: bool = False
    
    # Player configurations
    p1_character: str = "FOX"  # Will be converted to melee.Character
    p2_character: str = "FOX"
    
    # Reward configuration
    damage_ratio: float = 0.01
    ledge_grab_penalty: float = 0.0
    approaching_factor: float = 0.0
    stalling_penalty: float = 0.0
    stalling_threshold: float = 0.3
    
    def to_dolphin_kwargs(self) -> dict:
        """Convert to dolphin kwargs format."""
        import melee
        from slippi_ai import dolphin as dolphin_lib
        
        return {
            'path': self.dolphin_path,
            'iso': self.iso_path,
            'stage': getattr(melee.Stage, self.stage),
            'online_delay': self.online_delay,
            'headless': self.headless,
            'render': self.render,
            'console_timeout': self.console_timeout,
            'players': {
                1: dolphin_lib.AI(character=getattr(melee.Character, self.p1_character)),
                2: dolphin_lib.AI(character=getattr(melee.Character, self.p2_character)),
            }
        }
    
    def to_reward_config(self) -> RewardConfig:
        """Convert to reward config format."""
        return RewardConfig(
            damage_ratio=self.damage_ratio,
            ledge_grab_penalty=self.ledge_grab_penalty,
            approaching_factor=self.approaching_factor,
            stalling_penalty=self.stalling_penalty,
            stalling_threshold=self.stalling_threshold,
        )


class SlippiEnv(gym.Env):
    """Gym-compatible wrapper for Slippi AI environment.
    
    This wraps the existing Environment class to provide a standard Gym interface
    while reusing all the existing gamestate features and reward definitions.
    """
    observation_space = gym.spaces.Dict({
        agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(954,), dtype=np.float32) for agent in ["p1", "p2"]
    })
    action_space = gym.spaces.Dict({
        agent: gym.spaces.Dict({
            'button_A': gym.spaces.Discrete(2),
            'button_B': gym.spaces.Discrete(2),
            'button_X': gym.spaces.Discrete(2),
            'button_Y': gym.spaces.Discrete(2),
            'button_Z': gym.spaces.Discrete(2),
            'button_L': gym.spaces.Discrete(2),
            'button_R': gym.spaces.Discrete(2),
            'button_D_UP': gym.spaces.Discrete(2),
            'main_stick': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'c_stick': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'l_shoulder': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'r_shoulder': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        }) for agent in ["p1", "p2"]
    })
    
    def __init__(self, config: SlippiEnvConfig):
        super().__init__()
        
        self.config = config
        self._reward_config = config.to_reward_config()
        
        # Create the underlying environment
        dolphin_kwargs = config.to_dolphin_kwargs()
        self._env = env_lib.SafeEnvironment(
            dolphin_kwargs=dolphin_kwargs,
            swap_ports=False,  # We'll handle this at the RLlib level
        )
        
        # For pure self-play, we only control player 1
        self._controlled_port = 1
        self._opponent_port = 2
        
        # Initialize PyTorch-based embedding system
        self._game_embedding = create_torch_game_embedding()
        
        # Calculate observation space size from game embedding
        dummy_obs_size = self._game_embedding.get_embedding_size()
        
        # Define action space - using individual Discrete spaces for buttons (APPO compatibility)
        self.action_space = gym.spaces.Dict({
            'button_A': gym.spaces.Discrete(2),
            'button_B': gym.spaces.Discrete(2),
            'button_X': gym.spaces.Discrete(2),
            'button_Y': gym.spaces.Discrete(2),
            'button_Z': gym.spaces.Discrete(2),
            'button_L': gym.spaces.Discrete(2),
            'button_R': gym.spaces.Discrete(2),
            'button_D_UP': gym.spaces.Discrete(2),
            'main_stick': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'c_stick': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'l_shoulder': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'r_shoulder': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        
        # Observation space using proper embedding size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dummy_obs_size,), dtype=np.float32
        )
        
        self._last_game_state: Optional[Game] = None
    
    def _get_observation_size(self) -> int:
        """Calculate the observation size from the game embedding."""
        return self._game_embedding.get_embedding_size()
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial observation and info."""
        # Get initial state
        env_output = self._env.current_state()
        self._last_game_state = env_output.gamestates[self._controlled_port]
        
        obs = self._extract_observation(self._last_game_state)
        info = {
            'needs_reset': env_output.needs_reset,
        }
        
        return obs, info
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Convert RLlib action to controller format
        controller = self._action_to_controller(action)
        
        # For self-play, we need to provide actions for both players
        # In pure self-play, both players use the same policy
        controllers = {
            self._controlled_port: controller,
            self._opponent_port: controller,  # Same action for opponent in self-play
        }
        
        # Step the environment
        env_output = self._env.step(controllers)
        new_game_state = env_output.gamestates[self._controlled_port]
        
        # Calculate reward (simple example - can be made more sophisticated)
        reward = self._calculate_reward(self._last_game_state, new_game_state)
        
        # Check if episode is done - separate terminated and truncated
        terminated = env_output.needs_reset  # Game ended naturally (stock loss, etc.)
        truncated = False  # For now, disable time-based truncation since Game doesn't have frame
        
        # Create info dict
        info = {
            'needs_reset': env_output.needs_reset,
        }
        
        self._last_game_state = new_game_state
        return self._extract_observation(new_game_state), reward, terminated, truncated, info
    
    def _calculate_reward(self, old_state: Game, new_state: Game) -> float:
        """Calculate reward based on game state changes."""
        if old_state is None:
            return 0.0
            
        reward = 0.0
        
        # Get player states based on port
        if self._controlled_port == 0:
            old_player = old_state.p0
            new_player = new_state.p0
            old_opponent = old_state.p1
            new_opponent = new_state.p1
        else:
            old_player = old_state.p1
            new_player = new_state.p1
            old_opponent = old_state.p0
            new_opponent = new_state.p0
        
        # Reward for dealing damage
        damage_dealt = new_opponent.percent - old_opponent.percent
        reward += damage_dealt * 0.01
        
        # Penalty for taking damage
        damage_taken = new_player.percent - old_player.percent
        reward -= damage_taken * 0.01
        
        # Small survival reward
        reward += 0.001
        
        return reward
    
    def _action_to_controller(self, action: Dict[str, Any]) -> Controller:
        """Convert RLlib action format to controller format."""
        from slippi_ai.types import Controller, Buttons, Stick
        
        # Convert individual button actions to Buttons namedtuple
        buttons = Buttons(
            A=bool(action['button_A']),
            B=bool(action['button_B']),
            X=bool(action['button_X']),
            Y=bool(action['button_Y']),
            Z=bool(action['button_Z']),
            L=bool(action['button_L']),
            R=bool(action['button_R']),
            D_UP=bool(action['button_D_UP']),
        )
        
        # Convert stick arrays to Stick namedtuples
        main_stick = Stick(
            x=np.float32(action['main_stick'][0]),
            y=np.float32(action['main_stick'][1])
        )
        
        c_stick = Stick(
            x=np.float32(action['c_stick'][0]),
            y=np.float32(action['c_stick'][1])
        )
        
        # Create Controller namedtuple
        controller = Controller(
            main_stick=main_stick,
            c_stick=c_stick,
            shoulder=np.float32(action['l_shoulder'][0]),  # Use L shoulder for now
            buttons=buttons
        )
        
        return controller
    
    def _extract_observation(self, game_state: Game) -> np.ndarray:
        """Extract observation from game state using PyTorch embedding system."""
        try:
            # Convert game state to PyTorch tensor embedding
            torch_tensor = self._game_embedding.embed_game_state(game_state)
            
            # Convert to numpy array for RLlib compatibility
            obs_array = torch_tensor.detach().cpu().numpy()
            
            return obs_array
            
        except Exception as e:
            # Fallback to simple observation if embedding fails
            print(f"Warning: Failed to use PyTorch embedding system, falling back to simple obs: {e}")
            return self._extract_simple_observation(game_state)
    
    def _extract_simple_observation(self, game_state: Game) -> np.ndarray:
        """Fallback simple observation extraction."""
        obs_features = []
        
        # Player positions, percents, etc.
        if hasattr(game_state, 'p0') and hasattr(game_state, 'p1'):
            p0, p1 = game_state.p0, game_state.p1
            
            # Basic position and state features
            obs_features.extend([
                p0.x if hasattr(p0, 'x') else 0.0,
                p0.y if hasattr(p0, 'y') else 0.0,
                p0.percent if hasattr(p0, 'percent') else 0.0,
                p1.x if hasattr(p1, 'x') else 0.0,
                p1.y if hasattr(p1, 'y') else 0.0,
                p1.percent if hasattr(p1, 'percent') else 0.0,
            ])
        
        # Pad to expected observation size
        expected_size = self.observation_space.shape[0]
        while len(obs_features) < expected_size:
            obs_features.append(0.0)
        
        return np.array(obs_features[:expected_size], dtype=np.float32)
    
    def _compute_reward(self, prev_state: Optional[Game], current_state: Game) -> float:
        """Compute reward using existing reward system."""
        if prev_state is None:
            return 0.0
        
        # Use the existing reward computation
        # This assumes your reward system can handle single-frame transitions
        try:
            # Create a mini "game" with just two frames for reward computation
            # TODO: Adapt this to work with your actual reward.compute_rewards function
            
            # For now, return a simple reward based on damage dealt vs received
            if hasattr(current_state, 'p0') and hasattr(current_state, 'p1'):
                p0_curr = current_state.p0
                p1_curr = current_state.p1
                p0_prev = prev_state.p0
                p1_prev = prev_state.p1
                
                # Damage dealt to opponent minus damage received
                damage_dealt = getattr(p1_curr, 'percent', 0) - getattr(p1_prev, 'percent', 0)
                damage_received = getattr(p0_curr, 'percent', 0) - getattr(p0_prev, 'percent', 0)
                
                reward = damage_dealt - damage_received
                return float(reward) / 100.0  # Normalize
            
        except Exception:
            # Fallback to zero reward if computation fails
            pass
        
        return 0.0
    
    def close(self):
        """Clean up the environment."""
        if hasattr(self, '_env'):
            self._env.stop()


class SlippiMultiAgentEnv(MultiAgentEnv):
    """Multi-agent wrapper for Slippi AI environment.
    
    This extends MultiAgentEnv to support true multi-agent training where
    each player (p1, p2) is treated as a separate agent.
    """
    
    def __init__(self, config: SlippiEnvConfig):
        super().__init__()
        
        self.config = config
        self._reward_config = config.to_reward_config()
        
        # Get dolphin kwargs and assign a unique Slippi port for this worker
        dolphin_kwargs = config.to_dolphin_kwargs()
        
        # Assign a unique port to avoid conflicts between RLlib workers
        if 'slippi_port' not in dolphin_kwargs:
            dolphin_kwargs['slippi_port'] = portpicker.pick_unused_port()
            logging.info(f"Assigned Slippi port {dolphin_kwargs['slippi_port']} to multi-agent environment")
        
        # Create the underlying environment with process isolation to prevent Dolphin memory corruption
        self._env = AsyncEnvMP(
            dolphin_kwargs=dolphin_kwargs,
            num_envs=0,  # Single environment, not batched
            num_retries=3,
        )
        
        # Initialize PyTorch game embedding
        self._game_embedding = create_torch_game_embedding()
        
        # Track last game states for both players
        self._last_game_states = {"p1": None, "p2": None}
        
        # Define agent IDs
        self._agent_ids = {"p1", "p2"}
        
        # Port mapping (Slippi uses 1-indexed ports)
        self._port_mapping = {"p1": 1, "p2": 2}

        # Evaluation flag
        self.evaluation = config.evaluation
    
    def close(self):
        """Clean up resources when environment is closed."""
        if hasattr(self, '_env'):
            self._env.stop()
    
    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()
        
    @property
    def observation_space(self):
        """Multi-agent observation space."""
        return gym.spaces.Dict({
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(954,), dtype=np.float32) 
            for agent in self._agent_ids
        })
    
    @property 
    def action_space(self):
        """Multi-agent action space."""
        return gym.spaces.Dict({
            agent: gym.spaces.Dict({
                'button_A': gym.spaces.Discrete(2),
                'button_B': gym.spaces.Discrete(2),
                'button_X': gym.spaces.Discrete(2),
                'button_Y': gym.spaces.Discrete(2),
                'button_Z': gym.spaces.Discrete(2),
                'button_L': gym.spaces.Discrete(2),
                'button_R': gym.spaces.Discrete(2),
                'button_D_UP': gym.spaces.Discrete(2),
                'main_stick': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                'c_stick': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                'l_shoulder': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'r_shoulder': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }) for agent in self._agent_ids
        })
    
    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset the environment and return multi-agent observations and infos."""
        # Get initial state from async environment (it sends initial state on startup)
        env_output = self._env.recv()
        
        # Debug: print available ports
        print(f"Available ports in gamestates: {list(env_output.gamestates.keys())}")
        
        # Extract observations for both players
        observations = {}
        infos = {}
        
        for agent_id in self._agent_ids:
            port = self._port_mapping[agent_id]
            game_state = env_output.gamestates[port]
            
            # Extract observation for this agent
            observations[agent_id] = self._extract_observation(game_state)
            
            # Create info dict
            infos[agent_id] = {
                'needs_reset': env_output.needs_reset,
                'port': port,
            }
            
            # Store initial game state
            self._last_game_states[agent_id] = game_state
        
        return observations, infos
    
    def step(self, action_dict: Dict[str, Dict[str, Any]]) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Dict[str, Any]]
    ]:
        """Execute one multi-agent environment step."""
        # Convert actions for both players
        controllers = {}
        for agent_id, action in action_dict.items():
            port = self._port_mapping[agent_id]
            controllers[port] = self._action_to_controller(action)
        
        # Send controllers to async environment and receive result
        self._env.send(controllers)
        env_output = self._env.recv()
        
        # Prepare multi-agent returns
        observations = {}
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}
        
        for agent_id in self._agent_ids:
            port = self._port_mapping[agent_id]
            new_game_state = env_output.gamestates[port]
            old_game_state = self._last_game_states[agent_id]
            
            # Extract observation
            observations[agent_id] = self._extract_observation(new_game_state)
            
            # Calculate reward for this agent
            rewards[agent_id] = self._calculate_reward(old_game_state, new_game_state, agent_id)
            
            # Set termination flags for this agent
            terminated[agent_id] = env_output.needs_reset
            truncated[agent_id] = False
            
            # Create info dict
            infos[agent_id] = {
                'needs_reset': env_output.needs_reset,
                'port': port,
            }
            
            # Update last game state
            self._last_game_states[agent_id] = new_game_state
        
        # RLlib requires '__all__' key to indicate if the entire episode is done
        terminated['__all__'] = env_output.needs_reset
        truncated['__all__'] = False
        
        return observations, rewards, terminated, truncated, infos
    
    def _extract_observation(self, game_state: Game) -> np.ndarray:
        """Extract observation from game state using PyTorch embedding."""
        # Convert to tensor and get embedding
        obs_tensor = self._game_embedding.embed_game_state(game_state)
        
        # Convert to numpy
        return obs_tensor.detach().cpu().numpy()
    
    def _calculate_reward(self, old_state: Game, new_state: Game, agent_id: str) -> float:
        """Calculate reward for a specific agent based on game state changes."""
        if old_state is None:
            return 0.0
            
        reward = 0.0
        
        # Determine which player this agent controls
        if agent_id == "p1":
            player = new_state.p0
            opponent = new_state.p1
            old_player = old_state.p0
            old_opponent = old_state.p1
        else:  # agent_id == "p2"
            player = new_state.p1
            opponent = new_state.p0
            old_player = old_state.p1
            old_opponent = old_state.p0
        
        # Reward for dealing damage
        damage_dealt = opponent.percent - old_opponent.percent
        reward += damage_dealt * 0.01
        
        # Penalty for taking damage
        damage_taken = player.percent - old_player.percent
        reward -= damage_taken * 0.01
        
        # Small survival reward
        reward += 0.001
        
        return reward
    
    def _action_to_controller(self, action: Dict[str, Any]) -> Controller:
        """Convert RLlib action format to controller format."""
        from slippi_ai.types import Controller, Buttons, Stick
        
        # Convert individual button actions to Buttons namedtuple
        buttons = Buttons(
            A=bool(action['button_A']),
            B=bool(action['button_B']),
            X=bool(action['button_X']),
            Y=bool(action['button_Y']),
            Z=bool(action['button_Z']),
            L=bool(action['button_L']),
            R=bool(action['button_R']),
            D_UP=bool(action['button_D_UP']),
        )
        
        # Convert stick actions to Stick namedtuples
        main_stick = Stick(
            x=float(action['main_stick'][0]),
            y=float(action['main_stick'][1])
        )
        
        c_stick = Stick(
            x=float(action['c_stick'][0]),
            y=float(action['c_stick'][1])
        )
        
        # Handle shoulder triggers
        l_shoulder = float(action['l_shoulder'][0]) if 'l_shoulder' in action else 0.0
        r_shoulder = float(action['r_shoulder'][0]) if 'r_shoulder' in action else 0.0
        shoulder = max(l_shoulder, r_shoulder)  # Use the maximum as the shoulder value
        
        return Controller(
            main_stick=main_stick,
            c_stick=c_stick,
            shoulder=shoulder,
            buttons=buttons
        )
    
    def close(self):
        """Clean up the environment."""
        if hasattr(self, '_env'):
            self._env.stop()
