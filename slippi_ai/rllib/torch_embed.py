"""PyTorch-based embedding system for Slippi AI game states."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Union
import melee

from slippi_ai.types import Game, Player, Controller, Buttons, Stick


class TorchGameEmbedding:
    """PyTorch-based game state embedding that matches the TensorFlow version."""
    
    def __init__(self):
        # Embedding sizes (matching the TF version)
        self.action_size = 0x18F  # 399 actions
        self.character_size = 0x21  # 33 characters  
        self.stage_size = 64
        self.jumps_size = 6
        
        # Scaling factors (matching TF version)
        self.xy_scale = 0.05
        self.percent_scale = 0.01
        self.shield_scale = 0.01
        
    def embed_game_state(self, game_state: Game) -> torch.Tensor:
        """Convert game state to PyTorch tensor embedding."""
        features = []
        
        # Embed both players
        p0_features = self._embed_player(game_state.p0)
        p1_features = self._embed_player(game_state.p1)
        
        features.extend(p0_features)
        features.extend(p1_features)
        
        # Embed stage
        stage_features = self._embed_stage(game_state.stage)
        features.extend(stage_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _embed_player(self, player: Player) -> list:
        """Embed a single player's state."""
        features = []
        
        # Percent (scaled)
        features.append(player.percent * self.percent_scale)
        
        # Facing (bool to float, -1 for False, 1 for True)
        features.append(1.0 if player.facing else -1.0)
        
        # Position (scaled)
        features.append(player.x * self.xy_scale)
        features.append(player.y * self.xy_scale)
        
        # Action (one-hot encoding)
        action_onehot = self._one_hot_encode(
            int(player.action.value) if hasattr(player.action, 'value') else 0,
            self.action_size
        )
        features.extend(action_onehot)
        
        # Character (one-hot encoding)
        char_onehot = self._one_hot_encode(
            int(player.character.value) if hasattr(player.character, 'value') else 0,
            self.character_size
        )
        features.extend(char_onehot)
        
        # Invulnerable (bool to float)
        features.append(1.0 if player.invulnerable else 0.0)
        
        # Jumps left (one-hot encoding)
        jumps_onehot = self._one_hot_encode(
            min(player.jumps_left, self.jumps_size - 1),
            self.jumps_size
        )
        features.extend(jumps_onehot)
        
        # Shield strength (scaled)
        features.append(player.shield_strength * self.shield_scale)
        
        # On ground (bool to float)
        features.append(1.0 if player.on_ground else 0.0)
        
        return features
    
    def _embed_stage(self, stage) -> list:
        """Embed stage as one-hot encoding."""
        stage_id = int(stage.value) if hasattr(stage, 'value') else 0
        return self._one_hot_encode(stage_id, self.stage_size)
    
    def _one_hot_encode(self, index: int, size: int) -> list:
        """Create one-hot encoding as list of floats."""
        onehot = [0.0] * size
        if 0 <= index < size:
            onehot[index] = 1.0
        return onehot
    
    def get_embedding_size(self) -> int:
        """Calculate total embedding size."""
        # Per player: percent(1) + facing(1) + x(1) + y(1) + action(399) + char(33) + invuln(1) + jumps(6) + shield(1) + ground(1)
        player_size = 1 + 1 + 1 + 1 + self.action_size + self.character_size + 1 + self.jumps_size + 1 + 1
        # Two players + stage
        total_size = 2 * player_size + self.stage_size
        return total_size


def create_torch_game_embedding() -> TorchGameEmbedding:
    """Factory function to create the game embedding."""
    return TorchGameEmbedding()
