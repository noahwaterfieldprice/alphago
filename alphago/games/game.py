from typing import Any, Dict, Sequence

import abc

GameState, Action = Any, Any


class Game(abc.ABC):
    """Abstract base class for games playable by MCTS.

    This abstract base class defines the minimal interface a Game
    object needs to implement to be able run MCTS on it.
    """

    @abc.abstractmethod
    def which_player(self, state: GameState) -> int:
        """Returns the player to play in the given state."""

    @abc.abstractmethod
    def compute_next_states(self, state: GameState) -> Dict[Action, GameState]:
        """Computes the next states possible from the given state."""

    @abc.abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """Returns whether the given state is terminal."""

    @abc.abstractmethod
    def utility(self, state: GameState) -> Dict[int, float]:
        """Compute the utility of the given (terminal) state for each
        player."""
