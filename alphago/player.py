from .utilities import sample_distribution

from . import mcts, MCTSNode
from .backwards_induction import backwards_induction

# TODO: write tests and docstrings for all this!!!


class AbstractPlayer:

    def __init__(self, game):
        self.game = game

    def choose_action(self, game_state):
        raise NotImplementedError

    def update(self, action):
        pass

    def reset(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.game})"


class RandomPlayer(AbstractPlayer):

    def choose_action(self, game_state, return_probabilities=False):
        next_states = self.game.compute_next_states(game_state)
        action_probs = {action: 1 / len(next_states)
                        for action in next_states.keys()}

        action = sample_distribution(action_probs)

        if return_probabilities:
            return action, action_probs
        return action


class MCTSPlayer(AbstractPlayer):

    def __init__(self, game, estimator, mcts_iters, c_puct, tau=1):
        super().__init__(game)
        self.estimator = estimator
        self.mcts_iters = mcts_iters
        self.c_puct = c_puct
        self.tau = tau
        self.current_node = None

    def choose_action(self, game_state, return_probabilities=False):
        # TODO: need to test using existing MCTS tree vs creating new one
        # TODO: improve exception string, maybe define custom exception
        if self.current_node is None:
            player_no = self.game.which_player(game_state)
            self.current_node = MCTSNode(game_state, player_no)
        if game_state != self.current_node.game_state:
            raise ValueError("Input game state must match that of the "
                             "current node.")

        action_probs = mcts(self.current_node, self.game, self.estimator,
                            self.mcts_iters, self.c_puct, self.tau)

        action = sample_distribution(action_probs)

        if return_probabilities:
            return action, action_probs
        return action

    def update(self, action):
        """Update the position of the player in its MCTS tree
        corresponding to the action just played.

        Either the node has already been explored and we can update the
        position or it hasn't, in which case the current node is set to
        None such that next time choose_action is called, it will start
        a new tree.
        """

        if self.current_node is None:
            return
        print(self.current_node)
        print(action)
        print({k: v.N for k, v in self.current_node.children.items()})
        try:
            self.current_node = self.current_node.children[action]
        except KeyError:
            print("tree not kept!")
            self.current_node = None

    def reset(self):
        self.current_node = None


class OptimalPlayer(AbstractPlayer):  # TODO: Add UTs

    def choose_action(self, game_state, return_probabilities=False):
        value, action = backwards_induction(self.game, game_state)

        if return_probabilities:
            action_probs = {action: 1}
            return action, action_probs
        return action
