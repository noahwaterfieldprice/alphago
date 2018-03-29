from . import mcts, MCTSNode
from .backwards_induction import backwards_induction
from .utilities import sample_distribution


class AbstractPlayer:

    def __init__(self, player_no, game):
        self.player_no = player_no  # TODO: do Players need to know this
        self.game = game

    def choose_action(self, game_state):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.game})"


class RandomPlayer(AbstractPlayer):

    def choose_action(self, game_state):
        next_states = self.game.compute_next_states(game_state)
        action_probs = {action: 1 / len(next_states)
                        for action in next_states.keys()}

        action = sample_distribution(action_probs)

        return action, action_probs


class MCTSPlayer(AbstractPlayer):

    def __init__(self, player_no, game, estimator, mcts_iters, c_puct):
        super().__init__(player_no, game)
        self.estimator = estimator
        self.mcts_iters = mcts_iters
        self.c_puct = c_puct

    def choose_action(self, game_state, current_node=None):
        # TODO: need to test using existing MCTS tree vs creating new one
        if current_node is None:
            current_node = MCTSNode(game_state, self.player_no)

        assert current_node.game_state == game_state

        action_probs = mcts(current_node, self.game, self.estimator,
                            self.mcts_iters, self.c_puct)
        action = sample_distribution(action_probs)

        return action, action_probs


class OptimalPlayer(AbstractPlayer):  # TODO: Add UTs

    def choose_action(self, game_state):
        value, action = backwards_induction(self.game, game_state)

        return action, {action: 1}
