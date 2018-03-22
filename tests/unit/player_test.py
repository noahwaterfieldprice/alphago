import pytest

from alphago.player import AbstractPlayer, MCTSPlayer, RandomPlayer
from .games import mock_game


class TestAbstractPlayer:

    def test_string_representation_of_player(self, mocker):
        mock_player = mocker.MagicMock()
        mock_player.__repr__ = AbstractPlayer.__repr__
        mock_player.game = mock_game

        assert repr(mock_player) == "{0}({1})".format(
            mock_player.__class__.__name__, mock_player.game)
        assert str(mock_player) == "{0}({1})".format(
            mock_player.__class__.__name__, mock_player.game)

    def test_calculating_action_probabilities(self, mocker):
        mock_player = mocker.MagicMock()
        mock_player.action_probabilities = AbstractPlayer.action_probabilities

        with pytest.raises(NotImplementedError):
            mock_player.action_probabilities(mock_player, "some_game_state")


class TestRandomPlayer(TestAbstractPlayer):

    def test_calculating_action_probabilities(self, mocker):
        mock_player = mocker.MagicMock()
        mock_player.action_probabilities = RandomPlayer.action_probabilities
        mock_player.game = mock_game

        next_states = mock_game.compute_next_states(mock_game.INITIAL_STATE)
        expected_action_probs = {action: 1 / len(next_states)
                                 for action in next_states.keys()}

        action_probs = mock_player.action_probabilities(mock_player,
                                                        mock_game.INITIAL_STATE)
        assert action_probs == expected_action_probs


class TestMCTSPlayer(TestAbstractPlayer):

    def test_mcts_is_called_with_right_arguments(self, mocker):
        mock_evaluator = mocker.MagicMock()

        mock_mcts = mocker.patch("alphago.player.mcts")
        mock_mcts_node = mocker.MagicMock()
        mock_mcts_node_constructor = mocker.patch(
            "alphago.player.MCTSNode", return_value=mock_mcts_node)

        arg_names = ("game", "player_no", "evaluator", "mcts_iters", "c_puct")
        args = (mock_game, 1, mock_evaluator, 20, 0.5)
        player_info = {key: value for key, value in zip(arg_names, args)}
        mock_player = mocker.MagicMock(**player_info)
        mock_player.action_probabilities = MCTSPlayer.action_probabilities

        mock_player.action_probabilities(mock_player, mock_game.INITIAL_STATE)
        mock_mcts_node_constructor.assert_called_once_with(mock_game.INITIAL_STATE, 1)
        expected_args = (mock_mcts_node, mock_game, mock_evaluator) + args[3:]
        mock_mcts.assert_called_once_with(*expected_args)

    def test_calculating_action_probabilities(self, mocker):
        mock_player = mocker.MagicMock()
        mock_player.action_probabilities = MCTSPlayer.action_probabilities
        mocker.patch("alphago.player.mcts", return_value="some_action_probs")

        action_probs = mock_player.action_probabilities(mock_player, mock_game.INITIAL_STATE)
        assert action_probs == "some_action_probs"
