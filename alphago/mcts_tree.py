import numpy as np

from .utilities import compute_distribution

__all__ = ["mcts", "MCTSNode"]


def mcts(starting_node, game, estimator, mcts_iters, c_puct):
    # TODO: write a better docstring!
    """Perform a MCTS from a given starting node

    Parameters
    ----------
    starting_node: MCTSNode
        The root of a subtree of the game. We take actions at the root.
    game: Game
        An object representing the game to be played.
    estimator: func
        A function from states to probs, value. probs is a dictionary
        with keys the actions in the state and value given by the
        estimate of the value of the state.
    mcts_iters: int
        The number of iterations of MCTS.
    c_puct: float
        A hyperparameter determining the level of exploration in the
        select algorithm.

    Returns
    -------
    action_probs: dict
            A probability distribution over actions available in the
            root node, given as a dictionary from actions to
            probabilities.
    """

    for i in range(mcts_iters):
        # First select a leaf node from the MCTS tree. This actually
        # returns all nodes and actions taken, with the length of
        # actions being one less than the length of nodes. The last
        # element of nodes is the leaf node.
        nodes, actions = select(starting_node, c_puct)
        leaf = nodes[-1]

        if not leaf.is_terminal:
            # Evaluate the leaf node to get the probabilities and value
            # according to the net.
            prior_probs, value = estimator(leaf.game_state)

            # Store this as a value for player 1 and a value for player 2.
            # TODO: We could make this more general later.
            player = game.which_player(leaf.game_state)
            other_player = 1 if player == 2 else 2
            values = {player: value,
                      other_player: -value}

            # Compute the next possible states from the leaf node. This
            # returns a dictionary with keys the legal actions and
            # values the game states. Note that if the leaf is terminal
            # there will be no next_states.
            child_states = game.compute_next_states(leaf.game_state)

            # TODO: This should be replaced by a function that links the indices
            # for the neural network output to the actions in the game.
            prior_probs = {action: prior_probs[action]
                           for action in child_states}

            # Compute the players for the children states.
            child_players = {action: game.which_player(child_state)
                             for action, child_state in child_states.items()}

            child_terminals = {action: game.is_terminal(child_state)
                               for action, child_state in child_states.items()}

            # Expand the tree with the new leaf node
            leaf.expand(prior_probs, child_states, child_players, child_terminals)
        else:
            # We don't need prior probs if the node is terminal, but we
            # do still need the value of the node. The utility function
            # computes the value for the player to play.
            values = game.utility(leaf.game_state)

        # Backup the value up the tree.
        backup(nodes, values)

    action_counts = {action: child.N
                     for action, child in starting_node.children.items()}
    return compute_distribution(action_counts)


class MCTSNode:
    """A class to represent a Monte Carlo search tree node. This node
    keeps track of all quantities needed for the Monte Carlo tree
    search (MCTS) algorithm. These quantities correspond to those in
    the 'Mastering the game of Go without human knowledge' paper -
    (doi:10.1038/nature24270) and are named identically.

    Parameters
    ----------
    game_state: state
        An object describing the game state corresponding to this node.
    player: int
        The player to play at this node.
    is_terminal: bool
        A boolean indicating if the node is terminal.


    Attributes
    ----------

    Q: float
        The total action value of taking the action linking the parent
        node to this node.
    W: float
        The mean action value of taking the action linking the parent
        node to this node.
    N: int
        The number of times this node has been visited.
    children: dict
        A dictionary holding all the child nodes of this node with keys
        the legal actions from this node and values the nodes
        corresponding to taking said actions. Before the node has been
        expanded, this is empty.
    prior_probs: dict
        A dictionary where the keys are the available actions from
        the this node to the child nodes and the values are the prior
        probabilities of taking each action.
    game_state: state
        An object describing the game state corresponding to this node.
        This object holds sufficient information for the game object to
        understand the game -- in particular, given this state, the
        game object can return all the legal actions from this state.
    player: int
        The player to play at this node.
    is_terminal: bool
        A boolean indicating if the node is terminal.
    """

    __slots__ = "Q W N is_terminal children prior_probs game_state player".split()

    def __init__(self, game_state, player, is_terminal=False):
        self.Q = 0.0
        self.W = 0.0
        self.N = 0.0
        self.is_terminal = is_terminal
        self.player = player
        self.children = {}
        self.prior_probs = {}
        self.game_state = game_state

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.game_state}, "
                f"{self.player}, {self.is_terminal})")

    def __str__(self):
        return (f"{self.__class__.__name__}({self.game_state}, "
                f"{self.player}, {self.is_terminal}, "
                f"{self.Q}, {self.W}, {self.N})")

    def is_leaf(self):
        """Returns whether or not the node in the tree is a leaf."""
        return len(self.children) == 0

    def expand(self, prior_probs, child_states, child_players,
               child_terminals):
        """Expands the tree at the leaf node with the given
        probabilities.

        Parameters
        ---------
        prior_probs: dict
            A dictionary where the keys are the available actions from
            the node and the values are the prior probabilities of
            taking each action.
        child_states: dict
            A dictionary where the keys are the available actions from
            the node and the values are the corresponding game states
            resulting from taking each action.
        child_players: dict
            A dictionary where the keys are the available actions from the node
            and the values are the corresponding players to play in the child
            node.
        child_terminals: dict
            A dictionary where the keys are the available actions from the node
            and the values are booleans indicating whether the corresponding
            nodes are terminal.
        """
        assert self.is_leaf()

        # TODO: Enforce that prior probs is a distribution somehow
        self.prior_probs = prior_probs

        self.children = {action: MCTSNode(
            child_states[action], child_players[action],
            child_terminals[action])
            for action in child_states}


def compute_ucb(action_values, prior_probs, action_counts, c_puct):
    """Calculates the upper confidence bound, Q(s,a) + U(s,a), for each
    of the child nodes.

    U(s,a) is defined as
        c_puct * prior_prob * sqrt(sum(N)) / (1 + N)
    where N is the action count.

    Parameters
    ----------
    action_values, prior_probs, action_counts: dict
        These are all dictionaries where the keys are the available
        actions and the values are the corresponding action values,
        prior probabilities and action counts.
    c_puct: float
        A hyperparameter determining the level of exploration.

    Returns
    -------
    upper_confidence_bounds: dict
        A dictionary mapping each child node to: Q(s,a) + U(s,a).
    """
    # TODO: Check if this is the right way to define this. Currently we ignore
    # prior_probs if action_counts are 0. This is the case when we
    # select children for the first time, which is exactly the time
    # we want to be using prior_probs.
    num = 1.0 + np.sqrt(sum(action_counts.values()))
    # assert num > 0
    upper_confidence_bounds = {
        k: action_values[k] + prior_probs[k] / float(1 + action_counts[k]) *
        c_puct * num for k in action_values}
    return upper_confidence_bounds


def select(starting_node, c_puct):
    """Starting at a given node in the tree, traverse a path through
     child nodes until a leaf is reached. Return the sequence of nodes
     and actions taken along the path.

    At each node, the next node in the path is chosen to be the child
    node with the highest upper confidence bound, Q(s, a) + U(s, a).

    Parameters
    ----------
    starting_node: MCTSNode
        The node in the tree from which to start selection algorithm.
    c_puct: float
        A hyperparameter determining the level of exploration.

    Returns
    -------
    nodes: list
        The sequence of nodes that was traversed along the path.
    actions: list
        The sequence of actions that were taken along the path.
    """
    node = starting_node
    num_steps = 0

    actions = []
    nodes = [node]

    while not node.is_leaf():
        # The node is not a leaf, so has children. We select the one with
        # largest upper confidence bound.
        # TODO: maybe these should be arrays to vectorise compute_ucb
        # prior_probs = {action: child.prior_prob
        #                for action, child in node.children.items()}
        action_values = {action: child.Q
                         for action, child in node.children.items()}
        action_counts = {action: child.N
                         for action, child in node.children.items()}

        # Compute the upper confidence bound values
        upper_confidence_bounds = compute_ucb(action_values, node.prior_probs,
                                              action_counts, c_puct)

        # Take action with largest ucb
        action = max(upper_confidence_bounds, key=upper_confidence_bounds.get)
        node = node.children[action]

        # Append action to the list of actions, and node to nodes
        nodes.append(node)
        actions.append(action)
        num_steps += 1

    # We have reached a leaf node, or the maximum number of steps.
    # Return the sequence of nodes and actions.
    return nodes, actions


def backup(nodes, values):
    """Given the sequence of nodes (ending in the new expanded node)
    from the game tree, propagate back the Q-values and action counts.

    Parameters
    ----------
    nodes: list
        The list of nodes to backup.
    values: dict
        A dictionary with keys the players and values the value for
        that player. In a zero sum game with players 1 and 2, we have
        v[1] = -v[2].
    """
    parent_player = None
    for node in nodes:
        # Increment the visit count
        node.N += 1.0

        # We only set the W and Q values if the node has a parent.
        if parent_player:
            # Update the cumulative and mean action values
            node.W += values[parent_player]
            node.Q = node.W / node.N

        # Set the parent player as the player in the current node.
        parent_player = node.player


def print_tree(root):
    """Prints the tree rooted at 'root'. Prints in pre-order.
    """
    queue = [root]
    i = 0

    while i < len(queue):
        node = queue[i]
        print(node)
        queue.extend(node.children.values())
        i += 1
