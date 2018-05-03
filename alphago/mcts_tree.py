from typing import Any, Callable, Dict, List, Tuple

import numpy as np

__all__ = ["mcts", "MCTSNode"]


# TODO: sort this out:
State, Action = Any, Any
Player, Game = Any, Any


def mcts(starting_node: "MCTSNode",
         game: Game,
         estimator: Callable,
         mcts_iters: int,
         c_puct: float,
         tau: float = 1,
         dirichlet_epsilon: float = 0.25,
         dirichlet_alpha: float = 0.03
         ) -> Dict[Action, float]:
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
    tau: float
        A hyperparameter between 1 and 0 defining the 'temperature' of
        the calculated probability distribution. Here, a value of 1
        gives the true distribution based on node visit count and a value
        tending to 0 extremises the distribution such that effectively
        the most visited node has a corresponding probability of 1.
    dirichlet_epsilon: float
        Mixes the prior probabilities for starting_node with Dirichlet
        noise. Uses (1 - dirichlet_epsilon) * prior_prob +
        dirichlet_epsilon * dirichlet_noise, where dirichlet_noise is
        sampled from the Dirichlet distribution with parameter dirichlet_alpha.
    dirichlet_alpha: float
        The parameter to sample the Dirichlet distribution with.

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
        nodes, actions = select(starting_node, c_puct,
                                dirichlet_epsilon=dirichlet_epsilon,
                                dirichlet_alpha=dirichlet_alpha)
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

            # TODO: This should be replaced by a function that links the
            # indices for the neural network output to the actions in the game.
            prior_probs = {action: prior_probs[action]
                           for action in child_states.keys()}

            prior_probs = compute_distribution(prior_probs, tau=1)

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
    return compute_distribution(action_counts, tau)


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

    def __init__(self,
                 game_state: Any,
                 player: Player,
                 is_terminal: bool = False) -> None:
        self.Q = 0.0
        self.W = 0.0
        self.N = 0.0
        self.is_terminal = is_terminal
        self.player = player
        self.children = {}
        self.prior_probs = {}
        self.game_state = game_state

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.game_state}, "
                f"{self.player}, {self.is_terminal})")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}({self.game_state}, "
                f"{self.player}, {self.is_terminal}, "
                f"{self.Q}, {self.W}, {self.N})")

    def is_leaf(self) -> bool:
        """Returns whether or not the node in the tree is a leaf."""
        return len(self.children) == 0

    def expand(self,
               prior_probs: Dict[Action, float],
               child_states: Dict[Action, State],
               child_players: Dict[Action, Player],
               child_terminals: Dict[Action, bool]) -> None:
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


def compute_ucb(action_values: Dict[Action, float],
                prior_probs:  Dict[Action, float],
                action_counts: Dict[Action, int],
                c_puct: float) -> Dict[Action, float]:
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
    num = np.sqrt(sum(action_counts.values()))
    # assert num > 0
    upper_confidence_bounds = {
        k: action_values[k] + prior_probs[k] / float(1 + action_counts[k]) *
        c_puct * num for k in action_values.keys()}
    return upper_confidence_bounds


def mix_dirichlet_noise(d, epsilon, alpha):
    """Combine values in dictionary with Dirichlet noise. Samples
    dirichlet_noise according to dirichlet_alpha in each component. Then
    updates the value v for key k with (1-epsilon) * v + epsilon * noise_k.

    Parameters
    ----------
    d: dict
        Dictionary with floats as values.
    epsilon: float
        Mixes the prior probabilities for starting_node with Dirichlet
        noise. Uses (1 - dirichlet_epsilon) * prior_prob +
        dirichlet_epsilon * dirichlet_noise, where dirichlet_noise is
        sampled from the Dirichlet distribution with parameter dirichlet_alpha.
        Set to 0.0 if no Dirichlet perturbation.
    alpha: float
        The parameter to sample the Dirichlet distribution with.

    Returns
    -------
    d: dict
        The dictionary with perturbed values.
    """
    noise = np.random.dirichlet([alpha] * len(d))
    return {k: (1 - epsilon) * v + epsilon * noise for ((k, v), noise) in zip(
        d.items(), noise)}


def select(starting_node: "MCTSNode",
           c_puct: float,
           dirichlet_epsilon: float = 0.0,
           dirichlet_alpha: float = 0.03,
           ) -> Tuple[List["MCTSNode"], List[Action]]:
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
    dirichlet_epsilon: float
        Mixes the prior probabilities for starting_node with Dirichlet
        noise. Uses (1 - dirichlet_epsilon) * prior_prob +
        dirichlet_epsilon * dirichlet_noise, where dirichlet_noise is
        sampled from the Dirichlet distribution with parameter dirichlet_alpha.
        Set to 0.0 if no Dirichlet perturbation.
    dirichlet_alpha: float
        The parameter to sample the Dirichlet distribution with.

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

        # Add Dirichlet noise to the prior probs.
        prior_probs = mix_dirichlet_noise(node.prior_probs,
                                          dirichlet_epsilon, dirichlet_alpha)

        # Compute the upper confidence bound values
        upper_confidence_bounds = compute_ucb(action_values, prior_probs,
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


def backup(nodes: List["MCTSNode"],
           values: Dict[Player, float]) -> None:
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


def compute_distribution(d: Dict[Any, float],
                         tau: float = 1) -> Dict[Any, float]:
    """Calculate a probability distribution with probabilities
    proportional to the values in a dictionary

    Parameters
    ----------
    d: dict
        A dictionary with values equal to positive floats.
    tau: float
        A parameter between 1 and 0 defining the 'temperature' of the
        calculated probability distribution. Here, a value of 1 gives
        the true distribution based on the input (potentially
        non-normalised distribution) and a value tending to 0
        extremises the distribution such that effectively the most
        visited node has a corresponding probability of 1.

    Returns
    -------
    prob_distribution: dict:
        A probability distribution proportional to the values of d,
        given as a dictionary with keys equal to those of d and values
        the probability corresponding to the value.
    """
    assert tau > 0
    assert min(d.values()) >= 0
    max_value = max(d.values())
    d_tau = {k: (v / max_value) ** (1 / tau) for k, v in d.items()}
    total = sum(d_tau.values())
    assert total > 0

    prob_distribution = {k: v / total for k, v in d_tau.items()}
    return prob_distribution


def print_tree(root: "MCTSNode") -> None:
    """Prints the tree rooted at 'root'. Prints in pre-order.
    """
    queue = [root]
    i = 0

    while i < len(queue):
        node = queue[i]
        print(node)
        queue.extend(node.children.values())
        i += 1
