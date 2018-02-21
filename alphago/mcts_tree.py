import numpy as np


class MCTSNode:
    """A class to represent a Monte Carlo search tree node. This node
    keeps track of all quantities needed for the Monte Carlo tree
    search (MCTS) algorithm. These quantities correspond to those in
    the 'Mastering the game of Go without human knowledge' paper -
    (doi:10.1038/nature24270) and are named identically.

    Parameters
    ----------
    prior_prob: float

    Attributes
    ----------
    prior_prob: float
        The prior probability of visiting this node from the parent
        node.
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
    game_state: state
        An object describing the game state corresponding to this node.
        This object holds sufficient information for the game object to
        understand the game -- in particular, given this state, the
        game object can return all the legal actions from this state.
    """

    __slots__ = "prior_prob Q W N children game_state".split()

    def __init__(self, prior_prob, game_state):
        self.prior_prob = prior_prob
        self.Q = 0.0
        self.W = 0.0
        self.N = 0.0
        self.children = {}
        self.game_state = game_state

    def is_leaf(self):
        """Returns whether or not the node in the tree is a leaf."""
        return len(self.children) == 0

    def expand(self, prior_probs, children_states):
        """Expands the tree at the leaf node with the given
        probabilities.

        Parameters
        ---------
        prior_probs: dict
            A dictionary where the keys are the available actions from
            the node and the values are the prior probabilities of
            taking each action.
        children_states: dict
            A dictionary where the keys are the available actions from
            the node and the values are the corresponding game states
            resulting from taking each action.
        """
        assert self.is_leaf()

        prior_probs = compute_distribution(prior_probs)

        # Initialise the relevant data for each child of the leaf node
        self.children = {
            action: MCTSNode(prior_probs[action], children_states[action])
            for action in children_states
        }


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
    num = np.sqrt(sum(action_counts.values()))
    # assert num > 0
    upper_confidence_bounds = {
        k: action_values[k] + prior_probs[k] / float(1 + action_counts[k]) *
        c_puct * num for k in action_values
    }
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
        prior_probs = {action: child.prior_prob
                       for action, child in node.children.items()}
        action_values = {action: child.Q
                         for action, child in node.children.items()}
        action_counts = {action: child.N
                         for action, child in node.children.items()}

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


def backup(nodes, v):
    """Given the sequence of nodes (ending in the new expanded node) from
    the game tree, propagate back the Q-values and action counts.
    """
    for node in nodes:
        # Increment the visit count
        node.N += 1.0

        # Update the cumulative and mean action values
        node.W += v
        node.Q = float(node.W) / float(node.N)


def compute_distribution(d):
    """Calculate a probability distribution with probabilities
    proportional to the values in a dictionary

    Parameters
    ----------
    d: dict
        A dictionary with values equal to positive floats.

    Returns
    -------
    prob_distribution: dict:
        A probability distribution proportional to the values of d,
        given as a dictionary with keys equal to those of d and values
        the probability corresponding to the value.
    """
    total = sum(d.values())
    assert min(d.values()) >= 0
    assert total > 0
    prob_distribution = {k: float(v) / float(total)
                         for k, v in d.items()}
    return prob_distribution


def mcts(starting_node, evaluator, next_states_function,
         is_terminal, max_iters, c_puct):
    """Perform a MCTS from a given starting node

    Parameters
    ----------
    starting_node: MCTSNode
        The root of a subtree of the game. We take actions at the root.
    evaluator: func
        A function from states to probs, value. probs is a dictionary
        with keys the actions in the state and value given by the
        estimate of the value of the state.
    next_states_function: func
        A function that takes a state and returns a dictionary with
        keys the available actions in the state and values the
        resulting game states.
    max_iters: int
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

    for i in range(max_iters):
        # First select a leaf node from the MCTS tree. This actually
        # returns all nodes and actions taken, with the length of
        # actions being one less than the length of nodes. The last
        # element of nodes is the leaf node.
        nodes, actions = select(starting_node, c_puct)
        leaf = nodes[-1]

        # Evaluate the leaf node to get the probabilities and value
        # according to the net.
        probs, value = evaluator(leaf.game_state)

        if not is_terminal(leaf.game_state):
            # Compute the next possible states from the leaf node. This
            # returns a dictionary with keys the legal actions and
            # values the game states. Note that if the leaf is terminal
            # there will be no next_states.
            children_states = next_states_function(leaf.game_state)

            # Expand the tree with the new leaf node
            leaf.expand(probs, children_states)

        # Backup the value up the tree.
        backup(nodes, value)

    action_counts = {action: child.N
                     for action, child in starting_node.children.items()}
    return compute_distribution(action_counts)


def self_play(next_states_function, evaluator, initial_state, is_terminal,
              max_iters, c_puct):
    """Plays a game using MCTS to choose actions for both players.

    Parameters
    ----------
    next_states_function: func
        Gives the next states from the given state as a dictionary
        with keys the available actions and values the resulting
        states.
    evaluator: func
        An evaluator.
    initial_state: object
        An initial state to start the game in. This must be compatible
        with next_states_function, but is otherwise arbitrary.
    is_terminal: func
        A function that returns True if the state is terminal and
        otherwise returns False.
    max_iters: int
        Number of iterations to run MCTS for.
    c_puct: float
        Parameter for MCTS.

    Returns
    -------
    game_state_list: list
        A list of game states encountered in the self-play game. Starts
        with the initial state and ends with a terminal state.
    action_probs_list: list
        A list of action probability dictionaries, as returned by MCTS
        each time the algorithm has to take an action. The ith action
        probabilities dictionary corresponds to the ith game_state, and
        action_probs_list has length one less than game_state_list,
        since we don't have to move in a terminal state.
    """
    node = MCTSNode(None, initial_state)

    game_state_list = [node.game_state]
    action_probs_list = []

    while not is_terminal(node.game_state):
        # First run MCTS to compute action probabilities.
        action_probs = mcts(node, evaluator, next_states_function, is_terminal,
                            max_iters, c_puct)

        # Choose the action according to the action probabilities.
        # TODO: * unpacking is fast but only works in > Python3.5 - good idea?
        actions = [*action_probs]
        probs = [*action_probs.values()]
        ix = np.random.choice(len(actions), p=probs)
        action = actions[ix]

        # Play the action
        node = node.children[action]

        # Add the action probabilities and game state to the list.
        action_probs_list.append(action_probs)
        game_state_list.append(node.game_state)

    return game_state_list, action_probs_list


def build_training_data(states_, action_probs_, which_player, utility):
    """Takes a list of states and action probabilities, as returned by
    self_play, and creates training data from this. We build up a list
    consisting of (state, probs, z) tuples, where player is the player
    in state 'state', and 'z' is the utility to 'player' in 'last_state'.

    Parameters
    ----------
    states_: list
        A list of n states, with the last being terminal.
    action_probs_: list
        A list of n-1 dictionaries containing action probabilities. The ith
        dictionary applies to the ith state, representing the probabilities
        returned by self_play of taking each available action in the state.
    which_player: func
        A function taking a state to the player to play in that state.
    utility: func
        A function taking a terminal state to the outcome of the state.

    Returns
    -------
    training_data: list
        A list consisting of (state, probs, z) tuples, where player is the
        player in state 'state', and 'z' is the utility to 'player' in
        'last_state'.
    """

    training_data = []
    # Get the outcome for the game. This should be the last state in states_.
    last_state = states_.pop()
    outcome = utility(last_state)

    # Now action_probs_ and states_ are the same length.
    for state, probs in zip(states_, action_probs_):
        # Get the player in the state, and the value to this player of the
        # terminal state.
        player = which_player(state)
        z = outcome[player]
        training_data.append((state, probs, z))

    return training_data


def self_play_multiple(next_states_function, evaluator, initial_state,
                       is_terminal, utility, which_player, max_iters, c_puct,
                       num_self_play):
    """Combines self_play and build_training_data to generate training data
    given a game and an evaluator.

    Parameters
    ----------
    next_states_function: func
        Gives the next states from the given state as a dictionary with keys the
        available actions and values the resulting states.
    evaluator: func
        An evaluator.
    initial_state: object
        An initial state to start the game in. This must be compatible with
        next_states_function, but is otherwise arbitrary.
    is_terminal: func
        A function that returns True if the state is terminal and otherwise
        returns False.
    utility: func
        A function that returns the utility for terminal states.
    which_player: func
        A function that returns which player is to play in a given state.
    max_iters: int
        Number of iterations to run MCTS for.
    c_puct: float
        Parameter for MCTS.
    num_self_play: int
        Number of games to play in 'self-play'

    Returns
    -------
    training_data: list
        A list of training data tuples. See 'build_training_data'.
    """

    training_data = []
    for i in range(num_self_play):
        game_states_, action_probs_ = self_play(
            next_states_function, evaluator, initial_state, is_terminal,
            max_iters, c_puct)
        training_data.append(build_training_data(
            game_states_, action_probs_, which_player, utility))
    return training_data
