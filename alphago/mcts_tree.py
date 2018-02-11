import numpy as np


class MCTSNode:  # TODO: write a expand class docstring
    """ An MCTSNode stores Q, W, N for the action from the previous node to this
    node, where Q, W, N are from the paper. It also stores a list of the legal
    actions from this node. It also stores the game state corresponding to this
    node.
    """

    __slots__ = "prior_prob Q W N children game_state".split()

    def __init__(self, prior_prob, game_state):
        # We store the prior probability, Q-value, cumulative Q-value (called W)
        # and visit count (called N) from the parent node. These are all
        # scalars. In particular, prior_prob is the prior probability of
        # visiting this node from the parent node.
        self.prior_prob = prior_prob
        self.Q = 0.0
        self.W = 0.0
        self.N = 0.0

        # children will hold the children of the node, once the node is
        # expanded. It is a dictionary from actions to MCTSNodes.
        self.children = {}

        # We also store the game state in the node. This is sufficient for the
        # Game object to understand the game -- in particular, the Game object
        # can return the list of legal actions from this game state.
        self.game_state = game_state

    def is_leaf(self):
        """Returns whether or not the node in the tree is a leaf."""
        return len(self.children) == 0

    def expand(self, prior_probs, children_states):
        """Expands the tree at the leaf node with the given probabilities.

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
        A dictionary mapping each child node to: Q(s,a) + U(s,a),
    where U(s,a) = c_puct * prior_prob * sqrt(sum(action_counts)) / (1 +
    action_count).
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
    """ Given the sequence of nodes (ending in the new expanded node) from
    the game tree, propagate back the Q-values and action counts.
    """
    for node in nodes:
        # Increment the visit count
        node.N += 1.0

        # Update the cumulative and mean action values
        node.W += v
        node.Q = float(node.W) / float(node.N)


def compute_distribution(d):
    """Calculate a probability distribution with probabilities proportional to
    the values in a dictionary

    Parameters
    ----------
    d: dict
        A dictionary with values equal to positive floats.

    Returns
    -------
    prob_distribution: dict:
        A probability distribution proportional to the values of d, given as a
        dictionary with keys equal to those of d and values the probability
        corresponding to the value.
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
        with keys the actions in the state and value given by the estimate of the
        value of the state.
    next_states: func
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
        # First select a leaf node from the MCTS tree. This actually returns
        # all nodes and actions taken, with the length of actions being one
        # less than the length of nodes. The last element of nodes is the
        # leaf node.
        nodes, actions = select(starting_node, c_puct)
        leaf = nodes[-1]

        # Evaluate the leaf node to get the probabilities and value
        # according to the net.
        probs, value = evaluator(leaf.game_state)

        if not is_terminal(leaf.game_state):
            # Compute the next possible states from the leaf node. This returns a
            # dictionary with keys the legal actions and values the game states.
            # Note that if the leaf is terminal, there will be no next_states.
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
    node = MCTSNode(None, initial_state)

    game_state_list = [node.game_state]
    action_probs_list = []

    while not is_terminal(node.game_state) > 0:
        # First run MCTS to compute action probabilities.
        action_probs = mcts(node, evaluator, next_states_function, is_terminal,
                            max_iters, c_puct)

        # Choose the action according to the action probabilities.
        actions = [a for a in action_probs]
        probs = [p for a, p in action_probs.items()]
        print(actions)
        ix = np.random.choice(len(actions), p=probs)
        action = actions[ix]

        # Play the action
        node = node.children[action]

        # Add the action probabilities and game state to the list.
        action_probs_list.append(action_probs)
        game_state_list.append(node.game_state)

    return game_state_list, action_probs_list
