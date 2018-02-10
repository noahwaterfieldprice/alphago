import numpy as np


class MCTSNode:
    """ An MCTSNode stores Q, W, N for the action from the previous node to this
    node, where Q, W, N are from the paper. It also stores a list of the legal
    actions from this node. It also stores the game state corresponding to this
    node.
    """
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


def compute_ucb(action_values, prior_probs, action_counts, c_puct):
    """ Returns a dictionary mapping each child node to: Q(s,a) + U(s,a),
    where U(s,a) = c_puct * prior_prob * sqrt(sum(action_counts)) / (1 +
    action_count).
    """
    num = np.sqrt(sum(action_counts.values()))
    assert num > 0
    return {
        k: action_values[k] + prior_probs[k] / float(1 + action_counts[k]) *
        c_puct * num for k in action_values
    }


def select(root, max_steps, c_puct):
    """ Selects a new leaf node. Returns all nodes and actions on the path
    to the new leaf node. Terminates after max_steps.
    """
    node = root
    num_steps = 0

    actions = []
    nodes = [node]

    while not is_leaf(node) and num_steps < max_steps:
        # The node is not a leaf, so has children. We select the one with
        # largest upper confidence bound.
        prior_probs = {a: child.prior_prob for a, child in node.children.items()}
        action_values = {a: child.Q for a, child in node.children.items()}
        action_counts = {a: child.N for a, child in node.children.items()}

        # Compute the upper confidence bound values
        ucb = compute_ucb(action_values, prior_probs, action_counts, c_puct)

        # Take action with largest ucb
        action = max(ucb, key=ucb.get)
        node = node.children[action]

        # Append action to the list of actions, and node to nodes
        nodes.append(node)
        actions.append(action)
        num_steps += 1

    # We have reached a leaf node, or the maximum number of steps.
    # Return the sequence of nodes and actions.
    return nodes, actions


def is_leaf(node):
    """ Returns whether or not the node in the tree is a leaf.
    """
    return len(node.children) == 0


def expand(leaf, prior_probs, children_states):
    """ Expands the tree at the leaf node with the given probabilities. Note
    that prior_probs is a dictionary with keys the children of the leaf
    node and values the prior probability of visiting the child. Returns
    the list of expanded child nodes.
    """
    assert is_leaf(leaf)

    # Initialise the relevant data for each child of the leaf node
    leaf.children = {
        a: MCTSNode(prior_probs[a], children_states[a]) for a in children_states
    }


def backup(nodes, v):
    """ Given the sequence of nodes (ending in the new expanded node) from
    the game tree, backup the value 'v'.
    """
    for node in nodes:
        # Increment the visit count
        node.N += 1.0

        # Update the cumulative and mean action values
        node.W += v
        node.Q = float(node.W) / float(node.N)
