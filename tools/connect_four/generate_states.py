import argparse
from typing import Iterable
import random

from tqdm import tqdm

from alphago.games.connect_four import action_list_to_state, ConnectFour


def generate_states(num_states, min_length=1):
    """Generates num_states random connect four states, uniformly in length,
    and then uniformly across states of that length.

    Parameters
    ----------
    num_states: int
        The number of states to generate.

    Returns
    -------
    states: set
        A set of states.
    """
    states = set()
    game = ConnectFour()
    with tqdm(total=num_states) as pbar:
        while len(states) <= num_states:
            cols = [i for i in range(7)] * 6
            random.shuffle(cols)
            l = random.randint(min_length, 43)
            state = cols[:l]
            state_array = action_list_to_state(state)

            if not game.is_terminal(state_array):
                states.add(tuple(x + 1 for x in state))
                pbar.update(1)

    return states


def combine_connect_four_states_files(input_filepaths: Iterable[str],
                                      output_filepath: str) -> None:
    """Combine the results of multiple solved connect four games into a single file."""
    solutions = set()
    for filepath in input_filepaths:
        with open(filepath) as input_file:
            for line in input_file.readlines():
                solution = tuple(map(int, line.split()))
                solutions.add(solution)

    sorted_solutions = sorted(solutions, key=lambda x: x[0])
    with open(output_filepath, 'w') as output_file:
        lines = "\n".join([" ".join(map(str, solution)) for solution in sorted_solutions])
        output_file.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', help='File name to output result to')
    parser.add_argument('num_states', help='Number of states to generate')
    parser.add_argument('min_moves', help='The minimum number of moves for each sequence')

    args = parser.parse_args()

    min_moves = int(args.min_moves)
    num_states = int(args.num_states)
    states = generate_states(num_states, min_moves)

    with open(args.output_file, 'w') as f:
        string_states = [''.join((str(x) for x in state)) for state in states]
        f.write('\n'.join(string_states))
