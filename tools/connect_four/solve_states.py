import argparse

from tqdm import tqdm

from alphago.games.connect_four import optimal_moves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='File to read input from.')
    parser.add_argument('output_file', help='File name to output result to.')

    args = parser.parse_args()

    # First read in the states.
    states = []
    with open(args.input_file, 'r') as f:
        for line in f:
            state = map(int, line.strip())
            states.append(list(state))

    # Now solve the states
    with open(args.output_file, 'w') as f:
        for state in tqdm(states):
            state_str = ''.join(map(str, state))
            value, moves = optimal_moves(state)
            value_str = str(value)
            moves_str = ' '.join(map(str, moves))
            line = ' '.join([state_str, value_str, moves_str]) + '\n'
            f.write(line)
