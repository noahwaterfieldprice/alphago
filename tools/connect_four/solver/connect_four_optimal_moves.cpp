#include "solver.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using namespace GameSolver::Connect4;

/*
 * Given a file with states for connect four, one per line, this computes the
 * optimal moves for that position. The states should be a string of numbers in
 * the range 1 up to 7, denoting the action sequence taken to get to that
 * state. We print the state and then, on the same line, the optimal
 * actions, separated by spaces.
 * For example, if the state is '45', then 1 played in column 4 and 2 played in
 * column 5. The optimal moves are 1, 2 for player 1. Thus we print '45 1 2'.
 */

std::string solve_single_state(std::string state) {
  // Solves the state and returns a string consisting of the state, the value
  // of the state, and the optimal moves, all separated by spaces.
  Solver solver;

  std::pair<std::vector<int>, int> optimal_moves = solver.optimal_moves(state);
  std::vector<int> moves = optimal_moves.first;
  int value = optimal_moves.second;
  std::string moves_str = solver.vector_to_string(moves);

  std::string value_str = std::to_string(solver.sign(value));

  std::stringstream ss;
  ss << state << "," << value_str << "," << moves_str;
  return ss.str();
}


void solve_multiple_states(std::string input_file_name,
                           std::string output_file_name) {
  Solver solver;

  std::ifstream input_file (input_file_name);
  std::ofstream output_file (output_file_name);
  std::string state;

  int count = 0;
  if (input_file.is_open() && output_file.is_open())
  {
    while (std::getline(input_file, state))
    {
      if (count % 1000 == 0)
        std::cout << "Completed " << count << std::endl;

      std::string solution_str = solve_single_state(state);
      output_file << solution_str << std::endl;
      count += 1;
    }
    input_file.close();
    output_file.close();
  }
  return;
}


int main(int argc, char** argv) {

  if (argc == 2) {
    // Solve single state

    std::string state = argv[1];
    std::string solution_str = solve_single_state(state);

    std::cout << solution_str << std::endl;

    return 0;
  } else if (argc == 3) {
    // Solve all states in input_file and output them to output_file.
    std::string input_file_name = argv[1];
    std::string output_file_name = argv[2];
    std::cout << "Input file: " << input_file_name << std::endl;
    std::cout << "Output file: " << output_file_name << std::endl;

    solve_multiple_states(input_file_name, output_file_name);

    return 0;
  } else {
    // Invalid input.
    std::cout << "Either run as 'connect_four_optimal_moves <state>', or as 'connect_four_optimal_moves <input_file> <output_file>'.";
    return 0;
  }
}
