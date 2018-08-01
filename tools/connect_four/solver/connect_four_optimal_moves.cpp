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

int main(int argc, char** argv) {
  Solver solver;

  if (argc < 2) {
    std::cout << "Input and output file names must be specified. Run as 'connect_four_solve_state state'" << std::endl;
    return 0;
  }
  std::string state = argv[1];

  std::vector<int> optimal_moves = solver.optimal_moves(state);
  std::string optimal_moves_str = solver.vector_to_string(optimal_moves);

  std::cout << state << " " << optimal_moves_str << std::endl;
}
