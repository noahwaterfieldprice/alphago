#include "solver.hpp"
#include <iostream>
#include <fstream>

using namespace GameSolver::Connect4;

/*
 * Main function.
 * Reads Connect 4 positions, line by line, from standard input
 * and writes one line per position to standard output containing:
 *  - score of the position
 *  - number of nodes explored
 *  - time spent in microsecond to solve the position.
 *
 *  Any invalid position (invalid sequence of move, or already won game)
 *  will generate an error message to standard error and an empty line to standard output.
 */

int main(int argc, char** argv) {
  Solver solver;

  if (argc < 2) {
    std::cout << "Input and output file names must be specified. Run as 'connect_four_solve_state state'" << std::endl;
    return 0;
  }
  std::string state = argv[1];

  int optimal_move, optimal_score;
  std::pair<int, int> result = solver.optimal_move(state);
  optimal_move = result.first;
  optimal_score = result.second;

  std::cout << state << " " << optimal_move << " " <<
      solver.sign(optimal_score) << std::endl;
}
