/*
 * This file is part of Connect4 Game Solver <http://connect4.gamesolver.org>
 * Copyright (C) 2007 Pascal Pons <contact@gamesolver.org>
 *
 * Connect4 Game Solver is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Connect4 Game Solver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Connect4 Game Solver. If not, see <http://www.gnu.org/licenses/>.
 */

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
int sign(int x) {
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
}

int main(int argc, char** argv) {
  Solver solver;

  if (argc < 3) {
    std::cout << "Input and output file names must be specified. Run as 'c4solver input_name output_name'" << std::endl;
    return 0;
  }
  std::string input_file_name = argv[1];
  std::string output_file_name = argv[2];
  std::cout << "Input file: " << input_file_name << std::endl;
  std::cout << "Output file: " << output_file_name << std::endl;

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
      int max_score = -10000, best_move = -1;
      for (int move = 1; move <= 7; move++) {
        Position P;
        P.play(state);
        int score;
        if (P.public_isWinningMove(move - 1)) {
          score = 100000;
        } else {
          Position P;
          solver.reset();
          std::string next_state = state + std::to_string(move);
          unsigned long next_state_length = P.play(next_state);
          if (next_state_length != next_state.size()) {
            // illegal move
            continue;
          }
          score = -solver.solve(P, false);
          if (score > max_score) {
            max_score = score;
            best_move = move;
          }
        }
      }
      output_file << state << " " << best_move << " " << sign(max_score) << std::endl;
      count += 1;
    }
    input_file.close();
    output_file.close();
  }
}




