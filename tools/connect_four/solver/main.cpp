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

int main() {
  Solver solver;

  std::ifstream input_file ("connect_four_states.txt");
  std::ofstream output_file ("connect_four_solved_states.txt");
  std::string state;
  if (input_file.is_open() && output_file.is_open())
  {
    while (std::getline(input_file, state))
    { int max_score = -10000, best_move;
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
    }
    input_file.close();
    output_file.close();
  }
}




