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

#include <cassert>
#include "solver.hpp"
#include "MoveSorter.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

using namespace GameSolver::Connect4;

namespace GameSolver { namespace Connect4 {

    /**
     * Recursively score connect 4 position using negamax variant of alpha-beta algorithm.
     * @param: position to evaluate, this function assumes nobody already won and 
     *         current player cannot win next move. This has to be checked before
     * @param: alpha < beta, a score window within which we are evaluating the position.
     *
     * @return the exact score, an upper or lower bound score depending of the case:
     * - if actual score of position <= alpha then actual score <= return value <= alpha
     * - if actual score of position >= beta then beta <= return value <= actual score
     * - if alpha <= actual score <= beta then return value = actual score
     */
    int Solver::negamax(const Position &P, int alpha, int beta) {
      assert(alpha < beta);
      assert(!P.canWinNext());

      nodeCount++; // increment counter of explored nodes

      uint64_t possible = P.possibleNonLosingMoves();
      if(possible == 0)     // if no possible non losing move, opponent wins next move
        return -(Position::WIDTH*Position::HEIGHT - P.nbMoves())/2;

      if(P.nbMoves() >= Position::WIDTH*Position::HEIGHT - 2) // check for draw game
        return 0; 

      int min = -(Position::WIDTH*Position::HEIGHT-2 - P.nbMoves())/2;	// lower bound of score as opponent cannot win next move
      if(alpha < min) {
        alpha = min;                     // there is no need to keep beta above our max possible score.
        if(alpha >= beta) return alpha;  // prune the exploration if the [alpha;beta] window is empty.
      }

      int max = (Position::WIDTH*Position::HEIGHT-1 - P.nbMoves())/2;	// upper bound of our score as we cannot win immediately
      if(beta > max) {
        beta = max;                     // there is no need to keep beta above our max possible score.
        if(alpha >= beta) return beta;  // prune the exploration if the [alpha;beta] window is empty.
      }

      const uint64_t key = P.key();
      if(int val = transTable.get(key)) {
        if(val > Position::MAX_SCORE - Position::MIN_SCORE + 1) { // we have an lower bound
          min = val + 2*Position::MIN_SCORE - Position::MAX_SCORE - 2;
          if(alpha < min) {
            alpha = min;                     // there is no need to keep beta above our max possible score.
            if(alpha >= beta) return alpha;  // prune the exploration if the [alpha;beta] window is empty.
          }
        }
        else { // we have an upper bound
          max = val + Position::MIN_SCORE - 1;
          if(beta > max) {
            beta = max;                     // there is no need to keep beta above our max possible score.
            if(alpha >= beta) return beta;  // prune the exploration if the [alpha;beta] window is empty.
          }
        }
      }

      MoveSorter moves;

      for(int i = Position::WIDTH; i--; )
         if(uint64_t move = possible & Position::column_mask(columnOrder[i]))
           moves.add(move, P.moveScore(move));

      while(uint64_t next = moves.getNext()) {
          Position P2(P);
          P2.play(next);  // It's opponent turn in P2 position after current player plays x column.
          int score = -negamax(P2, -beta, -alpha); // explore opponent's score within [-beta;-alpha] windows:
          // no need to have good precision for score better than beta (opponent's score worse than -beta)
          // no need to check for score worse than alpha (opponent's score worse better than -alpha)

          if(score >= beta) {
            transTable.put(key, score + Position::MAX_SCORE - 2*Position::MIN_SCORE + 2); // save the lower bound of the position
            return score;  // prune the exploration if we find a possible move better than what we were looking for.
          }
          if(score > alpha) alpha = score; // reduce the [alpha;beta] window for next exploration, as we only 
          // need to search for a position that is better than the best so far.
        }

      transTable.put(key, alpha - Position::MIN_SCORE + 1); // save the upper bound of the position
      return alpha;
    }




    int Solver::solve(const Position &P, bool weak)
    {
      if(P.canWinNext()) // check if win in one move as the Negamax function does not support this case.
        return (Position::WIDTH*Position::HEIGHT+1 - P.nbMoves())/2;
      int min = -(Position::WIDTH*Position::HEIGHT - P.nbMoves())/2;
      int max = (Position::WIDTH*Position::HEIGHT+1 - P.nbMoves())/2;
      if(weak) {
        min = -1;
        max = 1;
      }

      while(min < max) {                    // iteratively narrow the min-max exploration window
        int med = min + (max - min)/2;
        if(med <= 0 && min/2 < med) med = min/2;
        else if(med >= 0 && max/2 > med) med = max/2;
        int r = negamax(P, med, med + 1);   // use a null depth window to know if the actual score is greater or smaller than med
        if(r <= med) max = r;
        else min = r;
      }
      return min;
    }
    
    // Constructor
    Solver::Solver() : nodeCount{0} {
      reset();
      for(int i = 0; i < Position::WIDTH; i++)
        columnOrder[i] = Position::WIDTH/2 + (1-2*(i%2))*(i+1)/2;   
      // initialize the column exploration order, starting with center columns
      // example for WIDTH=7: columnOrder = {3, 4, 2, 5, 1, 6, 0}
    }

    std::pair<int, int> Solver::optimal_move(std::string state)
    {
      // Returns the optimal move in the position. Uses the weak solver.
      int max_score = -10000, best_move = -1;
      for (int move = 1; move <= 7; move++) {
        Position P;
        P.play(state);
        int score;
        if (P.public_isWinningMove(move - 1)) {
          score = 100000;
        } else {
          Position P;
          reset();
          std::string next_state = state + std::to_string(move);
          unsigned long next_state_length = P.play(next_state);
          if (next_state_length != next_state.size()) {
            // illegal move
            continue;
          }
          score = -solve(P, false);
          if (score > max_score) {
            max_score = score;
            best_move = move;
          }
        }
      }
      return std::make_pair(best_move, max_score);
    }

    int Solver::sign(int x) {
      if (x > 0) return 1;
      if (x < 0) return -1;
      return 0;
    }


    std::string Solver::vector_to_string(std::vector<int> v) {
      std::stringstream v_str;
      std::copy(v.begin(), v.end(),
                std::ostream_iterator<int>(v_str, " "));
      return v_str.str();
    }


    std::vector<int> Solver::move_scores(std::string state)
    {
      // Computes the scores for each move in the position.
      std::vector<int> scores;
      int illegal_move_score = -10000;
      for (int move = 1; move <= 7; move++) {
        Position P;
        P.play(state);
        int score;
        if (P.public_isWinningMove(move - 1)) {
          score = 100000;
        } else {
          Position P;
          reset();
          std::string next_state = state + std::to_string(move);
          unsigned long next_state_length = P.play(next_state);
          if (next_state_length != next_state.size()) {
            // illegal move
            score = illegal_move_score;
          } else {
            // The move is legal, so solve the position.
            score = -solve(P, true);
          }
        }
        scores.push_back(score);
      }
      return scores;
    }


    bool Solver::is_legal_move(std::string state, int move)
    {
      // Returns whether the move is legal in the state. The moves are indexed
      // 1 to 7.
      Position P;
      P.play(state);
      return P.public_canPlay(move - 1);
    }


    std::vector<int> Solver::optimal_moves(std::string state)
    {
      // Returns the optimal moves in the position. Uses the weak solver. The
      // resulting array contains the moves in the range 1 up to 7 (i.e.
      // starting from 1 rather than 0. This returns the moves whose signs
      // equal the sign of the max score.
      // Note that if the max score is negative (all moves are losing), then we
      // return all moves.

      // Compute the scores for each move.
      std::vector<int> scores = Solver::move_scores(state);

      // Compute the max score.
      int max_score = -10000;
      for (int i = 0; i < 7; i++) {
        if (scores[i] > max_score) {
          max_score = scores[i];
        }
      }

      // Compute the moves whose scores have the same sign as the max score.
      std::vector<int> best_moves;
      for (int i = 0; i < 7; i++) {
        if ((Solver::sign(scores[i]) == Solver::sign(max_score)) &&
            Solver::is_legal_move(state, i + 1)) {
          best_moves.push_back(i + 1);
        }
      }
      return best_moves;
    }

}} // namespace GameSolver::Connect4
