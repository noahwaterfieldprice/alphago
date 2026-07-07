"""NAC behavioral gates: the migrated pipeline provably learns.

One seeded ~2.5-minute NAC 3x3 training run (the ``nac_agent_run`` session
fixture) produces the agent AND the logs that satisfy three E2E signals, then
the same champion is played head-to-head against ``RandomPlayer`` and
``OptimalPlayer``. Every test is ``@pytest.mark.gate`` so the default fast suite
(``-m 'not gate'``) skips them and stays under two minutes; run them explicitly
with ``uv run pytest -m gate``.

The bars are defended: a post-tuning failure here is a discovered
pipeline bug, never a threshold to relax.
"""

import math

import pytest

from alphago.device import enable_cpu_determinism
from alphago.evaluator import evaluate
from alphago.player import MCTSPlayer, OptimalPlayer, RandomPlayer


def _play_alternating_colors(run, make_opponent, games_per_color):
    """Play the champion vs a fresh opponent in both colors; aggregate results.

    Mirrors ``run_gauntlet``'s two-position pairing: the agent plays
    ``games_per_color`` games as player 1 and ``games_per_color`` as player 2,
    swapping the ``{1: ..., 2: ...}`` dict between batches. ``evaluate`` returns
    results from player 1's perspective (``{1: p1 wins, -1: p1 losses,
    0: draws}``), so agent wins/losses are read off the correct key per color.

    Args:
        run: The shared ``NacAgentRun`` fixture value.
        make_opponent: A zero-arg factory returning a fresh opponent ``Player``.
        games_per_color: Games played with the agent in each of the two colors.

    Returns:
        A ``(wins, losses, draws)`` tuple aggregated over both colors.
    """
    game = run.game

    def make_agent():
        return MCTSPlayer(
            game,
            run.champion.create_estimate_fn(),
            run.mcts_iters,
            run.cfg.mcts.c_puct,
            tau=0.01,
        )

    wins = losses = draws = 0
    for agent_position in (1, 2):
        opponent_position = 2 if agent_position == 1 else 1
        players = {agent_position: make_agent(), opponent_position: make_opponent()}
        results, _ = evaluate(game, players, games_per_color, verbose=False)
        if agent_position == 1:
            wins += results[1]
            losses += results[-1]
        else:
            wins += results[-1]
            losses += results[1]
        draws += results[0]
    return wins, losses, draws


@pytest.mark.gate
def test_e2e_three_signals(nac_agent_run):
    """One seeded run is NaN-free, fires champion replacement, and trends down.

    Three signals from a SINGLE run:
      1. every ``loss/total`` is finite — the run completing under the MCTS
         fail-loud guards is itself the NaN-free proof;
      2. ``eval/success_rate`` crosses ``win_rate`` at least once — champion
         replacement fired;
      3. the mean of the last quarter of losses is below the mean of the first
         quarter — a windowed downward trend (never strict monotonicity).
    """
    records = nac_agent_run.records
    win_rate = nac_agent_run.cfg.training.win_rate

    # (1) NaN-free losses.
    losses = [value for _, value in records["loss/total"]]
    assert losses, "no loss/total scalars were recorded"
    assert all(math.isfinite(value) for value in losses), (
        "loss/total has non-finite values"
    )

    # (2) Champion replacement fired at least once.
    success_rates = [value for _, value in records.get("eval/success_rate", [])]
    assert any(rate > win_rate for rate in success_rates), (
        f"champion never fired: no eval/success_rate exceeded win_rate={win_rate}"
    )

    # (3) Windowed downward loss trend (quarter vs quarter).
    q = max(1, len(losses) // 4)
    first_quarter_mean = sum(losses[:q]) / q
    last_quarter_mean = sum(losses[-q:]) / q
    assert last_quarter_mean < first_quarter_mean, (
        f"loss did not trend down: last-quarter mean {last_quarter_mean:.4f} "
        f">= first-quarter mean {first_quarter_mean:.4f}"
    )


@pytest.mark.gate
def test_vs_random(nac_agent_run):
    """The champion wins >= 80% of 50 alternating-color games vs Random."""
    enable_cpu_determinism(nac_agent_run.cfg.seed)
    wins, _losses, _draws = _play_alternating_colors(
        nac_agent_run, lambda: RandomPlayer(nac_agent_run.game), games_per_color=25
    )
    win_rate = wins / 50
    assert win_rate >= 0.80, (
        f"champion won only {wins}/50 ({win_rate:.0%}) vs RandomPlayer; expected >= 80%"
    )


@pytest.mark.gate
def test_vs_optimal(nac_agent_run):
    """The champion loses 0 of 20 alternating-color games vs Optimal."""
    enable_cpu_determinism(nac_agent_run.cfg.seed)
    _wins, losses, _draws = _play_alternating_colors(
        nac_agent_run, lambda: OptimalPlayer(nac_agent_run.game), games_per_color=10
    )
    assert losses == 0, (
        f"champion lost {losses}/20 games vs OptimalPlayer; expected 0 losses"
    )
