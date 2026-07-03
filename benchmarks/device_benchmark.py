"""CPU-vs-MPS timing sweep for the AlphaGo estimators.

Times single-state ``__call__`` inference and batched ``train_step`` for the
ConnectFour and NAC 3x3 nets on CPU and (when available) MPS, using
synchronize-bracketed warmup timing so async MPS dispatch is not
under-measured.

Run as a script::

    uv run python benchmarks/device_benchmark.py
"""

import time

import numpy as np
import torch

from alphago.estimator import ConnectFourNet, NACNetEstimator
from alphago.games import ConnectFour, NoughtsAndCrosses

# Sweep configuration: CF 6x7 + NAC 3x3; widen the batch set if no
# crossover appears.
NETS = [
    ("ConnectFourNet", ConnectFourNet, ConnectFour),
    ("NACNetEstimator", NACNetEstimator, NoughtsAndCrosses),
]
BATCH_SIZES = [32, 256, 2048]
WARMUP = 5
REPEATS = 50


def time_op(fn, device: torch.device, warmup: int = 5, repeats: int = 50) -> float:
    """Mean seconds per call of ``fn`` on ``device``, async-safe for MPS."""
    is_mps = device.type == "mps"
    for _ in range(warmup):  # warmup: trigger graph/kernel compilation & caching
        fn()
    if is_mps:
        torch.mps.synchronize()  # ensure warmup work is flushed before timing
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if is_mps:
        torch.mps.synchronize()  # WAIT for all enqueued GPU work before stopping clock
    return (time.perf_counter() - start) / repeats


def empty_board_state(est) -> np.ndarray:
    """Empty-board grid the estimator's net consumes via ``_state_to_vector``.

    The net takes a flat ``in_channels * board_h * board_w`` board grid;
    ConnectFour's game state already IS this flat board, while NAC's game state
    is a bitboard, so we build the net's grid representation of the empty board.
    The input flows through the real ``__call__`` / ``train_step`` path, not a
    fabricated raw tensor that would trip the forward shape guard.
    """
    cfg = est.cfg
    flat = cfg.in_channels * cfg.board_h * cfg.board_w
    return np.zeros(flat, dtype=np.float32)


def synthesize_batch(est, game, batch_size: int) -> list:
    """Build a ``batch_size``-row ``(state, uniform_pi, z=0.0)`` training batch."""
    state = empty_board_state(est)
    n_actions = len(game.action_space)
    pi = np.ones(n_actions) / n_actions
    return [(state, pi, 0.0) for _ in range(batch_size)]


def run_sweep() -> list[dict]:
    """Run the net x device x workload x batch timing sweep.

    Returns:
        A list of result rows, each a dict with keys ``net``, ``device``,
        ``workload``, ``batch`` and ``seconds_per_call``.
    """
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    else:
        print("MPS unavailable -- timing CPU only.")

    rows: list[dict] = []
    for net_name, net_cls, game_cls in NETS:
        game = game_cls()
        for dev_str in devices:
            est = net_cls(action_indices=game.action_indices, device=dev_str)
            dev = est.device

            # Single-state __call__ inference (MCTS path).
            state = empty_board_state(est)
            t_infer = time_op(
                lambda est=est, state=state: est(state),
                dev,
                warmup=WARMUP,
                repeats=REPEATS,
            )
            rows.append(
                {
                    "net": net_name,
                    "device": dev_str,
                    "workload": "infer",
                    "batch": 1,
                    "seconds_per_call": t_infer,
                }
            )

            # Batched train_step (exercises SGD.step) across the batch sweep.
            for bs in BATCH_SIZES:
                batch = synthesize_batch(est, game, bs)
                t_train = time_op(
                    lambda est=est, batch=batch: est.train_step(batch),
                    dev,
                    warmup=WARMUP,
                    repeats=REPEATS,
                )
                rows.append(
                    {
                        "net": net_name,
                        "device": dev_str,
                        "workload": "train_step",
                        "batch": bs,
                        "seconds_per_call": t_train,
                    }
                )
    return rows


def print_table(rows: list[dict]) -> None:
    """Print the timing rows as a fixed-width table (seconds per call)."""
    header = f"{'net':<16}{'device':<8}{'workload':<12}{'batch':>6}  {'s/call':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['net']:<16}{r['device']:<8}{r['workload']:<12}"
            f"{r['batch']:>6}  {r['seconds_per_call']:>12.6f}"
        )


def main() -> None:
    """Run the sweep and print the timing table."""
    print(f"torch {torch.__version__} | warmup={WARMUP} repeats={REPEATS}")
    rows = run_sweep()
    print_table(rows)


if __name__ == "__main__":
    main()
