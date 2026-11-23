"""Fast tests on a synthetic environment (no data download needed).

Run with:  pytest -q
"""

import numpy as np

from mobandit import config as C
from mobandit.algorithms import (run_etc_scalar, run_pareto_ts, run_pareto_ucb1,
                                 run_ts_scalar, run_ucb_scalar)
from mobandit.env import make_synthetic_env
from mobandit.evaluation import (eval_weight_grid, pareto_identification_f1,
                                 run_experiment, scalarized_regret, selection_regret)
from mobandit.pareto import _pareto_front_numpy, pareto_front


def test_pareto_front_simple():
    P = np.array([[1, 1, 1], [2, 2, 2], [3, 0, 0], [2, 2, 2]], dtype=float)
    # [1,1,1] is dominated; the two identical [2,2,2] rows do not dominate each other
    assert pareto_front(P).tolist() == [False, True, True, True]


def test_pareto_backends_agree():
    rng = np.random.default_rng(0)
    for _ in range(200):
        P = rng.integers(0, 4, size=(rng.integers(2, 30), 3)).astype(float)
        assert np.array_equal(pareto_front(P), _pareto_front_numpy(P))


def test_weight_grid():
    grid = eval_weight_grid()
    assert len(grid) == 36
    assert all(abs(w.sum() - 1) < 1e-9 and w.min() >= 0.1 - 1e-9 for w in grid)


def test_algorithms_run_and_learn():
    env = make_synthetic_env(n_arms=20, seed=1)
    rng = np.random.default_rng(1)
    T = 3000
    for fn in (run_ucb_scalar, run_ts_scalar, run_etc_scalar):
        chosen, counts, _ = fn(env, C.WEIGHTS["balanced"], T, rng)
        assert counts.sum() == T and len(chosen) == T
    # UCB's regret per round should shrink over time
    chosen, _, _ = run_ucb_scalar(env, C.WEIGHTS["balanced"], T, rng)
    reg = scalarized_regret(chosen, env, C.WEIGHTS["balanced"])
    assert reg[-1] - reg[-T // 4] < reg[T // 4]
    for fn in (run_pareto_ucb1, run_pareto_ts):
        _, counts, means = fn(env, T, rng)
        assert counts.sum() == T
        assert 0.0 <= pareto_identification_f1(env, means)["f1"] <= 1.0


def test_selection_regret_zero_for_best_arm():
    env = make_synthetic_env(n_arms=15, seed=2)
    w = C.WEIGHTS["offense"]
    best = int(np.argmax(env.truth_means @ w))
    assert selection_regret(env, [best], w) == 0.0


def test_experiment_is_reproducible():
    env = make_synthetic_env(n_arms=15, seed=3)
    a = run_experiment(env, 1500, np.random.default_rng(0), repeats=2)
    b = run_experiment(env, 1500, np.random.default_rng(0), repeats=2)
    assert a["report"] == b["report"]
    assert set(a["report"]) == {f"assume {k}" for k in C.WEIGHTS} | {"Pareto UCB1", "Pareto TS"}
