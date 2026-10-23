"""Hyperparameter tuning on a validation split carved out of TRAIN.

The rule this module exists to enforce: the test split is never seen while
choosing anything. `tune_content_weight` takes the training frame and
nothing else of that kind, so there is no argument through which test data
could leak in. The validation split reuses the outer split's strategy
(temporal cut or leave-one-out) one level down, so validation is to
train_inner what test is to train.

Components are fit once on train_inner and only re-scored per candidate
weight; the weight changes the fusion, not the fitted models.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .data import Dataset
from .evaluate import evaluate
from .recommenders import HybridRecommender
from .recommenders.base import Recommender
from .split import filter_test_to_known_users, global_temporal_split, leave_one_out_split

DEFAULT_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)

# Grids searched by run_eval --tune-models (documented in the README). The iALS alpha range was
# widened below the original default (20) after a validation-only pilot showed validation NDCG
# falling monotonically from alpha=2 to alpha=80 with the best tested value on the lower edge.
IALS_GRID = {
    "factors": [32, 64, 128],
    "regularization": [0.1, 1.0, 10.0],
    "alpha": [0.25, 0.5, 1.0, 2.0, 5.0],
}
# k was first searched up to 400, and the best value sat on that upper edge on both splits, so the
# range was extended up to 3883 = every neighbour kept (ml-1m's catalogue size; a larger k than the
# catalogue just means "no pruning").
# Stage 2 for iALS. Stage 1 put regularization on its upper edge on both splits (and `factors` on
# its lower edge on temporal). Stronger regularization is what makes larger models viable, so the
# extension varies factors and regularization together. Stage-1 trials are reused, not refit.
IALS_GRID_STAGE2 = {
    "factors": [16, 32, 64, 128],
    "regularization": [10.0, 30.0, 100.0, 300.0],
    "alpha": [0.5, 1.0, 2.0],
}
KNN_GRID = {"k": [50, 100, 200, 400, 800, 1600, 3883], "shrink": [0.0, 10.0, 50.0, 200.0]}


@dataclass
class TuningResult:
    best_weight: float
    best_ndcg: float
    curve: list[dict]  # [{content_weight, ndcg, recall, coverage}, ...] in grid order
    n_validation_users: int


def carve_validation(
    train: pd.DataFrame, strategy: str, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train into (train_inner, validation) using the same strategy as the outer split."""
    if strategy == "temporal":
        return global_temporal_split(train, val_frac)
    if strategy == "loo":
        return leave_one_out_split(train)
    raise ValueError(f"unknown split strategy {strategy!r}")


def tune_content_weight(
    dataset: Dataset,
    train: pd.DataFrame,
    strategy: str,
    *,
    grid: Sequence[float] = DEFAULT_GRID,
    k: int = 10,
    min_train: int = 5,
    val_frac: float = 0.2,
    cold_threshold: int = 3,
    make_content: Callable[[], Recommender],
    make_ials: Callable[[], Recommender],
) -> TuningResult:
    """Pick the content weight w in the fusion w*content + (1-w)*iALS by validation NDCG@k.

    Ties go to the smaller content weight: if the data cannot tell two weights apart, prefer
    the simpler model (w = 0 is plain iALS).
    """
    inner, val = carve_validation(train, strategy, val_frac)
    val = filter_test_to_known_users(inner, val, min_train)

    content = make_content().fit(dataset, inner)
    ials = make_ials().fit(dataset, inner)

    curve = []
    for w in grid:
        hybrid = HybridRecommender(
            {"content": (content, w), "ials": (ials, 1.0 - w)}, cold_threshold=cold_threshold
        ).fit(dataset, inner)
        m = evaluate(hybrid, dataset, inner, val, k=k)
        curve.append({
            "content_weight": float(w), "ndcg": m.ndcg, "recall": m.recall, "coverage": m.coverage,
        })

    best = max(curve, key=lambda c: (c["ndcg"], -c["content_weight"]))
    return TuningResult(
        best_weight=best["content_weight"],
        best_ndcg=best["ndcg"],
        curve=curve,
        n_validation_users=int(val["user_idx"].nunique()),
    )


@dataclass
class ModelTuning:
    """Outcome of a grid search for one model, scored on the validation split only."""

    best_params: dict[str, Any]
    best_ndcg: float
    trials: list[dict]  # [{params, ndcg, recall, fit_seconds}, ...] in grid order
    grid: dict[str, list]
    n_validation_users: int
    # Params whose best value is the smallest or largest searched value. The true optimum may lie
    # outside the grid; callers must say so instead of presenting the result as settled.
    edge_params: list[str]


def tune_model(
    dataset: Dataset,
    train: pd.DataFrame,
    strategy: str,
    *,
    make_model: Callable[..., Recommender],
    grid: Mapping[str, Sequence],
    k: int = 10,
    min_train: int = 5,
    val_frac: float = 0.2,
    prior_trials: Sequence[Mapping] = (),
) -> ModelTuning:
    """Exhaustive grid search of `make_model(**params)` by validation NDCG@k.

    `prior_trials` are trials from an earlier stage of the SAME search (same train, strategy, k,
    min_train, val_frac, and a deterministic model; the validation split is deterministic, so their
    scores are comparable). They are kept, not refit, and compete for best with the new ones. This
    is how a grid is extended after the optimum lands on its edge without redoing finished work.

    Same protocol as tune_content_weight: the validation split is carved out of `train`, models
    are fit on train_inner, and no test data can be passed in. Combinations are visited in
    itertools.product order (last key varies fastest) and the first maximum wins ties.
    """
    inner, val = carve_validation(train, strategy, val_frac)
    val = filter_test_to_known_users(inner, val, min_train)

    names = list(grid)
    trials = [dict(tr) for tr in prior_trials]
    done = [tr["params"] for tr in trials]
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values))
        if params in done:
            continue
        t0 = time.perf_counter()
        model = make_model(**params).fit(dataset, inner)
        fit_seconds = time.perf_counter() - t0
        m = evaluate(model, dataset, inner, val, k=k)
        trials.append({"params": params, "ndcg": m.ndcg, "recall": m.recall, "fit_seconds": round(fit_seconds, 2)})

    best = max(trials, key=lambda tr: tr["ndcg"])  # max() returns the first maximal element
    # the searched range is everything ever tried, so an extension can move the optimum off the edge
    searched = {n: sorted({tr["params"][n] for tr in trials}) for n in names}
    edge = [n for n in names if len(searched[n]) > 1 and best["params"][n] in (searched[n][0], searched[n][-1])]
    return ModelTuning(
        best_params=best["params"],
        best_ndcg=best["ndcg"],
        trials=trials,
        grid=searched,
        n_validation_users=int(val["user_idx"].nunique()),
        edge_params=edge,
    )
