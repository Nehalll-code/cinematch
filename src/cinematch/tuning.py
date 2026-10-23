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
) -> ModelTuning:
    """Exhaustive grid search of `make_model(**params)` by validation NDCG@k.

    Same protocol as tune_content_weight: the validation split is carved out of `train`, models
    are fit on train_inner, and no test data can be passed in. Combinations are visited in
    itertools.product order (last key varies fastest) and the first maximum wins ties.
    """
    inner, val = carve_validation(train, strategy, val_frac)
    val = filter_test_to_known_users(inner, val, min_train)

    names = list(grid)
    trials = []
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values))
        t0 = time.perf_counter()
        model = make_model(**params).fit(dataset, inner)
        fit_seconds = time.perf_counter() - t0
        m = evaluate(model, dataset, inner, val, k=k)
        trials.append({"params": params, "ndcg": m.ndcg, "recall": m.recall, "fit_seconds": round(fit_seconds, 2)})

    best = max(trials, key=lambda tr: tr["ndcg"])  # max() returns the first maximal element
    edge = [n for n in names if len(grid[n]) > 1 and best["params"][n] in (min(grid[n]), max(grid[n]))]
    return ModelTuning(
        best_params=best["params"],
        best_ndcg=best["ndcg"],
        trials=trials,
        grid={n: list(grid[n]) for n in names},
        n_validation_users=int(val["user_idx"].nunique()),
        edge_params=edge,
    )
