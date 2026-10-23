"""Fold-in: personalise for a user the model was not trained on, with one least-squares solve
against the fixed item factors (the user half-step of ALS). No retraining."""
import numpy as np
import pytest

from cinematch.recommenders import (
    ContentRecommender,
    HybridRecommender,
    IALSRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
)
from cinematch.recommenders.base import rank_normalize
from cinematch.split import global_temporal_split


@pytest.fixture(scope="module")
def train(dataset):
    return global_temporal_split(dataset.interactions, 0.2)[0]


@pytest.fixture(scope="module")
def ials(dataset, train):
    return IALSRecommender(factors=16, iterations=4).fit(dataset, train)


def _items_of(model, u):
    return model.train_matrix[u].indices


def test_fold_in_is_exactly_one_user_half_step(ials):
    # same function, same fixed item factors => same vector (this is the correctness argument)
    u = 3
    items = _items_of(ials, u)
    folded = ials.fold_in(items)

    users = np.zeros((ials.dataset.n_users, ials.factors))
    Cui = (ials.train_matrix * ials.alpha).tocsr()
    ials._solve(users, ials.item_factors_.astype(np.float64), Cui)
    assert np.allclose(folded, users[u], atol=1e-4)


def test_fold_in_user_scores_are_finite_and_shaped(ials):
    scores = ials.score_from_items(_items_of(ials, 5))
    assert scores.shape == (ials.dataset.n_items,) and np.isfinite(scores).all()


def test_fold_in_ranks_held_out_items_far_above_chance(dataset, ials):
    rng = np.random.default_rng(0)
    hits = chance = 0
    for u in range(40):
        items = _items_of(ials, u)
        if len(items) < 10:
            continue
        held = rng.choice(items, size=len(items) // 4, replace=False)
        keep = np.setdiff1d(items, held)
        scores = ials.score_from_items(keep)
        scores[keep] = -np.inf
        top = np.argsort(-scores)[:50]
        hits += np.isin(held, top).sum()
        chance += len(held) * 50 / dataset.n_items
    assert hits > 3 * chance


def test_no_interactions_gives_a_zero_vector_not_nan(ials):
    assert np.array_equal(ials.fold_in(np.array([], dtype=int)), np.zeros(ials.factors, dtype=np.float32))


def test_vector_is_cached_per_key_and_recomputed_only_when_the_stamp_changes(ials, monkeypatch):
    items, calls = _items_of(ials, 7), []
    original = ials._solve_row
    monkeypatch.setattr(ials, "_solve_row", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    ials.fold_in(items, cache_key=("u7", 3))
    ials.fold_in(items, cache_key=("u7", 3))
    assert len(calls) == 1  # second call served from the cache
    ials.fold_in(items[:-1], cache_key=("u7", 4))  # the user's interaction stamp moved
    assert len(calls) == 2
    ials.fold_in(items[:-1], cache_key=("u7", 4))
    assert len(calls) == 2


def test_cache_is_bounded(dataset, train):
    model = IALSRecommender(factors=8, iterations=1, fold_in_cache_size=3).fit(dataset, train)
    for i in range(10):
        model.fold_in(np.array([0, 1, 2]), cache_key=(f"u{i}", 1))
    assert len(model._fold_in_cache) == 3


def test_models_pickled_before_fold_in_existed_still_work(dataset, train):
    # an artifact trained by an older version has none of the fold-in state; loading it must not break
    model = IALSRecommender(factors=8, iterations=2).fit(dataset, train)
    for attr in ("_item_side", "_fold_in_cache", "fold_in_cache_size"):
        model.__dict__.pop(attr, None)
    vec = model.fold_in(_items_of(model, 2), cache_key=("legacy", 1))
    assert np.isfinite(vec).all() and np.array_equal(vec, model.fold_in(_items_of(model, 2), cache_key=("legacy", 1)))


# ---- every model can score a user described only by the items they liked --------------------

def test_every_model_scores_from_items(dataset, train):
    items = np.array([0, 3, 9, 12])
    for model in (PopularityRecommender(), ContentRecommender(), ItemKNNRecommender(k=30),
                  IALSRecommender(factors=8, iterations=2)):
        s = model.fit(dataset, train).score_from_items(items)
        assert s.shape == (dataset.n_items,) and np.isfinite(s).all(), model.name


def test_hybrid_uses_popularity_below_the_cold_threshold_and_skips_zero_weight(dataset, train):
    content, ials = ContentRecommender(), IALSRecommender(factors=8, iterations=2)
    hybrid = HybridRecommender({"content": (content, 0.0), "ials": (ials, 1.0)}, cold_threshold=3).fit(dataset, train)
    pop = rank_normalize(hybrid.fallback.scores_[None])[0]
    assert np.allclose(hybrid.score_from_items(np.array([1, 2])), pop)  # 2 < 3 => popularity
    warm = hybrid.score_from_items(np.array([1, 2, 3, 4]))
    assert np.allclose(warm, rank_normalize(ials.score_from_items(np.array([1, 2, 3, 4]))[None])[0])
