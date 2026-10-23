import inspect

import numpy as np
import pytest

from cinematch import tuning
from cinematch.evaluate import evaluate
from cinematch.recommenders import ContentRecommender, IALSRecommender
from cinematch.split import filter_test_to_known_users, global_temporal_split


def _key(df):
    return set(zip(df["user_idx"].tolist(), df["item_idx"].tolist(), df["timestamp"].tolist()))


@pytest.fixture(scope="module")
def train(dataset):
    tr, _ = global_temporal_split(dataset.interactions, 0.2)
    return tr


@pytest.mark.parametrize("strategy", ["temporal", "loo"])
def test_validation_is_carved_from_train_only(train, strategy):
    inner, val = tuning.carve_validation(train, strategy, val_frac=0.2)
    assert _key(inner) | _key(val) == _key(train)
    assert not (_key(inner) & _key(val))


def test_temporal_validation_is_strictly_after_inner_train(train):
    inner, val = tuning.carve_validation(train, "temporal", val_frac=0.2)
    assert inner["timestamp"].max() < val["timestamp"].min()


def test_tuning_has_no_way_to_receive_test_data():
    # the guardrail is structural: the function takes the train frame and nothing else of that kind
    params = set(inspect.signature(tuning.tune_content_weight).parameters)
    assert "test" not in params
    assert "train" in params


def test_tuning_returns_a_full_curve_and_picks_from_the_grid(dataset, train):
    grid = (0.0, 0.5, 1.0)
    res = tuning.tune_content_weight(
        dataset, train, "temporal", grid=grid, k=10, min_train=5,
        make_content=ContentRecommender, make_ials=lambda: IALSRecommender(factors=8, iterations=2),
    )
    assert [c["content_weight"] for c in res.curve] == list(grid)
    assert res.best_weight in grid
    assert res.best_ndcg == max(c["ndcg"] for c in res.curve)


def test_zero_content_weight_equals_ials_alone_on_the_same_validation_split(dataset, train):
    res = tuning.tune_content_weight(
        dataset, train, "temporal", grid=(0.0,), k=10, min_train=5,
        make_content=ContentRecommender, make_ials=lambda: IALSRecommender(factors=8, iterations=2),
    )
    inner, val = tuning.carve_validation(train, "temporal", 0.2)
    val = filter_test_to_known_users(inner, val, 5)
    ials = IALSRecommender(factors=8, iterations=2).fit(dataset, inner)
    expected = evaluate(ials, dataset, inner, val, k=10).ndcg
    assert np.isclose(res.curve[0]["ndcg"], expected, atol=1e-6)
