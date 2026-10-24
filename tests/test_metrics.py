"""Validate the metric implementations against models with known answers."""
import numpy as np
import pandas as pd
import pytest

from cinematch.evaluate import evaluate
from cinematch.recommenders.base import Recommender
from cinematch.split import filter_test_to_known_users, global_temporal_split


class OracleRecommender(Recommender):
    """Scores held-out items highest. Upper bound: recall@k must be ~1 when |truth| <= k."""

    name = "oracle"

    def __init__(self, test):
        super().__init__()
        self.truth = {int(u): g["item_idx"].to_numpy() for u, g in test.groupby("user_idx")}

    def fit(self, dataset, train):
        self.dataset = dataset
        self.train_matrix = dataset.matrix(train)
        return self

    def score_users(self, user_indices):
        out = np.zeros((len(user_indices), self.dataset.n_items), dtype=np.float32)
        for row, u in enumerate(user_indices):
            out[row, self.truth[int(u)]] = 1.0
        return out


class RandomRecommender(Recommender):
    name = "random"

    def fit(self, dataset, train):
        self.dataset = dataset
        self.train_matrix = dataset.matrix(train)
        self.rng = np.random.default_rng(0)
        return self

    def score_users(self, user_indices):
        return self.rng.random((len(user_indices), self.dataset.n_items)).astype(np.float32)


@pytest.fixture(scope="module")
def split(dataset):
    train, test = global_temporal_split(dataset.interactions, 0.2)
    return train, filter_test_to_known_users(train, test, 5)


def test_oracle_saturates_recall_and_ndcg(dataset, split):
    train, test = split
    # restrict to users with at most k held-out items so perfect recall is attainable
    sizes = test.groupby("user_idx")["item_idx"].size()
    small = test[test["user_idx"].isin(sizes[sizes <= 10].index)]
    m = evaluate(OracleRecommender(small).fit(dataset, train), dataset, train, small, k=10)
    assert m.recall == pytest.approx(1.0, abs=1e-6)
    assert m.ndcg == pytest.approx(1.0, abs=1e-6)


def test_random_is_a_weak_lower_bound(dataset, split):
    """Random must sit far below popularity.

    An absolute threshold would be wrong here: random recall@k scales with
    k / n_items, so it moves with catalogue size. The invariant that
    actually holds is the ordering against a real baseline.
    """
    from cinematch.recommenders import PopularityRecommender

    train, test = split
    rand = evaluate(RandomRecommender().fit(dataset, train), dataset, train, test, k=10)
    pop = evaluate(PopularityRecommender().fit(dataset, train), dataset, train, test, k=10)
    assert rand.ndcg < 0.5 * pop.ndcg
    assert rand.coverage > pop.coverage  # random spreads across the catalogue


def test_seen_items_are_never_recommended(dataset, split):
    train, test = split
    model = RandomRecommender().fit(dataset, train)
    tm = dataset.matrix(train)
    for u in test["user_idx"].unique()[:20]:
        recs = model.recommend(int(u), k=10)
        assert set(recs).isdisjoint(set(tm[int(u)].indices))


def test_metrics_are_in_valid_ranges(dataset, split):
    train, test = split
    m = evaluate(RandomRecommender().fit(dataset, train), dataset, train, test, k=10)
    for value in (m.recall, m.precision, m.ndcg, m.map, m.coverage, m.diversity):
        assert 0.0 <= value <= 1.0
    assert m.novelty > 0.0


def test_per_user_out_matches_the_reported_means(dataset, split):
    train, test = split
    model = RandomRecommender().fit(dataset, train)
    per_user: dict = {}
    m = evaluate(model, dataset, train, test, k=10, per_user_out=per_user)
    assert len(per_user["user_idx"]) == m.users_evaluated
    assert per_user["user_idx"].tolist() == sorted(test["user_idx"].unique().tolist())
    assert per_user["ndcg"].mean() == pytest.approx(m.ndcg)
    assert per_user["recall"].mean() == pytest.approx(m.recall)
    assert per_user["map"].mean() == pytest.approx(m.map)
    assert per_user["precision"].mean() == pytest.approx(m.precision)
