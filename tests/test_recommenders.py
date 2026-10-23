import numpy as np
import pytest

from cinematch.evaluate import evaluate
from cinematch.recommenders import (
    ContentRecommender,
    HybridRecommender,
    IALSRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
)
from cinematch.recommenders.base import rank_normalize
from cinematch.split import filter_test_to_known_users, global_temporal_split


@pytest.fixture(scope="module")
def split(dataset):
    train, test = global_temporal_split(dataset.interactions, 0.2)
    return train, filter_test_to_known_users(train, test, 5)


def test_rank_normalize_is_scale_invariant():
    a = np.array([[1.0, 5.0, 3.0]])
    assert np.allclose(rank_normalize(a), rank_normalize(a * 1000.0))
    assert rank_normalize(a).max() == 1.0 and rank_normalize(a).min() == 0.0


@pytest.mark.parametrize("factory", [
    PopularityRecommender,
    ContentRecommender,
    lambda: ItemKNNRecommender(k=50),
    lambda: IALSRecommender(factors=16, iterations=3),
])
def test_models_produce_finite_scores_of_correct_shape(dataset, split, factory):
    train, _ = split
    model = factory().fit(dataset, train)
    users = np.array([0, 1, 2])
    scores = model.score_users(users)
    assert scores.shape == (3, dataset.n_items)
    assert np.isfinite(scores).all()


def test_every_model_beats_random_on_ndcg(dataset, split):
    train, test = split
    rng = np.random.default_rng(0)

    class R(PopularityRecommender):
        name = "rand"

        def fit(self, ds, tr):
            super().fit(ds, tr)
            self.scores_ = rng.random(ds.n_items).astype(np.float32)
            return self

    baseline = evaluate(R().fit(dataset, train), dataset, train, test, k=10).ndcg
    for factory in (PopularityRecommender, ContentRecommender,
                    lambda: ItemKNNRecommender(k=50), lambda: IALSRecommender(factors=16, iterations=5)):
        m = evaluate(factory().fit(dataset, train), dataset, train, test, k=10)
        assert m.ndcg > baseline, f"{m.model} failed to beat random"


def test_ials_is_deterministic_under_a_fixed_seed(dataset, split):
    train, _ = split
    a = IALSRecommender(factors=16, iterations=3, random_state=1).fit(dataset, train)
    b = IALSRecommender(factors=16, iterations=3, random_state=1).fit(dataset, train)
    assert np.allclose(a.score_users(np.array([0])), b.score_users(np.array([0])))


def test_hybrid_falls_back_to_popularity_for_cold_users(dataset, split):
    train, _ = split
    content = ContentRecommender()
    ials = IALSRecommender(factors=16, iterations=3)
    hybrid = HybridRecommender({"c": (content, 0.3), "i": (ials, 0.7)}, cold_threshold=1000)
    hybrid.fit(dataset, train)  # threshold forces every user down the cold path
    pop = PopularityRecommender().fit(dataset, train)
    users = np.array([0, 1, 2])
    assert np.allclose(hybrid.score_users(users), rank_normalize(pop.score_users(users)))


def test_hybrid_weights_shift_output_toward_the_heavier_component(dataset, split):
    train, _ = split
    content = ContentRecommender().fit(dataset, train)
    ials = IALSRecommender(factors=16, iterations=3).fit(dataset, train)
    users = np.array([5, 6, 7])
    mostly_content = HybridRecommender({"c": (content, 0.99), "i": (ials, 0.01)}, cold_threshold=0)
    mostly_content.fit(dataset, train)
    pure_content = rank_normalize(content.score_users(users))
    pure_ials = rank_normalize(ials.score_users(users))
    blended = mostly_content.score_users(users)
    assert np.abs(blended - pure_content).mean() < np.abs(blended - pure_ials).mean()


def test_content_model_actually_uses_the_release_year(dataset, split):
    # content.py documents the year as a signal; the tokenizer used to drop every digit, so the
    # comment described a feature that did not exist
    train, _ = split
    model = ContentRecommender().fit(dataset, train)
    years = {t.rsplit("(", 1)[1].rstrip(")") for t in dataset.items["title"]}
    assert years <= set(model.vectorizer_.vocabulary_), "release years missing from the vocabulary"
