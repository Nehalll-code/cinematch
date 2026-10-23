"""Offline evaluation over the full catalogue.

Every model is scored against ALL items, with the user's training items
masked to -inf, and the top-K is compared to their held-out set.

Deliberately NOT using sampled negatives (rank the true item against 100
random others). That protocol is cheaper but it is not a consistent
estimator of full-ranking metrics -- the induced model ordering can
differ from the true one (Krichene & Rendle, KDD 2020). Full ranking is
affordable here and is what deployment actually does.

Beyond accuracy we report:
  coverage  -- fraction of the catalogue that ever appears in a top-K.
               Accuracy-optimal models often serve the same 200 items to
               everyone; coverage is what exposes that.
  diversity -- mean intra-list dissimilarity on TF-IDF metadata vectors.
  novelty   -- mean self-information, -log2(popularity). High accuracy
               with low novelty usually means the model relearned
               popularity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from .data import Dataset
from .recommenders.base import Recommender


@dataclass
class Metrics:
    model: str
    k: int
    users_evaluated: int
    recall: float
    precision: float
    ndcg: float
    map: float
    coverage: float
    diversity: float
    novelty: float

    def as_row(self) -> dict:
        return asdict(self)


def _dcg(hits: np.ndarray) -> np.ndarray:
    discounts = 1.0 / np.log2(np.arange(2, hits.shape[1] + 2))
    return (hits * discounts).sum(axis=1)


def _average_precision(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    csum = np.cumsum(hits, axis=1)
    ranks = np.arange(1, hits.shape[1] + 1)
    prec_at_hit = (csum / ranks) * hits
    denom = np.minimum(n_relevant, hits.shape[1])
    return prec_at_hit.sum(axis=1) / np.maximum(denom, 1)


def evaluate(
    model: Recommender,
    dataset: Dataset,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    batch_size: int = 512,
    item_features: sp.csr_matrix | None = None,
) -> Metrics:
    truth = {int(u): g["item_idx"].to_numpy() for u, g in test.groupby("user_idx", sort=True)}
    users = np.array(sorted(truth.keys()), dtype=np.int64)
    if len(users) == 0:
        raise ValueError("no evaluable users in test set")

    train_matrix = dataset.matrix(train).tocsr()
    pop = np.asarray(train_matrix.sum(axis=0)).ravel()
    pop_prob = pop / max(pop.sum(), 1.0)
    self_info = -np.log2(np.maximum(pop_prob, 1e-12))

    if item_features is None:
        item_features = _metadata_vectors(dataset)

    recalls, precisions, ndcgs, aps, novelties, diversities = [], [], [], [], [], []
    recommended_items: set[int] = set()

    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = model.score_users(batch).astype(np.float32, copy=True)

        # mask training interactions: recommending something already
        # consumed is not a hit, and leaving it unmasked inflates every metric
        for row, u in enumerate(batch):
            scores[row, train_matrix[u].indices] = -np.inf

        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)

        hits = np.zeros((len(batch), k), dtype=np.float32)
        n_rel = np.zeros(len(batch), dtype=np.float32)
        for row, u in enumerate(batch):
            relevant = truth[int(u)]
            n_rel[row] = len(relevant)
            hits[row] = np.isin(top[row], relevant).astype(np.float32)

        # IDCG: all min(|relevant|, k) hits packed at the top of the list
        cum_disc = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])
        ideal = cum_disc[np.minimum(n_rel.astype(int), k)]
        recalls.append(hits.sum(axis=1) / np.maximum(n_rel, 1))
        precisions.append(hits.sum(axis=1) / k)
        ndcgs.append(_dcg(hits) / np.maximum(ideal, 1e-12))
        aps.append(_average_precision(hits, n_rel))
        novelties.append(self_info[top].mean(axis=1))
        diversities.append(_intra_list_diversity(top, item_features))
        recommended_items.update(top.ravel().tolist())

    return Metrics(
        model=model.name,
        k=k,
        users_evaluated=int(len(users)),
        recall=float(np.concatenate(recalls).mean()),
        precision=float(np.concatenate(precisions).mean()),
        ndcg=float(np.concatenate(ndcgs).mean()),
        map=float(np.concatenate(aps).mean()),
        coverage=float(len(recommended_items) / dataset.n_items),
        diversity=float(np.concatenate(diversities).mean()),
        novelty=float(np.concatenate(novelties).mean()),
    )


def _metadata_vectors(dataset: Dataset) -> sp.csr_matrix:
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = dataset.items["genres"].fillna("").str.replace("|", " ", regex=False).tolist()
    return normalize(TfidfVectorizer().fit_transform(docs))


def _intra_list_diversity(top: np.ndarray, features: sp.csr_matrix) -> np.ndarray:
    out = np.zeros(top.shape[0], dtype=np.float32)
    k = top.shape[1]
    if k < 2:
        return out
    for row in range(top.shape[0]):
        V = features[top[row]]
        sim = (V @ V.T).toarray()
        iu = np.triu_indices(k, k=1)
        out[row] = 1.0 - float(sim[iu].mean())
    return out


def results_table(metrics: list[Metrics]) -> pd.DataFrame:
    return pd.DataFrame([m.as_row() for m in metrics])
