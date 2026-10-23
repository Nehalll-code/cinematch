import numpy as np
import pandas as pd

from cinematch.data import load_movielens


def _write(tmp_path, ratings, movies):
    pd.DataFrame(ratings, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(
        tmp_path / "ratings.csv", index=False)
    pd.DataFrame(movies, columns=["movieId", "title", "genres"]).to_csv(tmp_path / "movies.csv", index=False)
    return tmp_path


MOVIES = [(10, "A (2000)", "Drama"), (20, "B (2001)", "Comedy"), (30, "C (2002)", "Drama|Comedy"),
          (40, "D (2003)", "Action")]


def test_catalogue_indexes_every_movie_including_ones_with_no_positive_rating(tmp_path):
    # movie 30 is rated only 2.0 and movie 40 is never rated: both still get an item_idx
    ratings = [(1, 10, 5.0, 100), (1, 20, 4.0, 101), (2, 30, 2.0, 102), (2, 10, 4.5, 103)]
    ds = load_movielens(_write(tmp_path, ratings, MOVIES))
    assert ds.item_ids.tolist() == [10, 20, 30, 40]
    assert ds.items["movieId"].tolist() == [10, 20, 30, 40]
    assert ds.items["item_idx"].tolist() == [0, 1, 2, 3]


def test_users_with_no_positive_rating_keep_a_row(tmp_path):
    ratings = [(1, 10, 5.0, 100), (7, 20, 1.0, 101)]
    ds = load_movielens(_write(tmp_path, ratings, MOVIES))
    assert ds.user_ids.tolist() == [1, 7]
    assert ds.matrix().shape == (2, 4)


def test_item_idx_does_not_shift_when_a_new_rating_makes_a_movie_positive(tmp_path):
    base = [(1, 10, 5.0, 100), (1, 40, 4.0, 101)]
    before = load_movielens(_write(tmp_path, base, MOVIES))
    # movie 20 gains its first positive rating; under the old positives-only
    # indexing this would have renumbered movie 40
    after = load_movielens(_write(tmp_path, base + [(2, 20, 5.0, 102)], MOVIES))
    assert dict(zip(before.item_ids, range(before.n_items))) == dict(zip(after.item_ids, range(after.n_items)))


def test_only_positives_become_interactions_and_unknown_movies_are_dropped(tmp_path):
    ratings = [(1, 10, 5.0, 100), (1, 20, 3.5, 101), (1, 999, 5.0, 102)]
    ds = load_movielens(_write(tmp_path, ratings, MOVIES))
    assert len(ds.interactions) == 1
    assert ds.item_ids[ds.interactions["item_idx"].iloc[0]] == 10
    assert np.array_equal(ds.item_ids, [10, 20, 30, 40])
