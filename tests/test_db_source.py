"""The guardrail for the offline eval: the CSV path and the DB path must produce IDENTICAL data.

If these fail, `run_eval.py --source db` and `run_eval.py --source csv` are no longer evaluating
the same thing and every number in the README is suspect. They are strict on purpose: same values,
same dtypes, same row order, same original timestamps.
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from cinematch.data import load_movielens
from cinematch.db.indexing import ITEM_IDX, USER_IDX, peek
from cinematch.db.ingest import ingest_movielens
from cinematch.db.models import Interaction, Movie, User
from cinematch.db.source import load_dataset_from_db
from cinematch.split import global_temporal_split, leave_one_out_split


@pytest.fixture
def ingested(db, synthetic_dir):
    stats = ingest_movielens(db, synthetic_dir)
    db.commit()
    return db, stats


def test_csv_and_db_datasets_are_identical(ingested, synthetic_dir):
    db, _ = ingested
    csv_ds, db_ds = load_movielens(synthetic_dir), load_dataset_from_db(db)

    assert np.array_equal(csv_ds.user_ids, db_ds.user_ids)
    assert np.array_equal(csv_ds.item_ids, db_ds.item_ids)
    pd.testing.assert_frame_equal(csv_ds.interactions, db_ds.interactions, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(csv_ds.items, db_ds.items, check_dtype=True, check_exact=True)


def test_splits_of_both_sources_are_identical_with_heavy_timestamp_ties(db, tmp_path):
    # Leave-one-out holds out each user's LAST interaction; among equal timestamps "last" is
    # decided purely by row order, so this fails if the DB path returns rows in a different order
    # than the CSV. Real MovieLens users rate in same-second batches, so ties are the norm.
    rng = np.random.default_rng(0)
    users, movies = np.arange(1, 41), np.arange(1, 61)
    rows = [(u, m, 5.0, 1_000_000 + int(rng.integers(0, 3)))  # only 3 distinct timestamps
            for u in users for m in rng.choice(movies, 25, replace=False)]
    rng.shuffle(rows)  # file order unrelated to (user, movie) order
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(tmp_path / "ratings.csv", index=False)
    pd.DataFrame({"movieId": movies, "title": [f"M{m}" for m in movies], "genres": "Drama"}).to_csv(
        tmp_path / "movies.csv", index=False)

    ingest_movielens(db, tmp_path)
    db.commit()
    a, b = load_movielens(tmp_path).interactions, load_dataset_from_db(db).interactions
    assert a["timestamp"].duplicated().any(), "fixture must contain timestamp ties to be meaningful"
    for split in (leave_one_out_split, global_temporal_split):
        for x, y in zip(split(a), split(b)):
            pd.testing.assert_frame_equal(x, y, check_dtype=True, check_exact=True)


def test_original_timestamps_survive_the_round_trip(ingested, synthetic_dir):
    db, _ = ingested
    raw = pd.read_csv(synthetic_dir / "ratings.csv")
    positives = raw[raw["rating"] >= 4.0]
    assert sorted(load_dataset_from_db(db).interactions["timestamp"].tolist()) == sorted(positives["timestamp"].tolist())


def test_ingest_is_idempotent(db, synthetic_dir):
    first = ingest_movielens(db, synthetic_dir)
    db.commit()
    counts = [db.scalar(select(func.count()).select_from(m)) for m in (User, Movie, Interaction)]
    second = ingest_movielens(db, synthetic_dir)
    db.commit()
    assert (second.movies, second.users, second.interactions) == (0, 0, 0)
    assert counts == [db.scalar(select(func.count()).select_from(m)) for m in (User, Movie, Interaction)]
    assert first.interactions > 0


def test_counters_equal_the_number_of_rows_so_no_index_is_skipped(ingested):
    db, stats = ingested
    assert peek(db, USER_IDX) == stats.users and peek(db, ITEM_IDX) == stats.movies


def test_existing_indices_do_not_move_when_new_data_arrives(db, tmp_path):
    def write(dirpath, ratings, movies):
        dirpath.mkdir()
        pd.DataFrame(ratings, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(dirpath / "ratings.csv", index=False)
        pd.DataFrame(movies, columns=["movieId", "title", "genres"]).to_csv(dirpath / "movies.csv", index=False)
        return dirpath

    m1 = [(10, "A", "Drama"), (30, "C", "Comedy")]
    ingest_movielens(db, write(tmp_path / "v1", [(5, 10, 5.0, 100), (5, 30, 4.0, 101)], m1))
    db.commit()
    before = dict(db.execute(select(Movie.movie_id, Movie.item_idx)).all())

    # a movie with a LOWER id than existing ones, and a new user, arrive later
    m2 = m1 + [(20, "B", "Action")]
    ingest_movielens(db, write(tmp_path / "v2", [(5, 10, 5.0, 100), (3, 20, 4.0, 102)], m2))
    db.commit()
    after = dict(db.execute(select(Movie.movie_id, Movie.item_idx)).all())

    assert {k: after[k] for k in before} == before  # nothing moved
    assert after[20] == 2  # appended, not inserted in sorted position
    assert dict(db.execute(select(User.ml_user_id, User.user_idx)).all()) == {5: 0, 3: 1}
