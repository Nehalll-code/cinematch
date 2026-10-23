import pandas as pd

from cinematch.split import filter_test_to_known_users, global_temporal_split, leave_one_out_split


def test_temporal_split_has_no_future_leakage(dataset):
    train, test = global_temporal_split(dataset.interactions, 0.2)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(test) == len(dataset.interactions)


def test_temporal_split_never_straddles_a_timestamp():
    df = pd.DataFrame({"user_idx": range(10), "item_idx": range(10),
                       "rating": [5.0] * 10, "timestamp": [1] * 5 + [2] * 5})
    train, test = global_temporal_split(df, 0.5)
    assert set(train["timestamp"]).isdisjoint(set(test["timestamp"]))


def test_leave_one_out_holds_out_latest_per_user():
    df = pd.DataFrame({"user_idx": [0, 0, 0, 1, 1], "item_idx": [1, 2, 3, 4, 5],
                       "rating": [5.0] * 5, "timestamp": [10, 30, 20, 5, 9]})
    train, test = leave_one_out_split(df)
    assert sorted(test["item_idx"]) == [2, 5]  # user0's ts=30, user1's ts=9
    assert len(train) == 3


def test_single_interaction_user_is_not_held_out():
    df = pd.DataFrame({"user_idx": [0, 1, 1], "item_idx": [1, 2, 3],
                       "rating": [5.0] * 3, "timestamp": [1, 2, 3]})
    _, test = leave_one_out_split(df)
    assert 0 not in set(test["user_idx"])


def test_filter_drops_users_without_enough_history():
    train = pd.DataFrame({"user_idx": [0, 0, 0, 1], "item_idx": [1, 2, 3, 4],
                          "rating": [5.0] * 4, "timestamp": [1, 2, 3, 4]})
    test = pd.DataFrame({"user_idx": [0, 1], "item_idx": [9, 8],
                         "rating": [5.0] * 2, "timestamp": [5, 6]})
    kept = filter_test_to_known_users(train, test, min_train_interactions=3)
    assert kept["user_idx"].tolist() == [0]
