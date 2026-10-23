import pytest
from alembic import command
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from cinematch.db.indexing import ITEM_IDX, USER_IDX, allocate, peek
from cinematch.db.models import IndexCounter, Interaction, Movie, User, WatchlistItem


def test_migrations_match_the_models(migrated_engine, alembic_config_for):
    # `alembic check` raises if autogenerate would emit any operation: the migration history and
    # the ORM models describe the same schema.
    # str(url) masks the password as '***'; Alembic needs the real one
    command.check(alembic_config_for(migrated_engine.url.render_as_string(hide_password=False)))


def test_counters_are_seeded_by_the_migration(db):
    assert peek(db, USER_IDX) == 0 and peek(db, ITEM_IDX) == 0


def _user(idx, **kw):
    return User(user_idx=idx, **kw)


def _fails(db, *objs):
    db.add_all(objs)
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()


class TestUsers:
    def test_a_row_must_be_reachable_or_importable(self, db):
        _fails(db, _user(0))  # neither email nor ml_user_id

    def test_imported_user_without_credentials_is_valid(self, db):
        db.add(_user(0, ml_user_id=1))
        db.add(_user(1, email="a@example.com", password_hash="x"))
        db.flush()

    def test_email_must_be_lowercase(self, db):
        _fails(db, _user(0, email="Mixed@Example.com"))

    def test_password_requires_an_email(self, db):
        _fails(db, _user(0, ml_user_id=1, password_hash="x"))

    @pytest.mark.parametrize("field", ["email", "ml_user_id", "user_idx"])
    def test_uniqueness(self, db, field):
        a = dict(user_idx=0, ml_user_id=1, email="a@example.com")
        b = dict(user_idx=1, ml_user_id=2, email="b@example.com")
        b[field] = a[field]
        db.add(User(**a))
        db.flush()
        _fails(db, User(**b))


@pytest.fixture
def seeded(db):
    db.add_all([User(user_idx=0, ml_user_id=1), Movie(movie_id=10, item_idx=0, title="A", genres="Drama")])
    db.flush()
    user = db.scalars(select(User)).one()
    return db, user.id


class TestInteractions:
    def test_valid_rating_watch_and_dismiss(self, seeded):
        db, uid = seeded
        db.add(Interaction(user_id=uid, movie_id=10, type="rating", rating=4.5))
        db.add(Interaction(user_id=uid, movie_id=10, type="watch"))
        db.add(Interaction(user_id=uid, movie_id=10, type="dismiss"))
        db.flush()

    def test_dismiss_carries_no_rating(self, seeded):
        db, uid = seeded
        _fails(db, Interaction(user_id=uid, movie_id=10, type="dismiss", rating=3.0))

    @pytest.mark.parametrize("kwargs", [
        dict(type="like"),                       # unknown type
        dict(type="rating"),                     # rating required for type=rating
        dict(type="watch", rating=3.0),          # rating forbidden for other types
        dict(type="rating", rating=0.0),         # below range
        dict(type="rating", rating=5.5),         # above range
    ])
    def test_check_constraints(self, seeded, kwargs):
        db, uid = seeded
        _fails(db, Interaction(user_id=uid, movie_id=10, **kwargs))

    def test_one_row_per_user_movie_type(self, seeded):
        db, uid = seeded
        db.add(Interaction(user_id=uid, movie_id=10, type="rating", rating=4.0))
        db.flush()
        _fails(db, Interaction(user_id=uid, movie_id=10, type="rating", rating=2.0))

    def test_foreign_keys_are_enforced(self, seeded):
        db, uid = seeded
        _fails(db, Interaction(user_id=uid, movie_id=999, type="watch"))
        _fails(db, Interaction(user_id=uid + 100, movie_id=10, type="watch"))


class TestWatchlist:
    def test_movie_appears_once_per_user(self, seeded):
        db, uid = seeded
        db.add(WatchlistItem(user_id=uid, movie_id=10))
        db.flush()
        _fails(db, WatchlistItem(user_id=uid, movie_id=10))


class TestIndexAllocation:
    def test_allocations_are_consecutive_and_never_overlap(self, db):
        assert allocate(db, USER_IDX) == 0
        assert allocate(db, USER_IDX, count=5) == 1
        assert allocate(db, USER_IDX) == 6
        assert peek(db, ITEM_IDX) == 0  # independent spaces

    def test_rollback_returns_the_indices(self, db):
        db.commit()
        allocate(db, USER_IDX, count=3)
        db.rollback()
        assert allocate(db, USER_IDX) == 0

    def test_deleting_the_newest_row_does_not_free_its_index(self, db):
        # MAX(col)+1 would hand index 1 to the next user, who would then inherit the deleted
        # user's factors in a model trained while they existed.
        i0, i1 = allocate(db, USER_IDX), allocate(db, USER_IDX)
        db.add_all([User(user_idx=i0, ml_user_id=1), User(user_idx=i1, ml_user_id=2)])
        db.flush()
        db.delete(db.scalars(select(User).where(User.user_idx == i1)).one())
        db.flush()
        assert allocate(db, USER_IDX) == 2

    def test_unseeded_counter_is_an_error_not_a_silent_zero(self, db):
        db.execute(IndexCounter.__table__.delete())
        with pytest.raises(LookupError):
            allocate(db, USER_IDX)
