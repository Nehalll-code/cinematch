"""`python experiments/run_eval.py` must keep working after the data moves into the database, and
must report the same numbers from either source. This runs the real script as a subprocess."""
import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT
from cinematch.db.ingest import ingest_movielens

TIMING_KEYS = {"fit_seconds", "eval_seconds"}


def _run_eval(args, env_extra=None):
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "run_eval.py"), "--k", "10", *args],
        check=True, capture_output=True, text=True, env=env, cwd=ROOT,
    )


def _rows(path):
    payload = json.loads(path.read_text())
    return [{k: v for k, v in r.items() if k not in TIMING_KEYS} for r in payload["rows"]]


@pytest.mark.parametrize("split", ["temporal", "loo"])
def test_eval_from_the_database_matches_eval_from_csv(db, migrated_engine, synthetic_dir, tmp_path, split):
    ingest_movielens(db, synthetic_dir)
    db.commit()
    url = migrated_engine.url.render_as_string(hide_password=False)

    csv_out, db_out = tmp_path / "csv", tmp_path / "db"
    _run_eval(["--data", str(synthetic_dir), "--split", split, "--out", str(csv_out)])
    _run_eval(["--source", "db", "--split", split, "--out", str(db_out)], {"DATABASE_URL": url})

    csv_rows, db_rows = _rows(csv_out / f"results_{split}_k10.json"), _rows(db_out / f"results_{split}_k10.json")
    assert [r["model"] for r in csv_rows] == ["popularity", "content", "item_knn", "ials", "hybrid_0.3"]
    assert csv_rows == db_rows  # exact equality: same data in, same numbers out


def test_db_source_without_a_database_url_fails_with_a_clear_message():
    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    res = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "run_eval.py"), "--source", "db"],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    assert res.returncode != 0 and "DATABASE_URL is not set" in res.stderr
