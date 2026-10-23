import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic.py"),
         "--users", "200", "--items", "600", "--interactions", "12000", "--out", str(out)],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def dataset(synthetic_dir):
    from cinematch.data import load_movielens
    return load_movielens(synthetic_dir)
