#!/usr/bin/env python3
"""Download and unpack a MovieLens dataset into data/.

Usage:
    python scripts/download_data.py              # ml-latest-small (~1 MB)
    python scripts/download_data.py --size 1m    # ml-1m
"""
from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "small": ("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", "ml-latest-small"),
    "1m": ("https://files.grouplens.org/datasets/movielens/ml-1m.zip", "ml-1m"),
    "25m": ("https://files.grouplens.org/datasets/movielens/ml-25m.zip", "ml-25m"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(URLS), default="small")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    url, inner = URLS[args.size]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(out)

    src = out / inner
    for name in ("ratings.csv", "movies.csv"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)

    # ml-1m ships ::-delimited .dat files instead of csv
    if not (out / "ratings.csv").exists() and (src / "ratings.dat").exists():
        import pandas as pd

        r = pd.read_csv(src / "ratings.dat", sep="::", engine="python",
                        names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1")
        m = pd.read_csv(src / "movies.dat", sep="::", engine="python",
                        names=["movieId", "title", "genres"], encoding="latin-1")
        r.to_csv(out / "ratings.csv", index=False)
        m.to_csv(out / "movies.csv", index=False)

    print(f"ready: {out/'ratings.csv'}, {out/'movies.csv'}")


if __name__ == "__main__":
    main()
