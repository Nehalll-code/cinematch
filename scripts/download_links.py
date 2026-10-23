#!/usr/bin/env python3
"""Fetch MovieLens' links.csv (movieId -> imdbId, tmdbId) for a dataset that does not ship one.

ml-1m has no links.csv, and ml-latest-small's covers only ~76% of ml-1m's films. ml-25m's does cover
them, and MovieLens ids are consistent across releases (matching ids carry the same film). ml-25m is
used rather than the moving `ml-latest` because it is a frozen snapshot: the same file next year.

The zip is ~250 MB; it is streamed to a temp file (never held in memory) and only links.csv is read
out of it, filtered to the target dataset's movies.

    python scripts/download_links.py --into data/ml-1m
"""
from __future__ import annotations

import argparse
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MEMBER = "ml-25m/links.csv"


def extract_links(zip_path: Path, movies_csv: Path, out_csv: Path) -> dict:
    """Write links.csv restricted to the movies in `movies_csv`; return coverage stats."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("links.csv")]
        if not names:
            raise FileNotFoundError(f"no links.csv inside {zip_path}")
        with zf.open(names[0]) as fh:
            links = pd.read_csv(fh)
    movies = pd.read_csv(movies_csv)
    links = links[links["movieId"].isin(set(movies["movieId"]))][["movieId", "imdbId", "tmdbId"]]
    links.to_csv(out_csv, index=False)
    return {"movies": len(movies), "with_tmdb": int(links["tmdbId"].notna().sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", default="data/ml-1m", help="dataset directory containing movies.csv")
    args = ap.parse_args()
    target = Path(args.into)

    ctx = ssl.create_default_context(cafile=certifi.where())  # verify TLS; never disable it
    print(f"downloading {URL} (~250 MB) ...")
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "ml-25m.zip"
        with urllib.request.urlopen(URL, context=ctx) as resp, zpath.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
        stats = extract_links(zpath, target / "movies.csv", target / "links.csv")
    print(f"wrote {target / 'links.csv'}: tmdbId for {stats['with_tmdb']} of {stats['movies']} films "
          f"({stats['with_tmdb'] / stats['movies']:.1%})")


if __name__ == "__main__":
    main()
