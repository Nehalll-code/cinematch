import io
import sys
import zipfile

import pandas as pd

sys.path.insert(0, "scripts")
from download_links import extract_links  # noqa: E402


def _zip(path, links_csv):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ml-25m/README.txt", "x" * 10)
        zf.writestr("ml-25m/links.csv", links_csv)


def test_extracts_only_links_for_the_target_movies_and_reports_coverage(tmp_path):
    zpath = tmp_path / "big.zip"
    _zip(zpath, "movieId,imdbId,tmdbId\n1,114709,862\n2,113497,8844\n3,113228,15602\n99,1,2\n")
    pd.DataFrame({"movieId": [1, 2, 3, 4], "title": list("abcd"), "genres": "x"}).to_csv(tmp_path / "movies.csv", index=False)

    out = tmp_path / "links.csv"
    stats = extract_links(zpath, tmp_path / "movies.csv", out)

    got = pd.read_csv(out)
    assert got["movieId"].tolist() == [1, 2, 3]  # 99 is not in the target catalogue
    assert list(got.columns) == ["movieId", "imdbId", "tmdbId"]
    assert (stats["movies"], stats["with_tmdb"]) == (4, 3)  # film 4 has no link


def test_a_zip_without_links_csv_is_a_clear_error(tmp_path):
    zpath = tmp_path / "bad.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("ml-25m/ratings.csv", "a")
    pd.DataFrame({"movieId": [1], "title": "a", "genres": "x"}).to_csv(tmp_path / "movies.csv", index=False)
    try:
        extract_links(zpath, tmp_path / "movies.csv", tmp_path / "links.csv")
    except FileNotFoundError as exc:
        assert "links.csv" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
