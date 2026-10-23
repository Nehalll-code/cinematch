# CineMatch

Hybrid movie recommendation engine with a leakage-safe offline evaluation harness.

The point of this repo is the **evaluation**, not the model zoo. Five recommenders are
implemented so they can be compared against each other and against a popularity baseline
under a split that does not leak the future. Reporting "we built a hybrid recommender" without
that comparison says nothing.

## Quickstart

```bash
make install
make data          # downloads MovieLens ml-1m into data/ml-1m (headline dataset)
make eval          # trains all five models, prints the comparison table
make train && make bench && make api
```

Accounts, interactions and the watchlist need PostgreSQL (`make db-up`); see "Persistence and
accounts". The evaluation and the recommendation API do not.

No network? `make synth` generates a dataset with the same schema, and everything
(`make test`, `python experiments/run_eval.py --data data_synthetic`) runs offline.
`make data-small && make smoke` runs the same pipeline on ml-latest-small in about a minute.

## Results

MovieLens **ml-1m** (1,000,209 ratings; 575,281 positives at rating ≥ 4.0), k=10, full-catalogue
ranking over all **3,883 movies and 6,040 users** (the full catalogue, not just items that happen
to have a positive rating; see "One index space" below). These numbers were produced by
`python experiments/run_eval.py --source db` reading from PostgreSQL; `--source csv` gives
identical numbers, which a test enforces. `ml-latest-small` is kept as the fast smoke dataset
(`make data-small && make smoke`) but is not used for headline numbers: its temporal split leaves
only 26 evaluable users.

iALS and item-kNN hyperparameters are untuned defaults. The one thing that *was* tuned is the
hybrid's fusion weight, on a validation split carved out of train (next section).

**Global temporal split, 80/20 — 1,082 evaluable users** (the headline split; cut at 2000-12-01):

| model                | recall | precision |   ndcg |    map | coverage | diversity | novelty |
|----------------------|-------:|----------:|-------:|-------:|---------:|----------:|--------:|
| popularity           | 0.0446 |    0.1761 | 0.1892 | 0.1134 |   0.0234 |    0.7187 |    8.06 |
| content              | 0.0070 |    0.0312 | 0.0319 | 0.0126 |   0.1257 |    0.3767 |   13.53 |
| item_knn             | 0.0517 |    0.1936 | 0.2103 | 0.1278 |   0.0422 |    0.7507 |    8.37 |
| ials                 | 0.0406 |    0.1368 | 0.1437 | 0.0743 |   0.2954 |    0.7050 |    9.68 |
| hybrid (w=0.3)       | 0.0316 |    0.1101 | 0.1193 | 0.0588 |   0.2163 |    0.4960 |    9.87 |
| hybrid (tuned, w=0)  | 0.0406 |    0.1368 | 0.1437 | 0.0743 |   0.2954 |    0.7050 |    9.68 |

**Leave-one-out split — 6,028 evaluable users** (for comparability with published numbers):

| model                 | recall | precision |   ndcg |    map | coverage | diversity | novelty |
|-----------------------|-------:|----------:|-------:|-------:|---------:|----------:|--------:|
| popularity            | 0.0405 |    0.0040 | 0.0193 | 0.0131 |   0.0296 |    0.7205 |    8.08 |
| content               | 0.0196 |    0.0020 | 0.0093 | 0.0062 |   0.2161 |    0.3498 |   13.43 |
| item_knn              | 0.0604 |    0.0060 | 0.0306 | 0.0217 |   0.0788 |    0.7367 |    8.42 |
| ials                  | 0.0854 |    0.0085 | 0.0430 | 0.0303 |   0.3961 |    0.6662 |    9.63 |
| hybrid (w=0.3)        | 0.0733 |    0.0073 | 0.0366 | 0.0256 |   0.3155 |    0.4656 |    9.87 |
| hybrid (tuned, w=0.05)| 0.0818 |    0.0082 | 0.0405 | 0.0281 |   0.3680 |    0.5716 |    9.67 |

`w` is the content model's weight in the rank-normalised blend `w·content + (1−w)·iALS`.
"Tuned" rows use the weight chosen on validation (next section); at w = 0 the hybrid is
identical to iALS.

**Read these with three caveats.**

1. *The two splits disagree on ordering.* On temporal, NDCG ranks item_kNN > popularity > iALS >
   hybrid > content: **popularity beats iALS**. On LOO it ranks iALS > item_kNN > hybrid >
   popularity > content. Only content-last is consistent. LOO has 5.6× the users but leaks mildly
   (see below), and temporal is the split that matches deployment, so neither is a clean answer.
   A hypothesis for the temporal result, *not tested here*: item popularity persists across the
   cut while untuned iALS fits per-user taste that drifts. Tuning iALS on a validation split
   would be the way to find out.
2. *Temporal evaluates a minority of users.* ml-1m covers 2000-04 to 2003-02 but most ratings
   land in 2000. Only 1,785 of 6,040 users have any rating after the cutoff, and 1,082 of those
   have ≥ 5 training interactions. The headline is a statement about users active late, not all
   users.
3. *Coverage is over the full 3,883-movie catalogue.* iALS reaches 40% under LOO while item_kNN
   reaches 7.9%; that is a more honest denominator than "items that happen to appear in test".

### Hybrid fusion weight

The original hybrid blended content 0.3 / iALS 0.7. It scored *below iALS alone* on both
splits, so the weight was tuned — on a validation split carved out of **train** (temporal: the
last 20% of train by time; LOO: each user's last training interaction), never on test.
`tune_content_weight` takes the train frame and nothing else of that kind, so test data has no
way in. Validation NDCG@10 by content weight:

| content weight | 0.0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| temporal (297 validation users) | **0.1288** | 0.1151 | 0.1131 | 0.1054 | 0.1018 | 0.0909 |
| LOO (6,016 validation users) | 0.0433 | **0.0438** | 0.0431 | 0.0397 | 0.0376 | 0.0337 |

The optimum is **at or next to zero on both splits**, and beyond 0.1 the curve falls steadily.
That is the finding: **a genre-and-title content signal is too weak to contribute to the blend on
this dataset.** Content alone scores NDCG 0.0319 (temporal) and 0.0093 (LOO) against iALS's
0.1437 and 0.0430, and mixing a ranker that weak into a stronger one only dilutes it.

Two things not to over-read. On LOO the validation winner is w = 0.05, but it beats w = 0 by
0.0005 NDCG (about 1%), which is indistinguishable from noise; and on the held-out test split
that 0.05 hybrid scored 0.0405, *below* iALS alone at 0.0430. The small validation gain did not
generalise. The weight was not revisited after seeing test (that would be tuning on test).

The hybrid is not deleted: it carries the cold-start popularity fallback and the content
component that `/similar` reads. The served model uses w = 0 (`scripts/train.py
--content-weight`), my call: it is the optimum on the headline (temporal) validation split, and
the LOO difference is within noise. Reproduce with `python experiments/run_eval.py --source db
--split temporal --tune-hybrid`.

### Why content-based underperforms here

The item representation is weak, and no fusion weight can fix that:

- **Genres only, and few of them.** ml-1m has 18 genre labels forming 301 distinct combinations
  across 3,883 movies; 843 of them are plain "Drama". Most items are near-indistinguishable
  from many others.
- **No plot text and no tag data.** Nothing describes what a film is *about* — tone, setting,
  cast, director.
- **The only extra text is the title, tokenised naively.** Title words ("the", "of", "love")
  mostly add noise. The release year is dropped by the tokenizer's letters-only pattern (the code
  comment in `content.py` says year is used; it is not).

Fixing it needs richer item content, not a better blend: plot-synopsis or review embeddings
(e.g. a sentence-transformer over TMDB/IMDb overviews), or the MovieLens **Tag Genome**
(relevance scores for ~1,100 tags per film, shipped with ml-latest and ml-25m but not ml-1m).
Neither is done here; the content model is reported as it is.

### Serving latency

In-process, **not end-to-end over HTTP**: no network, no ASGI, no JSON serialisation. This is a
lower bound on deployed latency. 3,883-movie catalogue, 6,040 users, k=10, 500 requests, one run
on one machine, artifact trained from the database.

| path   | p50 | p95 | p99 |
|--------|----:|----:|----:|
| cold   | 2.998 ms | 4.583 ms | 5.610 ms |
| cached | 0.001 ms | 0.001 ms | 0.003 ms |

The "cached" row times the memoised top-k lookup only. The `/recommend` endpoint also calls
`score_users` again to attach display scores, so a cache hit does not make a full request
near-free. Fixing that is future work.

## Persistence and accounts

PostgreSQL + SQLAlchemy 2.0 + Alembic. `make db-up` starts Postgres 16 (Docker Compose, loopback
only); schema changes are Alembic migrations, and a test fails if the models and migrations
disagree.

```bash
make db-up                                  # docker compose up -d --wait
export DATABASE_URL=postgresql+psycopg://cinematch:cinematch_dev@127.0.0.1:5432/cinematch
make db-migrate                             # alembic upgrade head
python scripts/ingest.py --data data/ml-1m  # idempotent; ~5 min for 1M ratings
python scripts/train.py --source db         # artifact trained from the database
export CINEMATCH_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
make api
```

Tables: `users`, `movies`, `interactions (user_id, movie_id, type, rating, created_at)`,
`watchlist`, and `index_counters`.

### One index space

A trained model addresses movies and users by *position*. If the position of a movie changes, the
model still scores fine and returns plausible, wrong recommendations; nothing raises. So:

- `movies.item_idx` and `users.user_idx` are **persistent, append-only columns**. Each is
  allocated once from a counter table and never reused, even if a row is deleted (a
  `MAX(col)+1` scheme would reuse the highest index, and two concurrent inserts could read the
  same maximum; the counter's `UPDATE` takes a row lock and rolls back with the transaction).
- The **CSV loader uses the same mapping**: every movie in `movies.csv` and every rating user,
  sorted by id. A fresh database import reproduces it exactly; later imports append. There is
  deliberately one mapping, not a DB one plus a CSV one with a reconciliation test.
- The trained artifact carries the id vectors it was trained on. At startup the API compares them
  with the database and **refuses to start on a mismatch** (rows appended since training are
  fine; any disagreement at a known index, or fewer rows, is fatal). `/health` reports
  `index_verified`.
- `users` and `movies` are never hard-deleted by the application; the loader raises if the index
  columns are not a contiguous `0..n-1`.

### Offline eval still works

`python experiments/run_eval.py` reads CSV by default and needs no database. `--source db` reads
the same data from Postgres. `tests/test_db_source.py` asserts the two produce identical frames
(values, dtypes, row order, original MovieLens timestamps), including a tie-heavy dataset: among
equal timestamps leave-one-out's "last interaction" is decided by row order, and reversing the
database's row order makes that test fail. `tests/test_run_eval_source.py` runs the real script
from both sources and requires exactly equal metrics. Interactions keep their original timestamps
in `created_at`; stamping imports with `now()` would destroy the temporal split.

### Schema choices

- **One `users` table.** The eval needs 6,040 MovieLens users (no email, no password); the
  product needs accounts. `email` and `password_hash` are nullable, `ml_user_id` is unique, and
  CHECK constraints enforce: `email IS NOT NULL OR ml_user_id IS NOT NULL`, lowercase email (which
  makes the unique index case-insensitive without `citext`), and "a password needs an email".
- **`interactions.type` is `VARCHAR` + `CHECK`, not a Postgres `ENUM`.** Adding a value to a
  native enum needs `ALTER TYPE` and cannot be rolled back cleanly; a CHECK is an ordinary
  drop/create. Types are `rating` and `watch`; a rating is required (0.5–5.0) iff the type is
  `rating`.
- **One row per `(user, movie, type)`**, so re-rating updates in place (and loses rating
  history). Its leading `user_id` serves "all of a user's interactions", so there is no separate
  `user_id` index; `movie_id` has its own for the foreign-key reverse lookup.
- **`movies.movie_id` is the MovieLens id**, the id the API already exposes; a surrogate would be
  a second id space. `genres` stays a pipe-delimited string because the content model tokenises
  exactly that string.

### Accounts and auth

`POST /auth/register`, `POST /auth/login`, `GET /me`, plus authenticated `POST/GET /interactions`
and `PUT/DELETE/GET /watchlist`. The caller's identity comes only from the token.

- **Passwords: Argon2id via `argon2-cffi`**, used directly (no passlib, which is unmaintained).
  Salted per hash, parameters embedded, rehashed on login when the cost changes.
- **Tokens: HS256 JWT, access-only, 60 minutes.** Decoding pins `algorithms=["HS256"]` and
  requires `exp` and `sub`, which closes `alg: none` and algorithm-confusion forgeries. The secret
  comes from `CINEMATCH_JWT_SECRET` with no default and a 32-character minimum; the service
  refuses to start without it when a database is configured.
- Unknown email, wrong password and an imported (password-less) user all return the same 401,
  and a dummy hash is verified for unknown emails, so the response does not reveal which emails
  are registered (timing is comparable, not constant).

**Out of scope, on purpose:** refresh tokens, token revocation/logout and secret rotation (a
stolen token is valid until it expires; rotating the secret logs everyone out); rate limiting
and account lockout; email verification and password reset.

These writes are **stored, not learned from**: nothing reaches the trained model and there is no
scheduled retraining. A user registered after training has no row in the model, and
`/recommend/{user_id}` still takes MovieLens ids.

## Design decisions

These are the things an interviewer will push on. Each is a real choice with a real cost.

**Global temporal split, not random.** Interactions are sorted by timestamp and cut at a
quantile, so every training point precedes every test point. A random split lets the model see
a user's future behaviour while predicting their past, which inflates every metric. Leave-one-out
is also implemented (`make eval-loo`) because much of the literature uses it, but it leaks
mildly: one user's held-out item can predate another user's training items. Temporal is the
headline; LOO is for comparability.

**Binarized at rating ≥ 4.0, not rating prediction.** We rank, so we optimise ranking. Low
ratings are discarded rather than treated as negatives — a 2-star rating means the user watched
and disliked, which is not the same as the implicit negative of never having seen the film.

**Full-catalogue ranking, not sampled negatives.** Each model scores every item with the user's
training items masked to `-inf`. Sampling 100 negatives per positive is far cheaper but is not a
consistent estimator of full-ranking metrics — the induced model ordering can differ from the
true one (Krichene & Rendle, KDD 2020).

**iALS, not explicit-feedback MF.** Unobserved cells are missing-not-at-random, not negatives.
iALS (Hu/Koren/Volinsky 2008) models them as zero-preference with low confidence and observed
cells as one-preference with confidence `1 + αr`. The ALS step precomputes the Gramian `YᵀY`
once per half-iteration, so each solve is `O(f²|Iᵤ| + f³)` instead of `O(f²·n_items)`.

**Item-kNN is included on purpose.** On sparse data a tuned neighbourhood model frequently beats
matrix factorisation. Omitting it is how people end up claiming MF "won" when it was never tested
against the strong baseline.

**Rank normalisation before fusion.** Cosine lives in [-1,1], iALS dot products are unbounded,
popularity counts are integers. Min-max is outlier-dominated; rank normalisation is scale-free.
The cost is that score margins are discarded — two near-tied items become adjacent ranks.

**Coverage, diversity and novelty are reported alongside accuracy.** An accuracy-optimal model
often serves the same few hundred items to everyone. In the LOO table above, `item_knn` reaches
8.7% catalogue coverage against iALS's 43.3%: competitive NDCG, and a product that shows most
users the same films.
That trade-off is invisible if you only report Recall@10.

**Cold-start is a fallback, not a fudge.** Users below `cold_threshold` training interactions get
popularity, because a profile built from one or two items is noise. Test users with fewer than
`--min-train` interactions are excluded from the main table so the fallback's contribution does
not get silently credited to the personalised models.

## Deliberately not here

Claiming these would be dishonest, so they are listed as gaps:

- **No Redis.** Recommendations are served from an artifact loaded into process memory and
  memoised in an in-process LRU. PostgreSQL holds accounts, interactions and the watchlist but is
  not consulted per recommendation.
- **No online feedback loop.** Interactions can be written (`POST /interactions`) but are stored,
  not learned from: no scheduled retraining, and users registered after training get no
  personalised recommendations.
- **No refresh tokens, revocation, secret rotation, rate limiting, email verification** (see
  "Accounts and auth").
- **No LLM query interface.**
- **No frontend.**
- **No hyperparameter search beyond the fusion weight.** iALS factors/regularisation/α and
  item-kNN's `k`/shrinkage are reasonable defaults, not tuned. The hybrid's content weight was
  tuned on a validation split carved from train (see Results). Tuning anything on the test split
  would invalidate the tables; extending the validation split to iALS is the next step, and the
  temporal ordering (popularity ahead of iALS) is the reason to do it.

## Layout

```
src/cinematch/
  data.py              loading, binarization, the full-catalogue index mapping
  loading.py           csv-or-db Dataset entry point
  split.py             temporal and leave-one-out splits
  tuning.py            validation-split tuning (never sees test)
  evaluate.py          full-catalogue ranking metrics
  api.py               FastAPI service, LRU cache, latency percentiles, startup index check
  accounts.py          register / login / me          auth.py   Argon2id + JWT primitives
  user_data.py         interactions and watchlist endpoints
  db/                  models, session, index allocator, ingest, db-backed loader, verify
  recommenders/        popularity, content, item_knn, ials, hybrid
migrations/            Alembic
scripts/               download_data, make_synthetic, ingest, train
experiments/           run_eval, benchmark_api
tests/                 oracle, leakage, DB equality, index-drift, auth and endpoint tests;
                       DB tests run on SQLite and on Postgres (make test-pg)
```

`make test` runs the suite. It includes an oracle recommender that must score NDCG exactly 1.0
and a check that no model ever recommends an already-seen item — the two failure modes that
would silently invalidate every number above.
