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
to have a positive rating; see "One index space" below). `--source csv` and `--source db` give
identical numbers, which a test enforces. `ml-latest-small` is kept as the fast smoke dataset
(`make data-small && make smoke`) but is not used for headline numbers: its temporal split leaves
only 26 evaluable users.

Every `tuned` row below was tuned on a validation split carved out of train (see "Tuning
protocol"). The plain `item_knn` and `ials` rows are untuned defaults (item_knn k=200, shrink=10;
iALS 64 factors, regularization 0.05, alpha 20), kept in the table on purpose: the comparison
between them and the tuned rows is the first finding. Raw files: `results/ml1m_k10_tuned/`.

**Global temporal split, 80/20 — 1,082 evaluable users** (cut at 2000-12-01):

| model                | recall | precision |   ndcg |    map | coverage | diversity | novelty |
|----------------------|-------:|----------:|-------:|-------:|---------:|----------:|--------:|
| popularity           | 0.0446 |    0.1761 | 0.1892 | 0.1134 |   0.0234 |    0.7187 |    8.06 |
| content              | 0.0076 |    0.0328 | 0.0333 | 0.0129 |   0.1452 |    0.3520 |   13.33 |
| item_knn             | 0.0517 |    0.1936 | 0.2103 | 0.1278 |   0.0422 |    0.7507 |    8.37 |
| item_knn (tuned)     | 0.0519 |    0.1917 | 0.2092 | 0.1268 |   0.0402 |    0.7397 |    8.29 |
| ials                 | 0.0406 |    0.1368 | 0.1437 | 0.0743 |   0.2954 |    0.7050 |    9.68 |
| **ials (tuned)**     | 0.0574 |    0.1994 | 0.2141 | 0.1281 |   0.0466 |    0.7440 |    8.28 |
| hybrid (w=0.3)       | 0.0314 |    0.1072 | 0.1166 | 0.0568 |   0.2194 |    0.5017 |    9.90 |
| hybrid (tuned, w=0)  | 0.0574 |    0.1994 | 0.2141 | 0.1281 |   0.0466 |    0.7440 |    8.28 |

**Leave-one-out split — 6,028 evaluable users** (one relevant item per user):

| model                | recall | precision |   ndcg |    map | coverage | diversity | novelty |
|----------------------|-------:|----------:|-------:|-------:|---------:|----------:|--------:|
| popularity           | 0.0405 |    0.0040 | 0.0193 | 0.0131 |   0.0296 |    0.7205 |    8.08 |
| content              | 0.0212 |    0.0021 | 0.0102 | 0.0068 |   0.2439 |    0.3319 |   13.27 |
| item_knn             | 0.0604 |    0.0060 | 0.0306 | 0.0217 |   0.0788 |    0.7367 |    8.42 |
| item_knn (tuned)     | 0.0634 |    0.0063 | 0.0315 | 0.0220 |   0.0850 |    0.7239 |    8.40 |
| ials                 | 0.0854 |    0.0085 | 0.0430 | 0.0303 |   0.3961 |    0.6662 |    9.63 |
| **ials (tuned)**     | 0.1017 |    0.0102 | 0.0509 | 0.0358 |   0.2071 |    0.6731 |    9.22 |
| hybrid (w=0.3)       | 0.0768 |    0.0077 | 0.0387 | 0.0272 |   0.3217 |    0.4703 |    9.89 |
| hybrid (tuned, w=0)  | 0.1017 |    0.0102 | 0.0509 | 0.0358 |   0.2071 |    0.6731 |    9.22 |

`w` is the content model's weight in the rank-normalised blend `w·content + (1−w)·iALS`; at
w = 0 the hybrid is identical to iALS. **Absolute NDCG is not comparable across the two splits.**
Under LOO every user has exactly one relevant item, so the ideal DCG is fixed at 1; under the
temporal split users have anywhere from one to hundreds, and a list that hits some of many
relevant items scores far higher. Compare models within a split, never numbers across splits.

### Finding 1: tuning changes the conclusion

On the temporal split the default iALS **loses to popularity** (0.1437 vs 0.1892). Tuned, it
**beats** popularity (0.2141). An untuned model is not evidence about a model family: reporting
the default row alone would have concluded that matrix factorisation does not work on this data.
The same tuning moves LOO NDCG from 0.0430 to 0.0509, and item-kNN barely at all (0.2103 → 0.2092
temporal, 0.0306 → 0.0315 LOO; its default k=200 was already near a plateau).

### Finding 2: the gain comes from becoming conservative

Tuned iALS covers far less of the catalogue than the default:

| split | default iALS coverage | tuned iALS coverage | tuned regularization |
|---|---:|---:|---:|
| temporal | 29.5% | **4.7%** | 100 |
| LOO | 39.6% | **20.7%** | 30 |

Validation NDCG rewarded heavier regularization on both splits, and the tuned temporal model
(4.7%) now covers about as little as item-kNN (4.0%) and popularity (2.3%). **The tuned model wins
on accuracy by becoming more popularity-shaped.** This is why coverage is reported alongside NDCG
and not after it: the accuracy gain and the loss of catalogue reach are one trade, and a table of
NDCG alone would hide it. The temporal split tunes harder in this direction than LOO does. Why it
does is not established here; see "Why the two splits disagree" for what was and was not
supported.

### Finding 3: not every gap is real

Paired bootstrap over test users (1000 resamples; the same user draw is applied to every model,
so each interval is on the paired difference), 95% percentile intervals:

| difference in NDCG@10 | temporal (1,082 users) | LOO (6,028 users) |
|---|---|---|
| tuned iALS − tuned item-kNN | +0.0049 [−0.0009, +0.0105] — **not distinguishable** | +0.0194 [+0.0158, +0.0231] — distinguishable |
| tuned iALS − popularity | +0.0249 [+0.0174, +0.0332] — distinguishable | +0.0316 [+0.0271, +0.0364] — distinguishable |
| tuned item-kNN − popularity | +0.0200 [+0.0124, +0.0281] — distinguishable | +0.0122 [+0.0092, +0.0152] — distinguishable |

On the temporal split tuned iALS and tuned item-kNN are **comparable; no winner is declared**
(0.2141 vs 0.2092 should not be read as one). On LOO tuned iALS is clearly ahead. Both beat
popularity on both splits.

These intervals cover **test-user sampling noise only**. They do not include variance from the
iALS random initialisation (one seed was run) or from the choice of validation split, which on
the temporal split is only 297 users. Reproduce with `experiments/bootstrap_ci.py`.

### Tuning protocol

- **Validation only.** The validation split is carved out of train with the same strategy as the
  outer split (temporal: last 20% of train by time; LOO: each user's last training item).
  `tune_model` and `tune_content_weight` take the training frame and nothing else of that kind,
  so test data has no way in. Ties go to the first grid point.
- **item-kNN grid:** `k` ∈ {50, 100, 200, 400, 800, 1600, 3883} × `shrink` ∈ {0, 10, 50, 200}.
  The first grid stopped at k=400 and the optimum sat on that edge on both splits, so it was
  extended; k=1600 is interior on both. Optima: temporal k=1600, shrink=50; LOO k=1600, shrink=10.
- **iALS grid** (15 iterations fixed): stage 1 searched `factors` ∈ {32, 64, 128} ×
  `regularization` ∈ {0.1, 1, 10} × `alpha` ∈ {0.25, 0.5, 1, 2, 5}. The optimum landed on the
  regularization edge on both splits (and on the `factors` edge on temporal), so stage 2 added
  `factors` ∈ {16, 32, 64, 128} × `regularization` ∈ {10, 30, 100, 300} × `alpha` ∈ {0.5, 1, 2},
  reusing stage-1 trials rather than refitting. 84 distinct trials in total.
  **The full alpha range was not searched at the optimum:** alpha spans {0.25, 0.5, 1, 2, 5}
  overall, but only {0.5, 1, 2} at the tuned regularization. Alpha = 1 beat both neighbours there
  on both splits; 0.25 and 5 were not tried at regularization 30–100.
  Optima: temporal 128 factors / regularization 100 / alpha 1; LOO 128 / 30 / 1.
- **Grid edges.** Regularization and alpha are interior at the optimum. `factors` is on the
  upper edge (128) on both splits, so a validation-only probe fitted 256 factors at the
  tuned regularization and alpha. Temporal: 256 scores 0.1930 against 0.1935 for 128, so the
  optimum is interior. LOO: 256 scores 0.0531 against 0.0523 (and 0.0521 at 64), so the
  best value moves to the new edge, but the gains (64→128: +0.0002; 128→256: +0.0008) are small
  against a test-interval half-width of about 0.004 for the same number of users. **Over 64–256
  factors the factor count is flat within noise on LOO and is not identified by this
  procedure.** The one-standard-error rule (choose the simplest configuration within one standard
  error of the best) would pick a smaller model than the argmax; on LOO that includes 64
  factors. I kept 128, the argmax of the grid searched before the probe, and **64 would be
  equally defensible**. The 256-factor model was not evaluated on test.
- **How many times test was looked at.** Each `run_eval` run writes test metrics, and the stage-1
  run did: I saw tuned iALS (0.1798) trailing popularity (0.1892) on temporal. Nothing was
  *selected* on test: the stage-2 extension was triggered by validation edge optima, the
  factor probe is validation-only, and no configuration was chosen after seeing its test score.
  But the test split was viewed more than once, and the tables above are the final run.

### Why the two splits disagree

With tuned models the ordering is the same on both splits (iALS ≥ item-kNN > popularity), but the
size of the gap is not: tuned iALS is 1.13× popularity on temporal and 2.6× on LOO.
`experiments/analyze_splits.py` (tuned parameters read from the results JSON; nothing is chosen
from it) breaks per-user NDCG down by history length and by held-out count. One hypothesis I
started with was **refuted**, and the data supports a different account.

*Refuted: "the temporal test set is dominated by users with short histories."* The opposite
holds. Training-history length, in positives:

| | <20 | 20–49 | 50–99 | 100–199 | 200+ | share with <50 |
|---|---:|---:|---:|---:|---:|---:|
| temporal (1,082 users) | 100 | 208 | 258 | 296 | 220 | 28% |
| LOO (6,028 users) | 944 | 1,755 | 1,432 | 1,165 | 732 | 45% |

Temporal test users are *warmer*. Their held-out items are not more concentrated either: 17.0% of
temporal test interactions fall on the 100 most popular train films, against 20.2% for LOO.

*Supported: the temporal metric is weighted toward users with many held-out films.* Users still
active after the cutoff are heavy raters; 56% of temporal test users have 20 or more held-out
positives (LOO: always one). Popularity is strong for them because any popular film is a likely hit:

| held-out items per user | 1 | 2–4 | 5–19 | 20–49 | 50+ |
|---|---:|---:|---:|---:|---:|
| temporal users | 59 | 121 | 297 | 264 | 341 |
| popularity NDCG | 0.0253 | 0.0202 | 0.0803 | 0.1615 | 0.3938 |
| tuned iALS NDCG | 0.0688 | 0.0428 | 0.0933 | 0.1927 | 0.4217 |
| iALS ÷ popularity | 2.7× | 2.1× | 1.16× | 1.19× | 1.07× |

Matrix factorisation's edge is largest where a user has few relevant items to find, and shrinks to
a few percent for heavy raters. The temporal test set is mostly heavy raters; LOO is entirely
few-relevant-item users (iALS ÷ popularity there: 3.5× for users with under 20 training
positives, 4.0× for 20–49, 1.5–1.65× for 50–199, 1.07× for 200+). So the splits answer different
questions: LOO asks how well a model finds one more film for a typical user; temporal asks how
well it fills a long list for the users who were still active late.

*Not established:* why tuning pushes harder toward popularity-shaped output on temporal
(regularization 100, coverage 4.7%) than on LOO (30, 20.7%). Candidates, none tested: taste drift
across the cut rewards smoother, more popularity-like scores; the temporal validation split has
only 297 users and is a noisy target; or the heavy-rater weighting above carries over into the
validation split. The bucket figures come from one fit and carry no confidence intervals.

### Hybrid fusion weight

The hybrid blends content and iALS with weight `w`. The weight was tuned on validation against the
*tuned* iALS (so the partner is the strongest available model). Validation NDCG@10:

| content weight | 0.0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| temporal (297 validation users) | **0.1935** | 0.1672 | 0.1526 | 0.1429 | 0.1348 | 0.1184 |
| LOO (6,016 validation users) | **0.0523** | 0.0476 | 0.0456 | 0.0437 | 0.0413 | 0.0351 |

The optimum is exactly zero on both splits and the curve falls monotonically. A genre-and-title
content signal is too weak to help a strong collaborative model on this dataset: mixing in a
ranker that scores 0.0333 / 0.0102 NDCG against iALS's 0.2141 / 0.0509 only dilutes it. (Against
the *default* iALS the LOO optimum was 0.05, ahead of 0 by 0.0005, which is within noise; it
disappears once the partner is tuned.) The hybrid is kept because it carries the cold-start
popularity fallback and the content component that `/similar` reads. The served model uses w = 0
(`scripts/train.py --content-weight`). Reproduce the tuning with `python
experiments/run_eval.py --split temporal --tune-models --tune-hybrid --ials-stages 2`.

### Why content-based underperforms here

The item representation is weak, and no fusion weight can fix that:

- **Genres only, and few of them.** ml-1m has 18 genre labels forming 301 distinct combinations
  across 3,883 movies; 843 of them are plain "Drama". Most items are near-indistinguishable
  from many others.
- **No plot text and no tag data.** Nothing describes what a film is *about*.
- **The only extra text is the title.** Title words mostly add noise.
- **The release year did not help meaningfully.** The tokenizer used to drop digits, so the year
  (in the title, "Heat (1995)") was not a feature even though a code comment said it was. It now
  admits digits, and a test checks every year is in the vocabulary. Content NDCG moved from 0.0332
  to 0.0333 on temporal and from 0.0093 to 0.0102 on LOO. The temporal change is zero within
  rounding; the LOO change is small, and **I did not test it against noise**, so it should not be
  read as an effect.

Fixing it needs richer item content, not a better blend: plot-synopsis or review embeddings, or
the MovieLens **Tag Genome** (shipped with ml-latest and ml-25m but not ml-1m). Neither is done
here; the content model is reported as it is.

### Serving latency

In-process, **not end-to-end over HTTP**: no network, no ASGI, no JSON serialisation. This is a
**lower bound** on deployed latency. 3,883-movie catalogue, 6,040 users, k=10, 500 requests, one
run on one machine.

| path   | mean | p50 | p95 | p99 |
|--------|----:|----:|----:|----:|
| cold   | 3.30 ms | 3.10 ms | 4.59 ms | 5.68 ms |
| cached | 0.001 ms | 0.001 ms | 0.001 ms | 0.001 ms |

A cold request scores the user once and returns the top-k with scores. The cache stores the
`(item, score)` pairs themselves, so a hit is a dictionary lookup and never touches the model
(a test asserts a hit makes zero model calls and returns identical scores). No speedup ratio is
quoted: the cached figure is a lookup, so a ratio against it says nothing about the model.

### New users: fold-in

A registered user absent from training gets personalised recommendations from
`GET /me/recommendations` without retraining. `IALSRecommender.fold_in` solves one least-squares
problem for the user's latent vector against the **fixed** item factors; it is the user half-step
of training (`_solve_row`), so it takes milliseconds and needs no scheduler or worker. The vector
is cached per user and recomputed when the user's interactions change, keyed on
`(interaction count, newest timestamp)`: the count alone is not enough, since re-rating an item
leaves it unchanged while changing the user's liked set.

**Future work: periodic full retraining.** Fold-in is online and cheap, but the item factors never
learn from new users; only a full retrain does that.

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

These writes are **stored, not learned from by training**: nothing reaches the trained factors and
there is no scheduled retraining. A user registered after training has no row in the model, so
`/recommend/{user_id}` (which takes MovieLens ids) does not apply to them; their personalised
list comes from fold-in at `GET /me/recommendations` (see "New users: fold-in").

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
often serves the same few hundred items to everyone. In the LOO table above, default `item_knn`
reaches 7.9% catalogue coverage against default iALS's 39.6%, and tuning iALS for NDCG then
cuts its coverage to 20.7% (4.7% on temporal): accuracy and reach trade against each other.
That trade-off is invisible if you only report Recall@10 or NDCG.

**Cold-start is a fallback, not a fudge.** Users below `cold_threshold` training interactions get
popularity, because a profile built from one or two items is noise. Test users with fewer than
`--min-train` interactions are excluded from the main table so the fallback's contribution does
not get silently credited to the personalised models.

## Deliberately not here

Claiming these would be dishonest, so they are listed as gaps:

- **No Redis.** Recommendations are served from an artifact loaded into process memory and
  memoised in an in-process LRU. PostgreSQL holds accounts, interactions and the watchlist but is
  not consulted per recommendation.
- **No retraining loop.** Interactions can be written (`POST /interactions`) and a registered
  user is personalised by fold-in against the fixed item factors, but the item factors themselves
  are never updated: no scheduled or background retraining. Periodic full retraining is future
  work; fold-in is online and cheap, but the item factors never learn from new users.
- **No refresh tokens, revocation, secret rotation, rate limiting, email verification** (see
  "Accounts and auth").
- **No LLM query interface.**
- **No claim about the frontend's quality.** `frontend/` is a React + TypeScript + Vite control
  panel over the API; it is not part of the evaluation.
- **A narrow hyperparameter search.** iALS and item-kNN were grid-searched on validation only
  (see "Tuning protocol"). `iterations` was fixed at 15, one random seed was run, the validation
  split is small on the temporal side (297 users), and the LOO `factors` optimum is not identified
  (flat within noise over 64–256).

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
experiments/           run_eval, benchmark_api, analyze_splits, bootstrap_ci, probe_factors
results/               committed copies of the final tuned-run outputs cited in this README
frontend/              React + TypeScript + Vite UI (separate from the evaluation)
tests/                 oracle, leakage, DB equality, index-drift, auth and endpoint tests;
                       DB tests run on SQLite and on Postgres (make test-pg)
```

`make test` runs the suite. It includes an oracle recommender that must score NDCG exactly 1.0
and a check that no model ever recommends an already-seen item — the two failure modes that
would silently invalidate every number above.
