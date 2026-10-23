# CineMatch

Hybrid movie recommendation engine with a leakage-safe offline evaluation harness.

The point of this repo is the **evaluation**, not the model zoo. Five recommenders are
implemented so they can be compared against each other and against a popularity baseline
under a split that does not leak the future. Reporting "we built a hybrid recommender" without
that comparison says nothing.

## Quickstart

```bash
make install
make data          # downloads MovieLens ml-latest-small into data/
make eval          # trains all five models, prints the comparison table
make train && make bench && make api
```

No network? `make synth` generates a dataset with the same schema, and everything
(`make test`, `make eval --data data_synthetic`) runs offline.

## Results

MovieLens `ml-latest-small` (100,836 ratings; 48,580 positives at rating ≥ 4.0; 609 users,
6,298 items), k=10, full-catalogue ranking. iALS/item-kNN/hybrid hyperparameters are untuned
defaults (see "Deliberately not here").

**Leave-one-out split — 599 evaluable users** (the split with enough users to compare on):

| model      | recall | precision |   ndcg |    map | coverage | diversity | novelty |
|------------|-------:|----------:|-------:|-------:|---------:|----------:|--------:|
| popularity | 0.0384 |    0.0038 | 0.0205 | 0.0148 |   0.0097 |    0.7376 |    8.05 |
| content    | 0.0100 |    0.0010 | 0.0044 | 0.0026 |   0.0705 |    0.3430 |   13.20 |
| item_knn   | 0.0568 |    0.0057 | 0.0262 | 0.0171 |   0.0343 |    0.7446 |    8.57 |
| ials       | 0.0668 |    0.0067 | 0.0312 | 0.0205 |   0.1223 |    0.7531 |    9.51 |
| hybrid     | 0.0501 |    0.0050 | 0.0221 | 0.0139 |   0.1189 |    0.4381 |   10.41 |

**Global temporal split, 80/20 — only 26 evaluable users** (treat as indicative at best):

| model      | recall | precision |   ndcg |    map | coverage | diversity | novelty |
|------------|-------:|----------:|-------:|-------:|---------:|----------:|--------:|
| popularity | 0.0380 |    0.0885 | 0.1031 | 0.0647 |   0.0073 |    0.7520 |    8.08 |
| content    | 0.0091 |    0.0115 | 0.0089 | 0.0018 |   0.0160 |    0.3371 |   14.90 |
| item_knn   | 0.0411 |    0.0962 | 0.1081 | 0.0632 |   0.0133 |    0.7331 |    8.74 |
| ials       | 0.0285 |    0.0731 | 0.0796 | 0.0401 |   0.0284 |    0.7558 |    9.59 |
| hybrid     | 0.0286 |    0.0654 | 0.0733 | 0.0320 |   0.0248 |    0.4648 |   10.40 |

**Read these with three caveats.**

1. *The temporal split is data-starved on this dataset.* `ml-latest-small` spans 1996–2018 and
   most users rate in one short burst, so only 118 of 609 users have any rating after the cutoff
   (2016-02-04) and only 26 of those also have ≥ 5 training interactions. Differences of a few
   hundredths in NDCG over 26 users are not distinguishable from noise, and its coverage column
   is capped at 26×10 / 6,298 = 4.1% by construction, so it should not be compared across splits.
2. *The two splits disagree on ordering.* NDCG ranks item_kNN > popularity > iALS > hybrid on
   temporal but iALS > item_kNN > hybrid > popularity on LOO. Only content-last is consistent.
   LOO has 23× the users but leaks mildly (see below), so neither split is a clean answer.
3. *The hybrid does not beat its own best component.* It is below iALS on both splits and below
   popularity on temporal. It fuses iALS with the weakest model in the table (content), at
   untuned 0.3/0.7 weights, and rank-normalising a poor ranker into a good one dilutes it.

Serving latency, in-process, 1934-item catalogue, k=10:

| path   | p50 | p95 | p99 |
|--------|----:|----:|----:|
| cold   | 0.703 ms | 0.821 ms | 1.121 ms |
| cached | ~0 ms | ~0 ms | 0.001 ms |

In-process means no network or ASGI overhead — a lower bound on deployed latency, and it
should be described that way, not as an end-to-end service number.

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
3.4% catalogue coverage against iALS's 12.2%: competitive NDCG, and a product that shows most
users the same films.
That trade-off is invisible if you only report Recall@10.

**Cold-start is a fallback, not a fudge.** Users below `cold_threshold` training interactions get
popularity, because a profile built from one or two items is noise. Test users with fewer than
`--min-train` interactions are excluded from the main table so the fallback's contribution does
not get silently credited to the personalised models.

## Deliberately not here

Claiming these would be dishonest, so they are listed as gaps:

- **No PostgreSQL, no Redis.** Artifacts load into process memory; caching is an in-process LRU.
  A database and an external cache change the operational story, not recommendation quality.
- **No online feedback loop.** Interactions are batch inputs; there is no write path, no
  scheduled retraining.
- **No LLM query interface.**
- **No frontend.**
- **No hyperparameter search.** iALS factors/regularisation/α are reasonable defaults, not tuned.
  Tuning them on the test split would invalidate the table; a validation split is the next step.

## Layout

```
src/cinematch/
  data.py              loading, binarization, index mapping
  split.py             temporal and leave-one-out splits
  evaluate.py          full-catalogue ranking metrics
  api.py               FastAPI service, LRU cache, latency percentiles
  recommenders/        popularity, content, item_knn, ials, hybrid
scripts/               download_data, make_synthetic, train
experiments/           run_eval, benchmark_api
tests/                 26 tests incl. oracle and leakage checks
```

`make test` runs the suite. It includes an oracle recommender that must score NDCG exactly 1.0
and a check that no model ever recommends an already-seen item — the two failure modes that
would silently invalidate every number above.
