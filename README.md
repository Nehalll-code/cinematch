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

No network? `make synth` generates a dataset with the same schema, and everything
(`make test`, `python experiments/run_eval.py --data data_synthetic`) runs offline.
`make data-small && make smoke` runs the same pipeline on ml-latest-small in about a minute.

## Results

MovieLens **ml-1m** (1,000,209 ratings; 575,281 positives at rating ≥ 4.0; 6,038 users with at
least one positive; 3,533 items with at least one positive), k=10, full-catalogue ranking.
`ml-latest-small` is kept as the fast smoke dataset (`make data-small && make smoke`) but is not
used for headline numbers: its temporal split leaves only 26 evaluable users.

iALS and item-kNN hyperparameters are untuned defaults. The one thing that *was* tuned is the
hybrid's fusion weight, on a validation split carved out of train (next section).

**Global temporal split, 80/20 — 1,082 evaluable users** (the headline split; cut at 2000-12-01):

| model            | recall | precision |   ndcg |    map | coverage | diversity | novelty |
|------------------|-------:|----------:|-------:|-------:|---------:|----------:|--------:|
| popularity       | 0.0446 |    0.1761 | 0.1892 | 0.1134 |   0.0258 |    0.7179 |    8.06 |
| content          | 0.0072 |    0.0328 | 0.0332 | 0.0129 |   0.1407 |    0.3847 |   12.64 |
| item_knn         | 0.0517 |    0.1937 | 0.2103 | 0.1279 |   0.0464 |    0.7500 |    8.37 |
| ials             | 0.0409 |    0.1388 | 0.1474 | 0.0756 |   0.3207 |    0.7008 |    9.66 |
| hybrid (w=0.3)   | 0.0338 |    0.1137 | 0.1205 | 0.0589 |   0.2423 |    0.4982 |    9.86 |
| hybrid (tuned)   | 0.0409 |    0.1388 | 0.1474 | 0.0756 |   0.3207 |    0.7008 |    9.66 |

**Leave-one-out split — 6,028 evaluable users** (for comparability with published numbers):

| model            | recall | precision |   ndcg |    map | coverage | diversity | novelty |
|------------------|-------:|----------:|-------:|-------:|---------:|----------:|--------:|
| popularity       | 0.0406 |    0.0041 | 0.0194 | 0.0131 |   0.0326 |    0.7194 |    8.08 |
| content          | 0.0206 |    0.0021 | 0.0098 | 0.0066 |   0.2358 |    0.3563 |   12.55 |
| item_knn         | 0.0604 |    0.0060 | 0.0306 | 0.0217 |   0.0866 |    0.7360 |    8.42 |
| ials             | 0.0868 |    0.0087 | 0.0436 | 0.0306 |   0.4331 |    0.6643 |    9.63 |
| hybrid (w=0.3)   | 0.0718 |    0.0072 | 0.0356 | 0.0247 |   0.3487 |    0.4672 |    9.86 |
| hybrid (tuned)   | 0.0868 |    0.0087 | 0.0436 | 0.0306 |   0.4331 |    0.6643 |    9.63 |

`w` is the content model's weight in the rank-normalised blend `w·content + (1−w)·iALS`.
"Tuned" rows use w = 0 (below), which makes the hybrid identical to iALS.

**Read these with three caveats.**

1. *The two splits disagree on ordering.* On temporal, NDCG ranks item_kNN > popularity > iALS >
   hybrid > content: **popularity beats iALS**. On LOO it ranks iALS > item_kNN > hybrid >
   popularity > content. Only content-last is consistent. LOO has 5.6× the users but leaks mildly
   (see below), and temporal is the split that matches deployment, so neither is a clean answer.
   A hypothesis for the temporal result, *not tested here*: item popularity persists across the
   cut while untuned iALS fits per-user taste that drifts. Tuning iALS on a validation split
   would be the way to find out.
2. *Temporal evaluates a minority of users.* ml-1m covers 2000-04 to 2003-02 but most ratings
   land in 2000. Only 1,785 of 6,038 users have any rating after the cutoff, and 1,082 of those
   have ≥ 5 training interactions. The headline is a statement about users active late, not all
   users.
3. *Coverage is over the full 3,533-item catalogue*, not over items that happen to appear in
   test. iALS reaches 43% under LOO while item_kNN reaches 8.7%.

### Hybrid fusion weight

The original hybrid blended content 0.3 / iALS 0.7. It scored *below iALS alone* on both
splits, so the weight was tuned — on a validation split carved out of **train** (temporal: the
last 20% of train by time; LOO: each user's last training interaction), never on test.
`tune_content_weight` takes the train frame and nothing else of that kind, so test data has no
way in. Validation NDCG@10 by content weight:

| content weight | 0.0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| temporal (297 validation users) | **0.1277** | 0.1163 | 0.1122 | 0.1051 | 0.0981 | 0.0922 |
| LOO (6,016 validation users) | **0.0438** | 0.0430 | 0.0412 | 0.0399 | 0.0381 | 0.0340 |

The optimum is **zero on both splits** and the curve falls monotonically as content weight
rises. That is the finding: **a genre-and-title content signal is too weak to contribute to the
blend on this dataset.** Content alone scores NDCG 0.0332 (temporal) and 0.0098 (LOO) against
iALS's 0.1474 and 0.0436, and mixing a ranker that weak into a stronger one only dilutes it.
The hybrid is not deleted: it still carries the cold-start popularity fallback and the content
component that `/similar` reads. The served model uses w = 0 (`scripts/train.py
--content-weight`). Reproduce with `python experiments/run_eval.py --data data/ml-1m --split
temporal --tune-hybrid`.

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
lower bound on deployed latency. 3,533-item catalogue, 6,038 users, k=10, 500 requests, one run
on one machine.

| path   | p50 | p95 | p99 |
|--------|----:|----:|----:|
| cold   | 2.907 ms | 4.183 ms | 5.234 ms |
| cached | 0.001 ms | 0.001 ms | 0.004 ms |

The "cached" row times the memoised top-k lookup only. The `/recommend` endpoint also calls
`score_users` again to attach display scores, so a cache hit does not make a full request
near-free. Fixing that is future work.

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

- **No PostgreSQL, no Redis.** Artifacts load into process memory; caching is an in-process LRU.
  A database and an external cache change the operational story, not recommendation quality.
- **No online feedback loop.** Interactions are batch inputs; there is no write path, no
  scheduled retraining.
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
  data.py              loading, binarization, index mapping
  split.py             temporal and leave-one-out splits
  evaluate.py          full-catalogue ranking metrics
  api.py               FastAPI service, LRU cache, latency percentiles
  recommenders/        popularity, content, item_knn, ials, hybrid
scripts/               download_data, make_synthetic, train
experiments/           run_eval, benchmark_api
tests/                 32 tests incl. oracle, leakage and no-test-in-tuning checks
```

`make test` runs the suite. It includes an oracle recommender that must score NDCG exactly 1.0
and a check that no model ever recommends an already-seen item — the two failure modes that
would silently invalidate every number above.
