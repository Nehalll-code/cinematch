.PHONY: install data data-small synth smoke train eval eval-loo bench api test clean db-up db-down db-migrate db-ingest test-pg

install:
	pip install -r requirements.txt

data:            ## headline dataset: MovieLens 1M into data/ml-1m
	python scripts/download_data.py --size 1m --out data/ml-1m

data-small:      ## fast smoke dataset: ml-latest-small into data/small
	python scripts/download_data.py --size small --out data/small

synth:           ## generate offline synthetic data (no network)
	python scripts/make_synthetic.py --out data_synthetic

smoke:           ## quick end-to-end eval on the small dataset
	python experiments/run_eval.py --data data/small --split loo --k 10 --out artifacts_smoke

train:
	python scripts/train.py --data data/ml-1m --out artifacts/model.pkl

eval:            ## headline numbers: global temporal split
	python experiments/run_eval.py --data data/ml-1m --split temporal --k 10

eval-loo:        ## leave-one-out, for comparison with published results
	python experiments/run_eval.py --data data/ml-1m --split loo --k 10

bench:
	python experiments/benchmark_api.py --model artifacts/model.pkl --requests 500

api:
	CINEMATCH_MODEL=artifacts/model.pkl uvicorn cinematch.api:app --app-dir src --reload --port 8000

test:
	python -m pytest tests/ -q

db-up:           ## start Postgres (docker compose), wait until healthy
	docker compose up -d --wait

db-down:
	docker compose down

db-migrate:      ## apply Alembic migrations to $$DATABASE_URL
	python -m alembic upgrade head

test-pg:         ## run the suite with the Postgres variants of the DB tests enabled
	CINEMATCH_TEST_DATABASE_URL=postgresql+psycopg://cinematch:cinematch_dev@127.0.0.1:5432/cinematch_test python -m pytest tests/ -q -rs

clean:
	rm -rf artifacts artifacts_synth data_synthetic .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
