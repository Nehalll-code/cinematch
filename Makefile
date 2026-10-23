.PHONY: install data synth train eval eval-loo bench api test clean

install:
	pip install -r requirements.txt

data:            ## download real MovieLens into data/
	python scripts/download_data.py --size small

synth:           ## generate offline synthetic data (no network)
	python scripts/make_synthetic.py --out data_synthetic

train:
	python scripts/train.py --data data --out artifacts/model.pkl

eval:            ## headline numbers: global temporal split
	python experiments/run_eval.py --data data --split temporal --k 10

eval-loo:        ## leave-one-out, for comparison with published results
	python experiments/run_eval.py --data data --split loo --k 10

bench:
	python experiments/benchmark_api.py --model artifacts/model.pkl --requests 500

api:
	CINEMATCH_MODEL=artifacts/model.pkl uvicorn cinematch.api:app --app-dir src --reload --port 8000

test:
	python -m pytest tests/ -q

clean:
	rm -rf artifacts artifacts_synth data_synthetic .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
