.PHONY: install run serve test docker

install:
	python -m pip install -r requirements.txt

run:            ## train, evaluate, write outputs/ and reports/
	python -m src.pipeline

serve:          ## start the API on http://127.0.0.1:8000 (docs at /docs, chat at /assistant)
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

test:
	python -m pytest -q tests

docker:         ## one command, end to end
	docker compose up --build
