CONFIG ?= experiments/configs/baseline.yaml

.PHONY: setup test lint format download prepare-pdfs explain-config check-gpu index eval ablation models summarize notebook m1b m2 converse

setup:
	uv sync --extra ml
	uv run pre-commit install

test:
	uv run pytest

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

download:
	uv run paperlens download --split validation

prepare-pdfs:  ## download arXiv PDFs and parse them with Docling (needs --extra pdf)
	uv sync --extra ml --extra pdf
	uv run paperlens prepare-pdfs --config $(CONFIG)

explain-config:
	uv run paperlens explain-config --config $(CONFIG) --out docs/CONFIG.md

check-gpu:
	uv run paperlens check-gpu

index:
	uv run paperlens index --config $(CONFIG)

eval:
	uv run paperlens eval --config $(CONFIG)

ablation:
	uv run paperlens eval --config experiments/configs/baseline.yaml
	for f in experiments/configs/m1a/*.yaml; do uv run paperlens eval --config $$f || exit 1; done
	uv run paperlens summarize --out evals/results/SUMMARY.md

models:
	for f in experiments/configs/m1a_models/*.yaml; do uv run paperlens eval --config $$f || exit 1; done
	uv run paperlens summarize --out evals/results/SUMMARY.md

m1b:  ## every M1b ablation vs baseline, m1b_semantic_best and m1b_best, then judge and the summary table (hours of GPU; run `make prepare-pdfs` first)
	uv sync --extra ml --extra pdf
	for f in experiments/configs/m1b/*.yaml; do uv run paperlens eval --config $$f || exit 1; done
	uv run paperlens eval --config experiments/configs/m1b_semantic_best.yaml
	uv run paperlens eval --config experiments/configs/m1b_best.yaml
	uv run paperlens judge
	uv run paperlens summarize --out evals/results/SUMMARY.md

m2:  ## M2: corpus_rag floor and the LangGraph agent over the conversations, judge, summary (~1 h GPU)
	uv run --extra ml paperlens eval --config experiments/configs/m2_corpus_rag.yaml
	uv run --extra ml paperlens eval --config experiments/configs/m2_agent.yaml
	uv run paperlens judge --run m2_corpus_rag
	uv run paperlens judge --run m2_agent
	uv run paperlens summarize --out evals/results/SUMMARY.md

converse:  ## follow-up chats through the agent -> docs/m2_demo.md
	uv run --extra ml paperlens converse --out docs/m2_demo.md

summarize:
	uv run paperlens summarize --out evals/results/SUMMARY.md

notebook:  ## step-by-step RAG walkthrough in JupyterLab (http://localhost:8888)
	uv sync --extra ml --group notebook
	uv run jupyter lab --no-browser notebooks/
