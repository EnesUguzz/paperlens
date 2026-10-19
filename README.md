# paperlens

Agentic RAG over scientific papers (QASPER), built to measure every design choice.

> Work in progress — milestone M1a: baseline RAG + evaluation loop.

## Quickstart (WSL2 / Linux, NVIDIA GPU, Ollama)

    make setup
    make download
    make eval CONFIG=experiments/configs/baseline.yaml

Results: `evals/results/<run_name>/report.md`. Parameter reference: `docs/CONFIG.md`.

## Environment

Python 3.14 (see `.python-version`), CUDA 13 wheels for PyTorch.
