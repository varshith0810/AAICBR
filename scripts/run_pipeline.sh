#!/usr/bin/env bash
set -euo pipefail

python -m src.preprocess
python -m src.train
python app.py
