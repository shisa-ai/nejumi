#!/bin/bash
source ~/.bashrc
mamba activate nejumi

### Should be in env
# WANDB_API_KEY=<your WANDB_API_KEY>
# OPENAI_API_KEY=<your OPENAI_API_KEY>

export LANG=en_US.UTF-8
time python scripts/run_eval.py
