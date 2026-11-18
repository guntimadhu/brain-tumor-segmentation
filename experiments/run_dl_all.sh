#!/usr/bin/env bash
# Train both deep-learning models with the three seeds used in the base paper (sequential, CPU).
set -e
cd "$(dirname "$0")/.."
for seed in 42 2023 777; do
  for model in unet attention_unet; do
    if [ -f "models/${model}_seed${seed}.pt" ] && grep -q "best" "results/dl/${model}_seed${seed}.out" 2>/dev/null; then
      echo "skip ${model} seed ${seed}"; continue
    fi
    echo "=== ${model} seed ${seed} $(date)"
    python -u experiments/train_dl.py --model "$model" --seed "$seed" --epochs 20 > "results/dl/${model}_seed${seed}.out" 2>&1
    tail -1 "results/dl/${model}_seed${seed}.out"
  done
done
echo "ALL DONE $(date)"
