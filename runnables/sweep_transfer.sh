#!/bin/bash
# Sweep script for transfer learning experiments
# Usage: bash runnables/sweep_transfer.sh <pretrained_ckpt_path>
#
# Runs zero-shot, few-shot (varying sample sizes), and fine-tune (varying LR/epochs)
# experiments on the downstream tumour growth dataset.
# Parallelizes across 4 GPUs (up to 4 experiments at a time).

set -e

CKPT_PATH="${1:?Usage: bash runnables/sweep_transfer.sh <pretrained_ckpt_path>}"
PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
SEED=101
COEFF=10.0
RESULTS_DIR="outputs/transfer_sweep"
NUM_GPUS=4

cd /data1/letternight/DynamicCausalPFN
export PYTHONPATH=/data1/letternight/DynamicCausalPFN:$PYTHONPATH

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "Transfer Learning Sweep (4 GPUs parallel)"
echo "Checkpoint: $CKPT_PATH"
echo "Results dir: $RESULTS_DIR"
echo "=============================================="

# Helper: run an experiment on a specific GPU in the background
run_exp() {
    local gpu_id=$1
    local name=$2
    shift 2
    echo "[GPU $gpu_id] Starting $name..."
    $PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
        +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_tuned \
        +dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
        exp.gpus="[$gpu_id]" exp.logging=False \
        'model.dynamic_causal_pfn.pretrained_ckpt='"$CKPT_PATH" \
        "$@" \
        > "$RESULTS_DIR/${name}.log" 2>&1
    echo "[GPU $gpu_id] Finished $name"
}

# --- Batch 1: zero-shot + few-shot n=200,300,500 ---
echo ""
echo "=== Batch 1/3 ==="
run_exp 0 "zero_shot" \
    model.dynamic_causal_pfn.transfer_mode=zero_shot \
    hydra.run.dir="$RESULTS_DIR/zero_shot" &

run_exp 1 "few_shot_n200" \
    exp.max_epochs=25 \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=200 \
    hydra.run.dir="$RESULTS_DIR/few_shot_n200" &

run_exp 2 "few_shot_n300" \
    exp.max_epochs=25 \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=300 \
    hydra.run.dir="$RESULTS_DIR/few_shot_n300" &

run_exp 3 "few_shot_n500" \
    exp.max_epochs=25 \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=500 \
    hydra.run.dir="$RESULTS_DIR/few_shot_n500" &

wait
echo "Batch 1 complete."

# --- Batch 2: fine-tune lr=0.0001,0.00005 + fine-tune ep=25,50 ---
echo ""
echo "=== Batch 2/3 ==="
run_exp 0 "fine_tune_lr0.0001" \
    exp.max_epochs=25 \
    model.dynamic_causal_pfn.transfer_mode=fine_tune \
    model.dynamic_causal_pfn.optimizer.learning_rate=0.0001 \
    hydra.run.dir="$RESULTS_DIR/fine_tune_lr0.0001" &

run_exp 1 "fine_tune_lr0.00005" \
    exp.max_epochs=25 \
    model.dynamic_causal_pfn.transfer_mode=fine_tune \
    model.dynamic_causal_pfn.optimizer.learning_rate=0.00005 \
    hydra.run.dir="$RESULTS_DIR/fine_tune_lr0.00005" &

run_exp 2 "fine_tune_ep25" \
    exp.max_epochs=25 \
    model.dynamic_causal_pfn.transfer_mode=fine_tune \
    model.dynamic_causal_pfn.optimizer.learning_rate=0.0001 \
    hydra.run.dir="$RESULTS_DIR/fine_tune_ep25" &

run_exp 3 "fine_tune_ep50" \
    exp.max_epochs=50 \
    model.dynamic_causal_pfn.transfer_mode=fine_tune \
    model.dynamic_causal_pfn.optimizer.learning_rate=0.0001 \
    hydra.run.dir="$RESULTS_DIR/fine_tune_ep50" &

wait
echo "Batch 2 complete."

echo ""
echo "=============================================="
echo "Sweep complete. Results in: $RESULTS_DIR/"
echo "=============================================="

# --- Summary ---
echo ""
echo "=== RESULTS SUMMARY ==="
for logfile in "$RESULTS_DIR"/*.log; do
    name=$(basename "$logfile" .log)
    echo "--- $name ---"
    grep -E "(Val normalised RMSE|Test normalised RMSE)" "$logfile" || echo "  (no RMSE found)"
done
