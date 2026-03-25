#!/bin/bash
# Sweep script for transfer learning experiments on MIMIC
# Usage: bash runnables/sweep_transfer_mimic.sh <pretrained_ckpt_path>

set -e

CKPT_PATH="${1:?Usage: bash runnables/sweep_transfer_mimic.sh <pretrained_ckpt_path>}"
PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
SEED=101
RESULTS_DIR="outputs/mimic_transfer_sweep"

cd /data1/letternight/DynamicCausalPFN
export PYTHONPATH=/data1/letternight/DynamicCausalPFN:$PYTHONPATH

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "MIMIC Transfer Learning Sweep (4 GPUs parallel)"
echo "Checkpoint: $CKPT_PATH"
echo "Results dir: $RESULTS_DIR"
echo "=============================================="

run_exp() {
    local gpu_id=$1
    local name=$2
    shift 2
    echo "[GPU $gpu_id] Starting $name..."
    $PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
        +backbone/dynamic_causal_pfn_hparams=mimic3_pretrain_tuned \
        +dataset=mimic3_synthetic exp.seed=$SEED \
        exp.gpus="[$gpu_id]" exp.logging=False \
        'model.dynamic_causal_pfn.pretrained_ckpt='"$CKPT_PATH" \
        "$@" \
        > "$RESULTS_DIR/${name}.log" 2>&1
    echo "[GPU $gpu_id] Finished $name"
}

# --- Batch 1: zero-shot + few-shot n=100,200,500 (ep=50) ---
echo ""
echo "=== Batch 1/3 ==="
run_exp 0 "zero_shot" \
    model.dynamic_causal_pfn.transfer_mode=zero_shot \
    hydra.run.dir="$RESULTS_DIR/zero_shot" &

run_exp 1 "few_shot_n100_ep50" \
    exp.max_epochs=50 \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=100 \
    hydra.run.dir="$RESULTS_DIR/few_shot_n100_ep50" &

run_exp 2 "few_shot_n200_ep50" \
    exp.max_epochs=50 \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=200 \
    hydra.run.dir="$RESULTS_DIR/few_shot_n200_ep50" &

run_exp 3 "few_shot_n500_ep50" \
    exp.max_epochs=50 \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=500 \
    hydra.run.dir="$RESULTS_DIR/few_shot_n500_ep50" &

wait
echo "Batch 1 complete."

# --- Batch 2: few-shot n=100,200,500 (ep=100) ---
echo ""
echo "=== Batch 2/3 ==="
run_exp 0 "few_shot_n100_ep100" \
    exp.max_epochs=100 \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=100 \
    hydra.run.dir="$RESULTS_DIR/few_shot_n100_ep100" &

run_exp 1 "few_shot_n200_ep100" \
    exp.max_epochs=100 \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=200 \
    hydra.run.dir="$RESULTS_DIR/few_shot_n200_ep100" &

run_exp 2 "few_shot_n500_ep100" \
    exp.max_epochs=100 \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=500 \
    hydra.run.dir="$RESULTS_DIR/few_shot_n500_ep100" &

wait
echo "Batch 2 complete."

# --- Batch 3: fine-tune lr=0.0001 (ep=25,50,100) ---
echo ""
echo "=== Batch 3/3 ==="
run_exp 0 "fine_tune_ep25" \
    exp.max_epochs=25 \
    model.dynamic_causal_pfn.transfer_mode=fine_tune \
    model.dynamic_causal_pfn.optimizer.learning_rate=0.0001 \
    hydra.run.dir="$RESULTS_DIR/fine_tune_ep25" &

run_exp 1 "fine_tune_ep50" \
    exp.max_epochs=50 \
    model.dynamic_causal_pfn.transfer_mode=fine_tune \
    model.dynamic_causal_pfn.optimizer.learning_rate=0.0001 \
    hydra.run.dir="$RESULTS_DIR/fine_tune_ep50" &

run_exp 2 "fine_tune_ep100" \
    exp.max_epochs=100 \
    model.dynamic_causal_pfn.transfer_mode=fine_tune \
    model.dynamic_causal_pfn.optimizer.learning_rate=0.0001 \
    hydra.run.dir="$RESULTS_DIR/fine_tune_ep100" &

wait
echo "Batch 3 complete."

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
