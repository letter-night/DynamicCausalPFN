#!/bin/bash
# Evaluate DynamicCausalPFN on mimic3_synthetic across seeds and transfer modes
# Usage: bash runnables/eval_dcpfn_mimic.sh <pretrained_ckpt_path>

set -e

CKPT_PATH="${1:?Usage: bash runnables/eval_dcpfn_mimic.sh <pretrained_ckpt_path>}"
PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
SEEDS=(101 102 103 104 105)
RESULTS_DIR="outputs/mimic_dcpfn_eval"
NUM_GPUS=4

cd /data1/letternight/DynamicCausalPFN
export PYTHONPATH=/data1/letternight/DynamicCausalPFN:$PYTHONPATH

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "DynamicCausalPFN Evaluation on mimic3_synthetic"
echo "Checkpoint: $CKPT_PATH"
echo "Seeds: ${SEEDS[*]}"
echo "Modes: zero_shot, few_shot_n500, fine_tune"
echo "=============================================="

declare -a EXPERIMENTS
for SEED in "${SEEDS[@]}"; do
    EXPERIMENTS+=("zero_shot_seed${SEED}|${SEED}|zero_shot|0|25")
    EXPERIMENTS+=("few_shot_n500_seed${SEED}|${SEED}|few_shot|500|100")
    EXPERIMENTS+=("fine_tune_seed${SEED}|${SEED}|fine_tune|0|100")
done

TOTAL=${#EXPERIMENTS[@]}
echo "Total experiments: $TOTAL"

idx=0
batch=1
while [ $idx -lt $TOTAL ]; do
    echo ""
    echo "=== Batch $batch (experiments $((idx+1))-$((idx+NUM_GPUS < TOTAL ? idx+NUM_GPUS : TOTAL))/$TOTAL) ==="

    pids=()
    for gpu in $(seq 0 $((NUM_GPUS-1))); do
        if [ $idx -lt $TOTAL ]; then
            IFS='|' read -r name seed mode few_shot_n max_epochs <<< "${EXPERIMENTS[$idx]}"

            EXTRA_ARGS=""
            if [ "$mode" = "few_shot" ]; then
                EXTRA_ARGS="model.dynamic_causal_pfn.few_shot_samples=$few_shot_n"
            fi
            if [ "$mode" = "fine_tune" ]; then
                EXTRA_ARGS="model.dynamic_causal_pfn.optimizer.learning_rate=0.0001"
            fi

            echo "[GPU $gpu] $name"
            $PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
                +backbone/dynamic_causal_pfn_hparams=mimic3_pretrain_tuned \
                +dataset=mimic3_synthetic exp.seed=$seed \
                exp.gpus="[$gpu]" exp.logging=True \
                exp.max_epochs=$max_epochs \
                'model.dynamic_causal_pfn.pretrained_ckpt='"$CKPT_PATH" \
                model.dynamic_causal_pfn.transfer_mode=$mode \
                $EXTRA_ARGS \
                hydra.run.dir="$RESULTS_DIR/$name" \
                > "$RESULTS_DIR/${name}.log" 2>&1 &
            pids+=($!)

            idx=$((idx+1))
        fi
    done

    for pid in "${pids[@]}"; do
        wait $pid || echo "WARNING: process $pid exited with error"
    done

    batch=$((batch+1))
done

echo ""
echo "=============================================="
echo "Evaluation complete. Results in: $RESULTS_DIR/"
echo "=============================================="
