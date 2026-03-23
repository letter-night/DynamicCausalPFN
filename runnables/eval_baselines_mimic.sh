#!/bin/bash
# Evaluate all baselines on mimic3_synthetic across seeds
# Usage: bash runnables/eval_baselines_mimic.sh
#
# 7 baselines × 5 seeds = 35 experiments, parallelized across 4 GPUs

set -e

PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SEEDS=(101 102 103 104 105)
RESULTS_DIR="outputs/mimic_baseline_eval"
NUM_GPUS=4

cd /data1/letternight/DynamicCausalPFN
export PYTHONPATH=/data1/letternight/DynamicCausalPFN:$PYTHONPATH

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "Baseline Evaluation on mimic3_synthetic"
echo "Seeds: ${SEEDS[*]}"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

# Build experiment list: (name, script, backbone, hparams, extra_args)
declare -a EXPERIMENTS

for SEED in "${SEEDS[@]}"; do
    EXPERIMENTS+=("crn_seed${SEED}|train_enc_dec.py|crn|mimic3_synthetic_tuned|${SEED}|")
    EXPERIMENTS+=("tecde_seed${SEED}|train_enc_dec.py|tecde|mimic3_synthetic_tuned|${SEED}|+dataset.fill_missing=False")
    EXPERIMENTS+=("ct_seed${SEED}|train_multi.py|ct|mimic3_synthetic_tuned|${SEED}|")
    EXPERIMENTS+=("gnet_seed${SEED}|train_gnet.py|gnet|mimic3_synthetic_tuned|${SEED}|")
    EXPERIMENTS+=("gt_seed${SEED}|train_gtransformer.py|gt|mimic3_synthetic_tuned|${SEED}|")
    EXPERIMENTS+=("rmsn_seed${SEED}|train_rmsn.py|rmsn|mimic3_synthetic_tuned|${SEED}|")
    EXPERIMENTS+=("scip_seed${SEED}|train_scip.py|scip|mimic3_synthetic_tuned|${SEED}|+dataset.fill_missing=False")
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
            IFS='|' read -r name script backbone hparams seed extra <<< "${EXPERIMENTS[$idx]}"

            echo "[GPU $gpu] $name"
            $PYTHON runnables/$script \
                +backbone=$backbone \
                +backbone/${backbone}_hparams=$hparams \
                +dataset=mimic3_synthetic exp.seed=$seed \
                exp.gpus="[$gpu]" exp.logging=True \
                $extra \
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

# --- Summary ---
echo ""
echo "=== RESULTS SUMMARY ==="
printf "%-25s | %-15s | %-15s | %-15s\n" "Experiment" "Val RMSE" "Test RMSE" "2-step / 3-step"
echo "-------------------------------------------------------------------------------------"
for logfile in "$RESULTS_DIR"/*.log; do
    name=$(basename "$logfile" .log)
    val=$(grep "Val normalised RMSE (all)" "$logfile" | grep -oP "Val normalised RMSE \(all\): [\d.]+" | head -1 | grep -oP "[\d.]+$")
    test=$(grep "Test normalised RMSE (all)" "$logfile" | grep -oP "Test normalised RMSE \(all\): [\d.]+" | head -1 | grep -oP "[\d.]+$")
    nstep=$(grep "n-step prediction" "$logfile" | head -1 | grep -oP "'[23]-step': [\d.]+" | tr '\n' ' ')
    printf "%-25s | %-15s | %-15s | %s\n" "$name" "${val:--}" "${test:--}" "${nstep:--}"
done
