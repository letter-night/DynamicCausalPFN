#!/bin/bash
# Full evaluation of pretrained DynamicCausalPFN across coefficients, seeds, and transfer modes
# Usage: bash runnables/run_eval.sh <pretrained_ckpt_path>
#
# Runs 7 coefficients × 5 seeds × 3 transfer modes = 105 experiments
# Parallelizes across 4 GPUs

set -e

CKPT_PATH="${1:?Usage: bash runnables/run_eval.sh <pretrained_ckpt_path>}"
PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
RESULTS_DIR="outputs/eval_results"
NUM_GPUS=4

cd /data1/letternight/DynamicCausalPFN
export PYTHONPATH=/data1/letternight/DynamicCausalPFN:$PYTHONPATH

mkdir -p "$RESULTS_DIR"

COEFFS=(4 5 6 7 8 9 10)
SEEDS=(101 102 103 104 105)

echo "=============================================="
echo "DynamicCausalPFN Full Evaluation"
echo "Checkpoint: $CKPT_PATH"
echo "Coefficients: ${COEFFS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Modes: zero_shot, few_shot_n200, fine_tune_lr0.0001_ep50"
echo "Total experiments: $((${#COEFFS[@]} * ${#SEEDS[@]} * 3))"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

# Build list of all experiments
declare -a EXPERIMENTS
for COEFF in "${COEFFS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Zero-shot
        EXPERIMENTS+=("zero_shot|coeff${COEFF}_seed${SEED}_zero_shot|$COEFF|$SEED|zero_shot|0|0|25")
        # Few-shot n=200
        EXPERIMENTS+=("few_shot|coeff${COEFF}_seed${SEED}_few_shot_n200|$COEFF|$SEED|few_shot|200|0|25")
        # Fine-tune lr=0.0001 ep=50
        EXPERIMENTS+=("fine_tune|coeff${COEFF}_seed${SEED}_fine_tune|$COEFF|$SEED|fine_tune|0|0.0001|50")
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "Dispatching $TOTAL experiments in batches of $NUM_GPUS..."

# Run experiments in batches of NUM_GPUS
idx=0
batch=1
while [ $idx -lt $TOTAL ]; do
    echo ""
    echo "=== Batch $batch (experiments $((idx+1))-$((idx+NUM_GPUS < TOTAL ? idx+NUM_GPUS : TOTAL))/$TOTAL) ==="

    pids=()
    for gpu in $(seq 0 $((NUM_GPUS-1))); do
        if [ $idx -lt $TOTAL ]; then
            IFS='|' read -r mode name coeff seed transfer_mode few_shot_n lr max_epochs <<< "${EXPERIMENTS[$idx]}"

            EXTRA_ARGS=""
            if [ "$transfer_mode" = "few_shot" ]; then
                EXTRA_ARGS="model.dynamic_causal_pfn.few_shot_samples=$few_shot_n"
            fi
            if [ "$transfer_mode" = "fine_tune" ]; then
                EXTRA_ARGS="model.dynamic_causal_pfn.optimizer.learning_rate=$lr"
            fi

            echo "[GPU $gpu] $name"
            $PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
                +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_tuned \
                +dataset=cancer_sim dataset.coeff=$coeff exp.seed=$seed \
                exp.gpus="[$gpu]" exp.logging=True \
                exp.max_epochs=$max_epochs \
                'model.dynamic_causal_pfn.pretrained_ckpt='"$CKPT_PATH" \
                model.dynamic_causal_pfn.transfer_mode=$transfer_mode \
                $EXTRA_ARGS \
                hydra.run.dir="$RESULTS_DIR/$name" \
                > "$RESULTS_DIR/${name}.log" 2>&1 &
            pids+=($!)

            idx=$((idx+1))
        fi
    done

    # Wait for batch to complete
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
printf "%-45s | %-15s | %-15s | %-15s\n" "Experiment" "Val RMSE" "Test RMSE" "CF RMSE"
echo "--------------------------------------------------------------------------------------------------------------"
for logfile in "$RESULTS_DIR"/*.log; do
    name=$(basename "$logfile" .log)
    val=$(grep "Val normalised RMSE (all)" "$logfile" | grep -oP "Val normalised RMSE \(all\): [\d.]+" | head -1 | grep -oP "[\d.]+$")
    test=$(grep "Test normalised RMSE (all)" "$logfile" | grep -oP "Test normalised RMSE \(all\): [\d.]+" | head -1 | grep -oP "[\d.]+$")
    cf=$(grep "only counterfactual" "$logfile" | grep -oP "only counterfactual\): [\d.]+" | head -1 | grep -oP "[\d.]+$")
    printf "%-45s | %-15s | %-15s | %-15s\n" "$name" "${val:--}" "${test:--}" "${cf:--}"
done
