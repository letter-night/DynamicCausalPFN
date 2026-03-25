#!/bin/bash
# Run remaining MIMIC pretrain tuning trials with lr=0.0001, K=5, batch=128
# Covers hidden=32 and hidden=64 configs
set -e

PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
SEED=101
RESULTS_DIR="outputs/mimic_pretrain_tuning_remaining"

cd /data1/letternight/DynamicCausalPFN
export PYTHONPATH=/data1/letternight/DynamicCausalPFN:$PYTHONPATH

mkdir -p "$RESULTS_DIR"

run_trial() {
    local gpu=$1 name=$2 layers=$3 heads=$4 hidden=$5 br=$6 fc=$7
    echo "[GPU $gpu] $name: layers=$layers heads=$heads hidden=$hidden br=$br fc=$fc"
    $PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
        +backbone/dynamic_causal_pfn_hparams=mimic3_pretrain_tuned \
        +dataset=mimic3_pretrain exp.seed=$SEED \
        exp.gpus="[$gpu]" exp.logging=False exp.max_epochs=20 \
        dataset.num_pretrain_datasets_train=1000 \
        dataset.num_pretrain_dataset_val=100 \
        dataset.num_patients.train=100 \
        dataset.num_patients.val=100 \
        model.dynamic_causal_pfn.num_layer=$layers \
        model.dynamic_causal_pfn.num_heads=$heads \
        model.dynamic_causal_pfn.optimizer.learning_rate=0.0001 \
        model.dynamic_causal_pfn.batch_size=128 \
        model.dynamic_causal_pfn.seq_hidden_units=$hidden \
        model.dynamic_causal_pfn.br_size=$br \
        model.dynamic_causal_pfn.fc_hidden_units=$fc \
        model.dynamic_causal_pfn.gmm_n_components=5 \
        model.dynamic_causal_pfn.dropout_rate=0.1 \
        > "$RESULTS_DIR/${name}.log" 2>&1
    val=$(grep "Val normalised RMSE (all)" "$RESULTS_DIR/${name}.log" | grep -oP "[\d.]+$" | head -1)
    echo "[GPU $gpu] $name done: val_rmse=$val"
}

echo "=============================================="
echo "MIMIC pretrain tuning - remaining trials"
echo "All: lr=0.0001, K=5, batch=128, 1 GPU per trial"
echo "=============================================="

# Batch 1: hidden=32 configs
echo "=== Batch 1/3 (hidden=32) ==="
run_trial 0 "h32_l8_h8_br16_fc8"  8  8 32 16 8 &
run_trial 1 "h32_l10_h8_br64_fc4" 10 8 32 64 4 &
run_trial 2 "h32_l6_h8_br32_fc4"  6  8 32 32 4 &
run_trial 3 "h32_l8_h4_br32_fc4"  8  4 32 32 4 &
wait
echo "Batch 1 done."

# Batch 2: hidden=64 configs
echo "=== Batch 2/3 (hidden=64) ==="
run_trial 0 "h64_l6_h8_br64_fc2"  6  8 64 64 2 &
run_trial 1 "h64_l8_h4_br64_fc2"  8  4 64 64 2 &
run_trial 2 "h64_l10_h8_br64_fc2" 10 8 64 64 2 &
run_trial 3 "h64_l6_h4_br32_fc4"  6  4 64 32 4 &
wait
echo "Batch 2 done."

# Batch 3: hidden=64 more configs
echo "=== Batch 3/3 (hidden=64) ==="
run_trial 0 "h64_l8_h8_br32_fc8"  8  8 64 32 8 &
run_trial 1 "h64_l10_h4_br64_fc4" 10 4 64 64 4 &
run_trial 2 "h64_l6_h8_br16_fc4"  6  8 64 16 4 &
run_trial 3 "h64_l8_h4_br16_fc8"  8  4 64 16 8 &
wait
echo "Batch 3 done."

echo ""
echo "=============================================="
echo "All trials complete."
echo "=============================================="

echo ""
echo "=== RESULTS (sorted by val_rmse) ==="
for f in "$RESULTS_DIR"/*.log; do
    name=$(basename "$f" .log)
    val=$(grep "Val normalised RMSE (all)" "$f" | grep -oP "[\d.]+$" | head -1)
    echo "$name: $val"
done | sort -t: -k2 -n
