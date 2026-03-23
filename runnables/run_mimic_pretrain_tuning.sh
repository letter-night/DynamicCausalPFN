#!/bin/bash
# Run hyperparameter tuning for DynamicCausalPFN pretraining on MIMIC synthetic data
# Usage: bash runnables/run_mimic_pretrain_tuning.sh

set -e

PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
SEED=101

cd /data1/letternight/DynamicCausalPFN
export PYTHONPATH=/data1/letternight/DynamicCausalPFN:$PYTHONPATH

echo "=============================================="
echo "MIMIC Pretraining Hyperparameter Tuning"
echo "GPUs: [0,1,2,3]"
echo "=============================================="

$PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=mimic3_pretrain_grid \
    +dataset=mimic3_pretrain exp.seed=$SEED \
    'exp.gpus=[0,1,2,3]' exp.logging=False \
    dataset.num_pretrain_datasets_train=1000 \
    dataset.num_pretrain_dataset_val=100 \
    dataset.num_patients.train=100 \
    dataset.num_patients.val=100 \
    2>&1 | tee outputs/mimic_pretrain_tuning.log

echo ""
echo "=============================================="
echo "Tuning complete. Check log for best hyperparameters."
echo "=============================================="
