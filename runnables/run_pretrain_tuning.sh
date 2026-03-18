#!/bin/bash
# Run hyperparameter tuning for pretraining on synthetic data
# Usage: bash runnables/run_pretrain_tuning.sh
#
# Uses Ray Tune to search over the pretraining HP grid defined in
# cancer_sim_pretrain_grid.yaml. Best HPs should be copied to
# cancer_sim_pretrain_tuned.yaml after the run.
# Uses all 4 GPUs for maximum parallelism.

set -e

PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
SEED=101

cd /data1/letternight/DynamicCausalPFN
export PYTHONPATH=/data1/letternight/DynamicCausalPFN:$PYTHONPATH

echo "=============================================="
echo "Pretraining Hyperparameter Tuning"
echo "GPUs: [0,1,2,3]"
echo "=============================================="

$PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_grid \
    +dataset=cancer_sim_pretrain dataset.coeff=10.0 exp.seed=$SEED \
    'exp.gpus=[0,1,2,3]' exp.logging=False \
    dataset.num_pretrain_datasets_train=1000 \
    dataset.num_pretrain_dataset_val=100 \
    dataset.num_patients.train=100 \
    dataset.num_patients.val=100 \
    2>&1 | tee outputs/pretrain_tuning.log

echo ""
echo "=============================================="
echo "Tuning complete. Check log for best hyperparameters."
echo "Update cancer_sim_pretrain_tuned.yaml with the best values."
echo "=============================================="
