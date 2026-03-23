DynamicCausalPFN
==============================

DynamicCausalPFN for conditional average potential outcome estimation over time.

### Setup
Please set up a virtual environment and install the libraries as given in the requirements file.
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow
To start an experiments server, run:

`mlflow server --port=3335`

Then, one can go to the local browser <http://localhost:3335>.

## Project Structure

```
config/
  dataset/
    cancer_sim.yaml                  # Downstream tumour growth dataset
    cancer_sim_pretrain.yaml         # Pretraining multi-task synthetic dataset
    mimic3_synthetic.yaml            # Downstream MIMIC-III semi-synthetic dataset
    mimic3_pretrain.yaml             # Pretraining MIMIC-style synthetic dataset
  backbone/
    dynamic_causal_pfn.yaml          # Model architecture config
    dynamic_causal_pfn_hparams/
      cancer_sim_hparams_grid.yaml       # HP grid for downstream tuning (cancer)
      cancer_sim_tuned.yaml              # Tuned HPs for downstream (cancer)
      cancer_sim_pretrain_grid.yaml      # HP grid for pretraining tuning (cancer)
      cancer_sim_pretrain_tuned.yaml     # Tuned HPs for pretraining (cancer)
      cancer_sim_transfer_grid.yaml      # Transfer sweep settings (cancer)
      mimic3_synthetic_hparams_grid.yaml # HP grid for downstream tuning (MIMIC)
      mimic3_synthetic_tuned.yaml        # Tuned HPs for downstream (MIMIC)
      mimic3_pretrain_grid.yaml          # HP grid for pretraining tuning (MIMIC)
      mimic3_pretrain_tuned.yaml         # Tuned HPs for pretraining (MIMIC)
      mimic3_transfer_grid.yaml          # Transfer sweep settings (MIMIC)

runnables/
  train_dynamic_causal_pfn.py        # Main training script
  run_pretrain_tuning.sh             # HP tuning for pretraining (cancer)
  run_pretrain.sh                    # Full pretraining run (cancer)
  sweep_transfer.sh                  # Transfer learning sweep (cancer)
  run_mimic_pretrain_tuning.sh       # HP tuning for pretraining (MIMIC)
  tune_all_baselines_mimic.sh        # Tune all baselines on MIMIC
  eval_baselines_mimic.sh            # Evaluate all baselines on MIMIC
```

## Experiments

The main training script `config/config.yaml` is run automatically for all models and datasets.
___
The training `<script>` for each different models specified by:

**CRN**: `runnables/train_enc_dec.py`

**TE-CDE**: `runnables/train_enc_dec.py`

**CT**: `runnables/train_multi.py`

**RMSNs**: `runnables/train_rmsn.py`

**G-Net**: `runnables/train_gnet.py`

**GT**: `runnables/train_gtransformer.py`

**SCIP-Net**: `runnables/train_scip.py`

**DynamicCausalPFN**: `runnables/train_dynamic_causal_pfn.py`
___

The `<backbone>` is specified by:

**CRN**: `crn`

**TE-CDE**: `tecde`

**CT**: `ct`

**RMSNs**: `rmsn`

**G-Net**: `gnet`

**GT**: `gt`

**SCIP-Net**: `scip`

**DynamicCausalPFN**: `dynamic_causal_pfn`
___

The `<hyperparameter>` configuration for each model is specified by:

**CRN**: `backbone/crn_hparams='HPARAMS'`

**TE-CDE**: `backbone/tecde_hparams='HPARAMS'`

**CT**: `backbone/ct_hparams='HPARAMS'`

**RMSNs**: `backbone/rmsn_hparams='HPARAMS'`

**G-Net**: `backbone/gnet_hparams='HPARAMS'`

**GT**: `backbone/gt_hparams='HPARAMS'`

**SCIP-Net**: `backbone/scip_hparams='HPARAMS'`

**DynamicCausalPFN**: `backbone/dynamic_causal_pfn_hparams='HPARAMS'`


`HPARAMS` is either one of:

**Cancer Simulation:**

| Model | Grid (for tuning) | Tuned (for reproducing) |
|-------|-------------------|------------------------|
| CRN, TE-CDE, CT, G-Net, RMSN, SCIP | `hparams_grid` | `hparams_tuned` |
| GT | `cancer_sim_hparams_grid` | `cancer_sim_tuned` |
| DynamicCausalPFN (downstream) | `cancer_sim_hparams_grid` | `cancer_sim_tuned` |
| DynamicCausalPFN (pretraining) | `cancer_sim_pretrain_grid` | `cancer_sim_pretrain_tuned` |

**MIMIC-III:**

| Model | Grid (for tuning) | Tuned (for reproducing) |
|-------|-------------------|------------------------|
| CRN, TE-CDE, CT, G-Net, RMSN, SCIP, GT | `mimic3_synthetic_hparams_grid` | `mimic3_synthetic_tuned` |
| DynamicCausalPFN (downstream) | `mimic3_synthetic_hparams_grid` | `mimic3_synthetic_tuned` |
| DynamicCausalPFN (pretraining) | `mimic3_pretrain_grid` | `mimic3_pretrain_tuned` |

___

The `<dataset>` is specified by:

**Cancer Simulation (downstream)**: `cancer_sim`

**Cancer Simulation (pretraining)**: `cancer_sim_pretrain`

**MIMIC-III (downstream)**: `mimic3_synthetic`

**MIMIC-III (pretraining)**: `mimic3_pretrain`

___

Please use the following commands to run the experiments.
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices>
python3 <script> +dataset=<dataset> +backbone=<backbone> +<hyperparameter> exp.seed=<seed> exp.logging=True
```

> **Note:** RMSN and SCIP require `dataset.treatment_mode=multilabel` on the command line. SCIP and TE-CDE additionally require `dataset.fill_missing=False`.

## Example usage
To run our DynamicCausalPFN with optimized hyperparameters on synthetic data with random seeds 101--105 and confounding level 15.0, use the command:
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices>
python3 runnables/train_dynamic_causal_pfn.py --multirun +dataset=cancer_sim +backbone=dynamic_causal_pfn +backbone/dynamic_causal_pfn_hparams='cancer_sim_tuned' dataset.coeff=15.0 exp.seed=101,102,103,104,105
```

___

## Pretraining

DynamicCausalPFN supports pretraining on multi-task synthetic data followed by transfer to a downstream treatment effect estimation task. The pretraining pipeline has three stages: hyperparameter tuning, full pretraining, and transfer evaluation.

### 1. Hyperparameter Tuning for Pretraining

Tunes pretraining hyperparameters using Ray Tune over a grid defined in `cancer_sim_pretrain_grid.yaml`. Uses a reduced number of training datasets and epochs for faster search.

**Using the helper script:**
```console
bash runnables/run_pretrain_tuning.sh [gpu_id]
```

**Manually:**
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_grid \
    +dataset=cancer_sim_pretrain dataset.coeff=10.0 exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False \
    dataset.num_pretrain_datasets_train=5 \
    dataset.num_pretrain_dataset_val=1
```

After tuning completes, update `cancer_sim_pretrain_tuned.yaml` with the best hyperparameters reported in the log.

### 2. Full Pretraining

Runs pretraining with the tuned hyperparameters on a larger number of synthetic datasets and for more epochs. Saves a checkpoint for downstream transfer.

**Using the helper script:**
```console
bash runnables/run_pretrain.sh [gpu_id] [num_train_datasets] [max_epochs]
# Defaults: gpu_id=0, num_train_datasets=10, max_epochs=100
```

**Manually:**
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_tuned \
    +dataset=cancer_sim_pretrain dataset.coeff=10.0 exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False \
    exp.max_epochs=100 \
    dataset.num_pretrain_datasets_train=10 \
    dataset.num_pretrain_dataset_val=2
```

The checkpoint will be saved under `outputs/<date>/<time>/checkpoints/`.

## Transfer Learning (Evaluation and Fine-tuning)

After pretraining, the model can be transferred to a downstream task using three modes:

| Mode | Description |
|------|-------------|
| `zero_shot` | Evaluate pretrained model directly on downstream data (no training) |
| `few_shot` | Fine-tune output heads only on a small number of downstream samples |
| `fine_tune` | Fine-tune the entire model on the full downstream training set |

### Transfer Sweep

Runs all transfer settings (zero-shot, few-shot with varying sample sizes, fine-tune with varying learning rates and epochs) in a single sweep.

```console
bash runnables/sweep_transfer.sh <pretrained_ckpt_path> [gpu_id]
```

For example:
```console
bash runnables/sweep_transfer.sh outputs/2026-03-08/16-59-46/checkpoints/last.ckpt 0
```

Results are saved to `outputs/transfer_sweep/` with per-experiment logs and a summary printed at the end.

### Individual Transfer Experiments

**Zero-shot evaluation:**
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_tuned \
    +dataset=cancer_sim dataset.coeff=10.0 exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False \
    model.dynamic_causal_pfn.pretrained_ckpt=<ckpt_path> \
    model.dynamic_causal_pfn.transfer_mode=zero_shot
```

**Few-shot transfer** (fine-tune output heads on N samples):
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_tuned \
    +dataset=cancer_sim dataset.coeff=10.0 exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False exp.max_epochs=25 \
    model.dynamic_causal_pfn.pretrained_ckpt=<ckpt_path> \
    model.dynamic_causal_pfn.transfer_mode=few_shot \
    model.dynamic_causal_pfn.few_shot_samples=50
```

**Full fine-tuning:**
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_tuned \
    +dataset=cancer_sim dataset.coeff=10.0 exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False exp.max_epochs=50 \
    model.dynamic_causal_pfn.pretrained_ckpt=<ckpt_path> \
    model.dynamic_causal_pfn.transfer_mode=fine_tune \
    model.dynamic_causal_pfn.optimizer.learning_rate=0.001
```

> **Note:** The hyperparameter config for transfer experiments must match the architecture used during pretraining (`cancer_sim_pretrain_tuned`), not the downstream-tuned config (`cancer_sim_tuned`), so that the checkpoint weights can be loaded correctly.

### Hyperparameter Tuning for Downstream (without pretraining)

To tune hyperparameters for training directly on the downstream dataset (no pretraining):
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_hparams_grid \
    +dataset=cancer_sim dataset.coeff=10.0 exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False
```

After tuning, update `cancer_sim_tuned.yaml` with the best hyperparameters.

___

## MIMIC-III Experiments

### Data Preparation

MIMIC-III experiments require the processed `all_hourly_data.h5` file from [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract). Place it at `data/processed/all_hourly_data.h5`.

You can generate it using [MIMICExtractEasy](https://github.com/SphtKr/MIMICExtractEasy) (requires PhysioNet credentialed access to MIMIC-III):

```console
git clone https://github.com/SphtKr/MIMICExtractEasy && cd MIMICExtractEasy
git submodule update --init --recursive
# Download MIMIC-III CSVs
cd data && wget -r -N -c -np -nH --cut-dirs=1 --user <PHYSIONET_USER> --ask-password https://physionet.org/files/mimiciii/1.4/
# Build DuckDB and extract
cd ../mimic-code/mimic-iii/buildmimic/duckdb/
pip install -r requirements.txt
python3 import_duckdb.py <MIMIC_CSV_DIR> <DATA_DIR>/mimic3.db --skip-indexes
python3 import_duckdb.py <MIMIC_CSV_DIR> <DATA_DIR>/mimic3.db --skip-tables --make-concepts
cd ../../../../MIMIC_Extract/
pip install -r requirements.txt
python3 mimic_direct_extract.py --duckdb_database=<DATA_DIR>/mimic3.db --duckdb_schema=main --resource_path=./resources --out_path=<DATA_DIR>/extract --extract_notes=0
# Copy to project
cp <DATA_DIR>/extract/all_hourly_data.h5 <PROJECT_DIR>/data/processed/
```

### Baseline Tuning and Evaluation (MIMIC)

**Tune all baselines:**
```console
bash runnables/tune_all_baselines_mimic.sh "[0,1,2,3]"
```

**Evaluate all baselines across seeds:**
```console
bash runnables/eval_baselines_mimic.sh
```

### Example: Running a baseline on MIMIC

```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices>
python3 runnables/train_multi.py +dataset=mimic3_synthetic +backbone=ct +backbone/ct_hparams='mimic3_synthetic_tuned' exp.seed=101 exp.logging=True
```

> **Note:** TE-CDE and SCIP require `+dataset.fill_missing=False` on MIMIC.

### DynamicCausalPFN Pretraining on MIMIC

#### 1. Hyperparameter Tuning
```console
bash runnables/run_mimic_pretrain_tuning.sh
```
Or manually:
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=mimic3_pretrain_grid \
    +dataset=mimic3_pretrain exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False \
    dataset.num_pretrain_datasets_train=1000 \
    dataset.num_pretrain_dataset_val=100
```

#### 2. Full Pretraining
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=mimic3_pretrain_tuned \
    +dataset=mimic3_pretrain exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False \
    exp.max_epochs=100 \
    dataset.num_pretrain_datasets_train=1000 \
    dataset.num_pretrain_dataset_val=100
```

#### 3. Transfer to MIMIC Downstream

**Zero-shot:**
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=mimic3_pretrain_tuned \
    +dataset=mimic3_synthetic exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False \
    model.dynamic_causal_pfn.pretrained_ckpt=<ckpt_path> \
    model.dynamic_causal_pfn.transfer_mode=zero_shot
```

**Fine-tuning:**
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> \
python3 runnables/train_dynamic_causal_pfn.py \
    +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=mimic3_pretrain_tuned \
    +dataset=mimic3_synthetic exp.seed=101 \
    exp.gpus="[<gpu_id>]" exp.logging=False exp.max_epochs=50 \
    model.dynamic_causal_pfn.pretrained_ckpt=<ckpt_path> \
    model.dynamic_causal_pfn.transfer_mode=fine_tune \
    model.dynamic_causal_pfn.optimizer.learning_rate=0.0001
```

___
