# ArtPS Installation and Setup

This guide describes how to install ArtPS, prepare the HOI4D data used by the project, and run the training and evaluation pipelines. All commands assume a Linux system and should be run from the ArtPS repository root unless stated otherwise.

## Requirements

- Linux (x86-64)
- An NVIDIA GPU with a recent NVIDIA driver
- Conda or Miniconda
- Git
- A C/C++ compiler compatible with CUDA 13.0


## 1. Clone the Repository

```bash
git clone https://github.com/KaifengT/ArtPS.git
cd ArtPS
export ARTPS_ROOT="$(pwd)"
```

`ARTPS_ROOT` is used only as a convenient reference in this guide. Run the commands below from this directory.

## 2. Create the Conda Environment

```bash
conda create -n artps python=3.12 -y
conda activate artps
python -m pip install --upgrade pip setuptools wheel packaging
```

Install the PyTorch CUDA 13.0 wheels:

```bash
python -m pip install \
  torch==2.11.0 \
  torchvision==0.26.0 \
  torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Install the CUDA compiler and Ninja, which are required to build the bundled CUDA extensions:

```bash
conda install cuda-nvcc -c nvidia/label/cuda-13.0.2 -y
conda install ninja -c conda-forge -y
```

Expose both the Conda CUDA toolkit and the CUDA headers and libraries installed with the PyTorch wheel to extension builds:

```bash
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"

export ARTPS_CUDA_INCLUDE="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13/include"
export ARTPS_CUDA_LIB="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13/lib"

export CPATH="${ARTPS_CUDA_INCLUDE}:${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${ARTPS_CUDA_LIB}:${CONDA_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${ARTPS_CUDA_LIB}:${CONDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

These exports must be available whenever the CUDA extensions are rebuilt. Add them to the environment activation script if a persistent configuration is preferred.

## 3. Install Python Dependencies

```bash
python -m pip install \
  numpy accelerate omegaconf termcolor natsort tqdm trimesh \
  opencv-python scipy pillow pyyaml

python -m pip install \
  libigl einops tabulate h5py matplotlib scikit-image \
  comet_ml jaxtyping munch timm UMNN

python -m pip install transformers==4.57.3
```

## 4. Build the Bundled CUDA Extensions

ArtPS includes the required PointNet++, KNN, causal convolution, and Mamba sources under `thirdparty/`. Build these vendored versions instead of replacing them with upstream packages, because the bundled Mamba source contains compatibility changes used by this project.

```bash
python -m pip install ./thirdparty/Pointnet2_PyTorch/pointnet2_ops_lib \
  --no-build-isolation

python -m pip install ./thirdparty/KNN_CUDA \
  --no-build-isolation

CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install \
  ./thirdparty/causal-conv1d-1.1.1 \
  --no-build-isolation

MAMBA_FORCE_BUILD=TRUE python -m pip install \
  ./thirdparty/mamba-v1.1.1 \
  --no-build-isolation
```

The vendored Mamba code already contains the CUDA compatibility updates that replace deprecated CUB calls and omit the legacy `sm_70` build target. No separate Mamba clone or manual source edit is required.

## 5. Verify the Installation

Confirm that PyTorch can access the GPU and that all custom extensions import successfully:

```bash
python -c "import torch; import pointnet2_ops, knn_cuda, causal_conv1d, mamba_ssm; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

The command should finish without an import error, and `CUDA available` should be `True`.

## 6. Configure Data Paths

Choose a data directory that is writable on the local machine. The following example keeps all project data under the current user's home directory and does not rely on a machine-specific absolute path:

```bash
export ARTPS_DATA_DIR="${HOME}/datasets/artps"
export HOI4D_ROOT="${ARTPS_DATA_DIR}/HOI4D"
export HOI4D_CACHE="${ARTPS_DATA_DIR}/HOI4D_Cache"
export HOI4D_SDF_ROOT="${ARTPS_DATA_DIR}/processed"

mkdir -p "${HOI4D_ROOT}" "${HOI4D_CACHE}" "${HOI4D_SDF_ROOT}"
```

Set these variables again in each new shell, or place them in a local environment activation script. Do not commit personal paths, API keys, or credentials to the repository.

## 7. Prepare HOI4D

1. Download the official dataset from the [HOI4D project website](https://hoi4d.github.io/).
2. The hand annotations and official CAD models are not required by this project.
3. Download the [ArtPS CAD models](https://drive.google.com/file/d/1Su8NTGvugzjT5mqIJstTU45-qBJsR9Je/view?usp=sharing) and use them in place of the official CAD-model directory.

After extraction, the relevant portion of the dataset should have the following layout:

```text
${HOI4D_ROOT}/
├── ZY20210800001/
├── ZY20210800002/
├── ZY20210800003/
├── ZY20210800004/
└── cad_model/
    └── articulated/
        ├── Laptop/
        ├── Safe/
        ├── StorageFurniture/
        └── TrashCan/
```

ArtPS uses the following HOI4D categories:

| Category ID | Category name | SDF configuration | Pose configuration |
| --- | --- | --- | --- |
| `C3` | Laptop | `configs/sdf/config_sdf_hoi4d_Laptop.yaml` | `configs/pose/config_pose_hoi4d_Laptop.yaml` |
| `C4` | StorageFurniture | `configs/sdf/config_sdf_hoi4d_StorageFurniture.yaml` | `configs/pose/config_pose_hoi4d_StorageFurniture.yaml` |
| `C6` | Safe | `configs/sdf/config_sdf_hoi4d_Safe.yaml` | `configs/pose/config_pose_hoi4d_Safe.yaml` |
| `C14` | TrashCan | `configs/sdf/config_sdf_hoi4d_TrashCan.yaml` | `configs/pose/config_pose_hoi4d_TrashCan.yaml` |

### Generate the Dataset Cache

Generate both the training and validation caches for every supported category:

```bash
for category in C3 C4 C6 C14; do
  for split in train val; do
    python dataloader/hoi4d_dataset/hoi4d_dataset_arti.py \
      dataset_manual_cache \
      --root-dir "${HOI4D_ROOT}" \
      --cache-dir "${HOI4D_CACHE}" \
      --category "${category}" \
      --mode "${split}"
  done
done
```

### Generate the SDF Training Data

SDF generation is storage- and compute-intensive. Generate the training and validation files for each category:

```bash
for category in C3 C4 C6 C14; do
  for split in train val; do
    python perpare_sdf_training_data.py \
      --dataset hoi4d \
      --category "${category}" \
      --output_path "${HOI4D_SDF_ROOT}" \
      --hoi4d_dataset_root "${HOI4D_ROOT}" \
      --hoi4d_dataset_cache "${HOI4D_CACHE}" \
      --mode "${split}"
  done
done
```

The generated files are written below `${HOI4D_SDF_ROOT}/HOI4D_SDF/`, grouped by category name.

## 8. Train the SDF Shape Prior

Select one category configuration. The example below trains the Laptop (`C3`) model on GPUs 0 and 1:

```bash
SDF_CONFIG="configs/sdf/config_sdf_hoi4d_Laptop.yaml"

./run_accelerate.sh --cuda "0,1" \
  train_sdf_prior_accelerate.py \
  config="${SDF_CONFIG}" \
  dataset_root="${HOI4D_SDF_ROOT}"
```

The launcher reads `configs/accelerate.yaml`. Adjust that file or the GPU list when using a different number of GPUs. The training script asks for confirmation and whether Comet ML logging should be enabled.

Checkpoints are saved under `model_dict/sdf_hoi4d_<Category>/`. The SDF checkpoint required by later stages is named `sdf_misc_hoi4d_best.pth`.

### Optional: Evaluate the SDF Prior

```bash
SDF_CHECKPOINT="model_dict/sdf_hoi4d_Laptop/<experiment-directory>/sdf_misc_hoi4d_best.pth"

python eval_sdf_prior.py \
  --dataset hoi4d \
  --cuda 0 \
  --file "${SDF_CHECKPOINT}"
```

When evaluating a checkpoint on a different machine, pass `--dataset_path` explicitly if the dataset path stored in the checkpoint is no longer valid.

## 9. Train the Pose-Shape Model

Use the pose configuration that matches the SDF checkpoint. The following example trains the Laptop model:

```bash
POSE_CONFIG="configs/pose/config_pose_hoi4d_Laptop.yaml"
SDF_CHECKPOINT="model_dict/sdf_hoi4d_Laptop/<experiment-directory>/sdf_misc_hoi4d_best.pth"

./run_accelerate.sh --cuda "0,1" \
  train_pose_estimation_accelerate.py \
  config="${POSE_CONFIG}" \
  dataset.data_root="${HOI4D_ROOT}" \
  dataset.cache_root="${HOI4D_CACHE}" \
  sdf.misc_path="${SDF_CHECKPOINT}"
```

Pose checkpoints and intermediate evaluation results are saved under `model_dict/ape_hoi4d_<Category>/`. The script prompts for confirmation and optional Comet ML logging before training starts.

## 10. Run Hypothesis Selection and Evaluation

Evaluation expects a trained pose-model work directory. A point-completion checkpoint can also be supplied when available:

Download the [pretrained point-completion checkpoint](https://drive.google.com/file/d/1BqaipEfNaM4V26GkRf8U9ekjaaljm7AI/view?usp=sharing), extract it if necessary, and set `COMPLETION_CHECKPOINT` below to the path of the downloaded `best_cd_t_network.pth` file.

```bash
POSE_WORK_DIR="model_dict/ape_hoi4d_Laptop/<experiment-directory>"
COMPLETION_CHECKPOINT="model_dict/cmp_hoi4d_Laptop/<experiment-directory>/best_cd_t_network.pth"

./run_accelerate.sh --cuda "0,1" \
  eval_pose_estimation_h5_accelerate.py \
  --dataset hoi4d \
  --category C3 \
  --work_dir "${POSE_WORK_DIR}" \
  --cmp_model "${COMPLETION_CHECKPOINT}"
```

Replace the category, work directory, and completion checkpoint with the outputs for the category being evaluated. If no completion checkpoint is available, omit `--cmp_model`; the evaluator will run without the completion network.

## 11. Optional Comet ML Logging

Comet ML is optional. To enable it, configure credentials in the shell before starting training:

```bash
export COMET_WORKSPACE="your-comet-workspace"
export COMET_API_KEY="your-comet-api-key"
```

Never commit a real API key to Git. Answer `y` when the training script asks whether to enable Comet ML; answer `n` to train without remote logging.