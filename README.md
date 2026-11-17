# UA-SNN

Official implementation of **UA-SNN: Uncertainty-aware Spiking Neural Networks with Adaptive Time Steps for Robust Brain Disease Detection**.

UA-SNN takes a 3D brain MRI volume as input and predicts one of two classes.

[Method](#method) · [Setup](#setup) · [Data](#data-preparation) · [Training](#training) · [Evaluation](#evaluation) · [Five-fold protocol](#five-fold-protocol) · [Experiment settings](#experiment-settings)

<p align="center">
  <a href="./new_model2.png"><img src="./new_model2.png" alt="UA-SNN architecture showing 3D MRI feature extraction, spiking self-attention, evidential learning, and adaptive time-step selection"></a>
</p>

## Method

| Component | Implementation |
| --- | --- |
| 3D spiking Transformer | Projects volumetric features with 3D convolutions and leaky integrate-and-fire (LIF) neurons, then processes the features with spiking self-attention. |
| Evidential training | Maps logits to Dirichlet concentrations and trains with a Fisher-weighted evidential loss, including a Fisher information regularizer and a KL divergence term. |
| Adaptive inference | Averages the logits across the steps taken so far. Inference stops when the maximum softmax probability exceeds the current threshold or when the step budget is reached. |

## Setup

```bash
git clone https://github.com/UAIBC-Brain/UA-SNN.git
cd UA-SNN
```

Use a Python 3.9 environment and install the packages below. The examples use `cuda:0`, so select PyTorch and CuPy builds compatible with your CUDA environment.

| Package | Version |
| --- | --- |
| PyTorch | `1.13.0` |
| NumPy | `1.23.0` |
| timm | `0.6.12` |
| CuPy | `13.3.0` |
| SpikingJelly | `0.0.0.0.12` |
| SimpleITK | `2.4.0` |
| MONAI, scikit-learn, TensorBoard | Unpinned |

## Data preparation

Place your MRI volumes under `./dataset/data_round` or set `--root-dir` to their location. Create a `split.json` with `training`, `validation`, and `test` arrays. Each `image` path is relative to `--root-dir`, and each `label` is `0` or `1`. The example below uses NIfTI filenames.

```json
{
  "training": [
    {"image": "train/sub-001.nii.gz", "label": 0},
    {"image": "train/sub-002.nii.gz", "label": 1}
  ],
  "validation": [
    {"image": "val/sub-003.nii.gz", "label": 0},
    {"image": "val/sub-004.nii.gz", "label": 1}
  ],
  "test": [
    {"image": "test/sub-005.nii.gz", "label": 0},
    {"image": "test/sub-006.nii.gz", "label": 1}
  ]
}
```

The loader reads volumes with SimpleITK. For a `128 × 128 × 128` input, the center crop is `96 × 112 × 96` voxels (`D × H × W`). Batched training expects a consistent post-crop shape.

Supply your own image files and update the paths to match. [Example split manifest](./jsons/old/data_example.json)

## Training

Run from the repository root.

```bash
python UA-SNN.py --split-json ./split.json --root-dir ./dataset/data_round --device cuda:0
```

With the default output directory and experiment settings, training writes to `./logs/fold1_lambda1_0.01_lambda2_1.0/`.

| File | Contents |
| --- | --- |
| `checkpoint_best_validation.pth` | Checkpoint selected by validation accuracy |
| `checkpoint_latest.pth` | Latest checkpoint for resuming with `--resume` |
| `args.txt` | Command used for the completed run |

View the training loss, training accuracy, and validation accuracy with `tensorboard --logdir ./logs`.

## Evaluation

Evaluate the best validation checkpoint on the `test` split.

```bash
python test.py --split-json ./split.json --root-dir ./dataset/data_round --checkpoint ./logs/fold1_lambda1_0.01_lambda2_1.0/checkpoint_best_validation.pth
```

The evaluator reports accuracy (ACC), sensitivity (SEN), specificity (SPE), F1, and the mean number of time steps (Avg. T). It uses `1` as the positive class and saves `test_results.json` beside the checkpoint by default. Pass `--output` to choose another file.

## Repository map

| Path | Contents |
| --- | --- |
| [UA-SNN.py](./UA-SNN.py) and [test.py](./test.py) | Training and evaluation entry points |
| [model_test/model.py](./model_test/model.py) | 3D spiking Transformer backbone |
| [losses.py](./losses.py) | Evidential loss and regularization terms |
| [utils/runtime.py](./utils/runtime.py) | Model construction and adaptive evaluation |
| [utils/new_jsaon_data_utils.py](./utils/new_jsaon_data_utils.py) | MRI loading and center cropping |
| [energy_consumption_calculation/](./energy_consumption_calculation/README.md) | Operation counting and energy estimation utilities |
