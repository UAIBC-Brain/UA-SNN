# UA-SNN

Official implementation of **UA-SNN: Uncertainty-aware Spiking Neural Networks with Adaptive Time Steps for Robust Brain Disease Detection**.

## Requirements

- Python 3.9
- PyTorch 1.13.0
- NumPy 1.23.0
- timm 0.6.12
- CuPy 13.3.0
- SpikingJelly 0.0.0.0.12
- MONAI
- SimpleITK 2.4.0
- scikit-learn
- TensorBoard

## Data

The split file contains `training`, `validation`, and `test` subsets:

```json
{
  "training": [{"image": "sub-001", "label": 0}],
  "validation": [{"image": "sub-002", "label": 1}],
  "test": [{"image": "sub-003", "label": 0}]
}
```

Set `--root-dir` to the directory containing the MRI volumes.

## Training

```bash
python UA-SNN.py \
  --split-json ./jsons/old/data_example.json \
  --root-dir ./dataset/data_round \
  --lambda-1 0.01 \
  --lambda-2 1.0
```

## Evaluation

```bash
python test.py \
  --split-json ./jsons/old/data_example.json \
  --root-dir ./dataset/data_round \
  --checkpoint ./logs/fold1_lambda1_0.01_lambda2_1.0/checkpoint_best_validation.pth
```

## Model

<p align="center">
  <img src="https://github.com/UAIBC-Brain/UA-SNN/blob/main/new_model2.png">
</p>
