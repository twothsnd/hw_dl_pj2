# Project 2: CIFAR-10 and Batch Normalization

This folder contains a complete PyTorch implementation for Project 2 of
"Neural Network and Deep Learning".

## What Is Implemented

- CIFAR-10 data loading with optional augmentation and partial-dataset debugging.
- VGG-A baseline.
- VGG-A with BatchNorm after every convolution.
- VGG-A with Dropout.
- A configurable CIFAR CNN for filter/activation/loss/optimizer ablations.
- A residual CNN variant.
- Training, evaluation, checkpoint saving and metric plotting.
- First-layer filter visualization for model insight.
- Loss-landscape envelopes and gradient smoothness plots for the BN experiment.

## Environment

Create an isolated environment from this folder:

```bash
git clone https://github.com/twothsnd/hw_dl_pj2.git
cd hw_dl_pj2
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 12.8 PyTorch wheels verified on this machine
(`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`). If you run the code on a
different machine, adjust the PyTorch wheel to match that machine's driver.

The bundled `data/cifar-10-python.tar.gz` in the starter package is incomplete.
Either remove it or let `torchvision.datasets.CIFAR10(download=True)` replace it:

```bash
rm -f data/cifar-10-python.tar.gz
```

## Quick Smoke Test

Use a tiny subset first:

```bash
python train_cifar.py \
  --model cifar_cnn \
  --channels 32,64,128 \
  --epochs 1 \
  --batch-size 64 \
  --n-train 512 \
  --n-test 256 \
  --num-workers 2 \
  --out-dir reports/experiments/smoke
```

Expected outputs:

- `reports/experiments/smoke/config.json`
- `reports/experiments/smoke/metrics.csv`
- `reports/experiments/smoke/best.pt`
- `reports/experiments/smoke/last.pt`
- `reports/experiments/smoke/training_curve.png`
- `reports/experiments/smoke/first_layer_filters.png`
- `reports/experiments/smoke/model_summary.txt`

## Main CIFAR-10 Training

A strong default experiment:

```bash
python train_cifar.py \
  --model cifar_cnn \
  --channels 64,128,256,512 \
  --activation relu \
  --batch-norm \
  --dropout 0.2 \
  --loss ce \
  --label-smoothing 0.1 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --scheduler cosine \
  --augment \
  --epochs 100 \
  --batch-size 128 \
  --device auto \
  --out-dir reports/experiments/best_cifar_cnn
```

The final test error is `1 - best_test_accuracy` in
`reports/experiments/best_cifar_cnn/summary.json`.

## Required Ablations

Run all ablations required by the project:

```bash
python run_ablation.py \
  --suite all \
  --model cifar_cnn \
  --channels 64,128,256 \
  --epochs 40 \
  --batch-size 128 \
  --augment \
  --out-dir reports/experiments/ablation
```

For faster debugging, add:

```bash
--n-train 4096 --n-test 1024 --max-runs 2
```

The summary table is saved as:

```text
reports/experiments/ablation/ablation_summary.csv
```

This covers:

- Different number of filters: `32,64,128`, `64,128,256`, `64,128,256,512`.
- Different losses/regularization: CE, CE with weight decay, CE with label smoothing, focal loss.
- Different activations: ReLU, LeakyReLU, ELU.
- Different optimizers: SGD, Adam, AdamW.
- Optional components: no BN, BN+Dropout, residual connections.
- VGG-A vs VGG-A+BN.

## BatchNorm Loss Landscape

Run the required VGG-A/VGG-A+BN comparison:

```bash
python VGG_Loss_Landscape.py \
  --epochs 20 \
  --batch-size 128 \
  --lrs 1e-3 2e-3 1e-4 5e-4 \
  --optimizer adam \
  --out-dir reports/experiments/bn_loss_landscape
```

Outputs:

- `loss_landscape.png`: min-max training-loss envelope across learning rates.
- `gradient_smoothness.png`: gradient norm, gradient change and gradient cosine.
- `summary.json`: compact numerical comparison.
- `vgg_a/*/step_trace.csv` and `vgg_bn/*/step_trace.csv`: per-step loss/gradient traces.
- `vgg_a/*/last.pt` and `vgg_bn/*/last.pt`: trained model weights.

## Model Weights

Model weights are not committed to GitHub. Download them from the shared
network-disk folder:

```text
Baidu Netdisk: https://pan.baidu.com/s/1msDL24TwV2TTiIouIcM1KQ
Extraction code: pbh9
```

After downloading, put the three `.pt` files under the following paths if you
want to match the experiment paths used in the report:

```bash
mkdir -p reports/experiments/best_cifar_cnn_20e
mkdir -p reports/experiments/bn_loss_landscape_5e_subset/vgg_a/1em03
mkdir -p reports/experiments/bn_loss_landscape_5e_subset/vgg_bn/1em03

mv cifar_best_cnn_20e_best.pt \
  reports/experiments/best_cifar_cnn_20e/best.pt

mv vgg_a_lr1e-3_last.pt \
  reports/experiments/bn_loss_landscape_5e_subset/vgg_a/1em03/last.pt

mv vgg_a_bn_lr1e-3_last.pt \
  reports/experiments/bn_loss_landscape_5e_subset/vgg_bn/1em03/last.pt
```

Weight meanings:

- `cifar_best_cnn_20e_best.pt`: best full CIFAR-10 model, reported test error
  `12.23%`.
- `vgg_a_lr1e-3_last.pt`: VGG-A checkpoint for the BatchNorm comparison.
- `vgg_a_bn_lr1e-3_last.pt`: VGG-A + BatchNorm checkpoint for the BatchNorm
  comparison.

Example for loading the best CIFAR-10 checkpoint:

```python
import torch
from models.vgg import CIFARConvNet

model = CIFARConvNet(
    channels=(64, 128, 256, 512),
    activation="relu",
    batch_norm=True,
    dropout=0.2,
    classifier_hidden=512,
)
checkpoint = torch.load(
    "reports/experiments/best_cifar_cnn_20e/best.pt",
    map_location="cpu",
)
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

## Notes

Model weights, CIFAR-10 files and generated experiment outputs are intentionally
not tracked in this repository. They are ignored by `.gitignore` and should be
shared separately.
