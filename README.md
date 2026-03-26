# CodeSLAM

PyTorch implementation of [CodeSLAM: Learning a Compact, Optimisable Representation for Dense Visual SLAM](https://arxiv.org/pdf/1804.00874.pdf).

This repository now implements the complete algorithmic pipeline described in the paper:

- A conditioned depth autoencoder with a compact latent code
- The paper's hybrid proximity parametrization for depth
- A learned uncertainty decoder and multi-scale Laplace training loss
- A decoder structure that stays linear in the latent code for Jacobian precomputation
- Dense photometric and geometric residuals for direct optimization
- Affine brightness compensation, robustification, and occlusion/slanted-surface weighting
- Two-frame bootstrap with joint pose and code optimization
- Coarse-to-fine tracking for incoming monocular frames
- Sliding-window mapping with Schur-complement marginalization into a prior

## What Is In The Repo

### Core implementation

- `codeslam/config.py`
  - Configuration dataclasses for the model, losses, optimization, tracking, and SLAM system.
- `codeslam/network.py`
  - Conditioned depth VAE.
  - Zero-code depth prior for a single image.
  - Linear-in-code decoder path and Jacobian precomputation support.
- `codeslam/proximity.py`
  - Hybrid proximity/depth conversion from the paper.
- `codeslam/pose.py`
  - SE(3) and SO(3) Lie-group utilities.
- `codeslam/geometry.py`
  - Backprojection, projection, pyramids, and differentiable warping.
- `codeslam/losses.py`
  - Laplace likelihood, Huber-style robustification, and depth metrics.
- `codeslam/optimization.py`
  - Dense residual construction.
  - Levenberg-Marquardt optimization.
  - Pair optimization, tracking optimization, and prior construction by marginalization.
- `codeslam/system.py`
  - PTAM-style tracking/mapping system using the learned compact geometry representation.
- `codeslam/dataset.py`
  - SceneNet-style RGB-D dataset loader and a simple sequential image-folder loader.
- `codeslam/training.py`
  - Training and evaluation loops for the depth model.

### Entry points

- `scripts/train_codeslam.py`
  - Train the conditioned depth model.
- `scripts/evaluate_depth.py`
  - Evaluate single-image zero-code depth predictions against RGB-D ground truth.
- `scripts/infer_depth.py`
  - Run single-image inference and save depth and uncertainty outputs.
- `scripts/run_pair_optimization.py`
  - Reproduce the paper's two-frame initialization step.
- `scripts/run_slam.py`
  - Run the monocular sliding-window system on a sequence of images.

### Legacy utilities

- `preprocessing.py`
  - SceneNet preprocessing helpers for grayscale conversion, resizing, and depth normalization.
- `read_protobuf.py`
  - Utility to inspect SceneNet trajectory protobuf metadata.
- `U-Net/`
  - Older baseline code kept for comparison; not the main paper implementation.

## Paper Coverage

The table below maps the main technical pieces from the paper to this codebase.

| Paper concept | Where it is implemented |
| --- | --- |
| Image-conditioned compact depth representation | `codeslam/network.py` |
| 128-D optimizable code | `codeslam/config.py`, `codeslam/network.py` |
| Hybrid proximity depth parametrization | `codeslam/proximity.py` |
| Learned depth uncertainty | `codeslam/network.py`, `codeslam/losses.py` |
| Multi-scale negative log-likelihood training | `codeslam/training.py`, `codeslam/losses.py` |
| Decoder linear in the latent code | `codeslam/network.py` |
| Precomputable Jacobian wrt code | `CodeSLAMDepthModel.precompute_linear_jacobian(...)` |
| Dense photometric residuals | `codeslam/optimization.py` |
| Dense geometric residuals | `codeslam/optimization.py` |
| Affine brightness correction | `codeslam/optimization.py` |
| Robust and weighted residual handling | `codeslam/losses.py`, `codeslam/optimization.py` |
| Joint optimization of pose and latent geometry | `codeslam/optimization.py` |
| Two-frame initialization | `codeslam/system.py`, `scripts/run_pair_optimization.py` |
| Coarse-to-fine tracking | `codeslam/optimization.py` |
| Sliding-window monocular SLAM | `codeslam/system.py` |
| Schur-complement marginalization prior | `codeslam/optimization.py`, `codeslam/system.py` |

## Installation

### 1. Create an environment

Use Python `3.11` or `3.12`. The current `torch` wheels used here do not support Python `3.14`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Optional: generate the SceneNet protobuf module

```bash
make
```

This generates `scenenet_pb2.py` from `scenenet.proto`, which is only needed if you want to inspect SceneNet metadata with `read_protobuf.py`.

### 3. Quick sanity checks

```bash
make compile
make test
```

## Data Layout

The RGB-D training/evaluation loader expects a SceneNet-style directory tree like this:

```text
data/train/
  0/
    some_scene/
      photo/
        0.jpg
        1.jpg
      depth/
        0.png
        1.png
```

The loader pairs files by frame stem. It also tolerates intensity filenames created by the preprocessing utilities, such as `0_intensity.jpg` and `0_resized.jpg`.

## Preprocessing

If your SceneNet export still contains RGB images and you want grayscale/intensity preprocessing:

```bash
python3 preprocessing.py i r --data-root data/train --width 256 --height 192
```

Supported actions:

- `i`: convert RGB photos to grayscale intensity images
- `r`: resize intensity images
- `n`: normalize depth files using the repository's legacy helper
- `clean-intensity`: remove generated intensity/resized images
- `clean-depth`: remove generated normalized depth files

## Training

Train the image-conditioned depth model on RGB-D data:

```bash
python3 scripts/train_codeslam.py \
  --data-root data/train \
  --batch-size 8 \
  --epochs 30 \
  --learning-rate 1e-4 \
  --checkpoint-dir checkpoints
```

What training does:

1. Loads grayscale intensity images and metric depth.
2. Encodes depth into a compact posterior code conditioned on the image.
3. Decodes multi-scale proximity maps and uncertainty maps.
4. Optimizes a multi-scale Laplace likelihood plus KL regularization.
5. Saves `checkpoints/latest.pt` and `checkpoints/best.pt`.

Important training details implemented from the paper:

- The code dimension defaults to `128`.
- The decoder predicts depth at half resolution.
- The latent code is optimized around a zero-code prior.
- Uncertainty is predicted from image features, not from the latent code.
- The decoder path can stay linear in the latent code for optimization.
- The default schedule now follows the paper more closely: `1e-4` down to `1e-6` over `6` epochs.

## Depth Evaluation

Evaluate the learned single-image prior on RGB-D data:

```bash
python3 scripts/evaluate_depth.py \
  --data-root data/val \
  --checkpoint checkpoints/best.pt \
  --batch-size 8
```

This reports:

- `loss`
- `reconstruction`
- `kl`
- `abs_rel`
- `rmse`
- `delta1`

## Single-Image Inference

Predict a dense depth prior from one image:

```bash
python3 scripts/infer_depth.py \
  frame.png \
  --checkpoint checkpoints/best.pt \
  --output-prefix outputs/frame
```

Outputs:

- `outputs/frame_depth.npy`
- `outputs/frame_uncertainty.npy`
- `outputs/frame_depth.png`
- `outputs/frame_uncertainty.png`

Notes:

- This uses the zero-code prior, which is the paper's compact geometry prior before multi-view optimization.
- The output resolution is the model's half-resolution prediction level.

## Two-Frame Pair Optimization

Reproduce the paper's joint initialization for two overlapping frames:

```bash
python3 scripts/run_pair_optimization.py \
  frame_0001.png \
  frame_0002.png \
  --checkpoint checkpoints/best.pt \
  --fx 320 --fy 320 --cx 128 --cy 96
```

This jointly optimizes:

- Relative pose between the two views
- Latent code of the first keyframe
- Latent code of the second keyframe

using:

- Dense photometric residuals
- Dense geometric residuals
- Affine brightness compensation
- Robust weighting and uncertainty-aware geometry terms

## Monocular SLAM / Sequence Inference

Run the full sliding-window system on a folder of images:

```bash
python3 scripts/run_slam.py \
  /path/to/sequence \
  --checkpoint checkpoints/best.pt \
  --fx 320 --fy 320 --cx 128 --cy 96
```

The sequence loader accepts `.png` and `.jpg` images and processes them in sorted order.

The system flow is:

1. Insert the first frame as a seed keyframe.
2. Bootstrap the map from the next frame by jointly optimizing pose and both keyframe codes.
3. Track each new frame against the current keyframe with coarse-to-fine direct alignment.
4. Insert a new keyframe once the motion threshold is exceeded.
5. Run sliding-window mapping over active keyframes with joint pose/code optimization.
6. Marginalize old keyframes into a compact linear prior.

The default SLAM configuration now follows the paper more closely:

- 4 active keyframes
- consecutive mapping edges in the sliding window
- identity/zero initialization for poses and codes during bootstrap

## Implementation Notes

### Camera model

The direct optimization code assumes a calibrated pinhole camera model:

- `fx`, `fy`: focal lengths in pixels
- `cx`, `cy`: principal point in pixels
- `width`, `height`: input image size fed to the model/system

### Residual design

The optimizer implements:

- Symmetric pairwise residuals for mapping
- Photometric-only residuals during tracking
- Geometric residuals weighted by learned uncertainty
- Occlusion-aware down-weighting
- Slanted-surface weighting
- Huber-style robustification
- Affine brightness estimation per residual block

### Optimization

The code uses dense Levenberg-Marquardt with:

- Pose updates in SE(3)
- Code updates in Euclidean space
- Joint state vectors for active keyframes
- Gauge fixing through a pose prior on one keyframe
- Schur-complement marginalization to preserve old information

### Proximity parametrization

The paper's depth encoding is implemented as:

```text
p = a / (d + a)
```

where `a` is the average depth hyperparameter, exposed here as `ModelConfig.proximity_average_depth`.

### Jacobians wrt code

The model exposes:

```python
model.precompute_linear_jacobian(intensity, level=-1)
```

for the decoder path that remains linear in the latent code. This is the optimization-friendly design used in the paper.

## Testing

Run all tests with:

```bash
make test
```

Current coverage includes:

- Hybrid proximity conversion
- Lie-group SE(3) helpers
- Prior-factor shape/invariant checks
- Preprocessing argument handling
- SceneNet path helper utilities
- Legacy file discovery helpers
- Script bootstrap path handling

Some tests are skipped automatically when `torch` is not installed, because the geometry/optimization/model code depends on PyTorch.

## What "Full Paper Implementation" Means Here

This repository includes every major algorithmic component described in the paper and wires them together into a train/eval/inference/SLAM workflow.

What still depends on runtime validation rather than static code inspection:

- Reproducing the exact numbers from the paper
- Matching the original training schedule and dataset split perfectly
- Real-time performance claims
- Robustness on the same benchmarks as the original project

In other words: the paper's system is implemented here, but benchmark parity still requires trained weights, the intended datasets, and empirical evaluation.
