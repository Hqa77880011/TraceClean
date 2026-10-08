# TraceClean

[![Package checks](https://github.com/Hqa77880011/TraceClean/actions/workflows/ci.yml/badge.svg)](https://github.com/Hqa77880011/TraceClean/actions/workflows/ci.yml)

PyTorch implementation of **TraceClean: Class-Relative Learning Trajectory Modeling for Image Classification with Noisy Labels**.

TraceClean estimates label reliability from loss, prediction stability and agreement with the observed label. It normalizes these trajectories within each observed class, fits class-wise Gaussian mixtures and uses the resulting scores to train one PreAct ResNet-18 with a CE/GCE objective.

## Installation

Use Python 3.10 or newer. CI uses Python 3.11, PyTorch 2.7.1 and torchvision 0.22.1.

```bash
git clone https://github.com/Hqa77880011/TraceClean.git
cd TraceClean
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell. Install PyTorch and torchvision for your hardware using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). For CPU:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

Then install the package:

```bash
python -m pip install -e .
```

## Data

[CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) are downloaded through torchvision. For CIFAR-100 with 40% symmetric label noise:

```bash
traceclean prepare --dataset cifar100 --noise-rate 0.4 --noise-seed 0 --output data/cifar100-sym40.npz
```

Raw files go to `data/raw`. The prepared archive contains images, observed training labels, reference labels and clean test labels. Training and reliability estimation use the observed labels; reference labels are used for evaluation.

`--noise-rate` changes `floor(N × rate)` labels to a uniformly chosen different class. `--noise-seed` fixes the corruption; the training `seed` controls initialization, augmentation and sample order. Use the same prepared archive for all methods in a comparison.

The CIFAR configurations cover the paper's six dataset settings:

| Dataset | Noise | Configuration |
| --- | --- | --- |
| CIFAR-10 | Symmetric, 20% | [cifar10-sym20.yaml](configs/cifar10-sym20.yaml) |
| CIFAR-10 | Symmetric, 40% | [cifar10-sym40.yaml](configs/cifar10-sym40.yaml) |
| CIFAR-10 | Symmetric, 60% | [cifar10-sym60.yaml](configs/cifar10-sym60.yaml) |
| CIFAR-100 | Symmetric, 40% | [cifar100-sym40.yaml](configs/cifar100-sym40.yaml) |
| CIFAR-100 | Symmetric, 60% | [cifar100-sym60.yaml](configs/cifar100-sym60.yaml) |
| CIFAR-100N | Human annotations | [cifar100n.yaml](configs/cifar100n.yaml) |

For the other symmetric-noise settings, change `--dataset`, `--noise-rate` and the output filename to match the configuration. `--root` changes the download directory. `--train-per-class` selects a smaller training subset after corruption; its actual noise rate is recorded in the archive. Use `--overwrite` to replace an existing archive.

### CIFAR-100N

Download `CIFAR-100_human.pt` from the [CIFAR-N data repository](https://github.com/UCSC-REAL/cifar-10-100n/tree/main/data) and place it in `data/raw/`:

```bash
traceclean prepare --dataset cifar100n --annotations data/raw/CIFAR-100_human.pt --output data/cifar100n.npz
```

Preparation uses `noisy_label`, checks `clean_label` against torchvision's CIFAR-100 image order and adds no synthetic corruption. An `.npz` file containing numeric `clean_label` and `noisy_label` arrays is also accepted.

## Training

```bash
traceclean train --config configs/cifar100-sym40.yaml
```

Override YAML settings with `--set key=value`:

```bash
traceclean train --config configs/cifar100-sym40.yaml --set seed=1 device=cuda:0 output=runs/cifar100-sym40/full-seed1
```

`data` selects the prepared archive and `output` selects the run directory. `epochs`, `batch_size`, `lr`, `momentum` and `weight_decay` control optimization. `width` sets the first-stage channel count, normally 64. `device=auto` chooses CUDA when available and otherwise CPU; explicit `cpu`, `cuda:N` and `mps` devices are supported. `workers` sets loader processes and `threads` sets PyTorch CPU threads.

To resume, keep the original configuration and output directory:

```bash
traceclean train --config configs/cifar100-sym40.yaml --resume runs/cifar100-sym40/full-seed0/last.pt
```

The checkpoint restores model, optimizer, scheduler, trajectory and random-number states. Its epoch must match the last row of `history.csv`.

### Small CPU example

```bash
traceclean prepare --dataset toy --noise-rate 0.4 --noise-seed 0 --output data/toy.npz
traceclean train --config configs/toy.yaml
traceclean evaluate --checkpoint runs/toy/full-seed0/last.pt --device cpu
traceclean plot --run runs/toy/full-seed0
```

This example uses generated patterns, a narrower model and four epochs. Its metrics describe the synthetic data and are not CIFAR benchmark results.

## Method and paper correspondence

The core implementation follows Sections III-B–E:

| Paper | Implementation |
| --- | --- |
| EMA loss, stability and label agreement, Eqs. (3)–(6) | [`Trajectory.update`](src/traceclean/trajectory.py) |
| Class median/MAD and coordinate orientation, Eqs. (7)–(9) | [`robust_normalize`](src/traceclean/trajectory.py) |
| Diagonal two-component EM and reliable-component selection, Eqs. (10)–(13) | [`diagonal_mixture`](src/traceclean/trajectory.py) |
| Reliability smoothing, Eq. (14) | [`Trajectory.update`](src/traceclean/trajectory.py) |
| CE/GCE objective, Eqs. (15)–(16) | [`adaptive_loss` and `gce_loss`](src/traceclean/losses.py) |
| Training, trajectory collection and next-epoch weights | [`run_training`](src/traceclean/train.py) |
| PreAct ResNet-18 and dataset settings, Section IV | [`model.py`](src/traceclean/model.py) and [`configs/`](configs) |

Each post-warm-up epoch trains with the current weights, then records one prediction per training image in evaluation mode. The ring buffer measures adjacent-prediction stability and observed-label agreement over the last `window` predictions. Loss is negated after normalization, so all three coordinates point toward greater reliability. The component with the larger mean across those coordinates supplies the GMM posterior.

A three-dimensional class fit needs at least eight samples. Failed class fits use the epoch's global GMM. If that fit also fails, each sample retains its last valid posterior, initially one.

The CIFAR configurations use the paper's method parameters:

| Parameter | Setting |
| --- | --- |
| `warmup` | 10 CE epochs before recording |
| `window` | 5 predictions per sample |
| `alpha` | 0.9 for EMA loss |
| `rho` | 0.9 for reliability smoothing |
| `q` | 0.7 for GCE |
| `gce_lambda` | 0.5 for the GCE contribution |

Recording begins after epoch 11. The first full window is available after epoch 15, and its weights are used in epoch 16. Until then the objective is CE. Weights are fixed during minibatch optimization and detached from gradient computation.

For an observed-label probability `p` and smoothed reliability `w`, the per-sample loss is:

```text
w × (-log p) + gce_lambda × (1 - w) × (1 - p^q) / q
```

## Baselines and ablations

Run the baseline comparison with matched training seeds:

```bash
traceclean sweep --config configs/cifar100-sym40.yaml --suite baselines --seeds 0 1 2 --output runs/cifar100-sym40/baselines
traceclean summarize --root runs/cifar100-sym40/baselines --output results/cifar100-sym40/baselines
```

The suite includes TraceClean, CE, Label Smoothing, GCE, Co-teaching, ELR and Small-loss, using the same data and optimization settings. Run an individual baseline by setting `method`:

```bash
traceclean train --config configs/cifar100-sym40.yaml --set method=gce output=runs/cifar100-sym40/gce-seed0
```

Label Smoothing uses `label_smoothing=0.1`; GCE uses `q=0.7`. [Co-teaching](https://github.com/bhanML/Co-teaching) exchanges small-loss minibatch indices between two networks and evaluates their mean probabilities. Its removal rate rises from zero at epoch 1 to `forget_rate` at epoch `forget_ramp + 1`. [ELR](https://github.com/shengliu66/ELR) uses a detached EMA probability target with `elr_beta=0.7` and coefficient `elr_lambda=3.0`. Small-loss uses CE during warm-up, then retains the lowest-loss `floor((1 - forget_rate) × batch_size)` samples per minibatch, with a minimum of one.

Run the ablation comparison:

```bash
traceclean sweep --config configs/cifar100-sym40.yaml --suite ablations --seeds 0 1 2 --output runs/cifar100-sym40/ablations
traceclean summarize --root runs/cifar100-sym40/ablations --output results/cifar100-sym40/ablations
```

| `variant` | Loss EMA | Stability/agreement | Class normalization | Class fitting | Soft weights |
| --- | --- | --- | --- | --- | --- |
| `current-loss` | No | No | No | No | No |
| `ema-loss` | Yes | No | No | No | No |
| `trajectory` | Yes | Yes | No | No | No |
| `global-gmm` | Yes | Yes | No | No | Yes |
| `class-gmm` | Yes | Yes | Yes | Yes | No |
| `full` | Yes | Yes | Yes | Yes | Yes |
| `class-normalize-only` | Yes | Yes | Yes | No | Yes |
| `class-fit-only` | Yes | Yes | No | Yes | Yes |

The first six rows correspond to the components in Table III. The last two isolate normalization and mixture fitting. `trajectory`, `global-gmm`, `class-gmm` and `full` form the global/class × hard/soft comparison. All variants wait for the same full trajectory window.

`variant=no-ema` uses current loss in the full method; `variant=no-smoothing` disables reliability smoothing. These can be selected with `train --set`. `--suite all` combines both suites without repeating the full TraceClean run. Choose an empty output parent for each new sweep.

## Evaluation and results

```bash
traceclean evaluate --checkpoint runs/cifar100-sym40/full-seed0/last.pt --output runs/cifar100-sym40/full-seed0/evaluation.json
traceclean plot --run runs/cifar100-sym40/full-seed0
```

`evaluate --data` changes the location of the same prepared archive. `--threshold` changes the clean-detection cutoff without retraining. Classification uses clean test labels; detection treats samples with matching observed and reference labels as positives.

| Metric | Meaning |
| --- | --- |
| `top1` | Clean-test accuracy |
| `worst_class_accuracy` | Minimum accuracy across represented test classes |
| `auroc` | Clean-sample ranking across detection thresholds |
| `clean_precision` | Clean fraction among selected training samples |
| `clean_recall` | Selected fraction of all clean training samples |
| `per_class_clean_recall` | Clean recall grouped by observed training class |
| `selected_fraction` | Fraction selected at the threshold |

JSON and CSV store rates in `[0, 1]`; tables and figures show percentages. Undefined metrics are `null`. Runs ending before the first full window have no TraceClean detection metrics.

| Output | Contents |
| --- | --- |
| `config.yaml`, `data.json` | Resolved settings and dataset metadata |
| `history.csv` | Epoch losses, accuracy, available detection metrics and GMM diagnostics |
| `last.pt` | Latest model and training state |
| `summary.json` | Final-epoch results |
| `test_predictions.npz` | Clean-test predictions and labels |
| `scores.npz` | Training scores, losses, predictions and labels; TraceClean also saves raw posteriors and trajectory coordinates |
| `figures/training.png`, `figures/classes.png` | Learning curves and class-level accuracy/recall |
| `figures/reliability.png`, `figures/trajectories.png` | Score distributions, ROC and trajectory plots |

`summarize` writes `comparison.csv`, `comparison.json`, `comparison.md` and `comparison.png`. It groups matching dataset, corruption and training settings, then reports means and sample standard deviations across training seeds. A one-seed row has no standard deviation. Source paths and group settings are recorded in `comparison.json`.

## Implementation settings

The paper leaves the complete optimizer, augmentation, EM initialization and detection operating point unspecified. This implementation uses the following settings:

- **Training:** 200 epochs, batch size 128, SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=0.0005` and cosine decay. PreAct ResNet-18 has a CIFAR 3×3 stem, stage widths 64/128/256/512, two blocks per stage and a final BN/ReLU before global pooling. Training uses random cropping with four-pixel padding and horizontal flips. Trajectory and test passes use normalization without random augmentation.
- **GMM:** unscaled MAD with `epsilon=0.001`, also used as the variance floor. K-means uses three initializations on one thread. EM uses `gmm_iterations=100` and mean log-likelihood tolerance `gmm_tolerance=0.0001`. A fit is rejected if it does not converge, a component's mass is below `min_component_weight=0.01`, or its means coincide within epsilon. Global fallback fits the same class-normalized, oriented vectors. Loss-only ablations need four samples per fit.
- **Hard ablations:** smoothed reliability is thresholded at `threshold=0.5` before the CE/GCE objective. Low-reliability samples receive GCE. Global variants normalize over the entire dataset. The paper lists the ablation components but does not define this hard-weight rule.
- **Detection:** TraceClean uses smoothed continuous reliability, including in hard ablations. Baselines use observed-label probability as a common diagnostic score. Both use `threshold=0.5`; their precision/recall operating points have different calibrations. Raw TraceClean posteriors are saved in `scores.npz`.
- **Comparisons:** results use the final epoch, with no checkpoint selection from test accuracy. Corruption is fixed while training seeds vary. Selection baselines use the configured `forget_rate`, set to the synthetic noise rate for symmetric noise and an assumed 0.4 for CIFAR-100N. The included baselines cover CE, Label Smoothing, GCE, Co-teaching, ELR and Small-loss; DivideMix, DISC, DSS and DynaCor from the paper's tables are not bundled.

Full CIFAR benchmark results have not been measured for this implementation. CI checks installation, imports, syntax, command entry points and configuration validity.

## License

[MIT](LICENSE).
