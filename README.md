# TraceClean

Implementation of **TraceClean: Class-Relative Learning Trajectory Modeling for Image Classification with Noisy Labels**.

TraceClean compares samples within their observed class using EMA loss, prediction stability and label agreement. Median/MAD normalization and diagonal two-component Gaussian mixtures produce reliability scores. A single PreAct ResNet-18 learns with a reliability-weighted CE/GCE objective.

The workflow below prepares fixed noisy labels, trains the method, evaluates a checkpoint, runs comparisons and plots the results. Results are generated from completed runs; the repository does not contain prefilled experiment numbers.

## 1. Install

Use Python 3.10 or newer. Python 3.11 with PyTorch 2.7.1 and torchvision 0.22.1 is the CI environment.

```bash
git clone https://github.com/Hqa77880011/TraceClean.git
cd TraceClean
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell.

Install the matching PyTorch and torchvision builds for your hardware using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). For CPU:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
traceclean --help
```

GPU runs use `device: auto` by default. Override it with `--set device=cuda:0`, `device=cpu` or `device=mps`. CPU runs are supported, but full CIFAR experiments are intended for a GPU. There is no pretrained-weight download.

## 2. Prepare data

[CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) are downloaded through torchvision. The prepared archive keeps images, observed training labels, reference labels, clean test labels and the corruption settings together. Training receives observed labels; reference labels are used only for diagnostics.

Run the commands for the settings you need:

```bash
traceclean prepare --dataset cifar10 --noise-rate 0.2 --noise-seed 0 --output data/cifar10-sym20.npz
traceclean prepare --dataset cifar10 --noise-rate 0.4 --noise-seed 0 --output data/cifar10-sym40.npz
traceclean prepare --dataset cifar10 --noise-rate 0.6 --noise-seed 0 --output data/cifar10-sym60.npz
traceclean prepare --dataset cifar100 --noise-rate 0.4 --noise-seed 0 --output data/cifar100-sym40.npz
traceclean prepare --dataset cifar100 --noise-rate 0.6 --noise-seed 0 --output data/cifar100-sym60.npz
```

`--noise-rate` is the fraction changed to a uniformly chosen **different** class. Exactly `floor(N × noise_rate)` labels change. `--noise-seed` fixes the corruption; the training `seed` independently controls model initialization and sample order. All methods in a comparison should use the same prepared archive.

The default raw-data directory is `data/raw`; change it with `--root`. Existing prepared archives are protected from accidental replacement. Use `--overwrite` when intentionally rebuilding one. `--train-per-class 20` creates a small subset for local development, retaining the full clean test split.

### CIFAR-100N

Obtain `CIFAR-100_human.pt` from the [CIFAR-N data repository](https://github.com/UCSC-REAL/cifar-10-100n/tree/main/data) and save it under `data/raw/`. The released `clean_label` and `noisy_label` arrays use the torchvision/Python-version image order. The preparation command checks that the reference labels match that order and uses the noisy **fine** labels:

```bash
traceclean prepare --dataset cifar100n --annotations data/raw/CIFAR-100_human.pt --output data/cifar100n.npz
```

An `.npz` annotation file with numeric `clean_label` and `noisy_label` arrays is also accepted. `.pt` annotations use restricted deserialization with NumPy integer-array types; use the released file or an equivalent numeric archive. No synthetic corruption is added to CIFAR-100N.

### Small example without a download

The toy dataset contains generated colored patterns and is useful for exercising the workflow on a CPU. Its scores describe this synthetic example, not CIFAR experiments.

```bash
traceclean prepare --dataset toy --noise-rate 0.4 --noise-seed 0 --output data/toy.npz
traceclean train --config configs/toy.yaml
traceclean evaluate --checkpoint runs/toy/full-seed0/last.pt --device cpu
traceclean plot --run runs/toy/full-seed0
```

The toy configuration uses a narrower 18-layer model, one warm-up epoch and a two-epoch trajectory window. The fourth epoch uses weights computed after the third epoch.

## 3. Train TraceClean

```bash
traceclean train --config configs/cifar100-sym40.yaml
```

The six CIFAR configurations correspond to CIFAR-10 Sym20/40/60, CIFAR-100 Sym40/60 and CIFAR-100N. Each configuration uses the paper's TraceClean parameters. Choose another file to change the dataset setting:

```bash
traceclean train --config configs/cifar10-sym40.yaml --set seed=1 output=runs/cifar10-sym40/full-seed1
traceclean train --config configs/cifar100n.yaml --set output=runs/cifar100n/full-seed0
```

`--set` accepts space-separated `key=value` overrides of the YAML settings. Unknown keys are rejected. Paths with spaces can be passed as a single quoted argument, such as `"data=D:/my data/cifar100.npz"`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `data`, `output` | Dataset-specific | Prepared archive and run directory |
| `method` | `traceclean` | Training method; supported baselines are listed below |
| `variant` | `full` | TraceClean ablation |
| `seed` | 0 | Model, augmentation and minibatch seed |
| `epochs`, `batch_size` | 200, 128 | Training duration and minibatch size |
| `width` | 64 | First-stage channels in PreAct ResNet-18 |
| `lr`, `momentum`, `weight_decay` | 0.1, 0.9, 0.0005 | SGD with cosine learning-rate decay |
| `warmup` | 10 | Epochs before trajectory recording; also the Small-loss CE warm-up |
| `window` | 5 | Number of recorded predictions for stability/agreement |
| `alpha` | 0.9 | EMA loss coefficient |
| `rho` | 0.9 | Reliability smoothing coefficient |
| `q`, `gce_lambda` | 0.7, 0.5 | GCE exponent and coefficient in the adaptive objective |
| `epsilon` | 0.001 | MAD denominator addition and diagonal variance floor |
| `gmm_iterations`, `gmm_tolerance` | 100, 0.0001 | EM iteration limit and mean log-likelihood stopping tolerance |
| `min_component_weight` | 0.01 | Minimum mixture proportion for a valid fit |
| `threshold` | 0.5 | Clean-detection threshold and hard-variant cutoff |
| `label_smoothing` | 0.1 | Label Smoothing strength |
| `elr_beta`, `elr_lambda` | 0.7, 3.0 | ELR target EMA and regularization coefficient |
| `forget_rate`, `forget_ramp` | 0.4, 10 | Small-loss removal fraction and Co-teaching ramp duration |
| `device`, `workers`, `threads` | `auto`, 0, 4 | Compute device, loader processes and PyTorch CPU threads |

The symmetric-noise configurations set `forget_rate` to the specified corruption rate for the selection baselines. On CIFAR-100N it is a configurable assumed rate, defaulting to 0.4; it is never inferred from reference labels. TraceClean does not use `forget_rate`.

### Epoch order

1. Train with the latest stored weights.
2. After warm-up, run one deterministic forward pass over every training image.
3. Update EMA loss and the prediction ring buffer.
4. Once the buffer is full, normalize trajectory coordinates and fit GMMs.
5. Smooth the posteriors and store weights for the next epoch.

With `warmup=10` and `window=5`, recording starts after epoch 11. The first full window is available after epoch 15, and the first weighted training epoch is 16. Until then the objective is CE. Reliability fitting has no gradients and does not receive reference labels.

For an observed-label probability `p`, the per-sample objective is:

```text
w × (-log p) + gce_lambda × (1 - w) × (1 - p^q) / q
```

Stability is one minus the fraction of adjacent prediction changes in the window. Agreement is the fraction of predictions matching the observed label. Loss is negated after normalization, so all coordinates point toward greater reliability. The reliable GMM component has the larger mean across its oriented coordinates.

### Continue an interrupted run

```bash
traceclean train --config configs/cifar100-sym40.yaml --resume runs/cifar100-sym40/full-seed0/last.pt
```

Keep the original configuration, data and output directory. Resume restores both networks when applicable, optimizer/scheduler states, trajectory memory, ELR targets and random-number states. `last.pt` and `history.csv` must refer to the same epoch. An already completed run is not restarted. Loader process scheduling can still affect exact bitwise agreement when using multiple workers.

## 4. Evaluate and inspect results

```bash
traceclean evaluate --checkpoint runs/cifar100-sym40/full-seed0/last.pt --output runs/cifar100-sym40/full-seed0/evaluation.json
traceclean plot --run runs/cifar100-sym40/full-seed0
```

Use `--data` when the same prepared archive has moved and `--device cpu` to evaluate on a CPU. `--threshold 0.7` changes the detection operating point without changing the checkpoint or training weights.

Metrics in JSON and CSV are fractions in `[0, 1]`; figures and comparison tables display percentages.

| Metric | Interpretation |
| --- | --- |
| `top1` | Accuracy on clean test labels |
| `worst_class_accuracy` | Minimum accuracy over represented true test classes |
| `auroc` | Ranking of clean training samples as the positive class |
| `clean_precision` | Fraction of selected training samples whose observed/reference labels agree |
| `clean_recall` | Fraction of all clean training samples selected |
| `selected_fraction` | Fraction with a score at or above the threshold |
| `per_class_clean_recall` | Clean recall grouped by observed training class |
| `fallback_samples`, `retained_samples` | Samples using global fallback or an earlier posterior |

AUROC is `null` when there is only one clean/noisy category. Precision is `null` when nothing is selected, and recall is `null` when no clean samples are available. Missing test classes have `null` per-class accuracy and are excluded from the worst-class minimum. A TraceClean run ending before a full window reports no detection metrics.

### Run files

| File | Contents |
| --- | --- |
| `config.yaml`, `data.json` | Resolved settings and prepared-data metadata |
| `history.csv` | Per-epoch losses, accuracy, available detection metrics, fit diagnostics and elapsed training/refresh time |
| `last.pt` | Final or latest epoch, model and training state |
| `summary.json` | Final-epoch classification/detection metrics and configuration |
| `test_predictions.npz` | Clean-test predictions and labels |
| `scores.npz` | Training scores, labels, losses, predictions and original indices; TraceClean also saves raw posteriors and trajectories |
| `figures/training.png` | Objective/test loss, accuracy and detection curves |
| `figures/classes.png` | Per-class test accuracy and clean recall |
| `figures/reliability.png` | Clean/noisy score histograms and ROC curve |
| `figures/trajectories.png` | Loss versus stability/agreement colored by reliability |

TraceClean detection uses **smoothed continuous reliability**, including for hard ablations; `scores.npz` also stores the current raw GMM posterior. Baselines use the final observed-label probability as a shared detection proxy. This makes their ranking diagnostics available, but the probability threshold and the TraceClean threshold have different calibrations. AUROC does not depend on the selected cutoff.

Final-epoch results are the comparison endpoint. The code does not select a checkpoint using clean test accuracy. Test metrics are logged for reporting and never control training, GMM fitting or hyperparameters. Timing fields exclude checkpoint writes and clean-test evaluation; baseline diagnostic refresh occurs only at the final epoch.

## 5. Compare baselines

```bash
traceclean sweep --config configs/cifar100-sym40.yaml --suite baselines --seeds 0 1 2 --output runs/cifar100-sym40/baselines
traceclean summarize --root runs/cifar100-sym40/baselines --output results/cifar100-sym40/baselines
```

The sweep runs sequentially and includes TraceClean, CE, Label Smoothing, GCE, Co-teaching, ELR and Small-loss. All use the same data archive, architecture, optimization schedule and seed list. Co-teaching trains two independently initialized networks, exchanges each minibatch's small-loss indices and evaluates the mean of their probabilities.

A single baseline can be run directly:

```bash
traceclean train --config configs/cifar100-sym40.yaml --set method=gce output=runs/cifar100-sym40/gce-seed0
traceclean train --config configs/cifar100-sym40.yaml --set method=elr output=runs/cifar100-sym40/elr-seed0
```

Small-loss trains with CE during warm-up, then retains the lowest-loss `floor((1 - forget_rate) × batch_size)` examples per minibatch, with at least one retained. Co-teaching ramps its removal rate from zero at epoch 1 to `forget_rate` at epoch `forget_ramp + 1`. ELR uses a detached per-sample EMA probability target and the `log(1 - target · prediction)` regularizer. These implementations follow the mechanisms in the [Co-teaching](https://github.com/bhanML/Co-teaching) and [ELR](https://github.com/shengliu66/ELR) source projects under the shared settings above.

## 6. Run ablations

```bash
traceclean sweep --config configs/cifar100-sym40.yaml --suite ablations --seeds 0 1 2 --output runs/cifar100-sym40/ablations
traceclean summarize --root runs/cifar100-sym40/ablations --output results/cifar100-sym40/ablations
```

| `variant` | Loss EMA | Stability/agreement | Class normalization | Class fitting | Continuous weights |
| --- | --- | --- | --- | --- | --- |
| `current-loss` | No | No | No | No | No |
| `ema-loss` | Yes | No | No | No | No |
| `trajectory` | Yes | Yes | No | No | No |
| `global-gmm` | Yes | Yes | No | No | Yes |
| `class-gmm` | Yes | Yes | Yes | Yes | No |
| `full` | Yes | Yes | Yes | Yes | Yes |
| `class-normalize-only` | Yes | Yes | Yes | No | Yes |
| `class-fit-only` | Yes | Yes | No | Yes | Yes |

The first six rows match the components in the paper's ablation table. `trajectory` supplies the global-hard case for the global/class × hard/soft comparison. The last two rows separate class normalization from class fitting. Each variant waits for the same full post-warm-up window.

Two additional variants are available individually: `no-ema` uses current loss with the full three-coordinate class-relative method; `no-smoothing` sets the score smoothing coefficient to zero.

```bash
traceclean train --config configs/cifar100-sym40.yaml --set variant=no-smoothing output=runs/cifar100-sym40/no-smoothing-seed0
```

`--suite all` combines baseline and ablation runs and avoids repeating the full TraceClean run. Use a new output parent for each sweep; existing run directories are not overwritten.

## 7. Read aggregated results

`summarize` reads only actual `summary.json` files. It produces `comparison.csv`, `comparison.json`, `comparison.md` and `comparison.png`. Groups keep dataset metadata, noise realization and training settings separate. Each row reports the mean across training seeds and the sample standard deviation when at least two observations exist. One-seed rows have no standard deviation. Duplicate method/variant/seed observations are rejected rather than counted twice. `comparison.json` records the source files and settings for each group.

A sweep holds the corruption seed fixed and varies the training seed. To study corruption variation as well, prepare separate archives and run each setting separately. The summarized means are not significance tests.

## Implementation choices

The paper specifies the core formulas and TraceClean parameters but leaves several training, EM and ablation details open. This implementation uses the following conventions throughout:

- PreAct ResNet-18 has a CIFAR 3×3 stem, stage widths 64/128/256/512, two blocks per stage, a final BN/ReLU and global average pooling. Training uses SGD for 200 epochs with cosine decay, a four-pixel random crop and horizontal flip. Trajectory and test passes use evaluation mode and normalization without stochastic augmentation.
- Symmetric corruption excludes the original class. The prepared archive records both the requested rate and the actual changed-label fraction. CIFAR-100N uses the released noisy fine labels after an order check.
- MAD is unscaled. `epsilon=0.001` is added to each MAD denominator; no feature clipping is applied. The same epsilon floors diagonal variances during EM. K-means with three initializations and one thread seeds each two-component fit. Fits are rejected if EM does not converge, a component has less than 1% mass or its means coincide within epsilon.
- A fit needs at least `2(d + 1)` samples: eight for a full trajectory or four for loss-only variants. Failed class fits use a global GMM on the **same epoch's class-normalized, oriented vectors**. If it also fails, each affected sample keeps its last valid raw posterior, initially one. Reliability smoothing is then applied normally. No clean labels are used to identify a component.
- Global variants normalize over the complete training set. Hard variants threshold the smoothed score and use the same CE/GCE objective with binary coefficients. They do not discard the low-reliability examples. The paper does not specify that hard-weight convention or its cutoff.
- Detection uses smoothed reliability and a 0.5 cutoff; raw posteriors are available for alternative analysis. Baseline detection scores are observed-label probabilities. The paper does not specify its precision/recall operating point.
- The included classification baselines are CE, Label Smoothing, GCE, Co-teaching and ELR, with Small-loss for selection analysis. DivideMix, DISC, DSS and DynaCor appearing in the paper's tables are outside the included baseline set. The comparison commands report only methods run by this repository.

The implementation follows the supplied paper's method and evaluated dataset settings. Full CIFAR training, baseline sweeps and ablation sweeps must be run to obtain their experiment results. CI checks installation, syntax, CLI entry points and shipped configurations; it does not train models or run a test suite.

## Code layout

```text
configs/                    CIFAR and small-example settings
src/traceclean/config.py     Settings and ablation switches
src/traceclean/data.py       Data preparation, indexed datasets and transforms
src/traceclean/model.py      PreAct ResNet-18
src/traceclean/trajectory.py  Trajectory state, robust normalization and diagonal EM
src/traceclean/losses.py      GCE, adaptive CE/GCE, ELR and Co-teaching losses
src/traceclean/train.py       Training, checkpoint resume and evaluation
src/traceclean/metrics.py     Classification and clean-label detection metrics
src/traceclean/report.py      Per-run plots and result aggregation
src/traceclean/cli.py         Command-line interface
```

The source code is distributed under the MIT license. Dataset terms are supplied by their respective providers.
