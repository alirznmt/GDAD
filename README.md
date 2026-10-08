# GDAD: Learned Sensor Dependency Networks for Diffusion-Based Anomaly Detection

Official PyTorch implementation of **“Learned Sensor Dependency Networks for Diffusion-Based Anomaly Detection in Cyber-Physical Infrastructures.”**

GDAD is an unsupervised anomaly detector for multivariate time series. It conditions a diffusion denoiser on a learned, directed sensor-dependency network operating at two timescales:

- a **persistent layer** that captures sparse dependencies shared across all windows;
- an **instance-specific layer** that captures dense, transient dependencies inferred from each input window; and
- a learned gate that combines both layers inside every denoising block.

Instead of running a full reverse-diffusion chain at inference time, GDAD scores each window with a fixed set of one-step denoising probes. The implementation also separates label-free evaluation from oracle diagnostics: validation-fitted POT is the primary threshold, while best-F1 on test labels is reported only as an optimistic diagnostic.

## Highlights

- Dual-timescale, directed sensor graph with persistent and instance-specific dependencies
- Fully unsupervised training on normal data
- Leakage-safe preprocessing: the scaler is fitted only on the training split
- Label-free Peaks-Over-Threshold (POT) evaluation
- Pointwise and point-adjusted precision, recall, F1, AUROC, AUPRC, and detection delay
- Optional Range-AUC and VUS metrics through the reference VUS implementation
- Configurable graph ablations: identity, persistent, instance, and combined
- Reproducible checkpoints with configuration and data-provenance checks
- Support for MSL, PSM, SMAP, SWaT, and HAI 22.04

## Method overview

For a multivariate window, GDAD learns two row-stochastic adjacency matrices. The persistent graph is a global, top-*k* sparse backbone shared by every sample. The instance graph is inferred from the current noisy window and can adapt to changes in operating conditions. A learned scalar gate forms their convex combination, which conditions a compact graph-temporal diffusion denoiser.

Anomaly scores combine reconstruction and noise-prediction residuals from selected diffusion levels. With the default configuration, three diffusion levels and four noise samples require 12 denoiser evaluations per window, rather than a complete 100-step reverse chain.

## Results reported in the paper

The following values are five-seed means for the default GDAD model. R-AUC-PR and VUS-PR are macro averages across the five datasets.

| MSL F1 | PSM F1 | SMAP F1 | SWaT F1 | HAI F1 | Macro F1 | R-AUC-PR | VUS-PR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.218** | 0.437 | **0.254** | **0.650** | **0.256** | **0.363** | **0.310** | **0.296** |

The combined dependency network improved AUPRC over the identity, persistent-only, and instance-only variants on all five benchmarks. The largest thresholded-detection gains occurred on SWaT and HAI, where sensors are strongly coupled through physical processes and control logic.

> **Evaluation note:** `main.py` treats validation-fitted POT as the primary, label-free operating threshold. It also writes oracle best-F1 results for diagnosis and comparison, but those values use test labels to choose a threshold and must not be interpreted as deployable performance.

## Repository structure

```text
.
├── configs/                 # Base and dataset-specific configurations
│              
├── models/                  # Graph learner, denoiser, diffusion, and GDAD model
├── tools/                   # Diagnostics, scaling, SNR, and reviewer analyses
├── utils/                   # Data, metrics, scoring, logging, and reproducibility
├── datasets.py              # Dataset loading, normalization, and windowing
├── train.py                 # Train one dataset
├── evaluate.py              # Evaluate one checkpoint
└── main.py                  # Multi-dataset training/evaluation entry point
```

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- pandas
- scikit-learn
- SciPy
- PyYAML

A CUDA-capable GPU is recommended for training, especially for HAI. Install a PyTorch build suitable for your system, then install the remaining dependencies:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install torch numpy pandas scikit-learn scipy pyyaml
```

Range-AUC and VUS metrics require the reference VUS implementation to be importable as `vus.metrics` (or `metrics.vus.metrics`). If it is unavailable, the rest of the evaluation still runs and the four range-based metrics are returned as `NaN`.

## Data preparation

Download each dataset from its official source and follow its license and access requirements. Place the processed files under `data/` using the layout below:

```text
data/
├── MSL/
│   ├── MSL_train.npy
│   ├── MSL_test.npy
│   └── MSL_test_label.npy
├── SMAP/
│   ├── SMAP_train.npy
│   ├── SMAP_test.npy
│   └── SMAP_test_label.npy
├── PSM/
│   ├── train.csv
│   ├── test.csv
│   └── test_label.csv
├── SWaT/
│   ├── swat_train2.csv
│   └── swat2.csv
└── HAI/
    ├── train1.npy ... train6.npy
    ├── test1.npy  ... test4.npy
    └── test_label1.npy ... test_label4.npy
```

Expected formats:

- **MSL and SMAP:** two-dimensional `float` arrays shaped `[timestamps, sensors]`; labels are one-dimensional binary arrays.
- **PSM:** the first CSV column is treated as a timestamp and removed. Labels are read from the second column of `test_label.csv`.
- **SWaT:** the final column is treated as the test label and removed from the feature matrix. The training CSV is expected to use the same feature layout.
- **HAI:** the six training segments and four test segments are loaded in the order shown above. Each data file must contain 86 feature columns, and every test segment must have a matching binary label file.
- Labels must use `0/1`, with `1` indicating an anomaly. The loader also accepts the WADI-style `+1/-1` convention, where `-1` indicates an anomaly.

Dataset paths and filenames can be changed in `configs/<dataset>.yaml`. Windows are never allowed to cross file boundaries, which is important for segmented HAI data.

## Quick start

All commands should be run from the repository root.

### Train and evaluate one dataset

```bash
python main.py --datasets SMAP --mode train_eval
```

### Train and evaluate the four default benchmarks

```bash
python main.py --datasets all --mode train_eval
```

`all` expands to `SMAP MSL SWaT PSM`. HAI is supported but remains explicit because of its larger dense sensor graph:

```bash
python main.py --datasets HAI --mode train_eval
```

### Train only

```bash
python train.py --dataset PSM
```

The best validation checkpoint is saved to `checkpoints/<DATASET>/best.pt`.

### Evaluate a checkpoint

```bash
python evaluate.py --dataset SMAP --ckpt checkpoints/SMAP/best.pt
```


On Windows PowerShell, place the command on one line or replace each trailing `\` with a backtick.

## Configuration overrides

`configs/base.yaml` defines the shared defaults and each dataset YAML overrides dataset-specific values. Any setting can be changed from the command line with dotted `key=value` syntax:

```bash
python train.py --dataset PSM --override train.epochs=30 model.hidden_dim=128
```

To run graph ablations, set `model.graph_mode` to one of `identity`, `persistent`, `instance`, or `combined`:

```bash
python main.py \
  --datasets SWaT \
  --mode train_eval \
  --override model.graph_mode=persistent
```

Useful defaults include:

| Setting | Default |
|---|---:|
| Window size | 100 |
| Stride | 50 |
| Hidden width | 96 |
| Denoising blocks | 3 |
| Persistent top-*k* | 10 |
| Diffusion steps | 100 |
| Evaluation levels | 10, 25, 50 |
| Noise samples per level | 4 |
| Training epochs | 40 |

## Outputs

Training creates:

```text
checkpoints/<DATASET>/best.pt
checkpoints/<DATASET>/last.pt
```

Evaluation creates:

```text
results/<DATASET>/metrics.json
results/<DATASET>/test_scores.npy
```

Multi-dataset runs additionally create:

```text
results/summary.md
results/summary.json
```

`summary.md` contains two tables:

1. **Primary:** label-free POT fitted on validation scores.
2. **Diagnostic:** oracle best-F1, which uses test labels only for threshold analysis.

Logs are written under `logs/`.

## Reproducibility notes

- The default random seed is 42 unless overridden in the configuration.
- Deterministic PyTorch behavior is enabled by default.
- Standardization is fitted only on the optimization portion of the normal training data.
- Validation data is chronologically separated from the training portion.
- Checkpoints record the resolved configuration, sensor count, effective graph mode, and data provenance.
- Evaluation verifies checkpoint/data compatibility before loading model weights.
- For multi-seed results, run seeds 42–46 and aggregate metrics per dataset before computing macro averages.

## Citation

If you use this code or build on GDAD, please cite:

```bibtex
@misc{nemati2026gdad,
  title  = {Learned Sensor Dependency Networks for Diffusion-Based Anomaly Detection in Cyber-Physical Infrastructures},
  author = {Nemati, Alireza and Boudaghi, Ali and Zare, Hadi and Cherifi, Hocine},
  year   = {2026},
  note   = {Manuscript}
}
```



## Contact

For questions about the paper or implementation, please open a GitHub issue. Academic correspondence may be directed to **alinmt94@gmail.com** and **nemati.alireza@ut.ac.ir**.
