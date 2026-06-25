# T3DR — Trackerless 3D Freehand Ultrasound Reconstruction

Predict the per-frame 6-DoF probe pose of a freehand B-mode ultrasound sweep directly
from the 2D image stream (no external tracker, no IMU at inference). Accumulating the
per-frame transforms reconstructs the 3D sweep trajectory. Benchmarked against the
official **TUS-REC2024** challenge metrics.

This README covers how to run training and evaluation, what each script produces, the
expected data layout, and the results achieved.

---

## 1. What the model predicts

For each consecutive frame pair the model regresses 6 numbers:

```
(rz, ry, rx, tx, ty, tz)   # ZYX-intrinsic Euler + translation, in millimetres
```

These are the local transform **frame_{i+1} → frame_i** in the *image_mm* frame
(the TUS-REC "parameter" convention). Channel order is `(rz, ry, rx, ...)` — the train
and eval decoders both assume this; **do not reorder it.** `tz` is the out-of-plane
(elevational) component and is the hard part of the problem.

### Metrics (all in mm, lower is better)

| Metric | Meaning |
| --- | --- |
| **GPE** | Global Pixel reconstruction Error — accumulated trajectory, all pixels. The headline number. |
| **GLE** | Global Landmark Error — accumulated trajectory, at annotated landmarks. |
| **LPE** | Local Pixel Error — per-frame transform only (no accumulation). The local-accuracy ceiling. |
| **LLE** | Local Landmark Error — per-frame transform, at landmarks. |

All numbers are on the **validation set** (subjects 050 / 051 / 052, 72 scans, 24 each),
not held-out test.

---

## 2. Results achieved

| Configuration | GPE | GLE | LPE | LLE |
| --- | --- | --- | --- | --- |
| Bidirectional model + test-time inference (clean single model) | **11.86** | — | — | — |
| 6-model cross-resolution ensemble (best) | **11.60** | 10.80 | 0.1532 | 0.1393 |

- The clean single-model result is the bidirectional model evaluated with
  `--avg-window --reverse-tta` (see §6).
- The best number is a cross-resolution, multi-architecture ensemble of checkpoints (§9).
- The remaining gap to the leaderboard is a **local-accuracy ceiling** (LPE ≈ 0.15):
  accumulation is solved (the GPE/LPE ratio is healthy), but per-frame local accuracy
  plateaus. Higher input resolution is the only lever that moved LPE.

---

## 3. Repository layout

```
T3DR/
├── train_6dof.py                 # MAIN trainer — 6-DoF pose regression
├── eval_6dof.py                  # MAIN evaluator — official GPE/GLE/LPE/LLE
├── models/
│   ├── fimanet_mamba_6dof.py     # the production model (ResNet-18 + pair encoder + Mamba)
│   ├── fimanet_mamba.py          # precursor: scalar Δz model, real Mamba block
│   ├── fimanet.py                # precursor: scalar Δz model, lightweight SSM block
│   └── moglonet.py               # MoGLo-Net reimplementation (baseline)
├── lib/                          # precursor IMU / Kalman stack (see §10)
│   ├── eskf.py
│   ├── imu_simulator.py
│   ├── imu_verifier.py
│   └── complementary_filter.py
├── train_runner.py               # precursor trainer (scalar Δz + IMU verifier eval)
├── run_ablations.py              # precursor: sweeps filter/σ/TTA configs
├── requirements.txt
└── README.md
```

Outputs are written to `runs/<timestamp>_<run_name>/` (checkpoints) and
`6dof_eval_per_scan_<tag>.csv` (per-scan metrics). These plus `__pycache__/` and any
`*.png` are **gitignored** — model weights live on the training server, not in git.

---

## 4. Environment

- Python 3.12, CUDA 12.8, an NVIDIA A10 (24 GB) is the reference GPU.
- Key pins (full list in [requirements.txt](requirements.txt)):
  `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `mamba-ssm==2.3.2.post1`,
  `h5py`, `opencv-python-headless`, `scipy`, `numpy`, `matplotlib`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`mamba-ssm` builds CUDA kernels — install it on the GPU host, not a CPU-only laptop.

**GPU pinning is hard-coded** at the top of each script: `train_6dof.py` uses
`CUDA_VISIBLE_DEVICES=1` and `eval_6dof.py` uses `CUDA_VISIBLE_DEVICES=0`, so the two can
run concurrently on a 2-GPU box. Edit those lines if your machine differs.

---

## 5. Expected data layout

All paths are rooted at `BASE_DATA_DIR` (default `/home/123ghdh/datasets`, set in both
`train_6dof.py` and `eval_6dof.py`).

```
$BASE_DATA_DIR/
├── 000/ 001/ ... 049/                # training subjects, each a folder of *.h5
│   └── *.h5                          #   'frames' (N,H,W) uint8, 'tforms' (N,4,4) tool→world
├── valDataset/data/
│   ├── frames/<subj>/<scan>.h5       # subj ∈ {050,051,052}
│   ├── transfs/<subj>/<scan>.h5      # matching GT transforms
│   └── landmark/landmark_<subj>.h5   # (20,3) (frame_idx, x, y) per scan, for GLE/LLE
└── calib_matrix.csv                  # pixel→image_mm and image_mm→tool calibration
```

Scan naming: `LH/RH_Par/Per_C/L/S_DtP/PtD` (hand · parallel/perpendicular · shape ·
sweep direction). `DtP`/`PtD` are the same physical path swept in opposite directions.

---

## 6. Quickstart — evaluate a checkpoint

```bash
python eval_6dof.py --ckpt runs/<timestamp>_<run>/<run>_best.pth --avg-window --reverse-tta
```

Prints a GPE/GLE/LPE/LLE summary and writes `6dof_eval_per_scan_<tag>.csv` next to the
checkpoint. The evaluator **auto-detects the architecture** from the checkpoint (pooling
grid, sequence length, bidirectional, correlation/spatial-SSM blocks), so you normally
only pass `--backbone` if it isn't `resnet18`.

---

## 7. Training

```bash
python train_6dof.py --name my_run                                          # ImageNet init
python train_6dof.py --name hires_ft --init-ckpt runs/.../bidir_best.pth    # warm-start
```

The run config is a **constants block at the top of [train_6dof.py](train_6dof.py)** — there
is no config file; edit the constants to change the experiment. The main knobs:

| Constant | Role |
| --- | --- |
| `NET_H, NET_W` | network input resolution. **Eval must match this** (`--infer-res`). |
| `BIDIRECTIONAL` | `True` = BiMamba (forward+backward); resolves the `tz` sign. The production setting. |
| `SEQ_LEN_MIN/MAX` | sliding-window length (frames). |
| `WINDOW_STRIDE` | sampling stride between windows — drives sequences/epoch and epoch time. |
| `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM` | budget. |
| `LR_BACKBONE`, `LR_HEAD` | backbone fine-tunes slower than the head. |
| `USE_CORR`, `USE_SPATIAL_SSM` | optional encoder variants (off by default). |
| `W_GLOBAL`, `W_PEARSON` | loss weights for accumulation-consistency + case-wise Pearson; 0 = pure point-L1. |

**Produces** — per run, a directory `runs/<timestamp>_<run_name>/` containing:

- `<run_name>_best.pth` — best checkpoint by the **GPE proxy** (a few-scan sliding-window
  GPE evaluated each epoch; this tracks the real metric far better than val loss).
- `<run_name>_ep{N}.pth` — one checkpoint per epoch, for alternate selection or ensembling.

Each epoch logs `Train | Val | MaxCorner | GPEproxy | tz_resid`. Watch `GPEproxy`
(selection metric) and `tz_resid` (signed elevational bias — should trend toward 0).
`--init-ckpt` loads with `strict=False` and pops the length-dependent `pos_encoder.pe`
buffer, so a model trained at one sequence length can seed another.

---

## 8. Evaluation options

```bash
python eval_6dof.py --ckpt CKPT [CKPT ...] [options]
```

| Flag | Effect |
| --- | --- |
| `--ckpt A B C` | one path = single model; several = **ensemble** (mean of predicted 6-DoF params). |
| `--mode pure\|hybrid\|both` | `pure` = all 6 DoF from the model; `hybrid` = Lucas-Kanade supplies (tx,ty); `both` writes a comparison (default). |
| `--avg-window` | average each transition over all window positions with bidirectional context, instead of the last (zero-future-context) position. **Recommended for bidir models.** |
| `--reverse-tta` | also predict the reversed scan, invert + reorder, and average — cancels the direction-coherent `tz` bias. |
| `--infer-res H W` | network input resolution; **must equal the training `NET_H/NET_W`.** |
| `--ckpt-res HxW ...` | per-checkpoint resolution (parallel to `--ckpt`) for cross-resolution ensembles. |
| `--dump-trajectories DIR` | write per-scan predicted/GT corner trajectories + per-frame GPE as `.npz`. |
| `--backbone` | must match the checkpoint's backbone (default `resnet18`). |
| `--limit N` | evaluate only the first N scans (debugging). |

**Output:** `6dof_eval_per_scan_<tag>.csv` (tag encodes the mode/flags) in `--out-dir`
(default = the first checkpoint's directory), plus the printed summary.

---

## 9. Reproducing the best result (GPE 11.60)

A cross-resolution, multi-architecture ensemble — 6 checkpoints, two input resolutions,
with test-time inference:

```bash
python eval_6dof.py \
  --ckpt  bidir_ep9.pth bidir_ep10.pth bidir_ep11.pth tier1_best.pth tier2_best.pth hires_ep5.pth \
  --ckpt-res 240x320 240x320 240x320 240x320 240x320 360x480 \
  --avg-window --reverse-tta --mode pure
```

The clean single-model headline of 11.86 is just the bidir best checkpoint with
`--avg-window --reverse-tta`. Substitute your own checkpoint paths.

---

## 10. Precursor IMU pipeline

`train_runner.py`, `run_ablations.py`, the `lib/` stack, and `models/fimanet*.py` /
`models/moglonet.py` are an earlier track that predicted only the scalar out-of-plane
displacement (Δz) and fused it with a simulated IMU through a Kalman filter. It is
superseded by the 6-DoF model but kept for reference:

- [train_runner.py](train_runner.py) — trains the scalar-Δz model
  ([models/fimanet.py](models/fimanet.py), or [models/fimanet_mamba.py](models/fimanet_mamba.py)
  with `USE_MAMBA=1`); at eval it runs loosely-coupled visual+inertial fusion.
- [lib/](lib/) — error-state Kalman filter ([eskf.py](lib/eskf.py)), IMU simulator
  ([imu_simulator.py](lib/imu_simulator.py)), fusion wrapper
  ([imu_verifier.py](lib/imu_verifier.py)), classical complementary-filter baseline
  ([complementary_filter.py](lib/complementary_filter.py)).
- [run_ablations.py](run_ablations.py) — sweeps filter / σ / TTA / smoothing by repeatedly
  invoking `train_runner.py --eval-only`, then writes a comparison table.
- [models/moglonet.py](models/moglonet.py) — MoGLo-Net reimplementation (baseline).

---

## 11. Gotchas

- **Eval resolution must match training.** Images are resized to `NET_H × NET_W`
  (default 240×320); evaluating at a different resolution silently degrades accuracy. Pass
  `--infer-res` (or `--ckpt-res` for mixed ensembles) to match.
- **Bidirectional models need sliding-window inference, not `--fullscan`.** A whole-scan
  single pass is off-distribution for them (~70 mm GPE). Use `--avg-window`.
- **Selection is the GPE proxy, not val loss.** Validation point-loss is only weakly
  correlated with GPE; `*_best.pth` is chosen by the per-epoch GPE proxy.
- **GPU device is hard-coded** (train = GPU 1, eval = GPU 0) at the top of each script.
- **Checkpoints / CSVs / PNGs are gitignored** — they are produced at runtime, not stored
  in the repo.
