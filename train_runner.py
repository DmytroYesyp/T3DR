import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import cv2
import csv
import gc
import glob
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from datetime import datetime

from models.fimanet import FiMANet
from models.moglonet import MoGLoNet
from lib.imu_simulator import IMUSimulator
from lib.imu_verifier import IMUVerifier

# =============================================================================
# CONFIG
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20
BATCH_SIZE = 164
EPOCHS = 10

# In-model IMU fusion is OFF — IMU lives outside as a verifier (see below).
USE_IMU = False
WARMUP_EPOCHS = 2  # freeze layer2/3/4 for this many epochs

# Eval-time IMU verifier settings
USE_IMU_VERIFIER = True
PREDICT_UNCERTAINTY = False
# Two-head: |Δz| regression + binary sign classification. Single-head signed
# regression plateaued (val MAE ~0.14 mm, severe FDR on *_L_DtP cases) due to
# a directional bias the model can't unlearn. Decoupling magnitude from
# direction lets each head specialize.
PREDICT_SIGN_TWO_HEAD = True
SIGN_LOSS_WEIGHT = 0.3       # λ in total = mag_l1 + λ * BCE(sign)
SIGN_LABEL_MIN_DZ = 0.01     # mm — frames with |Δz| below this are masked
                              # from BCE since their sign is essentially noise

BASE_DATA_DIR = "/home/123ghdh/datasets"
TRAIN_FOLDERS = [os.path.join(BASE_DATA_DIR, str(i).zfill(3)) for i in range(50)]

VAL_FRAMES_ROOT = os.path.join(BASE_DATA_DIR, 'valDataset/data/frames')
VAL_TFORMS_ROOT = os.path.join(BASE_DATA_DIR, 'valDataset/data/transfs')
VAL_LAST_N = 10  # legacy: last N pairs alphabetically (monovariate)
VAL_PER_SUBJECT = 4  # stratified: N scans per validation subject

# =============================================================================
# UTILITIES
# =============================================================================
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

RUNS_ROOT = "runs"

def next_run_name(prefix):
    """Return '{prefix}_v{N}' where N is the lowest free integer version.
    Probes both legacy flat artifacts and the new runs/ directory layout
    so versions keep incrementing across the migration.
    """
    n = 1
    while glob.glob(f"{prefix}_v{n}_*") or glob.glob(os.path.join(RUNS_ROOT, f"*_{prefix}_v{n}")):
        n += 1
    return f"{prefix}_v{n}"


def make_run_dirs(run_name):
    """Create runs/{YYYY-MM-DD_HHMMSS}_{run_name}/{,val_images}/ and return paths."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(RUNS_ROOT, f"{ts}_{run_name}")
    val_dir = os.path.join(run_dir, "val_images")
    os.makedirs(val_dir, exist_ok=True)
    return run_dir, val_dir

def set_backbone_trainable(model, trainable):
    """Toggle requires_grad for layer2/3/4. stem and layer1 stay frozen."""
    for name in ['layer2', 'layer3', 'layer4']:
        for p in getattr(model, name).parameters():
            p.requires_grad = trainable

def collect_last_n_val_files(n=VAL_LAST_N):
    """Last n (frames_path, tforms_path, case_name) triples from VAL_FRAMES_ROOT,
    sorted by subject then filename. Tolerant of missing tform pairs and a
    flat (no-subject-subdir) layout.

    NOTE: Kept for backward compatibility. Prefer collect_val_files_stratified
    so the validation signal isn't dominated by a single subject.
    """
    pairs = []
    if not os.path.isdir(VAL_FRAMES_ROOT):
        return pairs
    for subj in sorted(os.listdir(VAL_FRAMES_ROOT)):
        sd = os.path.join(VAL_FRAMES_ROOT, subj)
        if not os.path.isdir(sd):
            continue
        for fname in sorted(os.listdir(sd)):
            if not fname.endswith('.h5'):
                continue
            tpath = os.path.join(VAL_TFORMS_ROOT, subj, fname)
            if os.path.exists(tpath):
                pairs.append((os.path.join(sd, fname), tpath, f"{subj}/{fname}"))
    return pairs[-n:]


def collect_val_files_stratified(n_per_subject=4):
    """Take n_per_subject scans from EACH validation subject. Avoids the
    pathology where the previous 'last 10 alphabetically' selector was
    monovariate (all from the alphabetically-last subject), making
    cross-subject generalization invisible to the val signal.
    """
    pairs = []
    if not os.path.isdir(VAL_FRAMES_ROOT):
        return pairs
    for subj in sorted(os.listdir(VAL_FRAMES_ROOT)):
        sd = os.path.join(VAL_FRAMES_ROOT, subj)
        if not os.path.isdir(sd):
            continue
        subj_pairs = []
        for fname in sorted(os.listdir(sd)):
            if not fname.endswith('.h5'):
                continue
            tpath = os.path.join(VAL_TFORMS_ROOT, subj, fname)
            if os.path.exists(tpath):
                subj_pairs.append((os.path.join(sd, fname), tpath, f"{subj}/{fname}"))
        pairs.extend(subj_pairs[:n_per_subject])
    return pairs

def load_weights_safe(model, path):
    print(f"[{get_time()}] Loading weights from {path}...")
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# =============================================================================
# LOSS
# =============================================================================
class MotionWeightedL1Loss(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        base_loss = torch.abs(pred - target)
        # |target| so negative-direction frames aren't down-weighted.
        weight = 1.0 + self.alpha * torch.abs(target)
        weighted_loss = base_loss * weight
        return torch.mean(weighted_loss)


class MotionWeightedHeteroscedasticLoss(nn.Module):
    """Gaussian NLL on (μ, log σ²) with the same motion-weighted prior.
    Lets the network predict its own per-frame σ — the ESKF reads σ as
    measurement noise so the filter trusts confident frames more.
    """
    def __init__(self, alpha=2.0, log_var_min=-6.0, log_var_max=6.0):
        super().__init__()
        self.alpha = alpha
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

    def forward(self, mu, log_var, target):
        log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)
        sq_err = (target - mu) ** 2
        nll = 0.5 * torch.exp(-log_var) * sq_err + 0.5 * log_var
        weight = 1.0 + self.alpha * torch.abs(target)
        return torch.mean(weight * nll)


class MotionWeightedTwoHeadLoss(nn.Module):
    """Magnitude L1 (motion-weighted) + binary cross-entropy on sign.

    Sign labels for very small motion (|Δz| ≈ 0) are essentially noise — we
    mask them out of the BCE term so the classifier isn't penalized for
    coin-flipping on near-stationary frames.

    Combined: total = mag_l1 + sign_weight * sign_bce
    Returns (total, mag_loss, sign_loss) so training can log both.
    """
    def __init__(self, alpha=2.0, sign_weight=0.3, sign_min_dz=0.01):
        super().__init__()
        self.alpha = alpha
        self.sign_weight = sign_weight
        self.sign_min_dz = sign_min_dz
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, mag_pred, sign_logit, target):
        target_abs = torch.abs(target)
        target_sign = (target > 0).float()

        mag_l1 = torch.abs(mag_pred - target_abs)
        mag_weight = 1.0 + self.alpha * target_abs
        mag_loss = (mag_weight * mag_l1).mean()

        sign_mask = (target_abs > self.sign_min_dz).float()
        sign_bce = self.bce(sign_logit, target_sign) * sign_mask
        denom = sign_mask.sum().clamp(min=1.0)
        sign_loss = sign_bce.sum() / denom

        return mag_loss + self.sign_weight * sign_loss, mag_loss, sign_loss


# =============================================================================
# DATASETS
# =============================================================================
class LargeUSDataset(Dataset):
    def __init__(self, root_dirs, seq_len=5, use_imu=False):
        self.samples = []
        self.seq_len = seq_len
        self.use_imu = use_imu
        self.imu_sim = IMUSimulator() if use_imu else None
        print(f"[{get_time()}] Indexing folders...")

        for folder in root_dirs:
            if not os.path.exists(folder): continue
            for filename in os.listdir(folder):
                if not filename.endswith('.h5'): continue
                filepath = os.path.join(folder, filename)

                try:
                    with h5py.File(filepath, 'r') as f:
                        if 'frames' not in f or 'tforms' not in f: continue
                        n_frames = f['frames'].shape[0]

                        # No motion filter — model must see full dz distribution.
                        # Step=2 gives 2x more samples than the old step=4.
                        for i in range(0, n_frames - seq_len, 2):
                            self.samples.append({'path': filepath, 'start': i})
                except Exception as e:
                    print(f"[{get_time()}] Error reading {filepath}: {e}")

        print(f"[{get_time()}] Indexed {len(self.samples)} sequences.")
        if len(self.samples) == 0: raise RuntimeError("0 sequences found.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        with h5py.File(meta['path'], 'r') as f:
            frames = f['frames'][meta['start'] : meta['start'] + self.seq_len]
            tforms = f['tforms'][meta['start'] : meta['start'] + self.seq_len + 1]

        seq_imgs = []
        for i in range(self.seq_len):
            img = frames[i]
            img = cv2.resize(img, (256, 256))
            if img.max() > 1.0: img = img / 255.0
            seq_imgs.append(np.expand_dims(img, axis=0))

        seq_tensor = torch.tensor(np.stack(seq_imgs), dtype=torch.float32)
        # Signed Δz — sign-from-IMU was unreliable; let the model learn direction.
        target = torch.tensor(tforms[-1, 2, 3] - tforms[-2, 2, 3], dtype=torch.float32)

        if self.use_imu:
            imu_data = self.imu_sim.generate(tforms)[:self.seq_len]
            imu_tensor = torch.tensor(imu_data, dtype=torch.float32)
            return seq_tensor, imu_tensor, target

        return seq_tensor, target

class MemoryUSDataset(Dataset):
    def __init__(self, frames_path, tforms_path, seq_len=5, use_imu=False):
        self.seq_len = seq_len
        self.samples = []
        self.use_imu = use_imu
        self.imu_sim = IMUSimulator() if use_imu else None

        print(f"[{get_time()}] Loading Validation Data...")
        with h5py.File(frames_path, 'r') as f: self.f = f['frames'][:]
        with h5py.File(tforms_path, 'r') as t: self.t = t['tforms'][:]

        # No motion filter — validation should match the full inference distribution.
        for i in range(0, len(self.f) - seq_len):
            self.samples.append(i)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        start = self.samples[idx]
        seq_imgs = []
        for i in range(self.seq_len):
            img = self.f[start+i]
            img = cv2.resize(img, (256, 256))
            if img.max() > 1.0: img = img/255.0
            seq_imgs.append(np.expand_dims(img, axis=0))
        seq_tensor = torch.tensor(np.stack(seq_imgs), dtype=torch.float32)

        tforms = self.t[start : start + self.seq_len + 1]
        target = torch.tensor(tforms[-1, 2, 3] - tforms[-2, 2, 3], dtype=torch.float32)

        if self.use_imu:
            imu_data = self.imu_sim.generate(tforms)[:self.seq_len]
            imu_tensor = torch.tensor(imu_data, dtype=torch.float32)
            return seq_tensor, imu_tensor, target

        return seq_tensor, target

# =============================================================================
# TRAINING
# =============================================================================
def train_model(run_name, run_dir, model_name, model, train_loader, val_loader, epochs, use_imu=False):
    print(f"\n{'='*50}")
    print(f"[{get_time()}] INITIALIZING TRAINING FOR: {model_name}")
    print(f"[{get_time()}] Run name: {run_name}")
    print(f"[{get_time()}] Active Device: {torch.cuda.get_device_name(DEVICE)} (Index 1)")
    print(f"[{get_time()}] IMU Fusion (in-model): {'ENABLED' if use_imu else 'DISABLED'}")
    print(f"[{get_time()}] Warmup (head-only) epochs: {WARMUP_EPOCHS}")
    print(f"{'='*50}")

    model = model.to(DEVICE)

    if WARMUP_EPOCHS > 0:
        set_backbone_trainable(model, False)
        print(f"[{get_time()}] Warmup: backbone layer2/3/4 frozen.")

    untrained_name = os.path.join(run_dir, f"{run_name}_untrained.pth")
    torch.save(model.state_dict(), untrained_name)

    backbone_params = list(model.stem.parameters()) + \
                      list(model.layer1.parameters()) + \
                      list(model.layer2.parameters()) + \
                      list(model.layer3.parameters()) + \
                      list(model.layer4.parameters())

    head_params = list(model.fusion.parameters()) + \
                  list(model.temporal_layers.parameters())

    # In two-head mode the regression head is split; otherwise there's a
    # single self.head sequential module.
    if getattr(model, 'predict_sign', False):
        head_params += list(model.head_mag.parameters())
        head_params += list(model.head_sign.parameters())
    else:
        head_params += list(model.head.parameters())

    if use_imu:
        head_params += list(model.imu_encoder.parameters())
        head_params += list(model.fusion_gate.parameters())

    optimizer = torch.optim.Adam([
        # 1e-5 destabilized post-unfreeze on the prior 4-epoch run (val NLL went
        # from -1.19 to -0.70 at Ep4); halving it tames that without crippling
        # adaptation of the ResNet features.
        {'params': backbone_params, 'lr': 5e-6},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=1e-5)

    # patience=3 so the scheduler doesn't drop LR on a single noisy val epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    if PREDICT_SIGN_TWO_HEAD:
        criterion = MotionWeightedTwoHeadLoss(
            alpha=2.0, sign_weight=SIGN_LOSS_WEIGHT, sign_min_dz=SIGN_LABEL_MIN_DZ
        )
    elif PREDICT_UNCERTAINTY:
        criterion = MotionWeightedHeteroscedasticLoss(alpha=2.0)
    else:
        criterion = MotionWeightedL1Loss(alpha=2.0)
    scaler = GradScaler('cuda')

    history = {'train': [], 'val': []}
    # Track best by val MAE (the L1 we care about) instead of val NLL —
    # heteroscedastic NLL can drop while MAE worsens if log_var overfits.
    best_val_mae = float('inf')
    best_model_path = os.path.join(run_dir, f"{run_name}_best.pth")
    total_batches = len(train_loader)

    for epoch in range(epochs):
        if epoch == WARMUP_EPOCHS and WARMUP_EPOCHS > 0:
            set_backbone_trainable(model, True)
            print(f"[{get_time()}] Epoch {epoch+1}: warmup over — unfreezing layer2/3/4.")

        model.train()
        train_loss = 0

        for i, batch in enumerate(train_loader):
            if use_imu:
                seqs, imus, targets = batch
                seqs, imus, targets = seqs.to(DEVICE), imus.to(DEVICE), targets.to(DEVICE)
            else:
                seqs, targets = batch
                seqs, targets = seqs.to(DEVICE), targets.to(DEVICE)
                imus = None

            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(seqs, imu=imus)
                if PREDICT_SIGN_TWO_HEAD:
                    mag, sign_logit = outputs
                    loss, _, _ = criterion(mag, sign_logit, targets)
                elif PREDICT_UNCERTAINTY:
                    mu, log_var = outputs
                    loss = criterion(mu, log_var, targets)
                else:
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            if i % 100 == 0:
                print(f"[{get_time()}] [{model_name}] Ep {epoch+1} | Batch {i}/{total_batches} | Loss: {loss.item():.4f}")

        avg_train = train_loss / total_batches
        history['train'].append(avg_train)

        model.eval()
        val_loss = 0
        val_mae_sum = 0
        pure_l1 = nn.L1Loss()

        with torch.no_grad():
            for batch in val_loader:
                if use_imu:
                    seqs, imus, targets = batch
                    seqs, imus, targets = seqs.to(DEVICE), imus.to(DEVICE), targets.to(DEVICE)
                else:
                    seqs, targets = batch
                    seqs, targets = seqs.to(DEVICE), targets.to(DEVICE)
                    imus = None

                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(seqs, imu=imus)
                    if PREDICT_SIGN_TWO_HEAD:
                        mag, sign_logit = outputs
                        v_loss, _, _ = criterion(mag, sign_logit, targets)
                        # Combined signed prediction so val MAE is comparable
                        # to single-head signed regression baselines.
                        signed_pred = mag * torch.tanh(sign_logit)
                        v_mae = pure_l1(signed_pred, targets)
                    elif PREDICT_UNCERTAINTY:
                        mu, log_var = outputs
                        v_loss = criterion(mu, log_var, targets)
                        v_mae = pure_l1(mu, targets)
                    else:
                        v_loss = criterion(outputs, targets)
                        v_mae = pure_l1(outputs, targets)
                    val_loss += v_loss.item()
                    val_mae_sum += v_mae.item()

        avg_val = val_loss / len(val_loader)
        avg_mae = val_mae_sum / len(val_loader)
        history['val'].append(avg_val)
        scheduler.step(avg_val)

        print(f"[{get_time()}] Epoch {epoch+1} Summary: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f} | (Real MAE: {avg_mae:.4f} mm)")

        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            torch.save(model.state_dict(), best_model_path)
            print(f"[{get_time()}] Best {model_name} saved as {best_model_path} (val MAE {avg_mae:.4f} mm).")

    # Training curve plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss (Weighted)', color='blue', linewidth=2)
    plt.plot(history['val'], label='Validation Loss (Weighted)', color='orange', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Motion Weighted L1 Loss")
    plt.title(f"{model_name} Training Curve (SEQ_LEN = {SEQ_LEN})")
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(run_dir, f"{run_name}_training_curve.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{get_time()}] Loss plot saved to {plot_filename}")

    return history, best_model_path

# =============================================================================
# EVALUATION
# =============================================================================
def calculate_metrics(z_pred, z_real):
    mae = np.mean(np.abs(z_pred - z_real))
    final_drift = abs(z_pred[-1] - z_real[-1])
    total_length = np.sum(np.abs(np.diff(z_real)))
    fdr = (final_drift / total_length) * 100 if total_length > 0 else float('inf')
    return mae, final_drift, total_length, fdr

def predict_z_trajectory(model, frames, tforms, imu_verifier, start_idx=0, end_idx=None):
    if end_idx is None: end_idx = len(frames)
    if end_idx - start_idx <= SEQ_LEN: return None, None

    curr_z_pred = float(tforms[start_idx + SEQ_LEN - 1, 2, 3])
    curr_z_real = float(tforms[start_idx + SEQ_LEN - 1, 2, 3])

    traj_pred = [curr_z_pred]
    traj_real = [curr_z_real]

    use_uncertainty = getattr(model, 'predict_uncertainty', False)
    use_two_head    = getattr(model, 'predict_sign', False)

    def _model_signed_dz(out):
        """Resolve a single signed Δz scalar from any of the supported model
        output shapes: two-head (mag, sign_logit) → mag·tanh(logit); hetero
        (mu, log_var) → mu; plain regression → out."""
        if use_two_head:
            mag_t, sign_logit_t = out
            return float((mag_t * torch.tanh(sign_logit_t)).item())
        if use_uncertainty:
            return float(out[0].item())
        return float(out.item())

    if imu_verifier is not None:
        imu_full = imu_verifier.precompute_imu(tforms)
        init_pos = np.array([0.0, 0.0, curr_z_pred], dtype=np.float64)
        init_rot = tforms[start_idx + SEQ_LEN - 1, :3, :3].astype(np.float64)

        # Bootstrap v_z by averaging the first N model predictions. A single
        # first-frame estimate was noisy enough to point v_z the wrong way on
        # some scans (Par_S_PtD, Par_L_DtP regressed in the bootstrap_v1 run);
        # averaging tames that variance without biasing direction.
        n_init = min(5, max(1, end_idx - start_idx - SEQ_LEN))
        init_dzs = []
        for k in range(n_init):
            seq_imgs = []
            for i in range(SEQ_LEN):
                img = frames[start_idx + k + i]
                img = cv2.resize(img, (256, 256))
                if img.max() > 1.0: img = img / 255.0
                seq_imgs.append(np.expand_dims(img, axis=0))
            inp = torch.tensor(np.stack(seq_imgs), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                init_dzs.append(_model_signed_dz(model(inp)))
        first_dz = float(np.mean(init_dzs))
        init_vel = np.array([0.0, 0.0, first_dz / imu_verifier.dt], dtype=np.float64)
        imu_verifier.reset(init_pos, init_rot, init_velocity=init_vel)

    with torch.no_grad():
        for t in range(start_idx, end_idx - SEQ_LEN):
            seq_imgs = []
            for i in range(SEQ_LEN):
                img = frames[t + i]
                img = cv2.resize(img, (256, 256))
                if img.max() > 1.0: img = img / 255.0
                seq_imgs.append(np.expand_dims(img, axis=0))

            inp = torch.tensor(np.stack(seq_imgs), dtype=torch.float32).unsqueeze(0).to(DEVICE)

            out = model(inp)
            z_visual = _model_signed_dz(out)
            if use_uncertainty:
                # log_var still available for any future σ-aware path; the
                # verifier currently uses fixed σ regardless.
                sigma_visual = float(torch.exp(0.5 * out[1]).item())
            else:
                sigma_visual = None

            real_step = tforms[t + SEQ_LEN, 2, 3] - tforms[t + SEQ_LEN - 1, 2, 3]

            if imu_verifier is not None:
                idx = t + SEQ_LEN - 1
                accel = imu_full[idx, :3]
                gyro  = imu_full[idx, 3:]
                signed_step = imu_verifier.step(accel, gyro, z_visual, sigma_visual)
                curr_z_pred += signed_step
            else:
                curr_z_pred += z_visual

            curr_z_real += real_step

            traj_pred.append(curr_z_pred)
            traj_real.append(curr_z_real)

    return np.array(traj_pred), np.array(traj_real)

def calculate_visual_odometry_2d(frames):
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    trajectory = [[0.0, 0.0]]
    curr_pos = np.array([0.0, 0.0])

    prev_gray = frames[0]
    if prev_gray.max() <= 1.0: prev_gray = (prev_gray * 255).astype(np.uint8)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    for i in range(len(frames) - 1):
        curr_gray = frames[i + 1]
        if curr_gray.max() <= 1.0: curr_gray = (curr_gray * 255).astype(np.uint8)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
        if p1 is not None:
            good_new, good_old = p1[st == 1], p0[st == 1]
            m, _ = cv2.estimateAffinePartial2D(good_old, good_new)
            if m is not None:
                curr_pos[0] -= m[0, 2]
                curr_pos[1] -= m[1, 2]
            trajectory.append(curr_pos.copy())
            if i % 5 == 0 or len(good_new) < 10:
                p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
            else:
                p0 = good_new.reshape(-1, 1, 2)
            prev_gray = curr_gray
        else:
            trajectory.append(curr_pos.copy())
    return np.array(trajectory)

def plot_z_trajectory(z_pred, z_real, title, filename, using_imu):
    plt.figure(figsize=(12, 6))
    z_r_norm = z_real - z_real[0]
    z_p_norm = z_pred - z_pred[0]
    label = 'FiMA + IMU-verifier' if using_imu else 'FiMA (visual only)'

    plt.plot(z_r_norm, 'g-', linewidth=3, alpha=0.5, label='Ground Truth (Sensor)')
    plt.plot(z_p_norm, 'r--', linewidth=2, label=label)
    plt.fill_between(range(len(z_r_norm)), z_r_norm, z_p_norm, color='red', alpha=0.1)

    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel("Relative Z Position (mm)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{get_time()}] Saved Z-plot: {filename}")

def run_evaluation(run_name, run_dir, val_dir, model, imu_verifier, val_pairs):
    print(f"\n{'='*50}")
    print(f"[{get_time()}] EVALUATION PHASE")
    print(f"[{get_time()}] IMU Verifier: {'ENABLED (ESKF)' if imu_verifier else 'DISABLED'}")
    print(f"{'='*50}")

    # ---------- PART 1: Per-case validation analysis ----------
    print(f"\n[{get_time()}] PART 1: Per-case validation ({len(val_pairs)} cases)")
    val_results = []
    for frames_path, tforms_path, case_name in val_pairs:
        try:
            with h5py.File(frames_path, 'r') as f_f, h5py.File(tforms_path, 'r') as f_t:
                v_frames = f_f['frames'][:]
                v_tforms = f_t['tforms'][:]
            z_pred, z_real = predict_z_trajectory(model, v_frames, v_tforms, imu_verifier)
            if z_pred is None:
                continue
            mae, drift, length, fdr = calculate_metrics(z_pred, z_real)
            val_results.append({
                'name': case_name, 'z_pred': z_pred, 'z_real': z_real,
                'mae': mae, 'drift': drift, 'length': length, 'fdr': fdr,
            })
            print(f"[{get_time()}]   {case_name}: MAE={mae:.2f}mm | FDR={fdr:.2f}%")
        except Exception as e:
            print(f"[{get_time()}]   ERROR on {case_name}: {e}")

    # Per-case Z plots → val_images/ subfolder
    for i, r in enumerate(val_results):
        safe = r['name'].replace('/', '_').replace('.h5', '')
        plot_z_trajectory(
            r['z_pred'], r['z_real'],
            f"Validation: {r['name']}\nFDR: {r['fdr']:.2f}% | MAE: {r['mae']:.2f}mm",
            os.path.join(val_dir, f"{run_name}_val_{i:02d}_{safe}.png"),
            using_imu=imu_verifier is not None,
        )

    # Single grid plot showing all validation trajectories at once
    val_avg_mae = float('nan'); val_avg_drift = float('nan'); val_avg_fdr = float('nan')
    if val_results:
        n = len(val_results)
        cols = 2
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 3.0))
        axes = np.atleast_1d(axes).flatten()
        for i, r in enumerate(val_results):
            ax = axes[i]
            zr = r['z_real'] - r['z_real'][0]
            zp = r['z_pred'] - r['z_pred'][0]
            ax.plot(zr, 'g-', linewidth=2, alpha=0.6, label='GT')
            ax.plot(zp, 'r--', linewidth=1.5, label='Pred')
            ax.set_title(f"{r['name']} | MAE: {r['mae']:.2f}mm | FDR: {r['fdr']:.2f}%", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Frame", fontsize=8)
            ax.set_ylabel("Δz (mm)", fontsize=8)
            if i == 0: ax.legend(fontsize=8, loc='best')
        for j in range(n, len(axes)):
            axes[j].axis('off')
        fig.suptitle(f"{run_name} — Validation Z-trajectories ({n} cases)", fontsize=12)
        plt.tight_layout()
        grid_path = os.path.join(run_dir, f"{run_name}_val_grid.png")
        plt.savefig(grid_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[{get_time()}] Saved validation grid: {grid_path}")

        val_avg_mae   = float(np.mean([r['mae']   for r in val_results]))
        val_avg_drift = float(np.mean([r['drift'] for r in val_results]))
        val_avg_fdr   = float(np.mean([r['fdr']   for r in val_results]))
        print(f"[{get_time()}] Validation avg -> MAE: {val_avg_mae:.2f}mm | Drift: {val_avg_drift:.2f}mm | FDR: {val_avg_fdr:.2f}%")

    # ---------- PART 2: Global dataset evaluation ----------
    print(f"\n[{get_time()}] PART 2: Global dataset evaluation")
    all_metrics = {'mae': [], 'drift': [], 'fdr': []}
    best_mae = float('inf'); best_data = None
    worst_mae = -1.0;        worst_data = None

    files_to_check = []
    for i in range(50):
        folder = os.path.join(BASE_DATA_DIR, str(i).zfill(3))
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith('.h5'):
                    files_to_check.append(os.path.join(folder, file))

    print(f"[{get_time()}] Found {len(files_to_check)} scans. Running inference...")
    detailed_metrics = []

    for idx, filepath in enumerate(files_to_check):
        try:
            with h5py.File(filepath, 'r') as f:
                frames = f['frames'][:]
                tforms = f['tforms'][:]

            z_p, z_r = predict_z_trajectory(model, frames, tforms, imu_verifier)
            if z_p is None: continue

            mae, drift, length, fdr = calculate_metrics(z_p, z_r)
            if length < 15.0 or fdr > 100.0: continue

            all_metrics['mae'].append(mae)
            all_metrics['drift'].append(drift)
            all_metrics['fdr'].append(fdr)

            scan_name = f"Subject {filepath.split('/')[-2]} - {os.path.basename(filepath)}"
            detailed_metrics.append([scan_name, mae, drift, fdr])

            if mae < best_mae:
                best_mae = mae
                best_data = (z_p, z_r, scan_name, fdr, mae)
            if mae > worst_mae:
                worst_mae = mae
                worst_data = (z_p, z_r, scan_name, fdr, mae)

            if idx % 10 == 0:
                print(f"[{get_time()}]   Processed {idx}/{len(files_to_check)} scans...")
        except Exception as e:
            pass

    total_scans = len(all_metrics['fdr'])
    avg_mae = np.mean(all_metrics['mae']) if total_scans > 0 else float('nan')
    avg_drift = np.mean(all_metrics['drift']) if total_scans > 0 else float('nan')
    avg_fdr = np.mean(all_metrics['fdr']) if total_scans > 0 else float('nan')

    summary_text = (
        f"=== EVALUATION METRICS (FiMA-Net) ===\n"
        f"Run tag:             {run_name}\n"
        f"IMU Verifier:        {'ENABLED (ESKF)' if imu_verifier else 'DISABLED'}\n"
        f"\n"
        f"--- Validation set ({len(val_results)} cases) ---\n"
        f"Validation MAE:      {val_avg_mae:.3f} mm (avg)\n"
        f"Validation Drift:    {val_avg_drift:.3f} mm (avg)\n"
        f"Validation FDR:      {val_avg_fdr:.2f} % (avg)\n"
        f"\n"
        f"--- Global dataset (all training scans re-eval) ---\n"
        f"Total valid scans:   {total_scans}\n"
        f"Average MAE:         {avg_mae:.3f} mm\n"
        f"Average Final Drift: {avg_drift:.3f} mm\n"
        f"Average Drift Rate:  {avg_fdr:.2f} %\n"
        f"===========================================\n"
    )
    print(summary_text)

    summary_path = os.path.join(run_dir, f"{run_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"[{get_time()}] Saved summary: {summary_path}")

    csv_path = os.path.join(run_dir, f"{run_name}_per_scan.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scan_Name", "MAE_mm", "Final_Drift_mm", "FDR_percent"])
        writer.writerows(detailed_metrics)
    print(f"[{get_time()}] Saved per-scan metrics: {csv_path}")

    if best_data:
        plot_z_trajectory(
            best_data[0], best_data[1],
            f"BEST CASE: {best_data[2]}\nFDR: {best_data[3]:.2f}% | MAE: {best_data[4]:.2f} mm",
            os.path.join(run_dir, f"{run_name}_bestscan.png"),
            using_imu=imu_verifier is not None,
        )
    if worst_data:
        plot_z_trajectory(
            worst_data[0], worst_data[1],
            f"WORST CASE: {worst_data[2]}\nFDR: {worst_data[3]:.2f}% | MAE: {worst_data[4]:.2f} mm",
            os.path.join(run_dir, f"{run_name}_worstscan.png"),
            using_imu=imu_verifier is not None,
        )

    print(f"\n[{get_time()}] Evaluation complete.")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FiMA-Net training + IMU-verified evaluation')
    parser.add_argument('--name', '-n', default=None,
                        help='Run name base tag (auto-suffixed with _v{N}). Defaults to a config-derived tag.')
    parser.add_argument('--eval-only', '-e', default=None, metavar='CKPT',
                        help='Skip training; load this .pth checkpoint and run evaluation only.')
    args = parser.parse_args()

    print(f"[{get_time()}] --- STEP 1: PREPARING DATASETS ---")

    val_pairs = collect_val_files_stratified(VAL_PER_SUBJECT)
    n_subjects = len({c.split('/')[0] for _, _, c in val_pairs}) if val_pairs else 0
    print(f"[{get_time()}] Validation set: {len(val_pairs)} cases across {n_subjects} subject(s)")

    if args.name:
        base_tag = args.name
    elif args.eval_only:
        ckpt_base = os.path.basename(args.eval_only)
        for suf in ('_best.pth', '.pth'):
            if ckpt_base.endswith(suf):
                ckpt_base = ckpt_base[:-len(suf)]
                break
        base_tag = ckpt_base + '_eval'
    else:
        base_tag = (
            f"fima_seq{SEQ_LEN}_e{EPOCHS}"
            f"{'_imufused' if USE_IMU else ''}"
            f"{'_eskf' if USE_IMU_VERIFIER else ''}"
            f"{'_unc' if PREDICT_UNCERTAINTY else ''}"
        )
    run_name = next_run_name(base_tag)
    run_dir, val_dir = make_run_dirs(run_name)
    print(f"[{get_time()}] Run tag: {run_name}")
    print(f"[{get_time()}] Run dir: {run_dir}")

    fima_model = None
    if args.eval_only:
        best_model_path = args.eval_only
        print(f"[{get_time()}] --- EVAL-ONLY: skipping training, loading {best_model_path} ---")
    else:
        train_ds = LargeUSDataset(TRAIN_FOLDERS, seq_len=SEQ_LEN, use_imu=USE_IMU)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_datasets = [
            MemoryUSDataset(fp, tp, seq_len=SEQ_LEN, use_imu=USE_IMU)
            for fp, tp, _ in val_pairs
        ]
        val_ds = ConcatDataset(val_datasets)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"[{get_time()}] --- STEP 2: TRAIN FIMA-NET ---")
        fima_model = FiMANet(seq_len=SEQ_LEN, use_imu=USE_IMU,
                             predict_uncertainty=PREDICT_UNCERTAINTY,
                             predict_sign=PREDICT_SIGN_TWO_HEAD)
        fima_history, best_model_path = train_model(
            run_name, run_dir, "FiMA_UnfrozenBackbone", fima_model, train_loader, val_loader, EPOCHS, use_imu=USE_IMU
        )

        del train_loader, train_ds, val_loader, val_ds, val_datasets
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[{get_time()}] --- STEP 3: EVALUATE WITH IMU VERIFIER ---")
    eval_model = FiMANet(seq_len=SEQ_LEN, use_imu=USE_IMU,
                         predict_uncertainty=PREDICT_UNCERTAINTY,
                         predict_sign=PREDICT_SIGN_TWO_HEAD).to(DEVICE)
    eval_model = load_weights_safe(eval_model, best_model_path)

    imu_verifier = IMUVerifier(IMUSimulator()) if USE_IMU_VERIFIER else None
    run_evaluation(run_name, run_dir, val_dir, eval_model, imu_verifier, val_pairs)

    del eval_model
    if fima_model is not None:
        del fima_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[{get_time()}] ALL TRAININGS + EVALUATIONS COMPLETED SUCCESSFULLY")
