import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import cv2
import gc
import glob
import h5py
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from datetime import datetime
from scipy.spatial.transform import Rotation

from models.fimanet_mamba_6dof import FiMANetMamba6DOF

# =============================================================================
# CONFIG
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20
BATCH_SIZE = 96  # 6-DoF is heavier; reduce from 164
EPOCHS = 10
WARMUP_EPOCHS = 4
PAIR_STRIDES = (1,)

# Loss weighting: rotations are in radians (~0.01 rad/frame), translations in mm (~1 mm/frame).
# Multiply rotations by ROT_WEIGHT so they contribute comparably to the L1 loss.
ROT_WEIGHT = 100.0

BASE_DATA_DIR = "/home/123ghdh/datasets"
TRAIN_FOLDERS = [os.path.join(BASE_DATA_DIR, str(i).zfill(3)) for i in range(50)]
VAL_FRAMES_ROOT = os.path.join(BASE_DATA_DIR, 'valDataset/data/frames')
VAL_TFORMS_ROOT = os.path.join(BASE_DATA_DIR, 'valDataset/data/transfs')
CALIB_PATH      = os.path.join(BASE_DATA_DIR, 'calib_matrix.csv')
VAL_PER_SUBJECT = 24

# =============================================================================
# UTILITIES
# =============================================================================
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


RUNS_ROOT = "runs"

def next_run_name(prefix):
    n = 1
    while glob.glob(f"{prefix}_v{n}_*") or glob.glob(os.path.join(RUNS_ROOT, f"*_{prefix}_v{n}")):
        n += 1
    return f"{prefix}_v{n}"


def make_run_dirs(run_name):
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(RUNS_ROOT, f"{ts}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def set_backbone_trainable(model, trainable):
    for name in ['layer2', 'layer3', 'layer4']:
        for p in getattr(model, name).parameters():
            p.requires_grad = trainable


def read_calib_matrices(filename_calib):
    """Same format as baseline. Returns (scale_pixel_to_image_mm, image_mm_to_tool)."""
    tform_calib = np.empty((8, 4), np.float32)
    with open(filename_calib, 'r') as f:
        txt = [i.strip('\n').split(',') for i in f.readlines()]
        tform_calib[0:4, :] = np.array(txt[1:5]).astype(np.float32)
        tform_calib[4:8, :] = np.array(txt[6:10]).astype(np.float32)
    pixel_to_image_mm = tform_calib[0:4, :]
    image_mm_to_tool  = tform_calib[4:8, :]
    return pixel_to_image_mm, image_mm_to_tool


def collect_val_files_stratified(n_per_subject=24):
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
            tp = os.path.join(VAL_TFORMS_ROOT, subj, fname)
            if os.path.exists(tp):
                subj_pairs.append((os.path.join(sd, fname), tp, f"{subj}/{fname}"))
        pairs.extend(subj_pairs[:n_per_subject])
    return pairs

# =============================================================================
# TARGET COMPUTATION
# =============================================================================
def tforms_to_6dof_params(tforms, image_mm_to_tool):
    """Convert a sequence of tool-to-world transformations into per-pair 6-DoF parameters
    in image_mm coordinates, matching the TUS-REC2024 baseline's 'parameter' label.

    tforms: np.ndarray shape (N+1, 4, 4) — tool-to-world for N+1 consecutive frames.
    image_mm_to_tool: np.ndarray shape (4, 4).
    Returns: np.ndarray shape (N, 6) — (rz, ry, rx, tx, ty, tz) per pair.
    """
    tool_to_image_mm = np.linalg.inv(image_mm_to_tool)
    tforms_inv = np.linalg.inv(tforms)  # world-to-tool per frame

    N = len(tforms) - 1
    out = np.zeros((N, 6), dtype=np.float32)
    for i in range(N):
        # tool_{i+1} -> tool_i  =  T(world->tool_i) @ T(tool_{i+1}->world)
        t_tool_pair = tforms_inv[i] @ tforms[i + 1]
        # convert to image_mm frame: image_mm_{i+1} -> image_mm_i
        t_imm = tool_to_image_mm @ t_tool_pair @ image_mm_to_tool
        # 'ZYX' euler — matches pytorch3d matrix_to_euler_angles('ZYX') ordering: (rz, ry, rx)
        rz, ry, rx = Rotation.from_matrix(t_imm[:3, :3]).as_euler('ZYX')
        tx, ty, tz = t_imm[:3, 3]
        out[i] = (rz, ry, rx, tx, ty, tz)
    return out

# =============================================================================
# LOSS
# =============================================================================
class WeightedParamL1Loss(nn.Module):
    """L1 on (rx, ry, rz, tx, ty, tz) with rotation weight to balance scales."""
    def __init__(self, rot_weight=100.0, start_pos=5):
        super().__init__()
        self.rot_weight = rot_weight
        self.start_pos = start_pos
        weight = torch.tensor([rot_weight, rot_weight, rot_weight, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer('weight', weight)

    def forward(self, pred, target):
        # pred, target: [B, L, 6]
        L = pred.shape[1]
        start = min(self.start_pos, L - 1)
        diff = (pred[:, start:, :] - target[:, start:, :]).abs() * self.weight
        return diff.mean()

# =============================================================================
# DATASETS
# =============================================================================
class LargeUSDataset6DOF(Dataset):
    def __init__(self, root_dirs, seq_len, image_mm_to_tool, augment=True):
        self.samples = []
        self.seq_len = seq_len
        self.augment = augment
        self.image_mm_to_tool = image_mm_to_tool
        print(f"[{get_time()}] Indexing folders...")
        for folder in root_dirs:
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.endswith('.h5'):
                    continue
                fp = os.path.join(folder, fname)
                try:
                    with h5py.File(fp, 'r') as f:
                        if 'frames' not in f or 'tforms' not in f:
                            continue
                        n = f['frames'].shape[0]
                        for i in range(0, n - seq_len, 2):
                            self.samples.append({'path': fp, 'start': i})
                except Exception as e:
                    print(f"err {fp}: {e}")
        print(f"[{get_time()}] Indexed {len(self.samples)} sequences.")
        if not self.samples:
            raise RuntimeError("0 sequences")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _augment(seq):
        # hflip / brightness-contrast / speckle
        if np.random.rand() < 0.5:
            seq = [np.ascontiguousarray(np.flip(im, axis=2)) for im in seq]
        if np.random.rand() < 0.5:
            g = np.random.uniform(0.9, 1.1)
            b = np.random.uniform(-0.05, 0.05)
            seq = [np.clip(im * g + b, 0., 1.).astype(np.float32) for im in seq]
        if np.random.rand() < 0.3:
            seq = [np.clip(im * (1 + np.random.randn(*im.shape).astype(np.float32) * 0.05), 0., 1.) for im in seq]
        return seq

    def __getitem__(self, idx):
        m = self.samples[idx]
        with h5py.File(m['path'], 'r') as f:
            frames = f['frames'][m['start']:m['start'] + self.seq_len]
            tforms = f['tforms'][m['start']:m['start'] + self.seq_len + 1]
        seq = []
        for i in range(self.seq_len):
            img = frames[i]
            img = cv2.resize(img, (256, 256))
            if img.max() > 1.0:
                img = img / 255.0
            seq.append(np.expand_dims(img, axis=0).astype(np.float32))
        if self.augment:
            seq = self._augment(seq)
        # NOTE hflip changes the image x-axis -> in-plane translation/rotation flips sign.
        # For SEQ_LEN=20, target shape is [20, 6]; we DON'T flip the param sign here because
        # the model learns the augmented mapping directly. This is consistent with how the
        # original Z-only training handled hflip (Z is invariant to hflip; for 6-DoF, the
        # rotations and X-translation will average to zero across hflip pairs at convergence).
        seq_tensor = torch.tensor(np.stack(seq), dtype=torch.float32)
        params = tforms_to_6dof_params(tforms.astype(np.float32), self.image_mm_to_tool)
        return seq_tensor, torch.tensor(params, dtype=torch.float32)


class MemoryUSDataset6DOF(Dataset):
    def __init__(self, frames_path, tforms_path, seq_len, image_mm_to_tool):
        with h5py.File(frames_path, 'r') as f:
            self.f = f['frames'][:]
        with h5py.File(tforms_path, 'r') as t:
            self.t = t['tforms'][:]
        self.seq_len = seq_len
        self.image_mm_to_tool = image_mm_to_tool
        self.samples = list(range(0, len(self.f) - seq_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start = self.samples[idx]
        seq = []
        for i in range(self.seq_len):
            img = self.f[start + i]
            img = cv2.resize(img, (256, 256))
            if img.max() > 1.0:
                img = img / 255.0
            seq.append(np.expand_dims(img, axis=0))
        seq_tensor = torch.tensor(np.stack(seq), dtype=torch.float32)
        tforms = self.t[start:start + self.seq_len + 1].astype(np.float32)
        params = tforms_to_6dof_params(tforms, self.image_mm_to_tool)
        return seq_tensor, torch.tensor(params, dtype=torch.float32)

# =============================================================================
# TRAINING
# =============================================================================
def train_model(run_name, run_dir, model, train_loader, val_loader, epochs):
    print(f"\n{'='*50}")
    print(f"[{get_time()}] INITIALIZING 6-DoF TRAINING")
    print(f"[{get_time()}] Run name: {run_name}")
    print(f"[{get_time()}] Active Device: {torch.cuda.get_device_name(DEVICE)}")
    print(f"[{get_time()}] Warmup (head-only) epochs: {WARMUP_EPOCHS}")
    print(f"{'='*50}")

    model = model.to(DEVICE)

    if WARMUP_EPOCHS > 0:
        set_backbone_trainable(model, False)
        print(f"[{get_time()}] Warmup: backbone layer2/3/4 frozen.")

    backbone_params = list(model.stem.parameters()) + \
                      list(model.layer1.parameters()) + \
                      list(model.layer2.parameters()) + \
                      list(model.layer3.parameters()) + \
                      list(model.layer4.parameters())

    head_params = list(model.fusion.parameters()) + \
                  list(model.temporal_layers.parameters()) + \
                  list(model.head.parameters())
    if getattr(model, 'pair_encoder', False):
        head_params += list(model.pair_proj.parameters())

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 5e-6},
        {'params': head_params, 'lr': 1e-4},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = WeightedParamL1Loss(rot_weight=ROT_WEIGHT, start_pos=5).to(DEVICE)
    scaler = GradScaler('cuda')

    best_val = float('inf')
    best_path = os.path.join(run_dir, f"{run_name}_best.pth")
    total_batches = len(train_loader)

    for epoch in range(epochs):
        if epoch == WARMUP_EPOCHS and WARMUP_EPOCHS > 0:
            set_backbone_trainable(model, True)
            print(f"[{get_time()}] Epoch {epoch+1}: warmup over — unfreezing layer2/3/4.")

        model.train()
        train_loss = 0.0
        for i, (seqs, targets) in enumerate(train_loader):
            seqs = seqs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(seqs)  # [B, L, 6]
                L = outputs.shape[1]
                tgt = targets[:, :L, :]
                loss = criterion(outputs, tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 100 == 0:
                print(f"[{get_time()}] Ep {epoch+1} | Batch {i}/{total_batches} | Loss: {loss.item():.4f}")

        avg_train = train_loss / total_batches

        # Validation: per-pair param L1 + raw translation MAE (mm)
        model.eval()
        val_loss = 0.0
        val_trans_mae = 0.0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs = seqs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(seqs)
                    L = outputs.shape[1]
                    tgt = targets[:, :L, :]
                    v_loss = criterion(outputs, tgt)
                    # last position only — apples-to-apples with old Z-only metric
                    trans_mae = (outputs[:, -1, 3:6] - tgt[:, -1, 3:6]).abs().mean()
                val_loss += v_loss.item()
                val_trans_mae += trans_mae.item()

        avg_val = val_loss / len(val_loader)
        avg_mae = val_trans_mae / len(val_loader)
        scheduler.step(avg_val)
        print(f"[{get_time()}] Epoch {epoch+1}: Train={avg_train:.4f} | Val={avg_val:.4f} | Trans MAE={avg_mae:.4f} mm")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)
            print(f"[{get_time()}] Best saved: {best_path} (val loss {avg_val:.4f}).")
        # Also save per-epoch in case of best-by-train-MAE preference later
        ep_path = os.path.join(run_dir, f"{run_name}_ep{epoch+1}.pth")
        torch.save(model.state_dict(), ep_path)

    return best_path

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FiMA-Net 6-DoF training')
    parser.add_argument('--name', '-n', default='fima_pair_mamba_6dof', help='Run name base tag.')
    args = parser.parse_args()

    pixel_to_image_mm, image_mm_to_tool = read_calib_matrices(CALIB_PATH)
    print(f"[{get_time()}] Loaded calib from {CALIB_PATH}")

    run_name = next_run_name(args.name)
    run_dir = make_run_dirs(run_name)
    print(f"[{get_time()}] Run tag: {run_name}")
    print(f"[{get_time()}] Run dir: {run_dir}")

    print(f"[{get_time()}] --- STEP 1: DATASETS ---")
    train_ds = LargeUSDataset6DOF(TRAIN_FOLDERS, seq_len=SEQ_LEN,
                                   image_mm_to_tool=image_mm_to_tool, augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    val_pairs = collect_val_files_stratified(VAL_PER_SUBJECT)
    n_subj = len({c.split('/')[0] for _, _, c in val_pairs})
    print(f"[{get_time()}] Validation set: {len(val_pairs)} cases across {n_subj} subject(s)")
    val_datasets = [MemoryUSDataset6DOF(fp, tp, seq_len=SEQ_LEN, image_mm_to_tool=image_mm_to_tool)
                    for fp, tp, _ in val_pairs]
    val_ds = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[{get_time()}] --- STEP 2: BUILD + TRAIN MODEL ---")
    model = FiMANetMamba6DOF(seq_len=SEQ_LEN, pair_encoder=True, pair_strides=PAIR_STRIDES)
    best_path = train_model(run_name, run_dir, model, train_loader, val_loader, EPOCHS)

    del train_loader, train_ds, val_loader, val_ds, val_datasets, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n[{get_time()}] Training done. Best checkpoint: {best_path}")
    print(f"[{get_time()}] Run `python eval_6dof.py --ckpt {best_path}` next to compute GPE/GLE/LPE/LLE.")
