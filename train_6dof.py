import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import cv2
import gc
import glob
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from datetime import datetime
from scipy.spatial.transform import Rotation

from models.fimanet_mamba_6dof import FiMANetMamba6DOF

# =============================================================================
# CONFIG
# =============================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20
BATCH_SIZE = 148
EPOCHS = 12
WARMUP_EPOCHS = 4
PAIR_STRIDES = (1,)
BACKBONE = os.environ.get("BACKBONE", "resnet18")  # 'resnet18' | 'resnet34' | 'resnet50' (R34 attempt regressed)

# Loss is point-based (corner-displacement L1 in mm) — no rotation weight needed because
# the projection automatically balances rotation and translation by their effect on pixels.
IMG_H, IMG_W = 480, 640
CORNER_DENSITY = 4  # 4x4 = 16 points (matches TUS-REC baseline)

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
# DIFFERENTIABLE GEOMETRY (torch — used in the loss)
# =============================================================================
def euler_zyx_to_matrix(euler):
    """euler: (..., 3) — (rz, ry, rx). Returns (..., 3, 3) rotation matrix.
    Matches pytorch3d.transforms.euler_angles_to_matrix(euler, 'ZYX'): R = Rz @ Ry @ Rx.
    Differentiable; runs on whichever device the input tensor lives on."""
    rz, ry, rx = euler.unbind(-1)
    cz, sz = torch.cos(rz), torch.sin(rz)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cx, sx = torch.cos(rx), torch.sin(rx)
    zero = torch.zeros_like(rz)
    one  = torch.ones_like(rz)

    Rz = torch.stack([
        torch.stack([  cz, -sz, zero], dim=-1),
        torch.stack([  sz,  cz, zero], dim=-1),
        torch.stack([zero, zero,  one], dim=-1),
    ], dim=-2)
    Ry = torch.stack([
        torch.stack([  cy, zero,   sy], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([ -sy, zero,   cy], dim=-1),
    ], dim=-2)
    Rx = torch.stack([
        torch.stack([ one, zero, zero], dim=-1),
        torch.stack([zero,   cx,  -sx], dim=-1),
        torch.stack([zero,   sx,   cx], dim=-1),
    ], dim=-2)
    return torch.matmul(torch.matmul(Rz, Ry), Rx)


def params_to_corner_points(params, image_points_mm):
    """params: (..., 6) — (rz, ry, rx, tx, ty, tz) in image_mm coords.
    image_points_mm: (4, P) homogeneous corner points in image_mm.
    Returns (..., 3, P) transformed corner points (xyz only)."""
    R = euler_zyx_to_matrix(params[..., :3])    # (..., 3, 3)
    t = params[..., 3:]                          # (..., 3)
    batch_shape = t.shape[:-1]
    T = torch.zeros(*batch_shape, 4, 4, device=params.device, dtype=params.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3]  = t
    T[..., 3, 3]   = 1.0
    return torch.matmul(T, image_points_mm)[..., :3, :]   # (..., 3, P)


def reference_image_points(image_size=(IMG_H, IMG_W), density=CORNER_DENSITY):
    """Returns (4, density*density) corner points in pixel coords, homogeneous.
    Matches the TUS-REC baseline's reference_image_points exactly."""
    pts = torch.flip(torch.cartesian_prod(
        torch.linspace(1, image_size[0], density),
        torch.linspace(1, image_size[1], density),
    ).t(), [0])
    pts = torch.cat([
        pts,
        torch.zeros(1, pts.shape[1]),
        torch.ones(1, pts.shape[1]),
    ], dim=0)
    return pts  # (4, density*density)


# =============================================================================
# TARGET COMPUTATION
# =============================================================================
def tforms_to_target_points(tforms, image_mm_to_tool, image_points_mm):
    """tforms: (N+1, 4, 4) tool-to-world.
    image_points_mm: (4, P) homogeneous corner points in image_mm (numpy).
    Returns: (N, 3, P) GT corner points after applying the local image_mm transform
    (frame_{i+1} -> frame_i) for each consecutive frame pair."""
    tool_to_image_mm = np.linalg.inv(image_mm_to_tool)
    tforms_inv = np.linalg.inv(tforms)
    N = len(tforms) - 1
    P = image_points_mm.shape[1]
    target = np.zeros((N, 3, P), dtype=np.float32)
    for i in range(N):
        t_tool = tforms_inv[i] @ tforms[i + 1]
        t_imm  = tool_to_image_mm @ t_tool @ image_mm_to_tool
        target[i] = (t_imm @ image_points_mm)[:3, :]
    return target


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
class PointBasedLoss(nn.Module):
    """L1 between predicted and GT corner-point positions in mm, after applying the
    predicted vs GT image_mm transform. Auto-balances rotation and translation by
    their effect on pixel coordinates — same idea as the TUS-REC baseline's
    LABEL_TYPE='point' (see baseline/utils/transform.py)."""
    def __init__(self, image_points_mm, start_pos=5):
        super().__init__()
        # image_points_mm: (4, P) homogeneous; pre-compute once.
        self.register_buffer('image_points_mm', image_points_mm)
        self.start_pos = start_pos

    def forward(self, pred_params, target_points):
        # pred_params:   [B, L, 6]
        # target_points: [B, L+, 3, P]  (sliced to L below)
        L = pred_params.shape[1]
        target = target_points[:, :L, :, :]
        pred   = params_to_corner_points(pred_params, self.image_points_mm)  # [B, L, 3, P]
        start  = min(self.start_pos, L - 1)
        diff   = (pred[:, start:, :, :] - target[:, start:, :, :]).abs()
        return diff.mean()

# =============================================================================
# DATASETS
# =============================================================================
class LargeUSDataset6DOF(Dataset):
    def __init__(self, root_dirs, seq_len, image_mm_to_tool, image_points_mm, augment=True):
        self.samples = []
        self.seq_len = seq_len
        self.augment = augment
        self.image_mm_to_tool = image_mm_to_tool
        self.image_points_mm = image_points_mm
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
                        for i in range(0, n - seq_len):
                            self.samples.append({'path': fp, 'start': i})
                except Exception as e:
                    print(f"err {fp}: {e}")
        print(f"[{get_time()}] Indexed {len(self.samples)} sequences.")
        if not self.samples:
            raise RuntimeError("0 sequences")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _augment_intensity(seq):
        """Intensity-only augmentation. NO hflip (it would invalidate 6-DoF targets).
        Conservative — gamma + stronger speckle were dropped after the R34 attempt
        plateaued (see thesis Section X: failed-experiment write-up)."""
        if np.random.rand() < 0.5:
            g = np.random.uniform(0.9, 1.1)
            b = np.random.uniform(-0.05, 0.05)
            seq = [np.clip(im * g + b, 0., 1.).astype(np.float32) for im in seq]
        if np.random.rand() < 0.3:
            seq = [np.clip(im * (1 + np.random.randn(*im.shape).astype(np.float32) * 0.05), 0., 1.)
                   for im in seq]
        return seq

    def __getitem__(self, idx):
        m = self.samples[idx]
        with h5py.File(m['path'], 'r') as f:
            frames = f['frames'][m['start']:m['start'] + self.seq_len]
            tforms = f['tforms'][m['start']:m['start'] + self.seq_len + 1]

        # Z-REVERSE augmentation removed — at p=0.5 it drove the model into a
        # predict-near-zero attractor (val loss plateaued, GPE regressed from
        # 44 to 82 mm in the R34 retrain). Documented as a negative result;
        # see thesis Section X for the write-up. A smaller probability (e.g.
        # p=0.1) might be safe but is future work.

        seq = []
        for i in range(self.seq_len):
            img = frames[i]
            img = cv2.resize(img, (256, 256))
            if img.max() > 1.0:
                img = img / 255.0
            seq.append(np.expand_dims(img, axis=0).astype(np.float32))
        if self.augment:
            seq = self._augment_intensity(seq)

        seq_tensor = torch.tensor(np.stack(seq), dtype=torch.float32)
        target = tforms_to_target_points(tforms.astype(np.float32),
                                          self.image_mm_to_tool, self.image_points_mm)
        return seq_tensor, torch.tensor(target, dtype=torch.float32)


class MemoryUSDataset6DOF(Dataset):
    def __init__(self, frames_path, tforms_path, seq_len, image_mm_to_tool, image_points_mm):
        with h5py.File(frames_path, 'r') as f:
            self.f = f['frames'][:]
        with h5py.File(tforms_path, 'r') as t:
            self.t = t['tforms'][:]
        self.seq_len = seq_len
        self.image_mm_to_tool = image_mm_to_tool
        self.image_points_mm = image_points_mm
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
        target = tforms_to_target_points(tforms, self.image_mm_to_tool, self.image_points_mm)
        return seq_tensor, torch.tensor(target, dtype=torch.float32)

# =============================================================================
# TRAINING
# =============================================================================
def train_model(run_name, run_dir, model, train_loader, val_loader, epochs, image_points_mm_torch):
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
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = PointBasedLoss(image_points_mm_torch, start_pos=5).to(DEVICE)
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
        for i, (seqs, target_pts) in enumerate(train_loader):
            seqs       = seqs.to(DEVICE, non_blocking=True)
            target_pts = target_pts.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(seqs)             # [B, L, 6]
                loss    = criterion(outputs, target_pts)  # mean corner-displacement |·| in mm
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 100 == 0:
                print(f"[{get_time()}] Ep {epoch+1} | Batch {i}/{total_batches} | Loss: {loss.item():.4f} mm")

        avg_train = train_loss / total_batches

        # Validation: same point-based loss + max-corner-error sanity metric
        model.eval()
        val_loss = 0.0
        val_max_err = 0.0
        with torch.no_grad():
            for seqs, target_pts in val_loader:
                seqs       = seqs.to(DEVICE, non_blocking=True)
                target_pts = target_pts.to(DEVICE, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(seqs)
                    v_loss  = criterion(outputs, target_pts)
                    # Sanity metric: last-position max corner displacement error in mm
                    pred_pts = params_to_corner_points(outputs[:, -1, :], criterion.image_points_mm)  # [B, 3, P]
                    last_pos = outputs.shape[1] - 1
                    gt_pts = target_pts[:, last_pos, :, :]
                    per_corner = torch.linalg.norm(pred_pts - gt_pts, dim=1)  # [B, P]
                    max_err = per_corner.max(dim=1).values.mean()             # avg across batch of per-scan max
                val_loss   += v_loss.item()
                val_max_err += max_err.item()

        avg_val     = val_loss / len(val_loader)
        avg_max_err = val_max_err / len(val_loader)
        scheduler.step(avg_val)
        print(f"[{get_time()}] Epoch {epoch+1}: Train={avg_train:.4f} mm | Val={avg_val:.4f} mm | Max corner err={avg_max_err:.4f} mm")

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

    # Reference corner points (4 corners of 480x640 image) in image_mm.
    image_points_pixel = reference_image_points((IMG_H, IMG_W), CORNER_DENSITY).numpy()  # (4, P) numpy
    image_points_mm    = pixel_to_image_mm @ image_points_pixel                          # (4, P) numpy
    image_points_mm_torch = torch.tensor(image_points_mm, dtype=torch.float32)
    print(f"[{get_time()}] Corner reference: {image_points_mm.shape[1]} points in image_mm")

    run_name = next_run_name(args.name)
    run_dir = make_run_dirs(run_name)
    print(f"[{get_time()}] Run tag: {run_name}")
    print(f"[{get_time()}] Run dir: {run_dir}")

    print(f"[{get_time()}] --- STEP 1: DATASETS ---")
    train_ds = LargeUSDataset6DOF(TRAIN_FOLDERS, seq_len=SEQ_LEN,
                                   image_mm_to_tool=image_mm_to_tool,
                                   image_points_mm=image_points_mm, augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    val_pairs = collect_val_files_stratified(VAL_PER_SUBJECT)
    n_subj = len({c.split('/')[0] for _, _, c in val_pairs})
    print(f"[{get_time()}] Validation set: {len(val_pairs)} cases across {n_subj} subject(s)")
    val_datasets = [MemoryUSDataset6DOF(fp, tp, seq_len=SEQ_LEN,
                                         image_mm_to_tool=image_mm_to_tool,
                                         image_points_mm=image_points_mm)
                    for fp, tp, _ in val_pairs]
    val_ds = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[{get_time()}] --- STEP 2: BUILD + TRAIN MODEL ---")
    print(f"[{get_time()}] Backbone: {BACKBONE}")
    model = FiMANetMamba6DOF(seq_len=SEQ_LEN, pair_encoder=True, pair_strides=PAIR_STRIDES,
                              backbone=BACKBONE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{get_time()}] Total params: {n_params/1e6:.1f}M")
    best_path = train_model(run_name, run_dir, model, train_loader, val_loader, EPOCHS,
                            image_points_mm_torch)

    del train_loader, train_ds, val_loader, val_ds, val_datasets, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n[{get_time()}] Training done. Best checkpoint: {best_path}")
    print(f"[{get_time()}] Run `python eval_6dof.py --ckpt {best_path}` next to compute GPE/GLE/LPE/LLE.")
