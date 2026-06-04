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
BATCH_SIZE = 48        # half-res frees memory; drop to 32/24 if OOM
EPOCHS = 15            # best ckpt saved each epoch — stop early once GPEproxy flattens
WARMUP_EPOCHS = 4
PAIR_STRIDES = (1,)
BACKBONE = os.environ.get("BACKBONE", "resnet18")  # resnet18 | resnet34 | resnet50

# Loss terms (weight 0 to ablate). point-L1 alone never penalized the coherent tz drift.
LOSS_START_POS = 5
W_GLOBAL  = 0.2   # global-accumulation corner consistency
W_PEARSON = 0.5   # case-wise Pearson on 6-DoF trajectory (FiMoNet)
# Architecture: restore the speckle/tz cue (was 2x2 pool + frozen stem/layer1).
POOL_SIZE    = 7
FREEZE_EARLY = False
ZREVERSE_P = 0.12     # frame-reversal aug; targets recomputed from reversed poses
GPE_PROXY_SCANS = 6   # full val scans for the per-epoch GPE proxy

# Loss is point-based (corner-displacement L1 in mm) — no rotation weight needed because
# the projection automatically balances rotation and translation by their effect on pixels.
IMG_H, IMG_W = 480, 640
CORNER_DENSITY = 4  # 4x4 = 16 points (matches TUS-REC baseline)

# Network input: half-res 240x320 (matches FiMoNet's 50% resize). GT corners/calibration
# stay in 480x640; only the input is resized. eval_6dof INFER_H/INFER_W must match.
NET_H, NET_W = 240, 320

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


def params_to_matrix_torch(params):
    """(..., 6) (rz,ry,rx,tx,ty,tz) -> (..., 4, 4); float32 for the accumulation chain."""
    params = params.float()
    R = euler_zyx_to_matrix(params[..., :3])
    t = params[..., 3:]
    batch_shape = t.shape[:-1]
    T = torch.zeros(*batch_shape, 4, 4, device=params.device, dtype=torch.float32)
    T[..., :3, :3] = R
    T[..., :3, 3]  = t
    T[..., 3, 3]   = 1.0
    return T


def accumulate_global_torch(local_T):
    """(B,L,4,4) locals -> globals via prev = prev @ local[i] (must match eval_6dof)."""
    B, L = local_T.shape[0], local_T.shape[1]
    prev = torch.eye(4, device=local_T.device, dtype=local_T.dtype).unsqueeze(0).expand(B, 4, 4).contiguous()
    outs = []
    for i in range(L):
        prev = prev @ local_T[:, i]
        outs.append(prev)
    return torch.stack(outs, dim=1)


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
# GPE PROXY (numpy — mirrors eval_6dof so checkpoint selection tracks real GPE)
# =============================================================================
def _np_params_to_matrix(p):
    """(6,) -> (4,4). Mirrors eval_6dof.params_to_matrix."""
    rz, ry, rx, tx, ty, tz = p
    R = Rotation.from_euler('ZYX', [rz, ry, rx]).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = (tx, ty, tz)
    return T


def _np_accumulate_global(locals_4x4):
    """Mirrors eval_6dof.accumulate_global: prev = prev @ local[i]."""
    out = np.zeros_like(locals_4x4)
    prev = np.eye(4, dtype=np.float32)
    for i in range(len(locals_4x4)):
        prev = prev @ locals_4x4[i]
        out[i] = prev
    return out


def _np_tforms_to_global(tforms, image_mm_to_tool):
    """GT global (frame_i -> frame_0) in image_mm. Mirrors eval_6dof.tforms_to_global_image_mm."""
    tool_to_image_mm = np.linalg.inv(image_mm_to_tool)
    tinv = np.linalg.inv(tforms)
    out = np.zeros((len(tforms) - 1, 4, 4), dtype=np.float32)
    for i in range(1, len(tforms)):
        out[i - 1] = tool_to_image_mm @ (tinv[0] @ tforms[i]) @ image_mm_to_tool
    return out


def _predict_local_params_np(model, frames):
    """Sliding-window inference over a full scan (mirrors eval_6dof.predict_local_params)."""
    N = len(frames)
    params = np.zeros((N - 1, 6), dtype=np.float32)
    last_pos_offset = SEQ_LEN - 2
    resized = np.zeros((N, 1, NET_H, NET_W), dtype=np.float32)
    for i in range(N):
        img = cv2.resize(frames[i], (NET_W, NET_H))
        if img.max() > 1.0:
            img = img / 255.0
        resized[i, 0] = img.astype(np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, N - SEQ_LEN + 1):
            window = torch.from_numpy(resized[start:start + SEQ_LEN]).unsqueeze(0).to(DEVICE)
            with autocast(device_type='cuda', dtype=torch.float16):
                out = model(window)
            out_np = out[0].float().cpu().numpy()
            if start == 0:
                L = out_np.shape[0]
                fb = out_np[min(LOSS_START_POS, L - 1)]
                for p in range(L - 1):
                    if 0 <= start + p < N - 1:
                        params[start + p] = out_np[p] if p >= LOSS_START_POS else fb
            t_idx = start + last_pos_offset
            if 0 <= t_idx < N - 1:
                params[t_idx] = out_np[-1]
    return params


def compute_gpe_proxy(model, proxy_scans, image_mm_to_tool, image_points_mm):
    """GPE proxy (mm) over a few full val scans + mean signed per-pair tz residual (pred-gt)."""
    gpe_errs, tz_resids = [], []
    tool_to_image_mm = np.linalg.inv(image_mm_to_tool)
    for frames, tforms in proxy_scans:
        local_params = _predict_local_params_np(model, frames)
        local_mats = np.stack([_np_params_to_matrix(p) for p in local_params], axis=0)
        pred_global = _np_accumulate_global(local_mats)
        gt_global = _np_tforms_to_global(tforms, image_mm_to_tool)
        n = min(len(pred_global), len(gt_global))
        pred_pts = np.einsum('nij,jp->nip', pred_global[:n], image_points_mm)[:, :3, :]
        gt_pts   = np.einsum('nij,jp->nip', gt_global[:n],  image_points_mm)[:, :3, :]
        gpe_errs.append(float(np.sqrt(((gt_pts - pred_pts) ** 2).sum(axis=1)).mean()))
        tinv = np.linalg.inv(tforms)
        gt_tz = np.array([(tool_to_image_mm @ (tinv[i] @ tforms[i + 1]) @ image_mm_to_tool)[2, 3]
                          for i in range(len(tforms) - 1)], dtype=np.float32)
        m = min(len(local_params), len(gt_tz))
        tz_resids.append(float((local_params[:m, 5] - gt_tz[:m]).mean()))
    gpe = float(np.mean(gpe_errs)) if gpe_errs else float('nan')
    tz_resid = float(np.mean(tz_resids)) if tz_resids else float('nan')
    return gpe, tz_resid

# =============================================================================
# LOSS
# =============================================================================
class PointBasedLoss(nn.Module):
    """Per-pair corner L1 + global-accumulation consistency + case-wise Pearson on the
    6-DoF trajectory (FiMoNet). The latter two penalize coherent tz drift that per-pair L1
    tolerates. Set w_global/w_pearson=0 for the original pure point loss."""
    def __init__(self, image_points_mm, start_pos=LOSS_START_POS, w_global=W_GLOBAL, w_pearson=W_PEARSON):
        super().__init__()
        # image_points_mm: (4, P) homogeneous; pre-compute once.
        self.register_buffer('image_points_mm', image_points_mm)
        self.start_pos = start_pos
        self.w_global = w_global
        self.w_pearson = w_pearson

    def forward(self, pred_params, target_points, target_params=None):
        # pred_params:   [B, L, 6]
        # target_points: [B, L+, 3, P]  (sliced to L below)
        # target_params: [B, L+, 6]     (GT 6-DoF; required for global/pearson terms)
        L = pred_params.shape[1]
        target = target_points[:, :L, :, :]
        pred   = params_to_corner_points(pred_params, self.image_points_mm)  # [B, L, 3, P]
        start  = min(self.start_pos, L - 1)
        local_loss = (pred[:, start:, :, :] - target[:, start:, :, :]).abs().mean()
        if target_params is None or (self.w_global == 0 and self.w_pearson == 0):
            return local_loss

        tgt_p = target_params[:, :L, :]
        loss = local_loss
        # fp32 for the accumulation chain (fp16 drifts over 19 matmuls).
        with torch.autocast(device_type='cuda', enabled=False):
            if self.w_global > 0:
                pts = self.image_points_mm.float()                                  # (4, P)
                pred_glob = accumulate_global_torch(params_to_matrix_torch(pred_params))   # [B,L,4,4]
                gt_glob   = accumulate_global_torch(params_to_matrix_torch(tgt_p))
                pred_gpts = torch.matmul(pred_glob, pts)[..., :3, :]                # [B,L,3,P]
                gt_gpts   = torch.matmul(gt_glob,   pts)[..., :3, :]
                global_loss = (pred_gpts[:, start:] - gt_gpts[:, start:]).abs().mean()
                loss = loss + self.w_global * global_loss
            if self.w_pearson > 0:
                loss = loss + self.w_pearson * self._pearson(pred_params[:, start:], tgt_p[:, start:])
        return loss

    @staticmethod
    def _pearson(pred, gt, eps=1e-6):
        # pred, gt: [B, L', 6]. Correlation across the time axis, per DoF, per sample.
        pred = pred.float(); gt = gt.float()
        pred = pred - pred.mean(dim=1, keepdim=True)
        gt   = gt   - gt.mean(dim=1, keepdim=True)
        num = (pred * gt).sum(dim=1)                                  # [B, 6]
        den = torch.sqrt((pred ** 2).sum(dim=1) * (gt ** 2).sum(dim=1) + eps)
        corr = num / (den + eps)                                     # [B, 6]
        return (1.0 - corr).mean()

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
            tforms = f['tforms'][m['start']:m['start'] + self.seq_len + 1].astype(np.float32)

        # Z-reverse aug at low p: reverse frame order so the model learns both sweep
        # directions (kills the tz sign prior). Targets recomputed from reversed poses.
        if self.augment and np.random.rand() < ZREVERSE_P:
            frames = frames[::-1].copy()
            poses = tforms[:self.seq_len][::-1]
            tforms = np.concatenate([poses, tforms[:1]], axis=0).copy()  # +1 throwaway pose

        seq = []
        for i in range(self.seq_len):
            img = frames[i]
            img = cv2.resize(img, (NET_W, NET_H))  # native 480x640, preserves speckle + aspect ratio
            if img.max() > 1.0:
                img = img / 255.0
            seq.append(np.expand_dims(img, axis=0).astype(np.float32))
        if self.augment:
            seq = self._augment_intensity(seq)

        seq_tensor = torch.tensor(np.stack(seq), dtype=torch.float32)
        target        = tforms_to_target_points(tforms, self.image_mm_to_tool, self.image_points_mm)
        target_params = tforms_to_6dof_params(tforms, self.image_mm_to_tool)
        return (seq_tensor,
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(target_params, dtype=torch.float32))


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
            img = cv2.resize(img, (NET_W, NET_H))  # native 480x640
            if img.max() > 1.0:
                img = img / 255.0
            seq.append(np.expand_dims(img, axis=0))
        seq_tensor = torch.tensor(np.stack(seq), dtype=torch.float32)
        tforms = self.t[start:start + self.seq_len + 1].astype(np.float32)
        target        = tforms_to_target_points(tforms, self.image_mm_to_tool, self.image_points_mm)
        target_params = tforms_to_6dof_params(tforms, self.image_mm_to_tool)
        return (seq_tensor,
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(target_params, dtype=torch.float32))

# =============================================================================
# TRAINING
# =============================================================================
def train_model(run_name, run_dir, model, train_loader, val_loader, epochs, image_points_mm_torch,
                proxy_scans=None, image_mm_to_tool=None, image_points_mm_np=None):
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
        for i, (seqs, target_pts, target_prm) in enumerate(train_loader):
            seqs       = seqs.to(DEVICE, non_blocking=True)
            target_pts = target_pts.to(DEVICE, non_blocking=True)
            target_prm = target_prm.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(seqs)             # [B, L, 6]
                loss    = criterion(outputs, target_pts, target_prm)  # point + global + pearson
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
            for seqs, target_pts, target_prm in val_loader:
                seqs       = seqs.to(DEVICE, non_blocking=True)
                target_pts = target_pts.to(DEVICE, non_blocking=True)
                target_prm = target_prm.to(DEVICE, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(seqs)
                    v_loss  = criterion(outputs, target_pts, target_prm)
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

        # Select checkpoint + step scheduler on the GPE proxy (per-pair val loss is
        # decoupled from GPE). tz_resid tracks the sign bias.
        gpe_proxy, tz_resid = (float('nan'), float('nan'))
        if proxy_scans:
            gpe_proxy, tz_resid = compute_gpe_proxy(model, proxy_scans, image_mm_to_tool, image_points_mm_np)
            model.train()  # restore (proxy set it to eval)

        select_metric = gpe_proxy if proxy_scans and not np.isnan(gpe_proxy) else avg_val
        scheduler.step(select_metric)
        print(f"[{get_time()}] Epoch {epoch+1}: Train={avg_train:.4f} | Val={avg_val:.4f} | "
              f"MaxCorner={avg_max_err:.4f} mm | GPEproxy={gpe_proxy:.2f} mm | tz_resid={tz_resid:+.4f} mm")

        if select_metric < best_val:
            best_val = select_metric
            torch.save(model.state_dict(), best_path)
            tag = f"GPEproxy {gpe_proxy:.2f} mm" if proxy_scans and not np.isnan(gpe_proxy) else f"val loss {avg_val:.4f}"
            print(f"[{get_time()}] Best saved: {best_path} ({tag}).")
        # Also save per-epoch in case a different selection is preferred later
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

    # Full-scan val subset for the per-epoch GPE proxy.
    proxy_scans = []
    seen_bases = []
    for fp, tp, cid in val_pairs:
        if len(proxy_scans) >= GPE_PROXY_SCANS:
            break
        try:
            with h5py.File(fp, 'r') as f:
                fr = np.array(f['frames'])
            with h5py.File(tp, 'r') as f:
                tf = np.array(f['tforms']).astype(np.float32)
            proxy_scans.append((fr, tf))
            seen_bases.append(cid)
        except Exception as e:
            print(f"[{get_time()}] proxy load skip {cid}: {e}")
    print(f"[{get_time()}] GPE-proxy scans ({len(proxy_scans)}): {', '.join(seen_bases)}")

    print(f"[{get_time()}] --- STEP 2: BUILD + TRAIN MODEL ---")
    print(f"[{get_time()}] Backbone: {BACKBONE}")
    model = FiMANetMamba6DOF(seq_len=SEQ_LEN, pair_encoder=True, pair_strides=PAIR_STRIDES,
                              backbone=BACKBONE, pool_size=POOL_SIZE, freeze_early=FREEZE_EARLY)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{get_time()}] Total params: {n_params/1e6:.1f}M ({n_train/1e6:.1f}M trainable) | "
          f"pool={POOL_SIZE}x{POOL_SIZE} freeze_early={FREEZE_EARLY} | "
          f"loss: point + {W_GLOBAL}*global + {W_PEARSON}*pearson | zrev_p={ZREVERSE_P}")
    best_path = train_model(run_name, run_dir, model, train_loader, val_loader, EPOCHS,
                            image_points_mm_torch,
                            proxy_scans=proxy_scans, image_mm_to_tool=image_mm_to_tool,
                            image_points_mm_np=image_points_mm)

    del train_loader, train_ds, val_loader, val_ds, val_datasets, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n[{get_time()}] Training done. Best checkpoint: {best_path}")
    print(f"[{get_time()}] Run `python eval_6dof.py --ckpt {best_path}` next to compute GPE/GLE/LPE/LLE.")
