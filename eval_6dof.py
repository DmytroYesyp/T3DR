"""Evaluate a 6-DoF FiMA-Net checkpoint against the official TUS-REC2024 metrics
(GPE, GLE, LPE, LLE). Self-contained: replicates the baseline's DDF computation
without requiring pytorch3d.

Modes:
    --mode pure    : all 6 DoF from the model (tx, ty, tz, rx, ry, rz)
    --mode hybrid  : model gives (rx, ry, rz, tz); Lucas-Kanade gives (tx, ty)
    --mode both    : runs both and writes a side-by-side comparison (default)

Usage:
    python eval_6dof.py --ckpt runs/.../fima_pair_mamba_6dof_v1_best.pth
    python eval_6dof.py --ckpt path/to/best.pth --mode hybrid
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import csv
import cv2
import glob
import h5py
import numpy as np
import torch
from datetime import datetime
from scipy.spatial.transform import Rotation

from models.fimanet_mamba_6dof import FiMANetMamba6DOF

SEQ_LEN = 20
PAIR_STRIDES = (1,)
IMG_H, IMG_W = 480, 640      # GT corners / calibration grid (NOT resized)
INFER_H, INFER_W = 240, 320  # network input — MUST match training NET_H/NET_W (half-res)

BASE_DATA_DIR = "/home/123ghdh/datasets"
VAL_FRAMES_ROOT = os.path.join(BASE_DATA_DIR, 'valDataset/data/frames')
VAL_TFORMS_ROOT = os.path.join(BASE_DATA_DIR, 'valDataset/data/transfs')
LANDMARK_ROOT   = os.path.join(BASE_DATA_DIR, 'valDataset/data/landmark')
CALIB_PATH      = os.path.join(BASE_DATA_DIR, 'calib_matrix.csv')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Calibration / geometry utilities (replicating baseline behavior)
# ---------------------------------------------------------------------------
def read_calib_matrices(filename_calib):
    tform_calib = np.empty((8, 4), np.float32)
    with open(filename_calib, 'r') as f:
        txt = [i.strip('\n').split(',') for i in f.readlines()]
        tform_calib[0:4, :] = np.array(txt[1:5]).astype(np.float32)
        tform_calib[4:8, :] = np.array(txt[6:10]).astype(np.float32)
    pixel_to_image_mm = tform_calib[0:4, :]
    image_mm_to_tool  = tform_calib[4:8, :]
    pixel_to_tool     = image_mm_to_tool @ pixel_to_image_mm
    return pixel_to_image_mm, image_mm_to_tool, pixel_to_tool


def reference_image_points(image_size=(IMG_H, IMG_W)):
    """Return homogeneous pixel coords for every pixel of an HxW image.
    Shape (4, H*W). Matches baseline.plot_functions.reference_image_points."""
    H, W = image_size
    pts = torch.flip(torch.cartesian_prod(
        torch.linspace(1, H, H),
        torch.linspace(1, W, W),
    ).t(), [0])  # (2, H*W)
    pts = torch.cat([
        pts,
        torch.zeros(1, pts.shape[1]),
        torch.ones(1, pts.shape[1]),
    ], dim=0)
    return pts  # (4, H*W)


# ---------------------------------------------------------------------------
# 6-DoF param ↔ 4x4 matrix conversion (scipy, ZYX Euler — matches baseline)
# ---------------------------------------------------------------------------
def infer_pool_size(state_dict, backbone):
    """Recover the pool grid from fusion.0.weight so old (2x2) and new ckpts both load."""
    chsum = 3584 if backbone == 'resnet50' else 896  # R18/R34: 128+256+512
    w = state_dict.get('fusion.0.weight')
    if w is None:
        return 2
    fusion_in = w.shape[1]
    area = fusion_in // chsum
    size = int(round(area ** 0.5))
    return max(1, size)


def infer_seq_len(state_dict, default=SEQ_LEN):
    """Recover seq_len from the positional-encoding buffer shape [1, seq_len, d]."""
    pe = state_dict.get('pos_encoder.pe')
    return int(pe.shape[1]) if pe is not None else default


def infer_bidirectional(state_dict):
    """Bidirectional checkpoints have temporal_layers.*.fwd/.bwd submodules."""
    return any('temporal_layers' in k and '.fwd.' in k for k in state_dict)


def params_to_matrix(params):
    """params: (6,) array (rz, ry, rx, tx, ty, tz). Returns (4, 4) image_mm transform."""
    rz, ry, rx, tx, ty, tz = params
    R = Rotation.from_euler('ZYX', [rz, ry, rx]).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = (tx, ty, tz)
    return T


def tforms_to_local_image_mm(tforms, image_mm_to_tool):
    """GT local transforms in image_mm frame. Returns (N-1, 4, 4)."""
    tool_to_image_mm = np.linalg.inv(image_mm_to_tool)
    tforms_inv = np.linalg.inv(tforms)
    N = len(tforms) - 1
    out = np.zeros((N, 4, 4), dtype=np.float32)
    for i in range(N):
        t_tool = tforms_inv[i] @ tforms[i + 1]
        out[i] = tool_to_image_mm @ t_tool @ image_mm_to_tool
    return out


def tforms_to_global_image_mm(tforms, image_mm_to_tool):
    """GT global transforms (frame_i -> frame_0) in image_mm. Returns (N-1, 4, 4)."""
    tool_to_image_mm = np.linalg.inv(image_mm_to_tool)
    tforms_inv = np.linalg.inv(tforms)  # world->tool per frame
    N = len(tforms) - 1
    out = np.zeros((N, 4, 4), dtype=np.float32)
    for i in range(1, len(tforms)):
        t_tool = tforms_inv[0] @ tforms[i]
        out[i - 1] = tool_to_image_mm @ t_tool @ image_mm_to_tool
    return out


def accumulate_global(locals_4x4):
    """Accumulate local transforms (frame_{i+1} -> frame_i) into globals (frame_{i+1} -> frame_0).
    locals_4x4: (N, 4, 4). Returns (N, 4, 4)."""
    N = locals_4x4.shape[0]
    out = np.zeros((N, 4, 4), dtype=np.float32)
    prev = np.eye(4, dtype=np.float32)
    for i in range(N):
        prev = prev @ locals_4x4[i]
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# DDF computation (matches baseline.Transf2DDFs)
# ---------------------------------------------------------------------------
def cal_allpts_DDF(transformations, pixel_to_image_mm, image_points):
    """transformations: (N, 4, 4) numpy.
    image_points: (4, P) numpy (pixel coords, homogeneous).
    pixel_to_image_mm: (4, 4) numpy.
    Returns: (N, 3, P) numpy displacements in mm."""
    pix_mm = pixel_to_image_mm @ image_points  # (4, P)
    transformed = np.einsum('nij,jp->nip', transformations, pix_mm)  # (N, 4, P)
    ddf = transformed[:, :3, :] - pix_mm[None, :3, :]  # (N, 3, P)
    return ddf


def cal_landmark_DDF(transformations_per_lm, landmarks, pixel_to_image_mm):
    """transformations_per_lm: (20, 4, 4) — the global/local transform corresponding to the
    landmark's frame_idx-1.
    landmarks: (20, 3) — (frame_idx, x, y) in pixel coords (1-based frame_idx).
    Returns: (3, 20) displacement vectors in mm."""
    out = np.zeros((3, len(landmarks)), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        pts = np.array([lm[1], lm[2], 0.0, 1.0], dtype=np.float32)
        pts_mm = pixel_to_image_mm @ pts
        out[:, i] = (transformations_per_lm[i] @ pts_mm)[:3] - pts_mm[:3]
    return out


def cal_dist(label, pred, mode='all'):
    if mode == 'all':
        return float(np.sqrt(((label - pred) ** 2).sum(axis=1)).mean())
    elif mode == 'landmark':
        return float(np.sqrt(((label - pred) ** 2).sum(axis=0)).mean())
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------
LOSS_START_POS = 5   # positions below this were masked from the training loss
WINDOW_BATCH = 16    # sliding windows per forward pass


def _resize_frames(frames):
    N = len(frames)
    resized = np.zeros((N, 1, INFER_H, INFER_W), dtype=np.float32)
    for i in range(N):
        img = cv2.resize(frames[i], (INFER_W, INFER_H))
        if img.max() > 1.0:
            img = img / 255.0
        resized[i, 0] = img.astype(np.float32)
    return resized


def predict_local_params(model, resized, seq_len=SEQ_LEN, avg_window=False):
    """Sliding-window prediction. resized: (N, 1, H, W). Returns (N-1, 6).
    avg_window=False: legacy — each transition read at its window's LAST output position,
    where the backward Mamba direction has zero future context.
    avg_window=True: average each transition over all window positions with >=5 frames of
    past AND future context (fallback: any supervised position, then the legacy early fill)."""
    N = len(resized)
    params = np.zeros((N - 1, 6), dtype=np.float32)
    starts = list(range(0, N - seq_len + 1))
    outs = np.zeros((len(starts), seq_len - 1, 6), dtype=np.float32)
    with torch.no_grad():
        for c in range(0, len(starts), WINDOW_BATCH):
            chunk = starts[c:c + WINDOW_BATCH]
            batch = torch.from_numpy(np.stack([resized[s:s + seq_len] for s in chunk])).to(DEVICE)
            outs[c:c + len(chunk)] = model(batch).float().cpu().numpy()

    first = outs[0]
    fallback = first[min(LOSS_START_POS, first.shape[0] - 1)]

    if not avg_window:
        for p in range(first.shape[0] - 1):
            if p < N - 1:
                params[p] = first[p] if p >= LOSS_START_POS else fallback
        for w, s in enumerate(starts):
            t = s + seq_len - 2
            if 0 <= t < N - 1:
                params[t] = outs[w][-1]
        return params

    p_hi = seq_len - 7  # >=5 future frames within the window
    sum_mid = np.zeros((N - 1, 6)); cnt_mid = np.zeros(N - 1, dtype=np.int64)
    sum_any = np.zeros((N - 1, 6)); cnt_any = np.zeros(N - 1, dtype=np.int64)
    for w, s in enumerate(starts):
        for p in range(LOSS_START_POS, seq_len - 1):
            t = s + p
            if t >= N - 1:
                break
            sum_any[t] += outs[w][p]; cnt_any[t] += 1
            if p <= p_hi:
                sum_mid[t] += outs[w][p]; cnt_mid[t] += 1
    for t in range(N - 1):
        if cnt_mid[t]:
            params[t] = sum_mid[t] / cnt_mid[t]
        elif cnt_any[t]:
            params[t] = sum_any[t] / cnt_any[t]
        else:
            params[t] = fallback
    return params


def predict_local_params_fullscan(model, resized):
    """Whole-scan single pass -> (N-1, 6). Needs the length-agnostic PositionalEncoding."""
    with torch.no_grad():
        seq = torch.from_numpy(np.ascontiguousarray(resized)).unsqueeze(0).to(DEVICE)
        out = model(seq)
    return out[0].float().cpu().numpy()  # (N-1, 6)


def invert_params_seq(params):
    """(M, 6) -> (M, 6): the 6-DoF params of each inverted transform."""
    out = np.zeros_like(params)
    for i, p in enumerate(params):
        Ti = np.linalg.inv(params_to_matrix(p))
        rz, ry, rx = Rotation.from_matrix(Ti[:3, :3]).as_euler('ZYX')
        out[i] = (rz, ry, rx, Ti[0, 3], Ti[1, 3], Ti[2, 3])
    return out


def predict_params_one(model, resized, seq_len, fullscan=False, avg_window=False, reverse_tta=False):
    """Per-model prediction with optional reverse TTA: predict the reversed scan, map its
    locals back to forward (reverse order + invert) and average — a direction-coherent tz
    bias flips sign under reversal and cancels."""
    def _run(r):
        return (predict_local_params_fullscan(model, r) if fullscan
                else predict_local_params(model, r, seq_len, avg_window))
    fwd = _run(resized)
    if not reverse_tta:
        return fwd
    rev = _run(resized[::-1].copy())
    return ((fwd + invert_params_seq(rev[::-1])) / 2.0).astype(np.float32)


def predict_global_local_matrices(models, frames, seq_lens, fullscan=False,
                                  avg_window=False, reverse_tta=False):
    """Ensemble-mean local params across models -> (global, local, params) matrices
    in the image_mm coordinate system."""
    N = len(frames)
    resized = _resize_frames(frames)
    all_params = [predict_params_one(m, resized, sl, fullscan, avg_window, reverse_tta)
                  for m, sl in zip(models, seq_lens)]
    local_params = np.mean(all_params, axis=0).astype(np.float32)
    local_mats = np.zeros((N - 1, 4, 4), dtype=np.float32)
    for i in range(N - 1):
        local_mats[i] = params_to_matrix(local_params[i])
    global_mats = accumulate_global(local_mats)
    return global_mats, local_mats, local_params


# ---------------------------------------------------------------------------
# Lucas-Kanade (hybrid mode): compute (tx_mm, ty_mm) per consecutive frame pair
# ---------------------------------------------------------------------------
_LK_FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def lucas_kanade_pair(prev_gray_u8, curr_gray_u8, pixel_to_image_mm):
    """Return (tx_mm, ty_mm) for the probe motion frame_{prev} -> frame_{curr}.
    Returns (None, None) when the flow can't be estimated reliably."""
    p0 = cv2.goodFeaturesToTrack(prev_gray_u8, mask=None, **_LK_FEATURE_PARAMS)
    if p0 is None or len(p0) < 10:
        return None, None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray_u8, curr_gray_u8, p0, None, **_LK_PARAMS)
    if p1 is None:
        return None, None
    good_new = p1[st.flatten() == 1]
    good_old = p0[st.flatten() == 1]
    if len(good_new) < 4:
        return None, None
    m, _ = cv2.estimateAffinePartial2D(good_old, good_new)
    if m is None:
        return None, None
    # m[:, 2] is feature motion in pixels. Probe motion in pixels is the negative.
    delta_pixel = np.array([-m[0, 2], -m[1, 2], 0.0], dtype=np.float32)
    delta_mm = pixel_to_image_mm[:3, :3] @ delta_pixel
    return float(delta_mm[0]), float(delta_mm[1])


def lk_translations_per_pair(frames, pixel_to_image_mm):
    """Per consecutive frame pair, return (N-1, 2) of (tx_mm, ty_mm) probe motion.
    Pairs where flow fails fall back to (0, 0) — the caller can decide what to do."""
    N = len(frames)
    out = np.zeros((N - 1, 2), dtype=np.float32)
    n_fail = 0
    prev_u8 = frames[0]
    if prev_u8.max() <= 1.0:
        prev_u8 = (prev_u8 * 255).astype(np.uint8)
    for i in range(N - 1):
        curr_u8 = frames[i + 1]
        if curr_u8.max() <= 1.0:
            curr_u8 = (curr_u8 * 255).astype(np.uint8)
        tx, ty = lucas_kanade_pair(prev_u8, curr_u8, pixel_to_image_mm)
        if tx is None:
            n_fail += 1
        else:
            out[i] = (tx, ty)
        prev_u8 = curr_u8
    return out, n_fail


def build_hybrid_local_mats(model_local_params, lk_tx_ty, model_local_mats):
    """For each pair, take rotations + tz from model, override tx, ty with LK.
    Falls back to model tx, ty where LK failed (signaled by zeros that came from fallback)."""
    N = len(model_local_params)
    out = np.zeros((N, 4, 4), dtype=np.float32)
    for i in range(N):
        params = model_local_params[i].copy()  # (rz, ry, rx, tx, ty, tz)
        # Override tx, ty with LK where it succeeded (non-zero); else keep model's.
        # Heuristic: lk_tx_ty[i] == (0, 0) means LK failed for this pair.
        if not (lk_tx_ty[i, 0] == 0.0 and lk_tx_ty[i, 1] == 0.0):
            params[3] = lk_tx_ty[i, 0]
            params[4] = lk_tx_ty[i, 1]
            out[i] = params_to_matrix(params)
        else:
            out[i] = model_local_mats[i]
    return out


# ---------------------------------------------------------------------------
# Landmark loader
# ---------------------------------------------------------------------------
def load_landmarks(landmark_root, subject_str, scan_name_no_ext):
    """Returns landmarks shape (20, 3) — (frame_idx, x, y), 1-based frame_idx."""
    # subject_str is like "050"; landmark file is landmark_050.h5
    lm_path = os.path.join(landmark_root, f"landmark_{subject_str}.h5")
    if not os.path.exists(lm_path):
        return None
    with h5py.File(lm_path, 'r') as f:
        if scan_name_no_ext not in f:
            return None
        return np.array(f[scan_name_no_ext])


# ---------------------------------------------------------------------------
# Val file discovery
# ---------------------------------------------------------------------------
def collect_val_files(n_per_subject=24):
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
                subj_pairs.append((os.path.join(sd, fname), tp, subj, fname[:-3]))
        pairs.extend(subj_pairs[:n_per_subject])
    return pairs


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, nargs='+',
                        help='Checkpoint path(s); pass several to ensemble (mean of predicted params).')
    parser.add_argument('--out-dir', default=None, help='Where to save metrics CSV.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of scans (debug).')
    parser.add_argument('--mode', choices=['pure', 'hybrid', 'both'], default='both',
                        help="pure: model gives all 6 DoF; hybrid: LK gives (tx, ty), model gives rest; both: comparison")
    parser.add_argument('--backbone', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18',
                        help="Must match the backbone used during training of the checkpoint.")
    parser.add_argument('--fullscan', action='store_true',
                        help="Feed the whole scan in one pass instead of sliding seq_len windows.")
    parser.add_argument('--avg-window', action='store_true',
                        help="Average each transition over all sliding-window positions with "
                             "bidirectional context instead of reading only the last position.")
    parser.add_argument('--reverse-tta', action='store_true',
                        help="Also predict the reversed scan, map back, and average (cancels "
                             "direction-coherent tz bias).")
    args = parser.parse_args()
    do_pure = args.mode in ('pure', 'both')
    do_hybrid = args.mode in ('hybrid', 'both')

    out_dir = args.out_dir or os.path.dirname(args.ckpt[0])
    os.makedirs(out_dir, exist_ok=True)
    tag = args.mode
    if args.fullscan:        tag += '_fullscan'
    if args.avg_window:      tag += '_avgwin'
    if args.reverse_tta:     tag += '_revtta'
    if len(args.ckpt) > 1:   tag += f'_ens{len(args.ckpt)}'
    csv_path = os.path.join(out_dir, f'6dof_eval_per_scan_{tag}.csv')

    print(f"[{get_time()}] Loading calibration...")
    pixel_to_image_mm, image_mm_to_tool, _ = read_calib_matrices(CALIB_PATH)
    image_points = reference_image_points((IMG_H, IMG_W)).numpy()  # (4, P)

    models, seq_lens = [], []
    for ck in args.ckpt:
        print(f"[{get_time()}] Loading model from {ck}... (backbone={args.backbone})")
        ckpt = torch.load(ck, map_location=DEVICE)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        pool_size = infer_pool_size(state_dict, args.backbone)
        seq_len = infer_seq_len(state_dict)
        bidirectional = infer_bidirectional(state_dict)
        print(f"[{get_time()}] Inferred from ckpt: pool_size={pool_size} seq_len={seq_len} bidirectional={bidirectional}")
        model = FiMANetMamba6DOF(seq_len=seq_len, pair_encoder=True, pair_strides=PAIR_STRIDES,
                                  backbone=args.backbone, pool_size=pool_size,
                                  bidirectional=bidirectional).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        seq_lens.append(seq_len)

    val_files = collect_val_files()
    if args.limit:
        val_files = val_files[:args.limit]
    print(f"[{get_time()}] Eval set: {len(val_files)} scans | "
          f"inference={'FULL-SCAN' if args.fullscan else 'sliding-window'} | "
          f"avg_window={args.avg_window} reverse_tta={args.reverse_tta} ensemble={len(models)}")

    metrics = {
        'pure':   {'gpe': [], 'gle': [], 'lpe': [], 'lle': []},
        'hybrid': {'gpe': [], 'gle': [], 'lpe': [], 'lle': []},
    }
    rows = []
    for idx, (frames_path, tforms_path, subj, scan_no_ext) in enumerate(val_files):
        try:
            with h5py.File(frames_path, 'r') as f:
                frames = np.array(f['frames'])
            with h5py.File(tforms_path, 'r') as f:
                tforms = np.array(f['tforms']).astype(np.float32)
            landmarks = load_landmarks(LANDMARK_ROOT, subj, scan_no_ext)
        except Exception as e:
            print(f"[{get_time()}] ERROR loading {subj}/{scan_no_ext}: {e}")
            continue

        # Model inference once
        pure_global, pure_local, pure_local_params = predict_global_local_matrices(
            models, frames, seq_lens, fullscan=args.fullscan,
            avg_window=args.avg_window, reverse_tta=args.reverse_tta)

        # GT once
        gt_local  = tforms_to_local_image_mm(tforms, image_mm_to_tool)
        gt_global = tforms_to_global_image_mm(tforms, image_mm_to_tool)
        gt_gp = cal_allpts_DDF(gt_global, pixel_to_image_mm, image_points)
        gt_lp = cal_allpts_DDF(gt_local,  pixel_to_image_mm, image_points)
        if landmarks is not None and len(landmarks) > 0:
            lm_frame_idx = (landmarks[:, 0] - 1).astype(np.int64).clip(0, pure_global.shape[0] - 1)
            gt_gl = cal_landmark_DDF(gt_global[lm_frame_idx], landmarks, pixel_to_image_mm)
            gt_ll = cal_landmark_DDF(gt_local [lm_frame_idx], landmarks, pixel_to_image_mm)
        else:
            lm_frame_idx = None
            gt_gl = gt_ll = None

        # Hybrid local matrices (optional)
        hybrid_local = hybrid_global = None
        n_lk_fail = 0
        if do_hybrid:
            lk_tx_ty, n_lk_fail = lk_translations_per_pair(frames, pixel_to_image_mm)
            hybrid_local = build_hybrid_local_mats(pure_local_params, lk_tx_ty, pure_local)
            hybrid_global = accumulate_global(hybrid_local)

        scan_label = f"{subj}/{scan_no_ext}"
        row = [scan_label]

        def _compute(pred_global, pred_local):
            pred_gp = cal_allpts_DDF(pred_global, pixel_to_image_mm, image_points)
            pred_lp = cal_allpts_DDF(pred_local,  pixel_to_image_mm, image_points)
            gpe = cal_dist(gt_gp, pred_gp, 'all')
            lpe = cal_dist(gt_lp, pred_lp, 'all')
            gle = lle = float('nan')
            if lm_frame_idx is not None:
                pred_gl = cal_landmark_DDF(pred_global[lm_frame_idx], landmarks, pixel_to_image_mm)
                pred_ll = cal_landmark_DDF(pred_local [lm_frame_idx], landmarks, pixel_to_image_mm)
                gle = cal_dist(gt_gl, pred_gl, 'landmark')
                lle = cal_dist(gt_ll, pred_ll, 'landmark')
            return gpe, gle, lpe, lle

        log_parts = [f"{idx+1}/{len(val_files)} {scan_label}"]
        if do_pure:
            gpe, gle, lpe, lle = _compute(pure_global, pure_local)
            metrics['pure']['gpe'].append(gpe);  metrics['pure']['lpe'].append(lpe)
            if not np.isnan(gle): metrics['pure']['gle'].append(gle)
            if not np.isnan(lle): metrics['pure']['lle'].append(lle)
            row += [gpe, gle, lpe, lle]
            log_parts.append(f"PURE GPE={gpe:.2f} GLE={gle:.2f}")
        if do_hybrid:
            gpe, gle, lpe, lle = _compute(hybrid_global, hybrid_local)
            metrics['hybrid']['gpe'].append(gpe);  metrics['hybrid']['lpe'].append(lpe)
            if not np.isnan(gle): metrics['hybrid']['gle'].append(gle)
            if not np.isnan(lle): metrics['hybrid']['lle'].append(lle)
            row += [gpe, gle, lpe, lle, n_lk_fail]
            log_parts.append(f"HYB  GPE={gpe:.2f} GLE={gle:.2f} (LK fail {n_lk_fail})")
        rows.append(row)
        print(f"[{get_time()}] " + " | ".join(log_parts))

    # CSV
    header = ['scan']
    if do_pure:   header += ['pure_GPE', 'pure_GLE', 'pure_LPE', 'pure_LLE']
    if do_hybrid: header += ['hybrid_GPE', 'hybrid_GLE', 'hybrid_LPE', 'hybrid_LLE', 'lk_fail_count']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n[{get_time()}] Per-scan metrics: {csv_path}")

    def m(x):
        return float(np.mean(x)) if x else float('nan')
    def s(x):
        return float(np.std(x)) if x else float('nan')

    print("\n=== 6-DoF Evaluation Summary ===")
    print(f"Scans: {len(val_files)}")
    print(f"{'Metric':<6} {'PURE (model 6-DoF)':<24} {'HYBRID (LK tx,ty)':<24}")
    print("-" * 60)
    for name in ('gpe', 'gle', 'lpe', 'lle'):
        cells = []
        for mode_key in ('pure', 'hybrid'):
            vals = metrics[mode_key][name]
            if vals:
                fmt = "{:.2f} ± {:.2f}" if name in ('gpe', 'gle') else "{:.4f} ± {:.4f}"
                cells.append(fmt.format(m(vals), s(vals)))
            else:
                cells.append('—')
        print(f"{name.upper():<6} {cells[0]:<24} {cells[1]:<24}")


if __name__ == '__main__':
    main()
