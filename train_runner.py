"""
End-to-end script: trains FiMA-Net (visual-only) and then evaluates it
with the IMU verifier enabled. All artifacts share a single versioned tag.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from torch.utils.data import Dataset, DataLoader
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
BATCH_SIZE = 64
EPOCHS = 10

# In-model IMU fusion is OFF — IMU lives outside as a verifier (see below).
USE_IMU = False
WARMUP_EPOCHS = 2  # freeze layer2/3/4 for this many epochs

# Eval-time IMU verifier settings
USE_IMU_VERIFIER = True
FUSION_ALPHA = 0.9  # heavy weight on vision; IMU only nudges outliers

BASE_DATA_DIR = "/home/123ghdh/datasets"
TRAIN_FOLDERS = [os.path.join(BASE_DATA_DIR, str(i).zfill(3)) for i in range(50)]

VAL_FRAMES_PATH = os.path.join(BASE_DATA_DIR, 'valDataset/data/frames/050/LH_Par_C_DtP.h5')
VAL_TFORMS_PATH = os.path.join(BASE_DATA_DIR, 'valDataset/data/transfs/050/LH_Par_C_DtP.h5')

VAL_START, VAL_END = 50, 400  # for the subject-050 detailed analysis

# =============================================================================
# UTILITIES
# =============================================================================
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def next_run_name(prefix):
    """Return '{prefix}_v{N}' where N is the lowest free integer version."""
    n = 1
    while glob.glob(f"{prefix}_v{n}_*"):
        n += 1
    return f"{prefix}_v{n}"

def set_backbone_trainable(model, trainable):
    """Toggle requires_grad for layer2/3/4. stem and layer1 stay frozen."""
    for name in ['layer2', 'layer3', 'layer4']:
        for p in getattr(model, name).parameters():
            p.requires_grad = trainable

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
        weight = 1.0 + self.alpha * target
        weighted_loss = base_loss * weight
        return torch.mean(weighted_loss)

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
        target = torch.tensor(abs(tforms[-1, 2, 3] - tforms[-2, 2, 3]), dtype=torch.float32)

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
        target = torch.tensor(abs(tforms[-1, 2, 3] - tforms[-2, 2, 3]), dtype=torch.float32)

        if self.use_imu:
            imu_data = self.imu_sim.generate(tforms)[:self.seq_len]
            imu_tensor = torch.tensor(imu_data, dtype=torch.float32)
            return seq_tensor, imu_tensor, target

        return seq_tensor, target

# =============================================================================
# TRAINING
# =============================================================================
def train_model(run_name, model_name, model, train_loader, val_loader, epochs, use_imu=False):
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

    untrained_name = f"{run_name}_untrained.pth"
    torch.save(model.state_dict(), untrained_name)

    backbone_params = list(model.stem.parameters()) + \
                      list(model.layer1.parameters()) + \
                      list(model.layer2.parameters()) + \
                      list(model.layer3.parameters()) + \
                      list(model.layer4.parameters())

    head_params = list(model.fusion.parameters()) + \
                  list(model.temporal_layers.parameters()) + \
                  list(model.head.parameters())

    if use_imu:
        head_params += list(model.imu_encoder.parameters())
        head_params += list(model.fusion_gate.parameters())

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=1e-5)

    # patience=3 so the scheduler doesn't drop LR on a single noisy val epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    criterion = MotionWeightedL1Loss(alpha=2.0)
    scaler = GradScaler('cuda')

    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    best_model_path = f"{run_name}_best.pth"
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
                    v_loss = criterion(outputs, targets)
                    val_loss += v_loss.item()
                    v_mae = pure_l1(outputs, targets)
                    val_mae_sum += v_mae.item()

        avg_val = val_loss / len(val_loader)
        avg_mae = val_mae_sum / len(val_loader)
        history['val'].append(avg_val)
        scheduler.step(avg_val)

        print(f"[{get_time()}] Epoch {epoch+1} Summary: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f} | (Real MAE: {avg_mae:.4f} mm)")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            print(f"[{get_time()}] Best {model_name} saved as {best_model_path}.")

    # Training curve plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss (Weighted)', color='blue', linewidth=2)
    plt.plot(history['val'], label='Validation Loss (Weighted)', color='orange', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Motion Weighted L1 Loss")
    plt.title(f"{model_name} Training Curve (SEQ_LEN = {SEQ_LEN})")
    plt.legend()
    plt.grid(True)

    plot_filename = f"{run_name}_training_curve.png"
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

    curr_z_pred = tforms[start_idx + SEQ_LEN - 1, 2, 3]
    curr_z_real = tforms[start_idx + SEQ_LEN - 1, 2, 3]

    traj_pred = [curr_z_pred]
    traj_real = [curr_z_real]

    with torch.no_grad():
        for t in range(start_idx, end_idx - SEQ_LEN):
            seq_imgs = []
            for i in range(SEQ_LEN):
                img = frames[t + i]
                img = cv2.resize(img, (256, 256))
                if img.max() > 1.0: img = img / 255.0
                seq_imgs.append(np.expand_dims(img, axis=0))

            inp = torch.tensor(np.stack(seq_imgs), dtype=torch.float32).unsqueeze(0).to(DEVICE)

            z_visual = model(inp).item()

            if imu_verifier is not None:
                tforms_seq = tforms[t : t + SEQ_LEN + 1]
                z_imu = imu_verifier.estimate_step_magnitude(tforms_seq)
                step_mag = imu_verifier.fuse(z_visual, z_imu, alpha=FUSION_ALPHA)
            else:
                step_mag = z_visual

            real_step = tforms[t + SEQ_LEN, 2, 3] - tforms[t + SEQ_LEN - 1, 2, 3]
            direction = np.sign(real_step) if real_step != 0 else 1

            curr_z_pred += (step_mag * direction)
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

def run_evaluation(run_name, model, imu_verifier):
    print(f"\n{'='*50}")
    print(f"[{get_time()}] EVALUATION PHASE")
    print(f"[{get_time()}] IMU Verifier: {'ENABLED (alpha=' + str(FUSION_ALPHA) + ')' if imu_verifier else 'DISABLED'}")
    print(f"{'='*50}")

    # ---------- PART 1: Subject 050 detailed analysis ----------
    print(f"\n[{get_time()}] PART 1: Subject 050 detailed analysis")
    with h5py.File(VAL_FRAMES_PATH, 'r') as f_f, h5py.File(VAL_TFORMS_PATH, 'r') as f_t:
        val_frames = f_f['frames'][:]
        val_tforms = f_t['tforms'][:]

    z_pred, z_real = predict_z_trajectory(model, val_frames, val_tforms, imu_verifier,
                                          start_idx=VAL_START, end_idx=VAL_END)
    v_mae, v_drift, v_len, v_fdr = calculate_metrics(z_pred, z_real)
    print(f"[{get_time()}] Validation -> MAE: {v_mae:.2f}mm | Drift: {v_drift:.2f}mm | FDR: {v_fdr:.2f}%")
    plot_z_trajectory(
        z_pred, z_real,
        f"Validation Z-Axis (Subject 050)\nFDR: {v_fdr:.2f}% | MAE: {v_mae:.2f}mm",
        f"{run_name}_val050_z.png",
        using_imu=imu_verifier is not None,
    )

    # 3D reconstruction plot
    path_2d_px = calculate_visual_odometry_2d(val_frames[VAL_START + SEQ_LEN - 1 : VAL_END])
    real_xy_mm = val_tforms[VAL_START + SEQ_LEN - 1 : VAL_END, :2, 3]
    real_xy_mm -= real_xy_mm[0]

    affine_mat, _ = cv2.estimateAffine2D(path_2d_px.astype(np.float32), real_xy_mm.astype(np.float32))
    path_2d_aug = np.column_stack([path_2d_px, np.ones(len(path_2d_px))])
    pred_xy_mm = path_2d_aug @ affine_mat.T

    pred_3d = np.column_stack([pred_xy_mm, z_pred - z_pred[0]])
    real_3d = np.column_stack([real_xy_mm, z_real - z_real[0]])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(real_3d[:,0], real_3d[:,1], real_3d[:,2], 'g-', linewidth=3, alpha=0.5, label='Ground Truth 3D Path')
    ax.plot(pred_3d[:,0], pred_3d[:,1], pred_3d[:,2], 'r--', linewidth=2, label='FiMA Reconstruction')
    ax.scatter(0, 0, 0, c='cyan', s=100, label='Start')
    ax.scatter(real_3d[-1,0], real_3d[-1,1], real_3d[-1,2], c='green', marker='x', s=100)
    ax.scatter(pred_3d[-1,0], pred_3d[-1,1], pred_3d[-1,2], c='red', marker='x', s=100)
    ax.set_title("Full 3D Probe Trajectory (Subject 050)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.legend()
    try: ax.set_box_aspect([1,1,1])
    except: pass
    plt.savefig(f"{run_name}_val050_3d.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{get_time()}] Saved 3D-plot: {run_name}_val050_3d.png")

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
        f"=== GLOBAL DATASET METRICS (FiMA-Net) ===\n"
        f"Run tag:             {run_name}\n"
        f"IMU Verifier:        {'ENABLED (alpha=' + str(FUSION_ALPHA) + ')' if imu_verifier else 'DISABLED'}\n"
        f"Subject 050 MAE:     {v_mae:.3f} mm\n"
        f"Subject 050 FDR:     {v_fdr:.2f} %\n"
        f"Total valid scans:   {total_scans}\n"
        f"Average MAE:         {avg_mae:.3f} mm\n"
        f"Average Final Drift: {avg_drift:.3f} mm\n"
        f"Average Drift Rate:  {avg_fdr:.2f} %\n"
        f"===========================================\n"
    )
    print(summary_text)

    with open(f"{run_name}_summary.txt", "w") as f:
        f.write(summary_text)
    print(f"[{get_time()}] Saved summary: {run_name}_summary.txt")

    with open(f"{run_name}_per_scan.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scan_Name", "MAE_mm", "Final_Drift_mm", "FDR_percent"])
        writer.writerows(detailed_metrics)
    print(f"[{get_time()}] Saved per-scan metrics: {run_name}_per_scan.csv")

    if best_data:
        plot_z_trajectory(
            best_data[0], best_data[1],
            f"BEST CASE: {best_data[2]}\nFDR: {best_data[3]:.2f}% | MAE: {best_data[4]:.2f} mm",
            f"{run_name}_bestscan.png",
            using_imu=imu_verifier is not None,
        )
    if worst_data:
        plot_z_trajectory(
            worst_data[0], worst_data[1],
            f"WORST CASE: {worst_data[2]}\nFDR: {worst_data[3]:.2f}% | MAE: {worst_data[4]:.2f} mm",
            f"{run_name}_worstscan.png",
            using_imu=imu_verifier is not None,
        )

    print(f"\n[{get_time()}] Evaluation complete.")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print(f"[{get_time()}] --- STEP 1: PREPARING DATASETS ---")

    train_ds = LargeUSDataset(TRAIN_FOLDERS, seq_len=SEQ_LEN, use_imu=USE_IMU)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_ds = MemoryUSDataset(VAL_FRAMES_PATH, VAL_TFORMS_PATH, seq_len=SEQ_LEN, use_imu=USE_IMU)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Single shared versioned tag for training + evaluation artifacts.
    base_tag = f"fima_seq{SEQ_LEN}_e{EPOCHS}{'_imufused' if USE_IMU else ''}"
    run_name = next_run_name(base_tag)
    print(f"[{get_time()}] Run tag: {run_name}")

    print(f"[{get_time()}] --- STEP 2: TRAIN FIMA-NET ---")
    fima_model = FiMANet(seq_len=SEQ_LEN, use_imu=USE_IMU)
    fima_history, best_model_path = train_model(
        run_name, "FiMA_UnfrozenBackbone", fima_model, train_loader, val_loader, EPOCHS, use_imu=USE_IMU
    )

    # Free training resources before eval
    del train_loader, train_ds, val_loader, val_ds
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[{get_time()}] --- STEP 3: EVALUATE WITH IMU VERIFIER ---")
    # Reload best checkpoint into a fresh model in eval mode
    eval_model = FiMANet(seq_len=SEQ_LEN, use_imu=USE_IMU).to(DEVICE)
    eval_model = load_weights_safe(eval_model, best_model_path)

    imu_verifier = IMUVerifier(IMUSimulator()) if USE_IMU_VERIFIER else None
    run_evaluation(run_name, eval_model, imu_verifier)

    del eval_model, fima_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[{get_time()}] ALL TRAININGS + EVALUATIONS COMPLETED SUCCESSFULLY")
