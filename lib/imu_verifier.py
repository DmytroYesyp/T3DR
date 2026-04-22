"""
IMU-based z-displacement estimator used as an independent second opinion
at evaluation time (NOT during training).

The network predicts |dz| from images; this module predicts |dz| from
(synthetic, noisy) IMU via proper double integration in world frame.
The two estimates are then fused externally — see vis notebook.

Swap the IMUSimulator for a real sensor feed once available.
"""

import numpy as np


class IMUVerifier:
    def __init__(self, imu_simulator, dt=1 / 30.0, gravity=9.81):
        """
        Args:
            imu_simulator: an IMUSimulator (or anything with .generate(tforms)
                           returning [N, 6] of [accel_xyz, gyro_xyz])
            dt:      time between frames (seconds)
            gravity: gravity magnitude (m/s²); set 0 if accel is already
                     gravity-compensated
        """
        self.imu_sim = imu_simulator
        self.dt = dt
        self.gravity = np.array([0.0, 0.0, gravity], dtype=np.float64)

    def estimate_step_magnitude(self, tforms):
        """
        Estimate |dz| between the last two frames of the given sequence
        by double-integrating synthetic IMU over the whole window.

        Args:
            tforms: [N, 4, 4] with N ≥ 3
        Returns:
            float — estimated |dz| (same units as tforms translation)
        """
        imu = self.imu_sim.generate(tforms)        # [N, 6]
        accel_sensor = imu[:, :3].astype(np.float64)
        rotations = tforms[:, :3, :3].astype(np.float64)

        # Derotate accel to world frame and subtract gravity
        accel_world = np.zeros_like(accel_sensor)
        for i in range(len(accel_sensor)):
            accel_world[i] = rotations[i] @ accel_sensor[i] - self.gravity

        # Double integrate. v0 = 0 assumption → some drift, but relative
        # dz between the last two samples stays bounded for short windows.
        velocity = np.cumsum(accel_world * self.dt, axis=0)
        position = np.cumsum(velocity * self.dt, axis=0)

        return float(abs(position[-1, 2] - position[-2, 2]))

    def fuse(self, z_visual, z_imu, alpha=0.7):
        """
        Weighted fusion of the two |dz| estimates.
        alpha=1.0 → trust vision only;  alpha=0.0 → trust IMU only.
        Default 0.7 leans on vision (usually more accurate in-plane) and
        uses IMU to pull outlier predictions back toward physical plausibility.
        """
        return alpha * z_visual + (1.0 - alpha) * z_imu
