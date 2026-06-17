"""Loosely-coupled visual + inertial fusion via Error-State Kalman Filter.

The visual model supplies (signed Δz, σ); the ESKF in lib/eskf.py drives
high-frequency IMU integration and uses σ as time-varying measurement noise.
Swap the IMUSimulator for a real sensor stream once available.
"""

import numpy as np

from lib.eskf import ESKF


class IMUVerifier:
    def __init__(self, imu_simulator, dt=1 / 30.0, gravity=9.81,
                 default_visual_sigma=0.3, use_fixed_sigma=True):
        # default_visual_sigma: σ when use_fixed_sigma or sigma_visual is None. 0.3 mm
        #   is looser than the model's overconfident heteroscedastic output yet tight
        #   enough that vision still dominates IMU drift.
        # use_fixed_sigma: ignore caller σ — the heteroscedastic head can overfit
        #   log_var and feed garbage R to the filter.
        self.imu_sim = imu_simulator
        self.dt = dt
        self.gravity = gravity
        self.default_visual_sigma = default_visual_sigma
        self.use_fixed_sigma = use_fixed_sigma
        self.eskf = ESKF(dt=dt, gravity=gravity)

    def reset(self, init_position, init_rotation, init_velocity=None):
        self.eskf.reset(position=np.asarray(init_position, dtype=np.float64),
                        rotation=np.asarray(init_rotation, dtype=np.float64),
                        velocity=init_velocity)

    def precompute_imu(self, tforms):
        return self.imu_sim.generate(tforms)

    def step(self, accel, gyro, z_visual, sigma_visual=None):
        """Advance one frame interval; return signed Δz."""
        if self.use_fixed_sigma or sigma_visual is None:
            sigma_visual = self.default_visual_sigma
        z_before = float(self.eskf.p[2])
        self.eskf.predict(accel, gyro)
        self.eskf.update_dz(z_visual, sigma_visual)
        self.eskf.commit_anchor()
        return float(self.eskf.p[2]) - z_before
