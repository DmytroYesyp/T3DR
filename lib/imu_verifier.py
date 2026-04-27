"""
Loosely-coupled visual + inertial fusion via Error-State Kalman Filter.

The visual model now supplies (|Δz|, σ); the ESKF in lib/eskf.py drives
high-frequency IMU integration and uses σ as time-varying measurement
noise. Replaces the previous open-loop double-integration + linear-α
fusion — both the sign of Δz and the vision/IMU weighting now come
from the filter's covariance rather than hand-tuned constants.

Swap the IMUSimulator for a real sensor stream once available.
"""

import numpy as np

from lib.eskf import ESKF


class IMUVerifier:
    def __init__(self, imu_simulator, dt=1 / 30.0, gravity=9.81,
                 default_visual_sigma=1.0):
        """
        Args:
            imu_simulator:        anything with .generate(tforms) → [N, 6]
            dt:                   frame interval (s)
            gravity:              gravity magnitude; pass 0 if the
                                  simulator is not adding it
            default_visual_sigma: fallback σ when the visual model does
                                  not predict its own uncertainty
        """
        self.imu_sim = imu_simulator
        self.dt = dt
        self.gravity = gravity
        self.default_visual_sigma = default_visual_sigma
        self.eskf = ESKF(dt=dt, gravity=gravity)

    def reset(self, init_position, init_rotation):
        self.eskf.reset(position=np.asarray(init_position, dtype=np.float64),
                        rotation=np.asarray(init_rotation, dtype=np.float64))

    def precompute_imu(self, tforms):
        return self.imu_sim.generate(tforms)

    def step(self, accel, gyro, z_visual_magnitude, sigma_visual=None):
        """Advance one frame interval; return signed Δz."""
        if sigma_visual is None:
            sigma_visual = self.default_visual_sigma
        z_before = float(self.eskf.p[2])
        self.eskf.predict(accel, gyro)
        self.eskf.update_dz_magnitude(z_visual_magnitude, sigma_visual)
        self.eskf.commit_anchor()
        return float(self.eskf.p[2]) - z_before
